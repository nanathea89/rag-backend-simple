import io
import json
import logging
import os
import time
import uuid
from functools import cache
from pathlib import Path
from typing import Any, Iterator, Optional

from dotenv import load_dotenv
from fastapi import APIRouter, FastAPI, File, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field
from pypdf import PdfReader

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-3.5-flash-lite")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-2")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "768"))
CHROMA_DIR = BASE_DIR / os.getenv("CHROMA_DIR", "data/chroma_db")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
RETRIEVER_K = int(os.getenv("RETRIEVER_K", "8"))

# embedding Gemini versi gratis dibatasi sekitar 100 teks per menit,
# jadi chunk dikirim per 50 dan diulang kalau kena limit
EMBED_BATCH_SIZE = 50
EMBED_RETRIES = 3
EMBED_RETRY_SECONDS = 20
MAX_UPLOAD_BYTES = 20 * 1024 * 1024

NO_CONTEXT_ANSWER = "Maaf, jawaban tidak tersedia dalam dokumen yang ada."
PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Anda adalah asisten yang menjawab pertanyaan hanya berdasarkan konteks dokumen yang diberikan. "
            "Jawab dengan detail dan akurat, dalam bahasa yang sama dengan pertanyaan. "
            f'Jika jawabannya tidak ada dalam konteks, jawab: "{NO_CONTEXT_ANSWER}" dan jangan mengarang.',
        ),
        ("human", "Konteks:\n{context}\n\nPertanyaan: {question}\n\nJawaban:"),
    ]
)

logger = logging.getLogger("rag-simple")


class ApiError(Exception):
    def __init__(self, message: str, code: str = "99", status: int = 400):
        super().__init__(message)
        self.message = message
        self.code = code
        self.status = status


def ok(data: Any = None) -> dict:
    return {"response_code": "00", "response_message": "Success", "data": data}


def error_message(e: Exception) -> str:
    # detail error aslinya cukup masuk log, user tidak perlu lihat
    logger.error("Gagal: %s", e)
    if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
        return "Kuota Gemini sedang habis, coba lagi beberapa saat."
    return "Gagal menghubungi Gemini. Periksa GEMINI_API_KEY dan koneksi internet."


# Objek Gemini dan Chroma dibuat saat pertama kali dipakai saja
@cache
def get_embeddings():
    if not GEMINI_API_KEY:
        raise ApiError("GEMINI_API_KEY belum diisi di backend-simple/.env", "10", 503)
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=GEMINI_API_KEY,
        output_dimensionality=EMBEDDING_DIM,
    )


@cache
def get_llm():
    if not GEMINI_API_KEY:
        raise ApiError("GEMINI_API_KEY belum diisi di backend-simple/.env", "10", 503)
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(model=LLM_MODEL, google_api_key=GEMINI_API_KEY)


@cache
def get_store() -> Chroma:
    # disimpan di disk, jadi dokumen tidak hilang kalau server restart
    return Chroma(
        collection_name="documents",
        embedding_function=get_embeddings(),
        persist_directory=str(CHROMA_DIR),
    )


def read_pages(filename: str, content: bytes) -> list[Document]:
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        reader = PdfReader(io.BytesIO(content))
        pages = []
        for number, page in enumerate(reader.pages, start=1):
            pages.append(Document(page_content=page.extract_text() or "", metadata={"page": number}))
    elif ext in (".txt", ".md", ".markdown"):
        text = content.decode("utf-8", errors="replace")
        pages = [Document(page_content=text, metadata={"page": 0})]
    else:
        raise ApiError("Format tidak didukung. Gunakan PDF, TXT, atau Markdown.", "06")

    pages = [p for p in pages if p.page_content.strip()]
    if not pages:
        raise ApiError("Tidak ada teks yang bisa dibaca (PDF hasil scan butuh OCR).", "06")
    return pages


def add_in_batches(store: Chroma, chunks: list[Document]) -> None:
    for start in range(0, len(chunks), EMBED_BATCH_SIZE):
        batch = chunks[start : start + EMBED_BATCH_SIZE]
        for attempt in range(EMBED_RETRIES):
            try:
                store.add_documents(batch)
                break
            except Exception as e:
                rate_limited = "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e)
                if not rate_limited or attempt == EMBED_RETRIES - 1:
                    raise
                time.sleep(EMBED_RETRY_SECONDS)


def list_documents() -> list[dict]:
    # tidak ada tabel dokumen, daftarnya dihitung dari metadata chunk di Chroma
    docs: dict[str, dict] = {}
    for meta in get_store().get(include=["metadatas"])["metadatas"]:
        doc_id = meta["doc_id"]
        if doc_id not in docs:
            docs[doc_id] = {
                "doc_id": doc_id,
                "doc_name": meta["source"],
                "doc_chunks": 0,
                "doc_status": "indexed",
                "doc_error": None,
            }
        docs[doc_id]["doc_chunks"] += 1
    return list(docs.values())


# Riwayat percakapan disimpan di memori, hilang kalau server restart
conversations: dict[int, dict] = {}


def save_exchange(cnv_id: Optional[int], question: str, answer: str, sources: list[dict]) -> int:
    if cnv_id is None:
        cnv_id = max(conversations, default=0) + 1
        conversations[cnv_id] = {"cnv_id": cnv_id, "cnv_title": question[:50], "messages": []}
    messages = conversations[cnv_id]["messages"]
    messages.append({"msg_role": "user", "msg_content": question, "msg_sources": []})
    messages.append({"msg_role": "assistant", "msg_content": answer, "msg_sources": sources})
    return cnv_id


def to_source(doc: Document) -> dict:
    return {
        "source": doc.metadata["source"],
        "page": doc.metadata.get("page") or None,
        "snippet": doc.page_content[:200],
    }


def format_docs(docs: list[Document]) -> str:
    return "\n\n".join(f"[{d.metadata['source']}]\n{d.page_content}" for d in docs)


def answer_events(question: str, cnv_id: Optional[int]) -> Iterator[dict]:
    """Urutan event: meta (sumber), token (potongan jawaban) berkali-kali, lalu done.
    Percakapan baru disimpan setelah jawaban selesai."""
    docs = get_store().similarity_search(question, k=RETRIEVER_K)
    sources = [to_source(d) for d in docs]
    yield {"event": "meta", "data": {"search_query": question, "sources": sources}}

    parts: list[str] = []
    if docs:
        chain = PROMPT | get_llm() | StrOutputParser()
        for token in chain.stream({"context": format_docs(docs), "question": question}):
            parts.append(token)
            yield {"event": "token", "data": {"text": token}}
    else:
        # belum ada dokumen, tidak perlu panggil LLM
        parts.append(NO_CONTEXT_ANSWER)
        yield {"event": "token", "data": {"text": NO_CONTEXT_ANSWER}}

    answer = "".join(parts)
    cnv_id = save_exchange(cnv_id, question, answer, sources)
    yield {"event": "done", "data": {"cnv_id": cnv_id, "answer": answer}}


def sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    cnv_id: Optional[int] = None

    model_config = {"str_strip_whitespace": True}


router = APIRouter(prefix="/api/v1")


def check_conversation(cnv_id: Optional[int]) -> None:
    if cnv_id is not None and cnv_id not in conversations:
        raise ApiError("Percakapan tidak ditemukan", "02", 404)


@router.post("/chat/stream", summary="Tanya dokumen, jawaban bertahap (SSE)")
def chat_stream(payload: ChatRequest):
    # dicek sebelum stream mulai supaya errornya berupa JSON biasa
    check_conversation(payload.cnv_id)

    def body() -> Iterator[str]:
        try:
            for ev in answer_events(payload.question, payload.cnv_id):
                yield sse(ev["event"], ev["data"])
        except ApiError as e:
            yield sse("error", {"message": e.message, "code": e.code})
        except Exception as e:
            yield sse("error", {"message": error_message(e), "code": "99"})

    return StreamingResponse(body(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


@router.post("/chat", summary="Tanya dokumen, jawaban sekaligus")
def chat(payload: ChatRequest):
    check_conversation(payload.cnv_id)
    meta, answer, cnv_id = {}, "", None
    try:
        for ev in answer_events(payload.question, payload.cnv_id):
            if ev["event"] == "meta":
                meta = ev["data"]
            elif ev["event"] == "done":
                answer = ev["data"]["answer"]
                cnv_id = ev["data"]["cnv_id"]
    except ApiError:
        raise
    except Exception as e:
        raise ApiError(error_message(e), "10", 503)
    return ok({"cnv_id": cnv_id, "answer": answer, **meta})


@router.get("/conversations")
def list_conversations():
    items = [{"cnv_id": c["cnv_id"], "cnv_title": c["cnv_title"]} for c in conversations.values()]
    # yang terbaru di atas
    return ok(list(reversed(items)))


@router.get("/conversations/{cnv_id}")
def get_conversation(cnv_id: int):
    check_conversation(cnv_id)
    return ok(conversations[cnv_id])


@router.delete("/conversations/{cnv_id}")
def delete_conversation(cnv_id: int):
    check_conversation(cnv_id)
    del conversations[cnv_id]
    return ok()


@router.get("/documents")
def documents():
    return ok(list_documents())


@router.post("/documents", summary="Unggah dan index dokumen (PDF, TXT, Markdown)")
def upload_document(file: UploadFile = File(...)):
    content = file.file.read()
    if len(content) > MAX_UPLOAD_BYTES:
        raise ApiError("File terlalu besar (maks 20 MB)", "06")

    name = Path(file.filename or "dokumen").name
    pages = read_pages(name, content)
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(pages)

    doc_id = uuid.uuid4().hex[:8]
    for chunk in chunks:
        chunk.metadata.update({"doc_id": doc_id, "source": name})

    store = get_store()
    # nama file sama dianggap versi baru, jadi yang lama dihapus dulu
    store.delete(where={"source": name})
    try:
        add_in_batches(store, chunks)
    except Exception as e:
        # buang chunk yang sudah sempat masuk
        store.delete(where={"doc_id": doc_id})
        raise ApiError(error_message(e), "10", 503)
    return ok({"doc_id": doc_id, "doc_name": name, "doc_chunks": len(chunks), "doc_status": "indexed"})


@router.delete("/documents/{doc_id}")
def delete_document(doc_id: str):
    if not any(d["doc_id"] == doc_id for d in list_documents()):
        raise ApiError("Dokumen tidak ditemukan", "02", 404)
    get_store().delete(where={"doc_id": doc_id})
    return ok()


app = FastAPI(title="RAG Chat (simple)")
app.include_router(router)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.exception_handler(ApiError)
async def api_error_handler(request: Request, exc: ApiError):
    return JSONResponse(
        status_code=exc.status,
        content={"response_code": exc.code, "response_message": exc.message, "data": None},
    )


@app.exception_handler(RequestValidationError)
async def validation_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=400,
        content={
            "response_code": "01",
            "response_message": "Input tidak valid (pertanyaan 1-2000 karakter)",
            "data": None,
        },
    )
