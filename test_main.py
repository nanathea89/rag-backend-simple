"""Tes backend. Gemini diganti model palsu jadi tidak perlu API key. Jalankan: pytest -q"""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import FakeListChatModel

import main

SOP = Path(__file__).resolve().parents[1] / "SOP Pengadaan Barang.pdf"


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "CHROMA_DIR", tmp_path / "chroma")
    monkeypatch.setattr(main, "get_embeddings", lambda: DeterministicFakeEmbedding(size=32))
    monkeypatch.setattr(main, "get_llm", lambda: FakeListChatModel(responses=["Ini jawaban tes."]))
    main.get_store.cache_clear()
    main.conversations.clear()
    yield TestClient(main.app)
    main.get_store.cache_clear()


def upload(client, name="cuti.txt", text="Karyawan berhak cuti 12 hari per tahun."):
    return client.post("/api/v1/documents", files={"file": (name, text.encode(), "text/plain")})


def events(resp):
    out, ev = [], None
    for line in resp.text.splitlines():
        if line.startswith("event:"):
            ev = line[6:].strip()
        elif line.startswith("data:"):
            out.append((ev, json.loads(line[5:])))
    return out


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_upload_list_delete(client):
    r = upload(client)
    assert r.json()["response_code"] == "00" and r.json()["data"]["doc_chunks"] >= 1
    docs = client.get("/api/v1/documents").json()["data"]
    assert [d["doc_name"] for d in docs] == ["cuti.txt"] and docs[0]["doc_status"] == "indexed"

    assert client.delete(f"/api/v1/documents/{docs[0]['doc_id']}").json()["response_code"] == "00"
    assert client.get("/api/v1/documents").json()["data"] == []
    assert client.delete("/api/v1/documents/tidakada").status_code == 404


def test_reupload_same_name_replaces(client):
    upload(client, text="versi lama")
    upload(client, text="versi baru")
    docs = client.get("/api/v1/documents").json()["data"]
    assert len(docs) == 1 and docs[0]["doc_chunks"] == 1


def test_unsupported_and_empty_file(client):
    assert client.post("/api/v1/documents", files={"file": ("a.exe", b"x")}).status_code == 400
    assert upload(client, "kosong.txt", "   ").json()["response_code"] == "06"


def test_stream_answer_sources_and_history(client):
    upload(client)
    r = client.post("/api/v1/chat/stream", json={"question": "Berapa hari cuti?"})
    evs = events(r)
    assert [e for e, _ in evs][0] == "meta" and evs[-1][0] == "done"
    assert evs[0][1]["sources"][0]["source"] == "cuti.txt"
    assert "".join(d["text"] for e, d in evs if e == "token") == "Ini jawaban tes."

    cnv = evs[-1][1]["cnv_id"]
    detail = client.get(f"/api/v1/conversations/{cnv}").json()["data"]
    assert [m["msg_role"] for m in detail["messages"]] == ["user", "assistant"]
    assert detail["messages"][1]["msg_sources"][0]["source"] == "cuti.txt"

    # pertanyaan kedua harus masuk ke percakapan yang sama
    evs2 = events(client.post("/api/v1/chat/stream", json={"question": "Lagi?", "cnv_id": cnv}))
    assert evs2[-1][1]["cnv_id"] == cnv
    assert len(client.get(f"/api/v1/conversations/{cnv}").json()["data"]["messages"]) == 4


def test_no_documents_does_not_call_llm(client, monkeypatch):
    monkeypatch.setattr(main, "get_llm", lambda: pytest.fail("LLM tidak seharusnya dipanggil"))
    evs = events(client.post("/api/v1/chat/stream", json={"question": "Halo?"}))
    assert "".join(d["text"] for e, d in evs if e == "token") == main.NO_CONTEXT_ANSWER


def test_llm_failure_sends_error_and_saves_nothing(client, monkeypatch):
    upload(client)

    def boom():
        raise RuntimeError("429 RESOURCE_EXHAUSTED detail internal")

    monkeypatch.setattr(main, "get_llm", boom)
    evs = events(client.post("/api/v1/chat/stream", json={"question": "cuti?"}))
    assert evs[-1][0] == "error" and "Kuota" in evs[-1][1]["message"]
    assert "internal" not in evs[-1][1]["message"]
    assert client.get("/api/v1/conversations").json()["data"] == []


def test_unknown_conversation_and_validation(client):
    r = client.post("/api/v1/chat/stream", json={"question": "x", "cnv_id": 99})
    assert r.status_code == 404 and r.json()["response_code"] == "02"
    r = client.post("/api/v1/chat/stream", json={"question": "   "})
    assert r.status_code == 400 and r.json()["response_code"] == "01"


def test_chat_non_stream_and_delete_conversation(client):
    upload(client)
    data = client.post("/api/v1/chat", json={"question": "cuti?"}).json()["data"]
    assert data["answer"] == "Ini jawaban tes." and data["sources"]
    assert client.get("/api/v1/conversations").json()["data"][0]["cnv_id"] == data["cnv_id"]
    client.delete(f"/api/v1/conversations/{data['cnv_id']}")
    assert client.get("/api/v1/conversations").json()["data"] == []


@pytest.mark.skipif(not SOP.exists(), reason="PDF contoh tidak ada")
def test_pdf_has_page_numbers(client):
    r = client.post("/api/v1/documents", files={"file": (SOP.name, SOP.read_bytes(), "application/pdf")})
    assert r.json()["data"]["doc_chunks"] > 1
    evs = events(client.post("/api/v1/chat/stream", json={"question": "pengadaan barang"}))
    assert all(s["page"] >= 1 for s in evs[0][1]["sources"])


def test_failed_embedding_leaves_no_partial_chunks(client, monkeypatch):
    """Batch pertama berhasil tapi batch kedua gagal, chunk batch pertama harus ikut dihapus."""
    monkeypatch.setattr(main, "EMBED_BATCH_SIZE", 1)
    monkeypatch.setattr(main, "CHUNK_SIZE", 40)
    monkeypatch.setattr(main, "CHUNK_OVERLAP", 0)
    real = main.get_store().add_documents
    calls = []

    def flaky(batch):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("embedding error internal")
        return real(batch)

    monkeypatch.setattr(main.get_store(), "add_documents", flaky)
    r = upload(client, text="kalimat pertama yang cukup panjang. kalimat kedua yang juga panjang. kalimat ketiga.")
    assert len(calls) == 2 and r.status_code == 503 and "internal" not in r.text
    assert main.get_store().get()["ids"] == []


def test_conversations_listed_newest_first(client):
    for q in ("pertama", "kedua"):
        events(client.post("/api/v1/chat/stream", json={"question": q}))
    assert [c["cnv_title"] for c in client.get("/api/v1/conversations").json()["data"]] == ["kedua", "pertama"]
