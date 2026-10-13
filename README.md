# RAG Chat (versi simple)

Alurnya: dokumen dipotong jadi chunk, di-embed pakai Gemini, disimpan di ChromaDB. Saat ada pertanyaan, chunk yang paling mirip diambil lalu dikirim ke Gemini untuk dijawab.

## Cara menjalankan

Backend:

```bash
cd backend-simple
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # isi GEMINI_API_KEY
uvicorn main:app --reload       # dokumentasi API di http://localhost:8000/docs
```

Frontend (di terminal lain):

```bash
cd frontend-simple
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run streamlit_app/app.py   # http://localhost:8501
```

Tes backend (tidak butuh API key):

```bash
cd backend-simple
pytest -q
```

## Endpoint

| Method     | Path                           | Fungsi                                                   |
| ---------- | ------------------------------ | -------------------------------------------------------- |
| GET        | `/health`                      | cek server                                               |
| POST       | `/api/v1/documents`            | unggah dan index dokumen (nama file sama akan menimpa)   |
| GET        | `/api/v1/documents`            | daftar dokumen                                           |
| DELETE     | `/api/v1/documents/{doc_id}`   | hapus dokumen beserta chunk-nya                          |
| POST       | `/api/v1/chat/stream`          | tanya, jawaban bertahap (SSE: `meta`, `token`, `done`)   |
| POST       | `/api/v1/chat`                 | tanya, jawaban sekaligus                                 |
| GET/DELETE | `/api/v1/conversations[/{id}]` | riwayat percakapan                                       |

## Batasan

- Tidak pakai database SQL. Daftar dokumen diambil dari metadata chunk di ChromaDB, sedangkan riwayat percakapan hanya ada di memori sehingga hilang saat server restart (dokumen tetap ada).
- Tiap pertanyaan berdiri sendiri. Riwayat chat tidak ikut dikirim ke prompt, jadi pertanyaan lanjutan seperti "yang kedua apa?" tidak akan dimengerti.
- Pencarian hanya berdasarkan vektor, dan chunking per jumlah karakter (bukan per pasal atau butir).
- Indexing berjalan saat upload, jadi dokumen besar membuat request menunggu cukup lama.
- Hanya PDF, TXT, dan Markdown. PDF hasil scan ditolak karena tidak ada teks yang bisa dibaca.
- Belum ada autentikasi.
