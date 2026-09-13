"""API evaluasi pemahaman mahasiswa. Satu endpoint: upload mp4 -> JSON hasil."""
import os
import shutil
import tempfile

from fastapi import FastAPI, UploadFile

import pipeline

app = FastAPI(title="Sistem Evaluasi Pemahaman Mahasiswa")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/evaluate")
async def evaluate(file: UploadFile):
    """Upload satu video interview, dapat satu JSON berisi semua hasil analisis.

    Catatan: proses ini sinkron dan bisa memakan waktu beberapa menit.
    Untuk video panjang, naikkan timeout di sisi client dan uvicorn.
    """
    suffix = os.path.splitext(file.filename or "")[1] or ".mp4"

    fd, video_path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "wb") as tmp:
        shutil.copyfileobj(file.file, tmp)

    try:
        return pipeline.evaluate(video_path)
    finally:
        if os.path.exists(video_path):
            os.unlink(video_path)
