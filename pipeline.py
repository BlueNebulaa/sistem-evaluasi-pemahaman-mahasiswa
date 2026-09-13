"""Pipeline evaluasi: satu video masuk, satu dict hasil keluar.

Dijalankan berurutan (sekuensial). Tidak ada scheduler, queue, atau DAG —
untuk skala MVP, urutan pemanggilan fungsi biasa sudah cukup.
"""
import os

import config
from services import (
    eye_tracking,
    media,
    nonverbal,
    object_detection,
    penilaian,
    transcription,
)


def safe(name, fn, *args, **kwargs):
    """Jalankan satu node. Kalau gagal, node lain tetap jalan."""
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        print("[WARN] node '{}' gagal: {}: {}".format(name, type(e).__name__, e))
        return {"error": "{}: {}".format(type(e).__name__, e)}


def evaluate(video_path: str) -> dict:
    """Jalankan semua node analisis, gabungkan jadi satu JSON."""
    # Tidak dibungkus safe(): tanpa audio hampir semua node kehilangan input,
    # jadi lebih baik gagal cepat dengan pesan yang jelas.
    wav = media.extract_audio(video_path)

    try:
        tr = safe("transcription", transcription.transcribe, wav)
        transcript = tr.get("text", "")
        segments = tr.get("segments", [])
        duration = tr.get("duration", 0.0)

        nv = safe(
            "nonverbal",
            nonverbal.analyze_audio_nonverbal,
            wav, transcript, duration,
        )
        sc = safe("scoring", penilaian.score_all, transcript)

        eye = safe(
            "eye_tracking",
            eye_tracking.analyze_eye,
            video_path, config.EYE_FRAME_SKIP, config.EYE_MAX_FRAMES,
        )
        dia = safe("diarization", object_detection.diarize_interview, wav, segments)

        # diarization menempelkan label speaker langsung ke objek segments,
        # jadi hasilnya sudah ikut terbawa di bagian "transcription".
        dia.pop("segments", None)

        return {
            "duration_sec": round(duration, 2),
            "transcription": tr,
            "nonverbal": nv,
            "scoring": sc,
            "cheating_detection": {
                "eye_tracking": eye,
                "diarization": dia,
            },
        }
    finally:
        if os.path.exists(wav):
            os.unlink(wav)
