"""Transkripsi audio dengan faster-whisper."""
from faster_whisper import WhisperModel

import config

_model = None


def _get_model():
    """Model di-load sekali lalu dipakai ulang. Load ulang tiap request itu mahal."""
    global _model
    if _model is None:
        _model = WhisperModel(
            config.WHISPER_MODEL,
            device=config.WHISPER_DEVICE,
            compute_type=config.WHISPER_COMPUTE_TYPE,
        )
    return _model


def transcribe(wav_path: str) -> dict:
    """Hasil: text lengkap, segments (start/end/text), dan durasi audio."""
    segments_iter, info = _get_model().transcribe(wav_path)

    segments = [
        {"start": s.start, "end": s.end, "text": s.text.strip()}
        for s in segments_iter
    ]

    return {
        "text": " ".join(s["text"] for s in segments).strip(),
        "segments": segments,
        "duration": float(getattr(info, "duration", 0.0)),
    }
