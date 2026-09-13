"""Ekstraksi audio dari video. Node paling awal di pipeline."""
import os
import shutil
import subprocess
import tempfile

import config


def extract_audio(video_path: str) -> str:
    """Ambil audio dari mp4 jadi wav mono 16kHz. Mengembalikan path wav."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg tidak ditemukan di PATH. Install ffmpeg dulu.")

    fd, wav_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)

    result = subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", video_path,
            "-vn",                            # buang video
            "-ac", "1",                       # mono
            "-ar", str(config.SAMPLE_RATE),   # 16kHz, sesuai kebutuhan librosa & whisper
            wav_path,
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        if os.path.exists(wav_path):
            os.unlink(wav_path)
        raise RuntimeError("ffmpeg gagal: " + result.stderr[-500:])

    return wav_path
