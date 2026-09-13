"""Konfigurasi terpusat. Semua nilai yang mungkin di-tuning ada di sini."""
import os
from dotenv import load_dotenv

load_dotenv()

# ===== LLM =====
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# ===== Transcription =====
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")

# ===== Eye tracking =====
# Naikkan MAX_FRAMES untuk video panjang. Dengan FRAME_SKIP=2 dan video 30fps,
# nilai 1000 hanya menganalisis ~33 detik pertama.
EYE_MAX_FRAMES = int(os.getenv("EYE_MAX_FRAMES", "1000"))
EYE_FRAME_SKIP = int(os.getenv("EYE_FRAME_SKIP", "2"))

# ===== Audio =====
SAMPLE_RATE = 16000
