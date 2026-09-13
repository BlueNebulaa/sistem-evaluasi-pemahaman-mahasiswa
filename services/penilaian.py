"""Penilaian jawaban interview dengan LLM (Gemini).

Satu panggilan LLM memetakan transkrip ke tiap pertanyaan sekaligus memberi skor,
sehingga tidak perlu node terpisah untuk memecah transkrip per soal.
"""
import json

import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted, GoogleAPIError

import config
from rubrics import QUESTIONS, format_rubrics

_model = None


def _get_model():
    """Model dibuat sekali saat pertama dipakai."""
    global _model
    if _model is None:
        if not config.GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY belum diisi di .env")
        genai.configure(api_key=config.GEMINI_API_KEY)
        _model = genai.GenerativeModel(config.GEMINI_MODEL)
    return _model


PROMPT = """You are an expert technical interviewer evaluating a candidate.

Below is the full transcript of a technical interview, followed by the questions
and their scoring rubrics.

TRANSCRIPT:
{transcript}

QUESTIONS AND RUBRICS:
{rubrics}

TASK:
- For each question, locate the part of the transcript that answers it
- Assign a score from 0 to 4 strictly based on that question's rubric
- Do NOT infer information that is not explicitly stated in the transcript
- If no relevant answer is found, return score 0
- Provide a brief justification referencing the rubric
- Return VALID JSON ONLY

JSON FORMAT:
{{"items": [{{"no": 1, "score": 0, "justification": "short explanation"}}]}}
"""


def _fallback(reason: str):
    return [{"no": q["no"], "score": 0, "justification": reason} for q in QUESTIONS]


def _wrap(items):
    total = sum(int(i.get("score", 0)) for i in items)
    return {"items": items, "total": total, "max_total": len(QUESTIONS) * 4}


def score_all(transcript: str) -> dict:
    """Nilai seluruh pertanyaan dalam satu panggilan LLM."""
    if not transcript or not transcript.strip():
        return _wrap(_fallback("Transcript empty. Manual review required."))

    # Sengaja di luar try: kalau API key belum diisi, itu salah konfigurasi
    # dan harus kelihatan sebagai error, bukan disamarkan jadi skor 0.
    model = _get_model()

    prompt = PROMPT.format(transcript=transcript, rubrics=format_rubrics())

    try:
        raw = model.generate_content(prompt).text.strip()
        payload = raw[raw.find("{"): raw.rfind("}") + 1]
        parsed = json.loads(payload)

        items = parsed.get("items")
        if not isinstance(items, list) or not items:
            raise ValueError("Invalid JSON structure")

    except ResourceExhausted:
        items = _fallback("AI quota exceeded. Manual review required.")
    except (GoogleAPIError, json.JSONDecodeError, ValueError, AttributeError):
        items = _fallback("AI evaluation failed. Manual review required.")

    return _wrap(items)
