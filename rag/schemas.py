from pydantic import BaseModel, Field


class AssessmentResult(BaseModel):
    """Hasil assessment jawaban interview mahasiswa."""

    # Urutan field = urutan model menulis JSON. score sengaja di akhir supaya
    # model menganalisis jawaban dulu (feedback, strengths, weaknesses) sebelum
    # menentukan angka — penting saat thinking dimatikan.
    feedback: str = Field(
        description="Penjelasan kualitas jawaban mahasiswa dalam Bahasa Indonesia."
    )
    strengths: list[str] = Field(
        description="Daftar kelebihan jawaban, satu poin per item."
    )
    weaknesses: list[str] = Field(
        description="Daftar kekurangan jawaban, satu poin per item."
    )
    recommendation: str = Field(
        description="Saran perbaikan agar jawaban lebih lengkap dan tepat."
    )
    score: int = Field(
        ge=0,
        le=100,
        description="Skor jawaban 0-100 sesuai rubrik penilaian.",
    )
