import os
from dotenv import load_dotenv
from qdrant import retrieve_knowledge
from schemas import AssessmentResult

load_dotenv()

# Pilih LLM lewat .env:
#   LLM_PROVIDER=ollama      -> model lokal via Ollama (default)
#   LLM_PROVIDER=openrouter  -> OpenRouter
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower()

# Untuk penilaian, temperature rendah supaya skor jawaban yang sama konsisten.
TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0"))

TOP_K = int(os.getenv("RAG_TOP_K", "3"))


def build_llm():
    """Buat LLM yang outputnya sudah dalam bentuk AssessmentResult."""
    if LLM_PROVIDER == "ollama":
        from langchain_ollama import ChatOllama

        model_name = os.getenv("OLLAMA_MODEL", "qwen3.5:4b")
        llm = ChatOllama(
            model=model_name,
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            temperature=TEMPERATURE,
            # Default context Ollama lebih kecil dari prompt kita (3 referensi,
            # sebagian chunk sampai ~6.500 karakter). Kalau kurang, prompt
            # terpotong diam-diam tanpa error.
            num_ctx=int(os.getenv("OLLAMA_NUM_CTX", "8192")),
            # Mode thinking Qwen3 jauh lebih lambat; nyalakan lewat .env
            # kalau ingin membandingkan kualitas.
            reasoning=os.getenv("OLLAMA_REASONING", "false").lower() == "true",
        )
        # json_schema = constrained decoding di Ollama: output dipaksa
        # mengikuti schema, jadi JSON selalu valid walau modelnya kecil.
        structured = llm.with_structured_output(AssessmentResult, method="json_schema")
        return structured, model_name

    if LLM_PROVIDER == "openrouter":
        from langchain_openrouter import ChatOpenRouter

        # ChatOpenRouter hanya membaca env OPENROUTER_API_KEY. Di Windows nama
        # env tidak case-sensitive sehingga "openrouter_api_key" kebetulan ikut
        # terbaca, tapi di Linux/Docker tidak — jadi key diteruskan eksplisit.
        api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("openrouter_api_key")
        if not api_key:
            raise RuntimeError(
                "OpenRouter API key belum diset. Isi OPENROUTER_API_KEY "
                "(atau openrouter_api_key) di file .env."
            )
        model_name = os.getenv(
            "OPENROUTER_MODEL",
            "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free"
        )
        llm = ChatOpenRouter(
            model=model_name,
            api_key=api_key,
            temperature=TEMPERATURE,
            reasoning={"effort": "high", "summary": "auto"}
        )
        # Model :free tidak mendukung response_format, jadi schema dikirim
        # sebagai tool yang wajib dipanggil (function_calling).
        structured = llm.with_structured_output(AssessmentResult, method="function_calling")
        return structured, model_name

    raise ValueError(
        f"LLM_PROVIDER tidak dikenal: {LLM_PROVIDER!r}. Pilih 'ollama' atau 'openrouter'."
    )


structured_llm, MODEL_NAME = build_llm()

# Provider :free di OpenRouter sering penuh ("ResourceExhausted") dan Ollama
# bisa gagal kalau model sedang di-load; coba ulang beberapa kali.
assessor = structured_llm.with_retry(stop_after_attempt=4)


def retrieve(question: str, top_k: int = TOP_K) -> list[dict]:
    """Ambil referensi dari knowledge base Qdrant berdasarkan pertanyaan."""
    return [
        {**result.payload, "similarity": result.score}
        for result in retrieve_knowledge(question, limit=top_k)
    ]


def format_references(references: list[dict]) -> str:
    reference_context = ""

    for i, ref in enumerate(references, start=1):

        reference_context += f"""
Reference {i}

Section:
{ref.get("section", "")}

Title:
{ref.get("title", "")}

Content:
{ref.get("content", "")}

Similarity Score:
{ref["similarity"]}

--------------------------------
"""
    return reference_context

system_prompt = """
Anda adalah sistem AI untuk melakukan assessment
jawaban interview mahasiswa.

Tugas Anda adalah menilai jawaban mahasiswa berdasarkan:

1. Pertanyaan interview.
2. Informasi referensi dari knowledge base yang
   disertakan bersama pertanyaan.

Reference dari knowledge base hanya
digunakan sebagai acuan.

Jangan mengharuskan jawaban mahasiswa memiliki
kata-kata yang sama persis dengan reference.

Tidak semua reference relevan dengan pertanyaan.
Gunakan hanya poin reference yang benar-benar
menjawab pertanyaan; abaikan reference lainnya.

Nilai jawaban berdasarkan:

1. Kesesuaian jawaban dengan pertanyaan.
2. Kelengkapan informasi.
3. Kejelasan jawaban.
4. Relevansi penjelasan.
5. Keselarasan dengan poin penting pada reference.

Yang dinilai adalah isi, bukan panjang jawaban.

Langkah penilaian:

1. Tentukan poin utama yang seharusnya ada untuk
   menjawab pertanyaan (dari reference yang relevan).
2. Cek apakah jawaban benar-benar menjawab pertanyaan.
3. Hitung poin utama yang ada di jawaban dan dijelaskan
   dengan benar.
4. Pilih kategori dari rubrik di bawah, lalu tentukan
   angka di dalam rentang kategori tersebut.

Rubrik score (pilih tepat satu kategori):

90-100 Sangat Baik   : hampir semua poin utama ada,
                       benar, jelas, disertai alasan
                       atau contoh.
80-89  Baik          : sebagian besar poin utama ada
                       dan dijelaskan dengan benar.
70-79  Cukup         : beberapa poin utama benar, tetapi
                       penjelasannya masih kurang.
60-69  Kurang        : dua atau lebih poin benar, tetapi
                       hanya disebut singkat.
40-59  Sangat Kurang : hanya satu poin benar, disebut
                       umum tanpa penjelasan.
16-39  Salah konsep  : menjawab pertanyaan, tetapi
                       alasan atau penjelasannya keliru.
0-15   Tidak menjawab: tidak relevan dengan pertanyaan,
                       atau hanya pernyataan umum yang
                       tidak menjawab yang ditanyakan.

Contoh kalibrasi (pertanyaan berbeda, hanya sebagai
patokan skor):

Pertanyaan: Mengapa kualitas data penting dalam KA?
- "AI sekarang sudah sangat pintar."
  -> 5 (Tidak menjawab)
- "Supaya AI bisa berjalan lebih cepat di komputer."
  -> 25 (Salah konsep)
- "Supaya hasil AI tidak salah."
  -> 50 (Sangat Kurang: satu poin, tanpa penjelasan)
- "Supaya hasil AI akurat dan tidak bias, dan
  datanya harus relevan."
  -> 65 (Kurang: beberapa poin, hanya disebut)
- "Model KA belajar dari data, sehingga data yang
  tidak lengkap atau bias membuat model meniru pola
  yang salah. Akibatnya hasilnya tidak akurat dan
  bisa merugikan kelompok tertentu. Karena itu data
  harus akurat, lengkap, relevan, dan representatif."
  -> 90 (Sangat Baik)

Berikan hasil assessment dengan field:

- feedback: penjelasan kualitas jawaban, sebutkan
  poin utama yang ada dan yang tidak ada
- strengths: daftar kelebihan jawaban
- weaknesses: daftar kekurangan jawaban
- recommendation: saran perbaikan
- score: angka 0-100 sesuai rubrik, konsisten dengan
  feedback, strengths, dan weaknesses di atas

(GUNAKAN BAHASA INDONESIA)
"""

def run_assessment(question, student_answer, top_k: int = TOP_K) -> dict:
    """Retrieve referensi, lalu nilai jawaban.

    Retrieval selalu dijalankan oleh kode (bukan diputuskan model), jadi
    penilaian tidak mungkin terjadi tanpa referensi.
    """
    references = retrieve(question, top_k=top_k)
    result = assessor.invoke([
        ("system", system_prompt),
        ("user", f"""
Pertanyaan Interview:
{question}

Jawaban Mahasiswa:
{student_answer}

Referensi dari knowledge base:
{format_references(references)}
"""),
    ])
    return {
        "result": result,
        "references": references,
        "provider": LLM_PROVIDER,
        "model": MODEL_NAME,
    }

def asses_answer(question, student_answer) -> AssessmentResult:
    return run_assessment(question, student_answer)["result"]

if __name__ == "__main__":

    question = """
    Mengapa transparansi penting dalam penerapan
    kecerdasan artifisial?
    """

    student_answer = """
    karena agar tidak terjadi kesalah pahaman atau bias
    terhadap suatu permasalahan atau sebuah kesimpulan.
    """

    result = asses_answer(
        question,
        student_answer
    )

    print(f"\n===== ASSESSMENT RESULT ({LLM_PROVIDER}: {MODEL_NAME}) =====\n")
    print(result.model_dump_json(indent=2))
