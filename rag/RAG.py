import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.tools import tool
from langchain.agents import create_agent
from qdrant import retrieve_knowledge

load_dotenv()
GROQ_API_KEY = os.getenv("groq_api")
MODEL_NAME = "openai/gpt-oss-120b"

model = ChatGroq(
    model=MODEL_NAME,
    api_key=GROQ_API_KEY,
    temperature=0.2
)

@tool
def retrieve(question: str, top_k: int = 3) -> str:
    """
    Mengambil informasi referensi dari knowledge base
    Qdrant berdasarkan pertanyaan interview.
    """

    results = retrieve_knowledge(
        question,
        limit=top_k
    )

    reference_context = ""

    for i, result in enumerate(results, start=1):

        payload = result.payload

        reference_context += f"""
Reference {i}

Section:
{payload.get("section", "")}

Title:
{payload.get("title", "")}

Content:
{payload.get("content", "")}

Similarity Score:
{result.score}

--------------------------------
"""
    return reference_context

system_prompt = """
Anda adalah sistem AI untuk melakukan assessment
jawaban interview mahasiswa.

Tugas Anda adalah menilai jawaban mahasiswa berdasarkan:

1. Pertanyaan interview.
2. Informasi referensi yang diperoleh dari knowledge base.

WAJIB menggunakan tool retrieve terlebih dahulu
sebelum melakukan assessment.

Reference yang diperoleh dari knowledge base hanya
digunakan sebagai acuan.

Jangan mengharuskan jawaban mahasiswa memiliki
kata-kata yang sama persis dengan reference.

Nilai jawaban berdasarkan:

1. Kesesuaian jawaban dengan pertanyaan.
2. Kelengkapan informasi.
3. Kejelasan jawaban.
4. Relevansi penjelasan.
5. Keselarasan dengan poin penting pada reference.

Berikan score dari 0 sampai 100.

Interpretasi umum score:

90-100 = Sangat Baik
80-89  = Baik
70-79  = Cukup
60-69  = Kurang
0-59   = Sangat Kurang

Berikan output dalam format:

Score: [0-100]

Feedback:
[Penjelasan kualitas jawaban]

Strengths:
- [kelebihan]
- [kelebihan]

Weaknesses:
- [kekurangan]
- [kekurangan]

Recommendation:
[Saran perbaikan]

(GUNAKAN BAHAS INDONESIA)
"""

agent = create_agent(
    model=model,
    tools=[retrieve],
    system_prompt=system_prompt
)

def asses_answer(question, student_answer):
    response = agent.invoke({
        "messages": [{
            "role": "user",
            "content": f"""
Pertanyaan Interview:
{question}

Jawaban Mahasiswa:
{student_answer}

Gunakan tool retrieve untuk mencari referensi yang relevan sebelum melakukan assessment.
"""
        }]
    })
    return response["messages"][-1].content

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

    print("\n===== ASSESSMENT RESULT =====\n")
    print(result)