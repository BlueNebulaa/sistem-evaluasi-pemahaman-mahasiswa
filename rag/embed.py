from fastembed import TextEmbedding

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

embedding_model = TextEmbedding(
    model_name=MODEL_NAME
)

def create_embedding(text):
    """
    Mengubah teks menjadi vector embedding.
    """
    embedding = list(
        embedding_model.embed([text])
    )[0]

    return embedding