import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from embed import create_embedding

load_dotenv()
urlqdrant= os.getenv("cluster_endpoint")
apiqdrant= os.getenv("api_qdrant_interview")
COLLECTION_NAME = "interview_knowledge"

client = QdrantClient(
    url=urlqdrant,
    api_key=apiqdrant
)

if not client.collection_exists(COLLECTION_NAME):

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=384,
            distance=Distance.COSINE
        )
    )

    print("Collection berhasil dibuat!")

else:

    print("Collection sudah tersedia.")

def retrieve_knowledge(query, limit=3):
    query_vector = create_embedding(query)
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector.tolist(),
        limit=limit
    )

    return results.points