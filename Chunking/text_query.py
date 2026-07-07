#this for testing purposes btw nothing to do with main code
# i messed up the testing directory structure

import chromadb
from sentence_transformers import SentenceTransformer

DB_PATH = "chroma_db"
COLLECTION_NAME = "banking_kb"
MODEL_NAME = "all-MiniLM-L6-v2"


def query_kb(question, top_k=3):
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_collection(name=COLLECTION_NAME)
    model = SentenceTransformer(MODEL_NAME)

    query_embedding = model.encode(question).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
    )

    print(f"\nQuestion: {question}\n")
    for i in range(len(results["documents"][0])):
        text = results["documents"][0][i]
        meta = results["metadatas"][0][i]
        distance = results["distances"][0][i]
        print(f"--- Match {i+1} (distance: {distance:.4f}) ---")
        print(f"Source: {meta['source_file']} | Category: {meta['category']}")
        print(text)
        print()


if __name__ == "__main__":
    query_kb("Can I send money in a currency other than Canadian dollars?")