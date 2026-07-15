import os
import pickle
import chromadb

#  CONFIG
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DB_PATH = os.path.join(BASE_DIR, "chroma_db")          # folder where the DB will live on disk
COLLECTION_NAME = "banking_kb"                         # name of your knowledge base collection
PKL_PATH = os.path.join(BASE_DIR, "chunks_with_embeddings.pkl") # Forces the script to look in its own folder

def load_chunks(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def build_vector_db(chunks):
    # Persistent client = saves to disk in DB_PATH, survives between runs
    client = chromadb.PersistentClient(path=DB_PATH)

    # get_or_create so re-running this script doesn't error if it already exists
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    ids = []
    documents = []
    embeddings = []
    metadatas = []

    for i, chunk in enumerate(chunks):
        ids.append(f"chunk_{i}")
        documents.append(chunk["text"])
        embeddings.append(chunk["embedding"].tolist())  # Chroma wants plain lists, not numpy arrays
        metadatas.append(chunk["metadata"])

    # add everything in one batch call
    BATCH_SIZE = 5000  # safely under the 5461 max

    for start in range(0, len(ids), BATCH_SIZE):
        end = start + BATCH_SIZE
        collection.add(
            ids=ids[start:end],
            documents=documents[start:end],
            embeddings=embeddings[start:end],
            metadatas=metadatas[start:end],
        )


    return collection

if __name__ == "__main__":
    chunks = load_chunks(PKL_PATH)
    print(f"Loaded {len(chunks)} chunks with embeddings")

    collection = build_vector_db(chunks)
    print(f"Stored {collection.count()} chunks in ChromaDB collection '{COLLECTION_NAME}'")
    print(f"Database saved to ./{DB_PATH}")