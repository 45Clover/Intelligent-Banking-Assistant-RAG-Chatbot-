
import pickle
import chromadb

#  CONFIG
DB_PATH = "chroma_db"          # folder where the DB will live on disk
COLLECTION_NAME = "banking_kb" # name of your knowledge base collection
PKL_PATH = "chunks_with_embeddings.pkl"


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
    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    return collection

if __name__ == "__main__":
    chunks = load_chunks(PKL_PATH)
    print(f"Loaded {len(chunks)} chunks with embeddings")

    collection = build_vector_db(chunks)
    print(f"Stored {collection.count()} chunks in ChromaDB collection '{COLLECTION_NAME}'")
    print(f"Database saved to ./{DB_PATH}")