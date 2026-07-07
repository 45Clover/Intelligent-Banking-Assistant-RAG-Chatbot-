import pickle
from sentence_transformers import SentenceTransformer
from ingest_pipeline import run_ingestion

# CONFIG
MODEL_NAME = "all-MiniLM-L6-v2"  # small, fast, runs fine on CPU
BASE_PATH = ".."  # same as ingest_pipeline: run this from inside Chunking folder


def generate_embeddings(chunks, model):
    """Takes list of chunk dicts (text + metadata), returns same list with 'embedding' added."""
    texts = [c["text"] for c in chunks]

    print(f"Embedding {len(texts)} chunks... (first run downloads the model, ~80MB)")
    embeddings = model.encode(texts, show_progress_bar=True)

    for chunk, embedding in zip(chunks, embeddings):
        chunk["embedding"] = embedding

    return chunks


if __name__ == "__main__":
    # Step 1: get chunks with metadata from your existing ingestion pipeline
    chunks = run_ingestion(base_path=BASE_PATH)
    print(f"Loaded {len(chunks)} chunks from ingestion pipeline\n")

    # Step 2: load the embedding model (downloads once, cached after that)
    model = SentenceTransformer(MODEL_NAME)

    # Step 3: generate embeddings for every chunk
    chunks_with_embeddings = generate_embeddings(chunks, model)

    # Step 4: save to disk so the vector DB step can load this directly
    with open("chunks_with_embeddings.pkl", "wb") as f:
        pickle.dump(chunks_with_embeddings, f)

    print(f"\n Saved {len(chunks_with_embeddings)} chunks with embeddings to chunks_with_embeddings.pkl")
    print(f"Embedding dimension: {len(chunks_with_embeddings[0]['embedding'])}")