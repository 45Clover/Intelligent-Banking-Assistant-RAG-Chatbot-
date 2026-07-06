from recursive_token_chunker import RecursiveTokenChunker

chunker = RecursiveTokenChunker(
    chunk_size=500,
    chunk_overlap=50,
)

with open("Interac_FAQ_2024_clean.txt", "r", encoding="utf-8") as f:
    text = f.read()

chunks = chunker.split_text(text)


# Remove chunks that are too short to be useful
chunks = [c for c in chunks if len(c.split()) > 15]

print(f"Total chunks after filtering: {len(chunks)}")

for i, chunk in enumerate(chunks):
    print(f"--- Chunk {i+1} ---")
    print(chunk)
    print()