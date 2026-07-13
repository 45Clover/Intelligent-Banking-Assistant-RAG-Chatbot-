import os
from pathlib import Path
from PyPDF2 import PdfReader
from recursive_token_chunker import RecursiveTokenChunker

# ---- CONFIG ----
# Map each folder to a category label. Add more folders here as your corpus grows.
SOURCE_FOLDERS = {
    "FAQ's-SBI": "faq",
    "RBI-policies": "policy",
}

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
MIN_WORDS = 15  # filter out chunks too short to be useful

chunker = RecursiveTokenChunker(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)


def extract_text_from_pdf(pdf_path):
    """Extract raw text from a PDF using PyPDF2."""
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text


def clean_text(text):
    """
    Basic cleaning pass. Extend this with whatever you already
    built for header-stripping / OCR artifact fixes on the FAQ doc.
    """
    lines = text.split("\n")
    cleaned_lines = [line.strip() for line in lines if line.strip()]
    return "\n".join(cleaned_lines)


def process_pdf(pdf_path, category, source_folder):
    """Extract, clean, and chunk a single PDF. Returns list of chunk dicts."""
    raw_text = extract_text_from_pdf(pdf_path)
    cleaned_text = clean_text(raw_text)
    raw_chunks = chunker.split_text(cleaned_text)

    filtered_chunks = [c for c in raw_chunks if len(c.split()) > MIN_WORDS]

    chunk_records = []
    for i, chunk_text in enumerate(filtered_chunks):
        chunk_records.append({
            "text": chunk_text,
            "metadata": {
                "source_file": pdf_path.name,
                "source_folder": source_folder,
                "category": category,
                "chunk_index": i,
            }
        })
    return chunk_records


def run_ingestion(base_path="."):
    """Walk all configured folders, process every PDF, return combined chunk list."""
    all_chunks = []
    base = Path(base_path)

    for folder_name, category in SOURCE_FOLDERS.items():
        folder_path = base / folder_name
        if not folder_path.exists():
            print(f"⚠ Folder not found, skipping: {folder_path}")
            continue

        pdf_files = list(folder_path.glob("*.pdf"))
        print(f" {folder_name}: found {len(pdf_files)} PDF(s)")

        for pdf_path in pdf_files:
            chunks = process_pdf(pdf_path, category, folder_name)
            print(f"   → {pdf_path.name}: {len(chunks)} chunks")
            all_chunks.extend(chunks)

    return all_chunks


if __name__ == "__main__":
    chunks = run_ingestion(base_path="..")
    print(f"\nTotal chunks across all documents: {len(chunks)}")

    # Write everything to one text file (all_chunks_output.txt) for inspection
    with open("all_chunks_output.txt", "w", encoding="utf-8") as out:
        for i, c in enumerate(chunks):
            out.write(f"--- Chunk {i+1} | {c['metadata']['category']} | {c['metadata']['source_file']} ---\n")
            out.write(c["text"])
            out.write("\n\n")

    print("Saved to all_chunks_output.txt")