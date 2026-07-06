
#USed PyPDF2 for text extraction this a test so it produces a txt file with what was extracted from the pdf

import re
from PyPDF2 import PdfReader


def clean_page_text(text: str) -> str:
    if not text:
        return ""

    # Remove repeated header/footer line: "Interac FAQ _ August 2024 ... Page X of 14"
    text = re.sub(r"Interac FAQ _ August 2024.*?Page \d+ of \d+", "", text, flags=re.DOTALL)

    # Collapse PDF's weird mid-word spacing artifacts, e.g. "e -Transfer" -> "e-Transfer"
    text = re.sub(r"\s+-\s*", "-", text)

    # Collapse multiple spaces/newlines into single space, but keep paragraph breaks
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)

    return text.strip()


reader = PdfReader("Interac_FAQ_2024.pdf")

with open("Interac_FAQ_2024_clean.txt", "w", encoding="utf-8") as output_file:
    for i, page in enumerate(reader.pages):
        raw_text = page.extract_text()
        cleaned = clean_page_text(raw_text)
        output_file.write(f"\n--- Page {i + 1} ---\n\n")
        output_file.write(cleaned if cleaned else "[No text found]")
        output_file.write("\n\n")

print("Cleaned extraction saved.")