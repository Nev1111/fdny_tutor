import os
from extract_pdf import extract_text_from_pdf

# Parameters
CHUNK_SIZE = 500  # number of words per chunk
OVERLAP = 50      # overlapping words for context

pdf_path = "Engines-Complete-Updated-and-Issued-8-5-2021.pdf"

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

if __name__ == "__main__":
    if not os.path.exists(pdf_path):
        print(f"❌ PDF not found: {pdf_path}")
    else:
        print("📖 Extracting text from PDF...")
        full_text = extract_text_from_pdf(pdf_path)
        chunks = chunk_text(full_text)
        print(f"✅ Created {len(chunks)} chunks.")
        print("\n🔍 Preview of first chunk:\n")
        print(chunks[0][:700], "...\n")
