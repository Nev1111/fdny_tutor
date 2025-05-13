import fitz  # PyMuPDF
import os

pdf_path = "Engines-Complete-Updated-and-Issued-8-5-2021.pdf"

def extract_text_from_pdf(path):
    doc = fitz.open(path)
    all_text = ""
    for i, page in enumerate(doc):
        text = page.get_text()
        print(f"\n--- Page {i + 1} ---\n{text[:300]}")
        all_text += text
    return all_text

if __name__ == "__main__":
    if not os.path.exists(pdf_path):
        print(f"❌ PDF not found at: {pdf_path}")
    else:
        content = extract_text_from_pdf(pdf_path)
        print("\n✅ Finished reading PDF.")
