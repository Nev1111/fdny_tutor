import os
from dotenv import load_dotenv
from extract_pdf import extract_text_from_pdf
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import pickle

load_dotenv()
pdf_path = "Engines-Complete-Updated-and-Issued-8-5-2021.pdf"
VECTOR_STORE_FILE = "fdny_faiss_index"
CHUNKS_FILE = "fdny_chunks.pkl"

# Chunk and embed
print("📖 Extracting text from PDF...")
text = extract_text_from_pdf(pdf_path)
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
docs = splitter.create_documents([text])

print("🔢 Generating embeddings...")
embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
faiss_index = FAISS.from_documents(docs, embeddings)

print("💾 Saving FAISS index to disk...")
faiss_index.save_local(VECTOR_STORE_FILE)

# Optional: Save chunks (for debug or future use)
with open(CHUNKS_FILE, "wb") as f:
    pickle.dump(docs, f)

print("✅ FAISS indexing complete!")
