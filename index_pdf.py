import os
from dotenv import load_dotenv
from extract_pdf import extract_text_from_pdf
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import DocArrayInMemorySearch
import pickle

load_dotenv()
pdf_path = "Engines-Complete-Updated-and-Issued-8-5-2021.pdf"
VECTOR_STORE_FILE = "fdny_docarray.pkl"

# Extract and chunk
print("📖 Extracting text...")
text = extract_text_from_pdf(pdf_path)
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
docs = splitter.create_documents([text])

# Embed and store
print("🔢 Generating embeddings...")
embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
vectorstore = DocArrayInMemorySearch.from_documents(docs, embeddings)

# Save vectorstore
with open(VECTOR_STORE_FILE, "wb") as f:
    pickle.dump(vectorstore, f)

print("✅ Saved vectorstore to fdny_docarray.pkl")
