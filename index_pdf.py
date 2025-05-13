import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from extract_pdf import extract_text_from_pdf

load_dotenv()
pdf_path = "Engines-Complete-Updated-and-Issued-8-5-2021.pdf"

persist_dir = "chroma_store"

def chunk_and_embed(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    chunks = splitter.create_documents([text])

    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory=persist_dir)
    vectorstore.persist()
    print(f"✅ Stored {len(chunks)} chunks in ChromaDB")

if __name__ == "__main__":
    print("📖 Extracting text from PDF...")
    text = extract_text_from_pdf(pdf_path)
    chunk_and_embed(text)
