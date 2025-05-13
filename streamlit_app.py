import os
import streamlit as st
from dotenv import load_dotenv
from extract_pdf import extract_text_from_pdf
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

# Load environment
load_dotenv()

# Rebuild the vectorstore in memory
pdf_path = "Engines-Complete-Updated-and-Issued-8-5-2021.pdf"
text = extract_text_from_pdf(pdf_path)
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
docs = splitter.create_documents([text])
embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
vectorstore = DocArrayInMemorySearch.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})

# Set up LLM-powered QA
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model_name="gpt-4"),
    retriever=retriever,
    return_source_documents=True
)

# UI
st.set_page_config(page_title="FDNY AI Tutor", page_icon="🚒")
st.title("🚒 FDNY Engine Company Tutor")
st.markdown("Ask me anything based on the Engine Operations Manual.")

if "history" not in st.session_state:
    st.session_state.history = []

user_input = st.chat_input("What do you want to learn about?")
if user_input:
    response = qa_chain(user_input)
    st.session_state.history.append((user_input, response["result"]))

for user_msg, bot_msg in st.session_state.history:
    st.chat_message("user").markdown(user_msg)
    st.chat_message("assistant").markdown(bot_msg)



