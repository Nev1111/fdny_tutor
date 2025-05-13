import os
import streamlit as st
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA

# Load API key
load_dotenv()

# Constants
VECTOR_STORE_FILE = "fdny_faiss_index"

# Load FAISS vectorstore
embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
vectorstore = FAISS.load_local(VECTOR_STORE_FILE, embeddings, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})

# Set up the retrieval-based tutor
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model_name="gpt-4"),
    retriever=retriever,
    return_source_documents=True
)

# Streamlit UI setup
st.set_page_config(page_title="FDNY AI Tutor", page_icon="🚒")
st.title("🚒 FDNY Engine Company Tutor")
st.markdown("Ask me anything based on the Engine Operations Manual.")

# Session state to store chat history
if "history" not in st.session_state:
    st.session_state.history = []

# Chat interface
user_input = st.chat_input("What do you want to learn about?")
if user_input:
    response = qa_chain(user_input)
    st.session_state.history.append((user_input, response["result"]))

# Display past messages
for user_msg, bot_msg in st.session_state.history:
    st.chat_message("user").markdown(user_msg)
    st.chat_message("assistant").markdown(bot_msg)

