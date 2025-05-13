import os
import pickle
import streamlit as st
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA

# Load environment variables
load_dotenv()

# Constants
VECTOR_STORE_FILE = "fdny_docarray.pkl"

# Load the vectorstore
with open(VECTOR_STORE_FILE, "rb") as f:
    vectorstore = pickle.load(f)

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})

# Set up the retrieval QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model_name="gpt-4"),
    retriever=retriever,
    return_source_documents=True
)

# Streamlit UI
st.set_page_config(page_title="FDNY AI Tutor", page_icon="🚒")
st.title("🚒 FDNY Engine Company Tutor")
st.markdown("Ask me anything from the Engine Operations Manual.")

# Maintain chat history
if "history" not in st.session_state:
    st.session_state.history = []

# Chat input
user_input = st.chat_input("What do you want to learn about?")
if user_input:
    result = qa_chain(user_input)
    st.session_state.history.append((user_input, result["result"]))

# Display chat history
for user_msg, bot_msg in st.session_state.history:
    st.chat_message("user").markdown(user_msg)
    st.chat_message("assistant").markdown(bot_msg)


