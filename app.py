import os
import streamlit as st
from streamlit_chat import message
import requests
import jwt

from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_cohere import CohereEmbeddings, ChatCohere
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# Constants
VERIFY_URL = "https://valobuy.shop/wp-json/chatbot-api/v1/verify"
JWT_SECRET = "abrakadabra"

os.environ["COHERE_API_KEY"] = "ZpvBSDjoZJae3CsAlFC6qcdiXNEvmVV0Z7ZwCxhz"

def get_user_from_token(token):
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
        return payload
    except Exception as e:
        st.error("Invalid or expired token.")
        return None

@st.cache_data
def doc_preprocessing():
    loader = PyPDFLoader("iesc111.pdf")
    docs = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    return text_splitter.split_documents(docs)

@st.cache_resource
def embeddings_store():
    embedding = CohereEmbeddings(model="embed-english-v3.0")
    texts = doc_preprocessing()
    vectordb = FAISS.from_documents(documents=texts, embedding=embedding)
    return vectordb.as_retriever()

@st.cache_resource
def conversational_qa():
    retriever = embeddings_store()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")
    return ConversationalRetrievalChain.from_llm(
        llm=ChatCohere(),
        memory=memory,
        retriever=retriever
    )

@st.cache_resource
def rag_qa_chain():
    retriever = embeddings_store()
    return RetrievalQA.from_chain_type(
        llm=ChatCohere(),
        chain_type="stuff",
        retriever=retriever
    )

def display_conversation(history):
    for i in range(len(history["generated"])):
        message(history["past"][i], is_user=True, key=f"{i}_user")
        message(history["generated"][i], key=str(i))

def main_f():
    st.title("LLM Chatbot with User Auth (JWT)")

    query_params = st.query_params
    token = query_params.get("token")

    if not token:
        st.warning("You are not logged in.")
        return

    user = get_user_from_token(token)
    if not user:
        return

    st.success(f"Welcome {user['name']}")

    rag_chain = rag_qa_chain()
    convo_chain = conversational_qa()

    if "generated" not in st.session_state:
        st.session_state["generated"] = ["I am ready to help you"]
    if "past" not in st.session_state:
        st.session_state["past"] = ["Hey there!"]

    user_query = st.text_input("Ask your question:")

    if user_query:
        with st.spinner("Thinking..."):
            if st.checkbox("Use RAG (PDF)", key="rag_toggle"):
                output = rag_chain.run(user_query)
            else:
                output = convo_chain({"question": user_query})["answer"]

            st.session_state.past.append(user_query)
            st.session_state.generated.append(output)

    if st.session_state["generated"]:
        display_conversation(st.session_state)

if __name__ == "__main__":
    main_f()
