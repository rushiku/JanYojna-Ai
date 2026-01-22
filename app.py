import os
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate


# -------------------------------
# 1. PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="CitizenScheme Bot",
    page_icon="🏛️",
    layout="centered"
)

st.title("🏛️ Government Scheme Eligibility Chatbot")
st.caption("Ask eligibility questions in any Indian language")


# -------------------------------
# 2. SIDEBAR – API KEY
# -------------------------------
api_key = st.sidebar.text_input(
    "Enter OpenAI API Key",
    type="password"
)

if api_key:
    os.environ["OPENAI_API_KEY"] = api_key


# -------------------------------
# 3. LOAD & CACHE VECTOR STORE
# -------------------------------
@st.cache_resource
def get_vector_store():
    """
    Load PDFs → Split → Embed → Store in FAISS
    """
    loader = PyPDFLoader("schemes.pdf")
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(documents)

    vectorstore = FAISS.from_documents(
        documents=chunks,
        embedding=OpenAIEmbeddings()
    )

    return vectorstore


# -------------------------------
# 4. MAIN LOGIC
# -------------------------------
if not api_key:
    st.warning("🔑 Please enter your OpenAI API key to continue.")
    st.stop()

try:
    vectorstore = get_vector_store()
except Exception as e:
    st.error("❌ Please add a `schemes.pdf` file in the project directory.")
    st.stop()


# -------------------------------
# 5. LLM + RAG CHAIN
# -------------------------------
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0
)

system_prompt = """
You are an expert Indian government assistant.

Use ONLY the provided context to answer questions about government schemes.

Rules:
- Strictly check eligibility conditions from the document
- Answer in the SAME LANGUAGE as the user's question
- If the answer is not found in the context, say "I don't know"
- Do NOT hallucinate schemes or benefits

Context:
{context}
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

qa_chain = create_stuff_documents_chain(llm, prompt)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

rag_chain = create_retrieval_chain(retriever, qa_chain)


# -------------------------------
# 6. CHAT UI
# -------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input(
    "Ask: Am I eligible for PM Kisan? / मैं पीएम किसान के लिए पात्र हूँ?"
)

if user_input:
    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        response = rag_chain.invoke({"input": user_input})
        answer = response["answer"]
        st.markdown(answer)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )
र
