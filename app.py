import streamlit as st
import tempfile
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

from transformers import pipeline as hf_pipeline   # FIX 1: avoid name clash with langchain

# ---------------- UI ----------------
st.set_page_config(page_title="AI Document Assistant", page_icon="📄")
st.title("📄 AI Document Assistant")

# ---------------- FILE UPLOADER ----------------
uploaded_file = st.file_uploader(
    "Upload your PDF document",
    type=["pdf"]
)

# ---------------- LOAD LLM ONCE ----------------
@st.cache_resource
def load_llm():
    pipe = hf_pipeline(
        task="text2text-generation",   # FIX 2: flan-t5 is seq2seq → must use text2text-generation
        model="google/flan-t5-base",
        max_new_tokens=256
    )
    llm = HuggingFacePipeline(pipeline=pipe)
    return llm

# ---------------- LOAD EMBEDDINGS ONCE ----------------
@st.cache_resource                     # FIX 3: cache embeddings to avoid reloading on every rerun
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

# ---------------- PROCESS DOCUMENT ----------------
if uploaded_file is not None:
    # FIX 4: only reprocess when a NEW file is uploaded (track by filename + size)
    file_id = f"{uploaded_file.name}_{uploaded_file.size}"

    if st.session_state.get("file_id") != file_id:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            file_path = tmp.name

        st.success("✅ Document Uploaded Successfully!")

        with st.spinner("Processing document…"):
            loader = PyPDFLoader(file_path)
            documents = loader.load()

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,
                chunk_overlap=100
            )
            docs = splitter.split_documents(documents)

            embeddings = load_embeddings()

            # FIX 5: use a unique persist directory per file so Chroma doesn't mix documents
            persist_dir = tempfile.mkdtemp()
            db = Chroma.from_documents(docs, embeddings, persist_directory=persist_dir)

            llm = load_llm()

            prompt_template = """
You are a friendly AI assistant. Your job is to explain information
like a teacher to a beginner student.

RULES:
- Use very simple English.
- Write in clear paragraph form.
- Do NOT use bullet points or numbering.
- Do NOT include citations, author names, or references.
- Ignore research paper indexes or DOI numbers.
- Explain the main idea only.
- Make the answer easy to understand.

If information is not available, say: "I don't know."

Context:
{context}

User Question:
{question}

Simple Explanation:
"""
            PROMPT = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )

            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=db.as_retriever(search_kwargs={"k": 4}),
                chain_type_kwargs={"prompt": PROMPT},
                return_source_documents=False,
            )

            # Persist chain and reset chat for the new document
            st.session_state.qa_chain = qa_chain
            st.session_state.file_id  = file_id
            st.session_state.messages = []           # FIX 6: clear old chat on new upload

        # Clean up temp file
        os.unlink(file_path)

# ---------------- CHAT SECTION ----------------
if "qa_chain" in st.session_state:
    st.subheader("💬 Chat With Your Document")

    # Initialize chat history if missing
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display existing chat history ABOVE the input
    for chat in st.session_state.messages:
        with st.chat_message("user"):
            st.write(chat["question"])
        with st.chat_message("assistant"):
            st.write(chat["answer"])

    # FIX 7: use st.chat_input instead of text_input — it auto-clears after submit
    question = st.chat_input("Ask anything from your document…")

    if question:
        # Show the user message immediately
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Generating answer…"):
                # FIX 8: use .invoke() — .run() is deprecated in newer LangChain
                result = st.session_state.qa_chain.invoke({"query": question})
                answer = result["result"].replace("\n", " ").strip()
            st.write(answer)

        st.session_state.messages.append({"question": question, "answer": answer})

else:
    st.info("👆 Upload a PDF above to get started.")