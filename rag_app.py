
from dotenv import load_dotenv

# PDF Loader
from langchain_community.document_loaders import PyPDFLoader

# Text Splitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

# FREE Embeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

# Vector Database
from langchain_community.vectorstores import Chroma

# FREE Local LLM
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline

# QA Chain
from langchain.chains import RetrievalQA

load_dotenv()

# Load PDF
loader = PyPDFLoader("sample.pdf")
documents = loader.load()

# Split text
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = splitter.split_documents(documents)

# Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Vector Database
db = Chroma.from_documents(
    chunks,
    embeddings,
    persist_directory="db"
)

retriever = db.as_retriever()

# LLM
pipe = pipeline(
    "text-generation",
    model="google/flan-t5-base",
    max_length=512
)

llm = HuggingFacePipeline(pipeline=pipe)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever
)

while True:
    query = input("Ask Question: ")
    answer = qa.run(query)
    print("\nAnswer:", answer)