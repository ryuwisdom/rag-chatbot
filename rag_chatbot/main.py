from fastapi import FastAPI
from langchain_community.vectorstores import FAISS
from pydantic import BaseModel
from rag_chatbot.document_loader import load_pdf
# from rag_chatbot.embedder import load_pdf
from rag_chatbot.embedder import create_chunks, create_vector_store, load_vector_store
from rag_chatbot.qa import ask_question
from fastapi.middleware.cors import CORSMiddleware

from langchain_community.embeddings import HuggingFaceEmbeddings

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 개발 중엔 "*"으로 해두고 배포 시 도메인 지정
    # allow_origins=["http://localhost:8080"],  # 개발 중엔 "*"으로 해두고 배포 시 도메인 지정
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "RAG chatbot is running"}

@app.get("/read-sample-pdf")
def read_pdf():
    text = load_pdf("data/documents/sample.pdf")
    return {"text_snippet": text[:300]}

@app.get("/build-vector-store")
def build_vector_store():
    print("📄 1. loading pdf")
    text = load_pdf("data/documents/sample.pdf")
    chunks = create_chunks(text)
    # embedding_model = OpenAIEmbeddings()
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    # vector_store = create_vector_store(chunks, embedding_model)

    persist_directory = "vectorstore"
    vector_store = FAISS.from_documents(chunks, embedding_model)
    vector_store.save_local(persist_directory)
    return {
        "chunk_count": len(chunks),
        "vector_count": vector_store.index.ntotal
    }

class Question(BaseModel): question: str

@app.post("/ask")
async def ask(req : Question):
# async def ask(req: Question, raw_request: Request):
#     body = await raw_request.body()
    print(f"Received body: {req.question}")
    response = ask_question(req.question)
    return {"answer": response}