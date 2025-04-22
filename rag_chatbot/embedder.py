# rag_chatbot/embedder.py

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import  HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

from torch.fx.passes.dialect.common.cse_pass import rand_ops


def create_chunks(text: str, chunk_size: int = 1500, chunk_overlap: int = 200):
    """텍스트를 chunk 단위로 나눔"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " "]
    )
    return splitter.create_documents([text])

# 문서 조각들(chunks)과 임베딩 모델(embedding_model)을 받아서, 벡터로 변환한 뒤, FAISS 벡터 저장소를 만들어 반환하는 함수
def create_vector_store(chunks, embedding_model) -> FAISS:
    print("📥 chunks received:", len(chunks))
    print("📦 creating FAISS vector store...")
    vector_store = FAISS.from_documents(chunks, embedding_model)
    return vector_store



def load_vector_store():
    persist_directory = "vectorstore"
    if not os.path.exists(persist_directory):
        raise FileNotFoundError(f"Vector store not found at {persist_directory}")

    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.load_local(persist_directory, embedding_model, allow_dangerous_deserialization=True )

# def load_pdf(pdf_path: str) -> str:
#     loader = PyPDFLoader(pdf_path)
#     document = loader.load()
#     text = " ".join([doc.page_content for doc in document])
#     return text