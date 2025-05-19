import fitz  # PyMuPDF
from pathlib import Path
from fastapi import File, UploadFile, HTTPException
import os
from docx import Document
import pandas as pd
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import  HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# 업로드 디렉토리 설정
UPLOAD_DIR = "./uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# 임베딩 모델 초기화
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def load_pdf(file_path: str) -> str:
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text

def load_docx(file_path: str) -> str:
    """Word 파일에서 텍스트 추출"""
    doc = Document(file_path)
    return "\n".join(para.text for para in doc.paragraphs)

def load_excel(file_path: str) -> str:
    """Excel 파일에서 텍스트 추출"""
    df = pd.read_excel(file_path)
    return df.to_string()

def extract_text(file_path: str, content_type: str) -> str:
    """파일 형식에 따라 텍스트 추출"""
    if content_type == "application/pdf":
        return load_pdf(file_path)
    elif content_type in [
        "application/msword",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ]:
        return load_docx(file_path)
    elif content_type in [
        "application/vnd.ms-excel",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ]:
        return load_excel(file_path)
    return ""

async def upload_files(files: list[UploadFile] = File(...)):
    """파일 업로드 및 RAG 파이프라인 처리"""
    try:
        for file in files:
            # 파일 저장
            file_path = os.path.join(UPLOAD_DIR, file.filename)
            with open(file_path, "wb") as f:
                f.write(await file.read())

            # 텍스트 추출
            text = extract_text(file_path, file.content_type)
            if not text:
                continue

            # 텍스트 분할
            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = text_splitter.split_text(text)

            # FAISS 벡터 DB에 저장
            vector_store = FAISS.from_texts(
                chunks, embeddings, metadatas=[{"source": file.filename}] * len(chunks)
            )
            vector_store.save_local("faiss_index")

        return {"message": "Files uploaded and processed successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

