from fastapi import FastAPI
from langchain_community.vectorstores import FAISS
from pydantic import BaseModel

from rag_chatbot.document_loader import load_pdf, upload_files
from rag_chatbot.embedder import create_chunks, create_vector_store, load_vector_store
from rag_chatbot.qa import ask_question, ask_question_stream
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
from langchain_community.embeddings import HuggingFaceEmbeddings

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 수정 필요
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

# @app.post("/ask")
# async def ask(req : Question):
# # async def ask(req: Question, raw_request: Request):
# #     body = await raw_request.body()
#     print(f"Received body: {req.question}")
#     response = ask_question(req.question)
#     return {"answer": response}
#
#
# async def event_generator(question):
#     for chunk in qa.stream(question):
#         yield {"event": "message","data" : chunk["answer"]}
#
# @app.post("/ask-stream")
# async def ask_stream(req : Question):
#     return EventSourceResponse(event_generator(req.question))

@app.post("/ask")
async def ask(req: Question):
    print(req.question)
    answer = ask_question(req.question)
    return {"answer": answer}

@app.post("/ask-stream")
async def ask_stream(req: Question):
    async def event_generator():
        try:
            print(f"Received question: {req.question}")
            async for chunk in ask_question_stream(req.question):
                if chunk:
                    print(f"Sending chunk: {chunk}")
                    yield {"data": chunk}  # SSE 포맷: data: {content}\n\n
            yield {"data": "[DONE]"}  # 명시적 종료 신호
        except Exception as e:
            yield {"data": f"Error: {str(e)}"}  # 에러 발생 시 클라이언트에 전달
            print(f"Stream error: {str(e)}")
        finally:
            print("Stream completed")
    # return EventSourceResponse(event_generator(),headers={"Cache-Control": "no-cache", "Connection": "keep-alive"})
    return EventSourceResponse(event_generator())

@app.post("/upload")
async def upload_endpoint(files: list):
    return await upload_files(files)

@app.post("/test-body")
async def test_body(req: Question):
    return {"received": req.question}