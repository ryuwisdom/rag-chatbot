# generate_vectorstore.py
from rag_chatbot.embedder import create_chunks, create_vector_store, load_pdf
from langchain_community.embeddings import HuggingFaceEmbeddings


from pypdf import PdfReader

file_path = "documents/sample.pdf"
reader = PdfReader(file_path)
print(f"페이지 수: {len(reader.pages)}")

print("📄 loading pdf")
text = load_pdf("data/documents/sample.pdf")

print("🧩 creating chunks")
chunks = create_chunks(text)

print("🔤 loading embedding model")
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

print("🧠 creating vector store")
vectorstore = create_vector_store(chunks, embedding_model)
print("✅ done! vectorstore created.")
