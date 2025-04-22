from openai.types import vector_store

from rag_chatbot.embedder import load_vector_store # FAISS 벡터 불러오기
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
import time
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
  당신은 회사 복리후생 규정을 바탕으로 질문에 정확하고 사실에 기반한 답변을 제공하는 HR 전문가입니다.
  아래의 문서를 읽고 질문에 대해 다음 중 하나로 명확히 답변하십시오:

    1. 문서에 **해당 상황이 명시되어 있으면**, 그 내용과 함께 근거를 구체적으로 제시하십시오.
    2. 문서에 **명시되지 않은 경우**, '언급 없음'으로 답하고, 그에 따라 지원이 어려울 수 있음을 설명하십시오.
    3. 문서와 관련없는 질문에는 회사 복리후생에 관련해서 질문해달라는 응답을 하십시오.

    [문서 내용]
    {context}
    
    [질문]
    {question}
    
    [정확하고 근거 있는 한국어 답변]
    """
)

# 백터 저장소 로드
vectorstore = load_vector_store()

# LLM + RAG chain 구성
llm = OllamaLLM(model="llama3")
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt_template},
    return_source_documents=True,
)

def ask_question(query: str) -> str:
    # result = qa.run(query)
    print(f"Query: {query}")
    start_time = time.time()
    print(query)
    result = qa.invoke(query)
    print(f"Processing time: {time.time() - start_time} seconds")
    return result



