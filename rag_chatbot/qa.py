from rag_chatbot.embedder import load_vector_store # FAISS 벡터 불러오기
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableMap
from langchain_ollama import ChatOllama
from  langchain.schema import StrOutputParser
import time

template = (
    """
당신은 회사 복리후생 규정에 따라 사용자의 질문에 정확하고 친절하게 답변하는 HR 담당자입니다. 
아래 문서를 참고하여 질문에 대해 다음 기준에 따라 **한국어**로 응답하세요:

1. 문서에 관련 항목이 명시되어 있다면, 그 내용을 자연스럽게 요약하고, 지원 조건, 금액, 대상자, 휴가 여부 등 구체적인 내용을 포함해 서술하십시오.
2. 문서에 명시적으로 언급되지 않았다면, '문서에 언급되어 있지 않습니다'라고 설명하고, 해당 항목에 대한 지원이 어려울 수 있음을 정중히 안내하십시오.
3. 복리후생과 관련 없는 질문일 경우, 복리후생과 관련된 질문을 요청하는 응답을 하십시오.

- 답변은 정중하고 자연스러운 문장으로 작성하며, 사용자가 이해하기 쉽도록 구성하세요.
- 질문과 상관없는 내용은 답변에 추가하지 말아주세요.
- 영어 응답은 하지 말아주세요.

[문서 내용]
{context}

[질문]
{question}

[정확한 한국어 답변]
"""
)

# prompt_template = PromptTemplate.from_template(
#     input_variables=["context", "question"],
#     template="""
#   당신은 회사 복리후생 규정을 바탕으로 질문에 정확하고 사실에 기반한 답변을 제공하는 HR 전문가입니다.
#   아래의 문서를 읽고 질문에 대해 다음 중 하나로 명확히 답변하십시오:
#
#     1. 문서에 **해당 상황이 명시되어 있으면**, 그 내용과 함께 근거를 구체적으로 제시하십시오.
#     2. 문서에 **명시되지 않은 경우**, '언급 없음'으로 답하고, 그에 따라 지원이 어려울 수 있음을 설명하십시오.
#     3. 문서와 관련없는 질문에는 회사 복리후생에 관련해서 질문해달라는 응답을 하십시오.
#
#     [문서 내용]
#     {context}
#
#     [질문]
#     {question}
#
#     [정확하고 근거 있는 한국어 답변]
#     """
# )

prompt_template = PromptTemplate.from_template(template)

# LLM 정의
llm = ChatOllama(model="llama3", temperature=0)

# Vectorstore 로딩
vectorstore = load_vector_store()
retriever=vectorstore.as_retriever()


# LLM + RAG chain 구성
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt_template
    | llm
    | StrOutputParser()
)

# rag_chain_stream = ChatOllama(model="llama3", temperature=0, disable_streaming=False)
streaming_llm = ChatOllama(model="llama3", temperature=0, streaming=True)

rag_chain_stream = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt_template
    | streaming_llm
)





# qa = RetrievalQA.from_chain_type(
#     llm=llm,
#     retriever=vectorstore.as_retriever(),
#     chain_type="stuff",
#     chain_type_kwargs={"prompt": prompt_template},
#     return_source_documents=True,
# )

# qa = ConversationalRetrievalChain.from_llm(
#     llm = llm,
#     retriever=vectorstore.as_retriever(),
#     memory=memory,
#     # 체인의 동작을 세부적으로 제어하기 위함
#     chain_type_kwargs={
#         "prompt" : prompt_template,
#     },
#     return_source_documents=True,
# )



# def ask_question(query: str) -> str:
#     # result = qa.run(query)
#     print(f"Query: {query}")
#     start_time = time.time()
#     print(query)
#     result = qa.invoke(query)
#     print(f"Processing time: {time.time() - start_time} seconds")
#     return result
#
# def ask_question_stream(query: str) -> str:
#     print(f"Query: {query}")
#     start_time = time.time()
#     stream = qa.stream(query)
#     for chunk in stream:
#         print(chunk["answer"], end="", flush=True)
#     print(f"\nProcessing time: {time.time() - start_time} seconds")

def ask_question(query: str) -> str:
    return rag_chain.invoke(query)

async def ask_question_stream(question: str):
    if isinstance(question, dict):
        question = question.get("question", "")  # 안전하게 문자열 추출
    # async for chunk in rag_chain_stream.astream({"question": question}):
    async for chunk in rag_chain_stream.astream(question):
        if chunk.content:
            yield chunk.content