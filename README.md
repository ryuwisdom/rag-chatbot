#### 🎈 Project Flow
```text

PDF/Word/Excel 문서 (사내 복리후생 안내 pdf 파일)
     ↓
Document Chunking
     ↓
HuggingFace Embeddings 
     ↓
→ FAISS 벡터 DB 저장 (벡터 + 메타데이터)
     ↓
사용자 질문 → 벡터화 → FAISS에서 유사 문서 검색
     ↓
→ 유사 문서를 context로 LLaMA3(Ollama)에 전달 → 응답 생성
     ↓
FastAPI → java → React로 응답 전달

```

#### 🎈 프로젝트 활용
```text
사내 Portal 프로젝트에 연동
[React] ──> [Spring Boot] ──(HTTP 요청)──> [FastAPI 서버 (Python)]
                       ↑                                 ↓
              DB, 서비스 로직                    모델 추론, AI 서비스
```



#### 🎈 Library & Tools
| 이름                  | 설명                                                                 | 용도                                      |
|-----------------------|---------------------------------------------------------------------|-------------------------------------------|
| `fastapi`             | 비동기 웹 프레임워크                                                 | API 서버 구축                             |
| `uvicorn`             | ASGI 서버 구현, FastAPI와 함께 사용                                  | FastAPI 애플리케이션 실행                 |
| `python-multipart`    | 멀티파트 데이터(예: 파일 업로드) 처리를 위한 라이브러리               | 파일 업로드 처리                          |
| `PyMuPDF`             | PDF 파일 처리 라이브러리                                             | PDF 텍스트 추출 및 처리                   |
| `langchain`           | 문서 분할, 임베딩, RAG 구현을 쉽게 해주는 프레임워크                 | LLM 기반 애플리케이션 개발                |
| `langchain-community` | 문서 chunking, embedding, retriever 등 커뮤니티 제공 유틸리티 포함   | 문서 처리 및 벡터 검색                    |
| `langchain-core`      | LangChain의 핵심 기능 제공                                           | LangChain의 기본 컴포넌트                 |
| `faiss-cpu`           | 효율적인 벡터 검색 및 클러스터링 라이브러리                          | 로컬 환경에서의 벡터 검색                 |
| `tiktoken`            | 토큰 길이 계산 라이브러리, OpenAI 모델과 호환                        | LLM 입력을 위한 토큰 계산 및 문서 분할    |

#### 🎈 추가할 기능
- 모델 튜닝
- 유저별 대화 저장
- 원하는 문서 업로드 
