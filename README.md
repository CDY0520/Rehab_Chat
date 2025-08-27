# Rehab_Chat 🧠
**RAG(Retrieval-Augmented Generation)** 기반 의료 지식 챗봇  
Streamlit + FAISS + SentenceTransformers를 활용해 PDF/TXT 지식소스를 업로드하면  
문서를 벡터화하여 빠르게 검색하고, 핵심 근거 기반의 답변을 생성합니다.

---

## 🚀 프로젝트 개요
- **프로젝트명**: Rehab_Chat
- **목적**: 의료·재활 분야 자료 기반 Q&A 챗봇 구현
- **기술 스택**:  
  - **Frontend**: Streamlit  
  - **Embedding Model**: `all-MiniLM-L6-v2` (SentenceTransformers)  
  - **Vector DB**: FAISS  
  - **PDF Parser**: PyMuPDF
- **주요 기능**
  1. PDF 및 TXT 파일 업로드 → 자동 벡터 인덱싱
  2. 사용자 질문 기반 상위 컨텍스트 검색
  3. 검색된 근거를 활용한 핵심 요약 응답 생성
  4. 출처 표시 및 안전 가이드라인 제공

---

## 🗂 디렉토리 구조
Rehab_Chat/
├── src/
│ ├── app.py # Streamlit 메인 앱
│ ├── compose.py # 답변 조합 및 요약 모듈
│ ├── safety.py # 안전 가이드라인 처리
│ └── vectorstore/ # FAISS 인덱스 및 메타데이터 저장
├── requirements.txt # 패키지 목록
└── README.md # 프로젝트 설명 문서

## 🛠 설치 방법

## 저장소 클론
```bash
git clone https://github.com/<username>/Rehab_Chat.git
cd Rehab_Chat

## 가상환경 생성 및 활성화
python -m venv .venv
.venv\Scripts\activate

## 패키지 설치
pip install -r requirements.txt

## 실행 방법
python -m streamlit run src/app.py