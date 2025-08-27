"""
파일명: src/config_rag.py
기능: RAG 지식소스 구축을 위한 공용 설정값을 정의한다.
     - 경로 상수(DOCS_DIR, PROCESSED_DIR)
     - 청크 규칙(윈도우/오버랩/최소 길이)
     - 전처리 옵션(목차 스킵/개행 정리)
     - 카테고리 정책: category = source_title (원문 머릿말 제목)  ← ingest.py에서 적용

블록 구성
 0) 라이브러리 임포트
 1) 경로/상수
 2) 청크 규칙
 3) 전처리 옵션

주의
 - 카테고리 매핑(CATEGORY_MAP)은 사용하지 않는다. (파일명/키워드 매핑 제거)
 - ingest.py에서 페이지 상단 머릿말을 source_title로 추출하고, category에 동일 값 저장.
 - 경로는 Pathlib 기반으로 운영체제에 독립적으로 동작한다.
"""
# 0) 라이브러리 임포트
from pathlib import Path

# 1) 경로/상수 ----------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = ROOT / "data" / "docs"           # 원본 PDF 폴더
PROCESSED_DIR = ROOT / "data" / "processed" # 청크 JSONL 저장 폴더
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# 2) 청크 규칙 ----------------------------------------------------------
# 더 촘촘한 청크: 섹션 혼합을 방지하고 검색 정밀도를 높임
CHUNK_WINDOW   = 400   # 한 청크 최대 길이(문자수)
CHUNK_OVERLAP  = 50    # 앞 청크와 겹치는 길이(문자수)
MIN_CHARS      = 100   # 너무 짧은 조각은 버림

# 3) 전처리 옵션 --------------------------------------------------------
# 페이지 텍스트에 아래 토큰이 있으면 '목차 페이지'로 간주하고 스킵
SKIP_TOC_IF_CONTAINS = ["목차"]

# 여러 줄 개행을 2줄로 정리하여 노이즈 제거
STRIP_MULTIPLE_NEWLINES = True
