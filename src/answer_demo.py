"""
파일명: src/answer_demo.py
기능: Retriever + compose를 연결해 최종 답변을 콘솔에 출력한다.
     - 입력: 질의문(명령행 인자) 또는 기본 예시
     - 처리: FAISS에서 상위 k개 청크 검색 → 섹션형 답변 조립
     - 출력: 요약/운동 방법/주의·준비/생활 지도/안전 보강/출처

블록 구성
 0) 라이브러리 임포트
 1) 파라미터/기본 쿼리
 2) 검색 → 조립 → 출력

주의
 - index_build.py 실행으로 rag/* (index.faiss, texts.jsonl, meta.jsonl)가 준비되어 있어야 한다.
 - 패키지 실행 권장: `python -m src.answer_demo "질문"` (직접 실행도 되도록 임포트 가드 포함)
"""
# 0) 라이브러리 임포트
from __future__ import annotations
import sys

# 패키지/직접 실행 모두 대응
try:
    from src.retrieve import Retriever
    from src.compose import compose_answer
except ImportError:
    from retrieve import Retriever
    from compose import compose_answer


# 1) 파라미터/기본 쿼리 ------------------------------------------------
DEFAULT_QUERY = "운동 전 주의사항 알려줘"
query = DEFAULT_QUERY if len(sys.argv) < 2 else " ".join(sys.argv[1:])
TOP_K = 6
CATEGORY = None  # 예: "생활지도", "운동처방", "보행프로그램", "가정운동", "뇌졸중기본"


# 2) 검색 → 조립 → 출력 ------------------------------------------------
def main():
    r = Retriever()
    hits = r.search(query, k=TOP_K, category=CATEGORY)
    answer = compose_answer(query, hits)
    print(answer)


if __name__ == "__main__":
    main()
