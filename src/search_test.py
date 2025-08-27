"""
파일명: src/search_test.py
기능: Retriever로 질의문을 테스트하고 사람이 읽기 쉬운 형태로 출력한다.
     - 입력: 질의 문자열(코드 내 또는 CLI 인자)
     - 출력: 상위 k개 결과(점수/카테고리/페이지/출처/요약)

블록 구성
 0) 라이브러리 임포트
 1) 실행 파라미터
 2) 검색 실행 및 pretty print
주의
 - 먼저 index_build.py가 완료되어 rag/* 가 존재해야 한다.
"""

# 0) 라이브러리 임포트
import sys
from textwrap import shorten
from retrieve import Retriever

# 1) 실행 파라미터
query = "무릎 아플 때 어떤 운동이 좋아요?" if len(sys.argv) < 2 else " ".join(sys.argv[1:])
k = 6
category = None  # 예: "생활지도"

# 2) 검색 실행
r = Retriever()
hits = r.search(query, k=k, category=category)

print(f"\n[Query] {query}")
if category:
    print(f"[Filter] category={category}")

if not hits:
    print("검색 결과가 없습니다.")
    sys.exit(0)

for i, h in enumerate(hits, 1):
    m = h["meta"]
    preview = shorten(h["text"].replace("\n", " "), width=120, placeholder=" ...")
    print(f"\n[{i}] score={h['score']:.3f} | page={m.get('page','?')} | cat={m.get('category','?')}")
    print(f"    source: {m.get('source_title','')}")
    print(f"    text  : {preview}")
