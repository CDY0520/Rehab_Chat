# -*- coding: utf-8 -*-
"""
04_answer_demo.py
- 03_search_demo 모듈을 importlib로 로드
- 동일한 리트리버 + 검색 API를 사용
- CLI 사용 예:
  python rag/04_answer_demo.py --query "근력운동 주의사항 알려줘" --top_k 3
"""

from __future__ import annotations
from typing import Any, Dict, List
from pathlib import Path
import argparse
import importlib.util
import sys

HERE = Path(__file__).resolve().parent
SD_PATH = HERE / "03_search_demo.py"

def _load_search_demo():
    spec = importlib.util.spec_from_file_location("search_demo", SD_PATH)
    if spec is None or spec.loader is None:
        raise ImportError("search_demo 로드 실패")
    sd = importlib.util.module_from_spec(spec)
    sys.modules["search_demo"] = sd
    spec.loader.exec_module(sd)
    return sd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True)
    parser.add_argument("--top_k", type=int, default=3)
    args = parser.parse_args()

    sd = _load_search_demo()
    retr = sd.get_retriever()

    # 공용 검색 API 사용
    results: List[Dict[str, Any]] = sd.search(retr, args.query, args.top_k)

    print("=== ANSWER DEMO ===")
    print(f"Query: {args.query}")
    print(f"Top-K: {args.top_k}\n")
    for r in results:
        cat = r.get("meta",{}).get("category","")
        print(f"- [{'?' if not cat else cat}] {r.get('text','').strip()}  (score={r.get('score',0):.3f})")
    print("\n[Draft Answer]")
    draft = " ".join([r.get("text","").strip() for r in results[:2] if r.get("text")])
    print(draft if draft else "(no draft)")

if __name__ == "__main__":
    main()
