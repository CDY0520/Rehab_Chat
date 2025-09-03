# 파일명: rag/04_answer_demo.py
# 목적:
#   - importlib으로 '03_search_demo.py' 동적 로드
#   - get_retriever() 반환형 자동 해석 (dict/object/tuple/list/callable)
#   - 카테고리 인지형 검색: 재정렬 + 오버페치 + 쿼리 확장
# 사용:
#   (.venv) python rag/04_answer_demo.py --query "근력운동 주의사항 알려줘" --top_k 3

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import importlib.util
from typing import Any, Dict, List, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# 0) importlib 로드
# ─────────────────────────────────────────────────────────────────────────────
def load_search_demo() -> Any:
    here = Path(__file__).resolve().parent
    module_path = here / "03_search_demo.py"
    if not module_path.exists():
        raise FileNotFoundError(f"동적 로드 대상 파일을 찾을 수 없습니다: {module_path}")

    spec = importlib.util.spec_from_file_location("search_demo", module_path)
    if spec is None or spec.loader is None:
        raise ImportError("importlib이 spec 로드를 실패했습니다 (spec/loader None).")

    search_demo = importlib.util.module_from_spec(spec)
    sys.modules["search_demo"] = search_demo
    spec.loader.exec_module(search_demo)
    return search_demo


# ─────────────────────────────────────────────────────────────────────────────
# 1) 타입 헬퍼
# ─────────────────────────────────────────────────────────────────────────────
def _looks_like_faiss_index(obj: Any) -> bool:
    return hasattr(obj, "search") and callable(getattr(obj, "search"))

def _looks_like_st_model(obj: Any) -> bool:
    return hasattr(obj, "encode") and callable(getattr(obj, "encode"))

def _looks_like_items(obj: Any) -> bool:
    return isinstance(obj, list) and (len(obj) == 0 or isinstance(obj[0], dict))


# ─────────────────────────────────────────────────────────────────────────────
# 2) (index, model, items)로 정규화
# ─────────────────────────────────────────────────────────────────────────────
def normalize_retriever_triplet(retr: Any) -> Tuple[Any, Any, List[Dict]]:
    # dict
    if isinstance(retr, dict):
        idx = retr.get("index") or retr.get("faiss") or retr.get("faiss_index")
        mdl = retr.get("model") or retr.get("encoder") or retr.get("st_model")
        itm = retr.get("items") or retr.get("docs") or retr.get("metas")
        if idx is not None and mdl is not None and itm is not None:
            return idx, mdl, itm

    # 객체 속성
    if hasattr(retr, "index") and hasattr(retr, "model") and hasattr(retr, "items"):
        return getattr(retr, "index"), getattr(retr, "model"), getattr(retr, "items")

    # tuple/list
    if isinstance(retr, (tuple, list)):
        idx = mdl = itm = None
        for x in retr:
            if idx is None and _looks_like_faiss_index(x):
                idx = x; continue
            if mdl is None and _looks_like_st_model(x):
                mdl = x; continue
            if itm is None and _looks_like_items(x):
                itm = x; continue
        if idx is not None and mdl is not None and itm is not None:
            return idx, mdl, itm

    raise TypeError("retriever를 (index, model, items)로 정규화할 수 없습니다.")


# ─────────────────────────────────────────────────────────────────────────────
# 3) FAISS 검색 (기본/오버페치 공용)
# ─────────────────────────────────────────────────────────────────────────────
def run_query_faiss(index: Any, model: Any, items: List[Dict], query: str, top_k: int) -> List[Dict]:
    try:
        import numpy as np
    except Exception as e:
        raise RuntimeError("numpy 임포트 실패. requirements를 확인하세요.") from e

    q_emb = model.encode([query], normalize_embeddings=True)
    if hasattr(q_emb, "astype"):
        q_emb = q_emb.astype("float32")
    else:
        q_emb = np.array(q_emb, dtype="float32")

    D, I = index.search(q_emb, top_k)
    scores = D[0].tolist()
    idxs = I[0].tolist()

    results: List[Dict] = []
    for rank, (i, sc) in enumerate(zip(idxs, scores), start=1):
        if i < 0 or i >= len(items):
            continue
        meta = items[i] or {}
        text = meta.get("text") or meta.get("sentence") or meta.get("content") or str(meta)
        results.append({"rank": rank, "text": text, "meta": meta, "score": float(sc)})
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 4) 카테고리 인지 유틸(추론/재정렬/확장)
# ─────────────────────────────────────────────────────────────────────────────
CATEGORY_ALIASES = {
    "운동 전 주의사항": ["운동 전", "운동전", "before exercise", "pre-exercise"],
    "운동 중 주의사항": ["운동 중", "운동중", "during exercise", "중간"],
    "운동 후 주의사항": ["운동 후", "운동후", "after exercise", "post-exercise"],
    "근력운동 주의사항": ["근력운동", "저항운동", "웨이트", "strength", "resistance"],
}

def infer_category_from_query(query: str) -> str | None:
    q = query.lower().replace(" ", "")
    for cat, aliases in CATEGORY_ALIASES.items():
        for a in aliases:
            if a.replace(" ", "") in q:
                return cat
    return None

def get_meta_category(meta: Dict) -> str:
    return meta.get("category") or meta.get("Category") or meta.get("섹션") or meta.get("section") or ""

def rerank_by_category(results: List[Dict], cat_hint: str | None) -> List[Dict]:
    if not cat_hint:
        return results
    same, other = [], []
    for r in results:
        (same if get_meta_category(r.get("meta", {}) or {}) == cat_hint else other).append(r)
    return same + other

def count_matches(results: List[Dict], cat_hint: str | None) -> int:
    if not cat_hint:
        return 0
    return sum(1 for r in results if get_meta_category(r.get("meta", {}) or {}) == cat_hint)

def build_expanded_query(query: str, cat_hint: str) -> str:
    aliases = CATEGORY_ALIASES.get(cat_hint, [])
    # 예: "근력운동 주의사항 알려줘 근력운동 저항운동 웨이트"
    return query + " " + " ".join(aliases)


# ─────────────────────────────────────────────────────────────────────────────
# 5) 메인 로직
# ─────────────────────────────────────────────────────────────────────────────
def answer_demo(query: str, top_k: int = 3) -> None:
    sd = load_search_demo()
    if not hasattr(sd, "get_retriever"):
        raise AttributeError("search_demo 모듈에 get_retriever 함수가 없습니다.")

    retr = sd.get_retriever()

    # 진단 로그
    print("=== DEBUG: get_retriever() 반환 타입 진단 ===")
    print("type:", type(retr))
    if isinstance(retr, dict):
        print("keys:", list(retr.keys()))
    elif isinstance(retr, (list, tuple)):
        print("len:", len(retr), "| elem types:", [type(x) for x in retr])
    else:
        attrs = [a for a in ["index", "model", "items", "search"] if hasattr(retr, a)]
        print("has attrs:", attrs)
    print("──────────────────────────────────────────")

    # 1차 검색
    if hasattr(sd, "search"):
        try:
            results = sd.search(retr, query, top_k=top_k)
        except TypeError:
            results = sd.search(retr, query)
        # (sd.search 사용 시에도 카테고리 오버페치 보장을 위해 triplet 확보)
        try:
            index, model, items = normalize_retriever_triplet(retr)
        except Exception:
            index = model = items = None
    elif callable(retr):
        try:
            results = retr(query, top_k=top_k)
        except TypeError:
            results = retr(query)
        try:
            index, model, items = normalize_retriever_triplet(retr)
        except Exception:
            index = model = items = None
    else:
        index, model, items = normalize_retriever_triplet(retr)
        results = run_query_faiss(index, model, items, query, top_k)

    # 정렬 + 카테고리 재정렬
    results = sorted(results, key=lambda x: x.get("score", 0.0), reverse=True)
    cat_hint = infer_category_from_query(query)
    results = rerank_by_category(results, cat_hint)

    # 매칭 수 확인
    m_cnt = count_matches(results, cat_hint)
    print(f"DEBUG: cat_hint={cat_hint!r}, 1st_match_count={m_cnt}")

    # 2차: 오버페치 → 재정렬
    if cat_hint and m_cnt == 0 and index is not None:
        over_k = max(top_k * 10, 30)
        results_big = run_query_faiss(index, model, items, query, over_k)
        results_big = sorted(results_big, key=lambda x: x.get("score", 0.0), reverse=True)
        results_big = rerank_by_category(results_big, cat_hint)
        m_cnt2 = count_matches(results_big, cat_hint)
        print(f"DEBUG: overfetch_k={over_k}, 2nd_match_count={m_cnt2}")
        if m_cnt2 > 0:
            results = results_big[:top_k]

    # 3차: 쿼리 확장 → 오버페치
    if cat_hint and count_matches(results, cat_hint) == 0 and index is not None:
        q_exp = build_expanded_query(query, cat_hint)
        results_big = run_query_faiss(index, model, items, q_exp, max(top_k * 10, 30))
        results_big = sorted(results_big, key=lambda x: x.get("score", 0.0), reverse=True)
        results_big = rerank_by_category(results_big, cat_hint)
        m_cnt3 = count_matches(results_big, cat_hint)
        print(f"DEBUG: expanded_query={q_exp!r}, 3rd_match_count={m_cnt3}")
        if m_cnt3 > 0:
            results = results_big[:top_k]

    # 출력
    print("\n=== ANSWER DEMO ===")
    print(f"Query: {query}")
    print(f"Top-K: {top_k}\n")

    if not results:
        print("검색 결과가 없습니다.")
        return

    for r in results:
        meta = r.get("meta", {}) or {}
        cat = get_meta_category(meta)
        line = r.get("text", "").strip()
        score = float(r.get("score", 0.0))
        if cat:
            print(f"- [{cat}] {line}  (score={score:.3f})")
        else:
            print(f"- {line}  (score={score:.3f})")

    # Draft Answer
    top_texts = [r.get("text", "") for r in results[:2]]
    draft_answer = " ".join(t.strip() for t in top_texts if t.strip())
    print("\n[Draft Answer]")
    print(draft_answer if draft_answer else "상위 문맥을 기반으로 요약 답안을 생성할 수 없습니다.")


# ─────────────────────────────────────────────────────────────────────────────
# 6) CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RAG Answer Demo (category-aware with overfetch & expansion)")
    p.add_argument("--query", type=str, default="운동 전 주의사항 알려줘")
    p.add_argument("--top_k", type=int, default=3)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    answer_demo(query=args.query, top_k=args.top_k)
