# -*- coding: utf-8 -*-
"""
파일: rag/03_search_demo.py
목적: 질문(q)에 대해 FAISS 유사도 검색 후 상위 K개 결과 + 출처(원본 페이지 우선) 표시
"""

from pathlib import Path
import pickle
from sentence_transformers import SentenceTransformer
import faiss

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR  = ROOT / "data" / "vectorstore"
FAISS_INDEX_PATH = OUT_DIR / "faiss.index"
META_PATH        = OUT_DIR / "meta.pkl"

def load_index_and_meta():
    index = faiss.read_index(str(FAISS_INDEX_PATH))
    with open(META_PATH, "rb") as f:
        meta = pickle.load(f)
    model = SentenceTransformer(meta["model_name"])
    return index, meta, model

def search(q, top_k=5, category=None, severity=None, intent=None):
    index, meta, model = load_index_and_meta()

    # 간단 후보 필터(메타 기반)
    candidates = list(range(len(meta["records"])))
    if category:
        candidates = [i for i in candidates if meta["records"][i].get("category") == category]
    if severity:
        candidates = [i for i in candidates if meta["records"][i].get("severity") == severity]
    if intent:
        candidates = [i for i in candidates if meta["records"][i].get("intent") == intent]
    if not candidates:
        return []

    # 검색
    q_emb = model.encode([q], normalize_embeddings=True).astype("float32")
    D, I = index.search(q_emb, k=min(top_k*10, len(meta["texts"])))
    hits = []
    for dist, idx in zip(D[0], I[0]):
        if idx in candidates:
            rec = meta["records"][idx]
            hits.append({
                "score": float(dist),
                "id": rec["id"],
                "text": rec["canonical_ko"],
                "category": rec.get("category"),
                "severity": rec.get("severity"),
                "intent": rec.get("intent"),
                "source_title": rec.get("source_title"),
                "source_page": rec.get("source_page"),
                "source_orig_page": rec.get("source_orig_page"),  # ✅ 추가
            })
        if len(hits) >= top_k:
            break
    return hits

def pretty_print(results):
    if not results:
        print("결과 없음")
        return
    for i, r in enumerate(results, 1):
        print(f"[{i}] ({r['score']:.3f}) {r['text']}")
        print(f"    - id: {r['id']} | cat: {r['category']} | sev: {r['severity']} | intent: {r['intent']}")
        page = r.get("source_orig_page") or r.get("source_page")  # ✅ 원본 페이지 우선
        print(f"    - 출처: {r['source_title']} (p.{page})")
        print()

if __name__ == "__main__":
    # 간단 테스트
    for q in ["식사 직후 운동해도 돼?", "운동 전 혈압 확인해야 해?", "균형이 불안정할 때?"]:
        print("Q:", q)
        pretty_print(search(q, top_k=3))
        print("-"*80)
