"""
# 파일명: rag/rag_eval.py
# 목적: RAG 파이프라인 오프라인 성능 평가 (포트폴리오 2단계 지표 포함)
# 지표:
#   [Retrieval]  Hit@k, MRR@k
#   [Pipeline]   Top-1 Category Accuracy
#   [Routing]    Routing Trigger Rate, Routing Accuracy(when triggered)
#
# 필요 파일:
#   - rag/04_answer_demo.py (ResponseEngineRAG 정의)
#   - data/processed/precautions.jsonl
#   - data/processed/train_queries.csv / dev_queries.csv / test_queries.csv
#
# 실행:
#   python rag/rag_eval.py --eval-file data/processed/test_queries.csv
#
# 코드 구성 (블록별):
# 0) 임포트 & 경로 설정
# 1) 엔진 모듈 동적 로드
# 2) 데이터 로더
# 3) 간단 TF-IDF 리트리버 (fallback용)
# 4) 평가용 엔진 래퍼 (라우팅 플래그 포함)
# 5) 지표 함수 (Hit@k, MRR@k)
# 6) 메인 evaluate(): Retrieval/파이프라인/라우팅 지표 출력
"""

from __future__ import annotations
import csv
import importlib.util
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from collections import Counter
import argparse
import json
import re
import unicodedata
import numpy as np

# ------------------------------------------------
# 0) 경로 설정
# ------------------------------------------------
THIS_DIR = Path(__file__).resolve().parent      # .../Rehab_Chat/rag
ROOT = THIS_DIR.parent                          # .../Rehab_Chat
ENGINE_PATH = THIS_DIR / "04_answer_demo.py"    # 동적 로드 대상
JSONL = ROOT / "data" / "processed" / "precautions.jsonl"

# ------------------------------------------------
# 1) 엔진 모듈 동적 로드
# ------------------------------------------------
def load_engine_module(engine_path: Path):
    if not engine_path.exists():
        raise FileNotFoundError(f"엔진 파일을 찾을 수 없습니다: {engine_path}")
    spec = importlib.util.spec_from_file_location("answer_engine_mod", engine_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)  # type: ignore
    return mod

engine_mod = load_engine_module(ENGINE_PATH)
ResponseEngineRAG = engine_mod.ResponseEngineRAG
route_category_by_keywords = engine_mod.route_category_by_keywords
is_high = engine_mod.is_high

# ------------------------------------------------
# 2) 데이터 로더
# ------------------------------------------------
def normalize_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = s.lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s

def load_items(jsonl_path: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            obj.setdefault("category", "기타")
            obj.setdefault("severity", "Low")
            text = (obj.get("canonical_ko") or obj.get("text") or "").strip()
            if not text:
                continue
            obj["text"] = text
            items.append(obj)
    if not items:
        raise ValueError("no items in JSONL")
    return items

def load_eval(csv_path: Path) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            pairs.append((r["query"], r["gt_category"]))
    return pairs

# ------------------------------------------------
# 3) 간단 TF-IDF 리트리버 (top-k)
# ------------------------------------------------
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class SimpleTFIDFRetriever:
    def __init__(self, items: List[Dict[str, Any]], ngram=(1,2)):
        self.items = items
        self.vectorizer = TfidfVectorizer(min_df=1, ngram_range=ngram)
        self.mat = self.vectorizer.fit_transform([normalize_text(x["text"]) for x in items])

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        qv = self.vectorizer.transform([normalize_text(query)])
        sims = cosine_similarity(qv, self.mat)[0]
        idxs = np.argsort(-sims)[:top_k]
        out = []
        for i in idxs:
            it = dict(self.items[int(i)])
            it["score"] = float(sims[int(i)])
            out.append(it)
        return out

# ------------------------------------------------
# 4) 평가용 엔진 래퍼
# ------------------------------------------------
class ResponseEngineForEval(ResponseEngineRAG):
    def build_response_with_flags(self, user_query: str) -> Dict[str, Any]:
        q = (user_query or "").strip()
        flags = {
            "routed": False,
            "routed_category": None,
            "used_retriever": self.retriever is not None,
        }

        best_item: Optional[Dict[str, Any]] = None
        if self.retriever is not None:
            try:
                cands = self.retriever.retrieve(q, top_k=self.top_k) or []
                cands = [c for c in cands if str(c.get("text") or c.get("canonical_ko") or "").strip()]
                if cands:
                    best_item = {
                        "text": (cands[0].get("text") or cands[0].get("canonical_ko") or "").strip(),
                        "category": cands[0].get("category", "기타"),
                        "severity": cands[0].get("severity", "Low"),
                    }
            except Exception:
                best_item = None

        if best_item is None:
            idx, max_sim, has_overlap = self.fallback.top1(q)
            if has_overlap and max_sim > 0:
                best_item = self.items[idx]
            else:
                routed_cat = route_category_by_keywords(q)
                if routed_cat:
                    flags["routed"] = True
                    flags["routed_category"] = routed_cat
                    cat_items = [x for x in self.items if x.get("category") == routed_cat]
                    highs = [x for x in cat_items if is_high(x.get("severity", "Low"))]
                    best_item = highs[0] if highs else (cat_items[0] if cat_items else self.items[0])
                else:
                    best_item = self.items[0]

        main = best_item["text"]
        pred_cat = best_item.get("category", "기타")
        return {"main": main, "pred_category": pred_cat, **flags}

# ------------------------------------------------
# 5) 지표 함수
# ------------------------------------------------
def hit_at_k(pred_cats: List[str], gt_cat: str, k: int) -> int:
    return int(gt_cat in pred_cats[:k])

def mrr_at_k(pred_cats: List[str], gt_cat: str, k: int) -> float:
    for rank, c in enumerate(pred_cats[:k], start=1):
        if c == gt_cat:
            return 1.0 / rank
    return 0.0

# ------------------------------------------------
# 6) 메인 평가
# ------------------------------------------------
def evaluate(eval_file: Path):
    assert JSONL.exists(), f"not found: {JSONL}"
    assert eval_file.exists(), f"not found: {eval_file}"

    items = load_items(JSONL)
    eval_pairs = load_eval(eval_file)

    engine = ResponseEngineForEval(JSONL, retriever=None, retriever_top_k=6)
    ext_retriever = SimpleTFIDFRetriever(items, ngram=(1,2))

    total = 0
    top1_cat_correct = 0
    routed_cnt = 0
    routed_cat_correct = 0
    per_cat_total = Counter()
    per_cat_correct = Counter()

    ks = [3, 5]
    hits_k = {k: [] for k in ks}
    mrrs_k = {k: [] for k in ks}

    for q, gt_cat in eval_pairs:
        total += 1
        per_cat_total[gt_cat] += 1

        # Retrieval 지표
        results = ext_retriever.retrieve(q, top_k=max(ks))
        pred_cats = [r["category"] for r in results]
        for k in ks:
            hits_k[k].append(hit_at_k(pred_cats, gt_cat, k))
            mrrs_k[k].append(mrr_at_k(pred_cats, gt_cat, k))

        # Pipeline + Routing
        r = engine.build_response_with_flags(q)
        pred_cat = r["pred_category"]

        if pred_cat == gt_cat:
            top1_cat_correct += 1
            per_cat_correct[gt_cat] += 1

        if r["routed"]:
            routed_cnt += 1
            if r["routed_category"] == gt_cat:
                routed_cat_correct += 1

    # 결과 출력
    print("=== RAG / Routing Evaluation ===")
    print(f"Eval file: {eval_file.name}")
    print(f"Total: {total}")
    for k in ks:
        print(f"Hit@{k}: {np.mean(hits_k[k]):.3f} | MRR@{k}: {np.mean(mrrs_k[k]):.3f}")
    print(f"Top-1 Category Accuracy: {top1_cat_correct/total:.3f}")
    print(f"Routing Trigger Rate: {routed_cnt/total:.3f} ({routed_cnt}/{total})")
    if routed_cnt > 0:
        print(f"Routing Accuracy (when triggered): {routed_cat_correct/routed_cnt:.3f}")

    print("\nPer-Category Accuracy:")
    for cat in sorted(per_cat_total.keys()):
        acc = per_cat_correct[cat]/per_cat_total[cat]
        print(f" - {cat}: {acc:.3f} ({per_cat_correct[cat]}/{per_cat_total[cat]})")

# ------------------------------------------------
# 7) CLI
# ------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-file", type=str, required=True, help="CSV 평가 파일 경로 (예: data/processed/test_queries.csv)")
    args = parser.parse_args()
    evaluate(Path(args.eval_file))
