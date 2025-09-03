"""
# 파일명: rag/rag_eval_plot.py
# 목적: rag_eval.py와 동일하게 평가하되,
#       결과를 표 형태로 터미널에 출력 (저장 X)
#
# 실행:
#   python rag/rag_eval_plot.py --eval-file data/processed/test_queries.csv
"""

# 라이브러리 임포트
from __future__ import annotations
import csv
import argparse
from pathlib import Path
from collections import Counter
import numpy as np
import importlib.util
import json, re, unicodedata

# 경로 설정
THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parent
ENGINE_PATH = THIS_DIR / "04_answer_demo.py"
JSONL = ROOT / "data" / "processed" / "precautions.jsonl"

# ----------- 유틸 -----------
def normalize_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s

def load_items(jsonl_path: Path):
    items = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            obj.setdefault("category", "기타")
            obj.setdefault("severity", "Low")
            text = (obj.get("canonical_ko") or obj.get("text") or "").strip()
            if text:
                obj["text"] = text
                items.append(obj)
    return items

def load_eval(csv_path: Path):
    pairs = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            pairs.append((r["query"], r["gt_category"]))
    return pairs

# TF-IDF 리트리버 (간단)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class SimpleTFIDFRetriever:
    def __init__(self, items, ngram=(1,2)):
        self.items = items
        self.vectorizer = TfidfVectorizer(min_df=1, ngram_range=ngram)
        self.mat = self.vectorizer.fit_transform([normalize_text(x["text"]) for x in items])
    def retrieve(self, query, top_k=5):
        qv = self.vectorizer.transform([normalize_text(query)])
        sims = cosine_similarity(qv, self.mat)[0]
        idxs = np.argsort(-sims)[:top_k]
        return [dict(self.items[i], score=float(sims[i])) for i in idxs]

# ----------- 엔진 로드 -----------
spec = importlib.util.spec_from_file_location("engine_mod", str(ENGINE_PATH))
engine_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(engine_mod)  # type: ignore
ResponseEngineRAG = engine_mod.ResponseEngineRAG
route_category_by_keywords = engine_mod.route_category_by_keywords
is_high = engine_mod.is_high

class ResponseEngineForEval(ResponseEngineRAG):
    def build_response_with_flags(self, q: str):
        flags = {"routed": False, "routed_category": None}
        best_item = None
        if self.retriever:
            cands = self.retriever.retrieve(q, top_k=self.top_k)
            if cands:
                best_item = dict(text=cands[0]["text"], category=cands[0]["category"], severity=cands[0]["severity"])
        if best_item is None:
            routed_cat = route_category_by_keywords(q)
            if routed_cat:
                flags["routed"] = True
                flags["routed_category"] = routed_cat
                cat_items = [x for x in self.items if x["category"] == routed_cat]
                highs = [x for x in cat_items if is_high(x.get("severity", "Low"))]
                best_item = highs[0] if highs else (cat_items[0] if cat_items else self.items[0])
            else:
                best_item = self.items[0]
        return {"main": best_item["text"], "pred_category": best_item["category"], **flags}

# ----------- 지표 -----------
def hit_at_k(pred_cats, gt_cat, k):
    return int(gt_cat in pred_cats[:k])

def mrr_at_k(pred_cats, gt_cat, k):
    for rank, c in enumerate(pred_cats[:k], 1):
        if c == gt_cat:
            return 1.0 / rank
    return 0.0

# ----------- 평가 메인 -----------
def evaluate(eval_file: Path):
    items = load_items(JSONL)
    eval_pairs = load_eval(eval_file)
    engine = ResponseEngineForEval(JSONL, retriever=None, retriever_top_k=6)
    retriever = SimpleTFIDFRetriever(items)

    total, top1_cat_correct, routed_cnt, routed_cat_correct = 0,0,0,0
    per_cat_total, per_cat_correct = Counter(), Counter()
    ks = [3,5]
    hits, mrrs = {k: [] for k in ks}, {k: [] for k in ks}

    for q, gt_cat in eval_pairs:
        total += 1
        per_cat_total[gt_cat]+=1

        # retrieval
        results = retriever.retrieve(q, top_k=max(ks))
        pred_cats = [r["category"] for r in results]
        for k in ks:
            hits[k].append(hit_at_k(pred_cats, gt_cat, k))
            mrrs[k].append(mrr_at_k(pred_cats, gt_cat, k))

        # pipeline
        r = engine.build_response_with_flags(q)
        pred_cat = r["pred_category"]
        if pred_cat == gt_cat:
            top1_cat_correct+=1
            per_cat_correct[gt_cat]+=1
        if r["routed"]:
            routed_cnt+=1
            if r["routed_category"]==gt_cat:
                routed_cat_correct+=1

    # === 출력 ===
    print("=== RAG Evaluation ===")
    print(f"Eval file: {eval_file.name}")
    print(f"Total: {total}")
    for k in ks:
        print(f"Hit@{k}: {np.mean(hits[k]):.3f} | MRR@{k}: {np.mean(mrrs[k]):.3f}")
    print(f"Top-1 Category Accuracy: {top1_cat_correct/total:.3f}")
    print(f"Routing Trigger Rate: {routed_cnt/total:.3f} ({routed_cnt}/{total})")
    if routed_cnt > 0:
        print(f"Routing Accuracy (when triggered): {routed_cat_correct/routed_cnt:.3f}")

    print("\nPer-Category Accuracy:")
    print("{:<20} {:<10} {:<10} {:<10}".format("Category","Accuracy","Correct","Total"))
    for cat in sorted(per_cat_total.keys()):
        acc = per_cat_correct[cat]/per_cat_total[cat]
        print("{:<20} {:<10.3f} {:<10} {:<10}".format(cat, acc, per_cat_correct[cat], per_cat_total[cat]))

# ----------- CLI -----------
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-file",type=str,required=True)
    args=parser.parse_args()
    evaluate(Path(args.eval_file))
