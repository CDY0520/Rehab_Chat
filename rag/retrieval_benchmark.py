"""
# 파일명: rag/retrieval_benchmark.py
# 목적: TF-IDF / BM25 / 임베딩(여러 모델) 리트리버 성능 비교
# 지표:
#   (A) Retrieval만: Hit@k, MRR@k
#   (B) Pipeline: Top-1 Category Accuracy
#     - 우리 엔진(ResponseEngineRAG)에 주입하여 E2E로 측정
#
# 실행 예:
#   python rag/retrieval_benchmark.py --k 5
#   python rag/retrieval_benchmark.py --k 5 --embed-models "intfloat/multilingual-e5-base"
"""# ============================================"""

# 라이브러리 임포트
from __future__ import annotations
import argparse
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
from pathlib import Path
import numpy as np
import json, re, unicodedata
import importlib.util

ROOT = Path(__file__).resolve().parents[1]
JSONL = ROOT / "data" / "processed" / "precautions.jsonl"
EVAL_CSV = ROOT / "data" / "processed" / "eval_queries.csv"
ENGINE_PATH = Path(__file__).resolve().parent / "04_answer_demo.py"

# 엔진 로드 (E2E 평가용)
spec = importlib.util.spec_from_file_location("engine_mod", str(ENGINE_PATH))
engine_mod = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(engine_mod)  # type: ignore
ResponseEngineRAG = engine_mod.ResponseEngineRAG

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
            if not line: continue
            obj = json.loads(line)
            obj.setdefault("category", "기타")
            obj.setdefault("severity", "Low")
            text = (obj.get("canonical_ko") or obj.get("text") or "").strip()
            if not text: continue
            obj["text"] = text
            items.append(obj)
    if not items:
        raise ValueError("no items")
    return items

def load_eval(csv_path: Path) -> List[Tuple[str, str]]:
    import csv
    rows = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append((r["query"], r["gt_category"]))
    return rows

# ---- Retriever 인터페이스 ----
class BaseRetriever:
    def __init__(self, items: List[Dict[str, Any]]):
        self.items = items
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        raise NotImplementedError

# TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class TFIDFRetriever(BaseRetriever):
    def __init__(self, items: List[Dict[str, Any]], ngram=(1,2)):
        super().__init__(items)
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

# BM25
from rank_bm25 import BM25Okapi
class BM25Retriever(BaseRetriever):
    def __init__(self, items: List[Dict[str, Any]]):
        super().__init__(items)
        self.tokenized = [normalize_text(x["text"]).split() for x in items]
        self.bm25 = BM25Okapi(self.tokenized)
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        q = normalize_text(query).split()
        scores = self.bm25.get_scores(q)
        idxs = np.argsort(-scores)[:top_k]
        out = []
        for i in idxs:
            it = dict(self.items[int(i)])
            it["score"] = float(scores[int(i)])
            out.append(it)
        return out

# Embedding
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

class EmbeddingRetriever(BaseRetriever):
    def __init__(self, items: List[Dict[str, Any]], model_name: str):
        if SentenceTransformer is None:
            raise ImportError("sentence-transformers가 설치되어야 합니다.")
        super().__init__(items)
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.is_e5 = "e5" in model_name.lower()
        corpus = [("passage: " + x["text"]) if self.is_e5 else x["text"] for x in items]
        self.emb = self.model.encode(corpus, convert_to_numpy=True, show_progress_bar=True)
        self.emb = self.emb / (np.linalg.norm(self.emb, axis=1, keepdims=True) + 1e-12)
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        q = "query: " + query if self.is_e5 else query
        qv = self.model.encode([q], convert_to_numpy=True)[0]
        qv = qv / (np.linalg.norm(qv) + 1e-12)
        sims = self.emb @ qv
        idxs = np.argsort(-sims)[:top_k]
        out = []
        for i in idxs:
            it = dict(self.items[int(i)])
            it["score"] = float(sims[int(i)])
            out.append(it)
        return out

# ---- Retrieval 지표 ----
def hit_at_k(pred_cats: List[str], gt_cat: str, k: int) -> int:
    return int(gt_cat in pred_cats[:k])

def mrr_at_k(pred_cats: List[str], gt_cat: str, k: int) -> float:
    for rank, c in enumerate(pred_cats[:k], start=1):
        if c == gt_cat:
            return 1.0 / rank
    return 0.0

def eval_retrieval(retriever: BaseRetriever, eval_pairs: List[Tuple[str,str]], k: int = 5) -> Dict[str, float]:
    hits, mrrs = [], []
    for q, gt_cat in eval_pairs:
        results = retriever.retrieve(q, top_k=k)
        pred_cats = [r["category"] for r in results]
        hits.append(hit_at_k(pred_cats, gt_cat, k))
        mrrs.append(mrr_at_k(pred_cats, gt_cat, k))
    return {f"Hit@{k}": float(np.mean(hits)), f"MRR@{k}": float(np.mean(mrrs))}

# ---- 파이프라인 E2E 정확도 ----
class RetrieverAdapter:
    def __init__(self, retriever: BaseRetriever):
        self._retriever = retriever
    def retrieve(self, query: str, top_k: int = 6):
        return self._retriever.retrieve(query, top_k=top_k)

def eval_pipeline(retriever: BaseRetriever, eval_pairs: List[Tuple[str,str]]) -> float:
    engine = ResponseEngineRAG(JSONL, retriever=RetrieverAdapter(retriever), retriever_top_k=6)
    correct = 0
    for q, gt_cat in eval_pairs:
        r = engine.build_response(q)
        if r["category_title"] == gt_cat:
            correct += 1
    return correct / len(eval_pairs)

# ---- 메인 ----
@dataclass
class SystemSpec:
    name: str
    make: callable  # items -> BaseRetriever

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--embed-models", nargs="*", default=[
        "intfloat/multilingual-e5-base",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    ])
    parser.add_argument("--no-embed", action="store_true")
    args = parser.parse_args()

    items = load_items(JSONL)
    eval_pairs = load_eval(EVAL_CSV)

    systems: List[SystemSpec] = [
        SystemSpec(name="TF-IDF(1,2)", make=lambda its: TFIDFRetriever(its, ngram=(1,2))),
        SystemSpec(name="BM25",        make=lambda its: BM25Retriever(its)),
    ]
    if not args.no_embed:
        for m in args.embed_models:
            systems.append(SystemSpec(name=f"Embed:{m}", make=lambda its, mn=m: EmbeddingRetriever(its, mn)))

    print(f"총 평가 쿼리 수: {len(eval_pairs)} | top-k={args.k}")
    print("-"*80)

    for spec in systems:
        print(f"[{spec.name}] 인덱싱/평가 중...")
        retr = spec.make(items)

        r_metrics = eval_retrieval(retr, eval_pairs, k=args.k)
        e2e_acc  = eval_pipeline(retr, eval_pairs)

        print(f"  Retrieval -> Hit@{args.k}: {r_metrics[f'Hit@{args.k}']:.3f} | MRR@{args.k}: {r_metrics[f'MRR@{args.k}']:.3f}")
        print(f"  Pipeline  -> Top-1 Category Accuracy: {e2e_acc:.3f}")
        print("-"*80)

if __name__ == "__main__":
    main()
