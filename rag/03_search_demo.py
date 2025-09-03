"""
파일명: rag/03_search_demo.py
목적: FAISS 인덱스 + E5 임베딩으로 검색 테스트
"""

from pathlib import Path
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import unicodedata, re, json

ROOT = Path(__file__).resolve().parents[1]
VEC_DIR = ROOT / "data" / "vectorstore"
INDEX_PATH = VEC_DIR / "faiss.index"
META_PATH = VEC_DIR / "meta.npy"

# ✅ 확정 모델
MODEL_NAME = "intfloat/multilingual-e5-base"

def normalize_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s

def get_retriever():
    model = SentenceTransformer(MODEL_NAME)
    index = faiss.read_index(str(INDEX_PATH))
    items = np.load(META_PATH, allow_pickle=True).tolist()

    class Retriever:
        def __init__(self, model, index, items):
            self.model = model
            self.index = index
            self.items = items
        def retrieve(self, query: str, top_k: int = 5):
            q = "query: " + query  # E5 포맷
            qv = self.model.encode([q], convert_to_numpy=True)[0]
            qv = qv / (np.linalg.norm(qv) + 1e-12)
            D, I = self.index.search(np.array([qv]), top_k)
            return [dict(self.items[i], score=float(D[0][j])) for j,i in enumerate(I[0])]
    return Retriever(model, index, items)

if __name__ == "__main__":
    retr = get_retriever()
    q = "운동 전 주의사항 알려줘"
    results = retr.retrieve(q, top_k=3)
    for r in results:
        print(r["category"], r["text"], f"score={r['score']:.3f}")
