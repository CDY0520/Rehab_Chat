"""
파일명: rag/03_build_faiss.py
목적: precautions.jsonl 데이터를 임베딩 → FAISS 인덱스 빌드
모델: intfloat/multilingual-e5-base (확정)
"""

from pathlib import Path
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import unicodedata, re

ROOT = Path(__file__).resolve().parents[1]
JSONL = ROOT / "data" / "processed" / "precautions.jsonl"
OUT_DIR = ROOT / "data" / "vectorstore"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def normalize_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s

# ✅ 확정 모델
MODEL_NAME = "intfloat/multilingual-e5-base"

def main():
    model = SentenceTransformer(MODEL_NAME)
    items = []
    with JSONL.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            text = (obj.get("canonical_ko") or obj.get("text") or "").strip()
            if not text:
                continue
            obj["text"] = text
            items.append(obj)
    texts = [("passage: " + x["text"]) for x in items]  # E5 포맷

    emb = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)

    dim = emb.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(emb)

    faiss.write_index(index, str(OUT_DIR / "faiss.index"))
    np.save(OUT_DIR / "meta.npy", items)

    print(f"FAISS index built with {len(items)} items using {MODEL_NAME}")

if __name__ == "__main__":
    main()
