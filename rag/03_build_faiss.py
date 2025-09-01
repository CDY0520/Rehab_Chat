"""
파일: rag/03_build_faiss.py
목적: precautions.jsonl -> 임베딩 생성 -> FAISS 인덱스 + 메타데이터 저장
요구: sentence-transformers, faiss-cpu, pandas, numpy
"""

from pathlib import Path
import json
import pickle
import numpy as np

# 1) 한국어 포함 멀티링궐 임베딩 모델
from sentence_transformers import SentenceTransformer
import faiss

ROOT = Path(__file__).resolve().parents[1]
IN_JSONL = ROOT / "data" / "processed" / "precautions.jsonl"
OUT_DIR  = ROOT / "data" / "vectorstore"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 저장 파일들
FAISS_INDEX_PATH = OUT_DIR / "faiss.index"
META_PATH        = OUT_DIR / "meta.pkl"
MODEL_NAME       = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"  # KO 지원, 가벼움

def load_records(path: Path):
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records

def main():
    if not IN_JSONL.exists():
        raise FileNotFoundError(f"입력 파일 없음: {IN_JSONL}")
    print("[1/4] JSONL 로드")
    records = load_records(IN_JSONL)

    texts = [r["canonical_ko"] for r in records]
    ids   = [r["id"] for r in records]

    print(f"[2/4] 임베딩 생성 (model={MODEL_NAME}, n={len(texts)})")
    model = SentenceTransformer(MODEL_NAME)
    emb = model.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    emb = np.asarray(emb, dtype="float32")

    print("[3/4] FAISS 인덱스 빌드")
    dim = emb.shape[1]
    index = faiss.IndexFlatIP(dim)  # 코사인 유사도용: normalize_embeddings=True + Inner Product
    index.add(emb)
    faiss.write_index(index, str(FAISS_INDEX_PATH))

    print("[4/4] 메타데이터 저장")
    # 검색 시 문장/메타 참조용
    meta = {
        "ids": ids,
        "texts": texts,
        "records": records,  # 전체 메타 포함(category, severity, intent, source_title/page ...)
        "model_name": MODEL_NAME,
        "normalize": True,
        "metric": "cosine/IP"
    }
    with open(META_PATH, "wb") as f:
        pickle.dump(meta, f)

    print(f"[OK] 인덱스 저장: {FAISS_INDEX_PATH}")
    print(f"[OK] 메타 저장   : {META_PATH}")
    print(f"총 {len(records)}개 문장 인덱싱 완료")

if __name__ == "__main__":
    main()
