"""
파일명: src/index_build.py
기능: 인제스트된 청크(JSONL)를 로드하여 임베딩을 생성하고 FAISS 인덱스를 구축한다.
     - (메모리 최적화) e5-base 임베딩 사용
     - FAISS IndexFlatIP(내적)로 벡터 인덱스 생성
     - 텍스트/메타 매핑 파일 저장(texts.jsonl, meta.jsonl)
     - 산출물: rag/index.faiss, rag/texts.jsonl, rag/meta.jsonl, rag/stats.json

블록 구성
 0) 라이브러리 임포트
 1) 경로/모델 설정(메모리 절약 값으로 조정)
 2) 입력(JSONL) 로드
 3) 임베딩 생성(배치 인코딩)
 4) FAISS 인덱스 생성/저장
 5) 매핑 파일 저장(텍스트/메타)
 6) 실행 엔트리포인트(main) + 간단 통계 출력

주의
 - 모델: intfloat/multilingual-e5-base (약 700~800MB) — Windows 페이징 파일 부족(1455) 방지
 - normalize_embeddings=True → 코사인 유사도와 동등한 내적 검색 가능
 - 배치/메모리 조절: BATCH_SIZE 값으로 튜닝(기본 8)
 - data/processed 폴더에 .jsonl 파일이 있어야 한다(2단계 결과물)
"""

# 0) 라이브러리 임포트
from __future__ import annotations
from pathlib import Path
import json
import numpy as np

import faiss
from sentence_transformers import SentenceTransformer

# 1) 경로/모델 설정(메모리 절약)
ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
RAG_DIR = ROOT / "rag"
RAG_DIR.mkdir(parents=True, exist_ok=True)

# ⬇️ large → base 로 교체 / 배치 축소
MODEL_NAME = "intfloat/multilingual-e5-base"   # 대안: "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
BATCH_SIZE = 8                                  # 메모리 부족 시 4까지 낮추세요

TEXTS_OUT = RAG_DIR / "texts.jsonl"
META_OUT  = RAG_DIR / "meta.jsonl"
INDEX_OUT = RAG_DIR / "index.faiss"
STATS_OUT = RAG_DIR / "stats.json"


# 2) 입력(JSONL) 로드
def load_jsonl_files(processed_dir: Path):
    """
    processed 디렉토리의 모든 .jsonl을 로드하고
    texts(list[str]), metas(list[dict])를 반환한다.
    - 각 줄은 {"text": "...", "meta": {...}} 형식이어야 한다.
    """
    files = sorted(processed_dir.glob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"인제스트 결과(JSONL)가 없습니다: {processed_dir}")
    texts, metas = [], []
    for fp in files:
        with fp.open(encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                txt = (obj.get("text") or "").strip()
                meta = obj.get("meta") or {}
                if not txt:
                    continue
                texts.append(txt)
                metas.append(meta)
    return texts, metas


# 3) 임베딩 생성
def embed_texts(texts: list[str], model_name: str = MODEL_NAME, batch_size: int = BATCH_SIZE):
    """
    Sentence-Transformers로 임베딩을 생성한다.
    - normalize_embeddings=True: 내적(IndexFlatIP)과 호환
    - 메모리 절약을 위해 배치 크기를 낮춰 사용
    """
    model = SentenceTransformer(model_name)
    emb = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    emb = np.asarray(emb, dtype="float32")
    return emb


# 4) FAISS 인덱스 생성/저장
def build_faiss_index(embeddings: np.ndarray, out_path: Path):
    """
    내적 기반(IndexFlatIP) FAISS 인덱스를 생성 후 저장한다.
    """
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embeddings)
    faiss.write_index(index, str(out_path))
    return index


# 5) 매핑 파일 저장
def save_mapping(texts: list[str], metas: list[dict], texts_out: Path, meta_out: Path):
    """
    검색 결과에서 인덱스 → 원문/메타를 복원하기 위해
    텍스트/메타를 각각 jsonl로 저장한다.
    """
    with texts_out.open("w", encoding="utf-8") as ft:
        for t in texts:
            ft.write(json.dumps(t, ensure_ascii=False) + "\n")
    with meta_out.open("w", encoding="utf-8") as fm:
        for m in metas:
            fm.write(json.dumps(m, ensure_ascii=False) + "\n")


# 6) 실행 엔트리포인트
def main():
    print("[1/4] JSONL 로드...")
    texts, metas = load_jsonl_files(PROCESSED_DIR)
    print(f" - 로드 완료: {len(texts)} chunks")

    print("[2/4] 임베딩 생성...")
    emb = embed_texts(texts)

    print("[3/4] FAISS 인덱스 구축...")
    _ = build_faiss_index(emb, INDEX_OUT)

    print("[4/4] 매핑 파일 저장...")
    save_mapping(texts, metas, TEXTS_OUT, META_OUT)

    stats = {
        "num_chunks": len(texts),
        "dim": int(emb.shape[1]),
        "model": MODEL_NAME,
        "batch_size": BATCH_SIZE,
        "index_path": str(INDEX_OUT.name),
        "texts_path": str(TEXTS_OUT.name),
        "meta_path": str(META_OUT.name),
    }
    with STATS_OUT.open("w", encoding="utf-8") as fs:
        json.dump(stats, fs, ensure_ascii=False, indent=2)

    print("✅ 완료")
    print(f" - 인덱스: {INDEX_OUT}")
    print(f" - 텍스트: {TEXTS_OUT}")
    print(f" - 메타   : {META_OUT}")
    print(f" - 통계   : {STATS_OUT}")


if __name__ == "__main__":
    main()
