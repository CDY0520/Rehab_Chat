"""
파일명: src/retrieve.py
기능: FAISS 인덱스와 동일 임베딩 모델을 사용해 사용자의 질의를 벡터화하고,
     top-k 관련 청크를 검색하여 (text, meta, score) 형태로 반환한다.
     - 로드: rag/index.faiss, rag/texts.jsonl, rag/meta.jsonl
     - 임베딩: intfloat/multilingual-e5-base (index_build.py와 동일 모델 사용)
     - 카테고리 필터 옵션 제공(category ∈ {생활지도, 운동처방, 보행프로그램, 가정운동, 뇌졸중기본})
     - 반환: [{idx, score, text, meta}, ...]

블록 구성
 0) 라이브러리 임포트
 1) 경로/모델 상수
 2) 유틸: jsonl 로더
 3) Retriever 클래스 (search)
 4) 모듈 단독 테스트(직접 실행 시)

주의
 - index_build.py에서 생성된 rag/* 파일이 있어야 한다.
 - 인덱싱과 검색은 동일 임베딩 공간을 사용해야 한다(모델명 일치).
 - 점수는 내적 값(IP). normalize_embeddings=True이면 코사인 유사도와 동등하게 해석 가능.
"""
# 0) 라이브러리 임포트
from __future__ import annotations
from pathlib import Path
import json
from typing import List, Dict, Optional

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# 1) 경로/모델 상수 ------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
RAG_DIR = ROOT / "rag"

INDEX_PATH = RAG_DIR / "index.faiss"
TEXTS_PATH = RAG_DIR / "texts.jsonl"
META_PATH  = RAG_DIR / "meta.jsonl"

# 인덱싱과 동일 모델 사용 (index_build.py와 동일해야 함)
MODEL_NAME = "intfloat/multilingual-e5-base"


# 2) 유틸: jsonl 로더 ---------------------------------------------
def _load_jsonl_strings(path: Path) -> List[str]:
    """
    목적: 한 줄당 문자열(JSON 인코딩된 str)을 담은 jsonl 로드
    형식 예: "텍스트 본문..."
    """
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def _load_jsonl_lines(path: Path) -> List[Dict]:
    """
    목적: 한 줄당 dict를 담은 jsonl 로드
    형식 예: {"source_title": "...", "page": 3, "category": "..."}
    """
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


# 3) Retriever 클래스 ----------------------------------------------
class Retriever:
    """
    RAG 검색기
    - 질의 임베딩 → FAISS 검색 → (옵션) 카테고리 필터 → 정렬/정리
    """
    def __init__(
        self,
        index_path: Path = INDEX_PATH,
        texts_path: Path = TEXTS_PATH,
        meta_path: Path = META_PATH,
        model_name: str = MODEL_NAME,
    ):
        # 인덱스 로드
        if not index_path.exists():
            raise FileNotFoundError(f"FAISS 인덱스가 없습니다: {index_path}")
        self.index = faiss.read_index(str(index_path))

        # 매핑 파일 로드
        if not texts_path.exists() or not meta_path.exists():
            raise FileNotFoundError(f"매핑 파일이 없습니다: {texts_path}, {meta_path}")
        self.texts: List[str] = _load_jsonl_strings(texts_path)
        self.metas:  List[Dict] = _load_jsonl_lines(meta_path)

        # 질의 임베딩 모델
        self.model = SentenceTransformer(model_name)

        # 간단 검증: 매핑 수와 인덱스 크기 일치 여부(경고 수준)
        if len(self.texts) != len(self.metas):
            print(f"[경고] texts({len(self.texts)}) != metas({len(self.metas)}) 크기 불일치")
        if self.index.ntotal != len(self.texts):
            print(f"[경고] index.ntotal({self.index.ntotal}) != chunks({len(self.texts)}) 불일치")

    def search(self, query: str, k: int = 6, category: Optional[str] = None) -> List[Dict]:
        """
        질의문을 임베딩 후, 내적(top-k) 검색 결과를 반환한다.
        - category가 주어지면 해당 카테고리만 필터링
        - 반환: [{idx, score, text, meta}, ...] (score는 내적값)
        """
        if not query or not query.strip():
            return []

        q = self.model.encode([query], normalize_embeddings=True)
        D, I = self.index.search(np.asarray(q, dtype="float32"), k)

        results: List[Dict] = []
        for score, idx in zip(D[0], I[0]):
            if idx < 0:  # 방어코드(검색 실패 슬롯)
                continue
            meta = self.metas[idx]
            if category and meta.get("category") != category:
                continue
            results.append({
                "idx": int(idx),
                "score": float(score),
                "text": self.texts[idx],
                "meta": meta,
            })
        return results


# 4) 모듈 단독 테스트 ---------------------------------------------
if __name__ == "__main__":
    r = Retriever()
    q = "무릎 아플 때 운동 후 주의사항 알려줘"
    hits = r.search(q, k=6, category=None)
    print(f"질문: {q}")
    for i, h in enumerate(hits, 1):
        m = h["meta"]
        print(f"[{i}] s={h['score']:.3f} p.{m.get('page','?')} {m.get('category','?')} | {m.get('source_title','')}")
        print(" └", h["text"][:120].replace("\n", " "), "...")
