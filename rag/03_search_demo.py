# -*- coding: utf-8 -*-
"""
03_search_demo.py
- 인덱스 로드(get_retriever)
- 카테고리 인지형 검색(search)
- app_demo/answer_demo에서 그대로 재사용 가능
"""

from __future__ import annotations
from typing import Any, Dict, List, Tuple
from pathlib import Path
import re
import numpy as np

# ====== 파일 경로 설정 ======
HERE = Path(__file__).resolve().parent
VSTORE = HERE / "vectorstore"
FAISS_PATH = VSTORE / "faiss.index"
META_PATH = VSTORE / "meta.npy"

# ====== 공통 유틸 ======
def _nz(s: str) -> str:
    return re.sub(r"\s+", "", str(s or "")).lower()

def _looks_like_faiss_index(obj: Any) -> bool:
    return hasattr(obj, "search") and callable(getattr(obj, "search"))

def _looks_like_st_model(obj: Any) -> bool:
    return hasattr(obj, "encode") and callable(getattr(obj, "encode"))

def _looks_like_items(obj: Any) -> bool:
    return isinstance(obj, list) and (len(obj) == 0 or isinstance(obj[0], dict))

def _normalize_triplet(retr) -> Tuple[Any, Any, List[Dict]]:
    if isinstance(retr, dict):
        idx = retr.get("index") or retr.get("faiss") or retr.get("faiss_index")
        mdl = retr.get("model") or retr.get("encoder") or retr.get("st_model")
        itm = retr.get("items") or retr.get("docs") or retr.get("metas")
        if idx is not None and mdl is not None and itm is not None:
            return idx, mdl, itm
    if hasattr(retr, "index") and hasattr(retr, "model") and hasattr(retr, "items"):
        return retr.index, retr.model, retr.items
    if isinstance(retr, (tuple, list)):
        idx = mdl = itm = None
        for x in retr:
            if idx is None and _looks_like_faiss_index(x): idx = x; continue
            if mdl is None and _looks_like_st_model(x):   mdl = x; continue
            if itm is None and _looks_like_items(x):       itm = x; continue
        if idx is not None and mdl is not None and itm is not None:
            return idx, mdl, itm
    raise TypeError("retriever를 (index, model, items)로 정규화할 수 없습니다.")

def _get_item_category(meta: Dict) -> str:
    return meta.get("category") or meta.get("Category") or meta.get("섹션") or meta.get("section") or ""

# ====== 인덱스 로드 ======
def get_retriever():
    """FAISS + SentenceTransformer + items 로드. 반환 형식은 dict로 고정."""
    import faiss                     # faiss-cpu
    from sentence_transformers import SentenceTransformer

    if not FAISS_PATH.exists() or not META_PATH.exists():
        raise FileNotFoundError("vectorstore가 없습니다. 03_build_faiss.py 먼저 실행하세요.")

    index = faiss.read_index(str(FAISS_PATH))
    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    items = np.load(str(META_PATH), allow_pickle=True).tolist()
    # items: [{text, category, severity, intent, keywords, ...}, ...]
    return {"index": index, "model": model, "items": items}

# ====== Dense 검색 ======
def _faiss_search(index, model, items, query: str, top_k: int) -> List[Dict]:
    q_emb = model.encode([query], normalize_embeddings=True)
    if hasattr(q_emb, "astype"):
        q_emb = q_emb.astype("float32")
    else:
        q_emb = np.array(q_emb, dtype="float32")
    D, I = index.search(q_emb, top_k)
    scores = D[0].tolist(); idxs = I[0].tolist()
    out = []
    for rk, (i, sc) in enumerate(zip(idxs, scores), 1):
        if i < 0 or i >= len(items): continue
        meta = items[i] or {}
        text = meta.get("text") or meta.get("sentence") or meta.get("content") or str(meta)
        out.append({"rank": rk, "text": text, "meta": meta, "score": float(sc)})
    return out

# ====== 자동 카테고리 학습 ======
def _top_keywords(texts: List[str], topn=12) -> List[str]:
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        vec = TfidfVectorizer(analyzer="word", token_pattern=r"(?u)\b\w+\b",
                              ngram_range=(1,2), max_features=5000)
        X = vec.fit_transform(texts)
        scores = np.asarray(X.sum(axis=0)).ravel()
        feats = np.array(vec.get_feature_names_out())
        order = np.argsort(scores)[::-1]
        return [feats[i] for i in order[:topn].tolist()]
    except Exception:
        from collections import Counter
        toks=[]
        for t in texts: toks += [w for w in re.split(r"[,\s/;]+", t) if w]
        return [w for w,_ in Counter(toks).most_common(topn)]

def build_auto_categories(items: List[Dict], _model) -> Dict[str, Dict[str, Any]]:
    """카테고리별 프로토타입/별칭/별칭임베딩 생성(_model은 캐시 해시 제외용 이름)."""
    cat2txt = {}
    for it in items:
        cat = _get_item_category(it or {})
        txt = (it or {}).get("text") or (it or {}).get("sentence") or (it or {}).get("content") or ""
        if not cat or not txt: continue
        cat2txt.setdefault(cat, []).append(str(txt))
    auto={}
    for cat, texts in cat2txt.items():
        sample = texts[:200]
        emb = _model.encode(sample, normalize_embeddings=True)
        if isinstance(emb, list): emb = np.array(emb, dtype="float32")
        proto = emb.mean(axis=0)
        aliases = _top_keywords(sample, topn=12)
        alias_emb = _model.encode(aliases, normalize_embeddings=True)
        alias_emb = np.array(alias_emb, dtype="float32")
        auto[cat] = {"prototype": proto.astype("float32"),
                     "aliases": aliases, "alias_emb": alias_emb}
    return auto

def _cos(a,b): return float(np.dot(a,b))

def infer_category_auto(query: str, model, AUTO: Dict[str, Dict[str,Any]],
                        proto_w=1.0, alias_w=0.8, th=0.35) -> str|None:
    if not AUTO: return None
    q = np.array(model.encode([query], normalize_embeddings=True)[0], dtype="float32")
    best_cat, best = None, -1.0
    for cat, info in AUTO.items():
        s_proto = _cos(q, info["prototype"])
        s_alias = float(np.max(info["alias_emb"] @ q)) if info["alias_emb"].size else 0.0
        s = proto_w*s_proto + alias_w*s_alias
        if s>best: best, best_cat = s, cat
    return best_cat if best>=th else None

# ---- 강건 매칭(핵심키 + 안전 alias) ----
def _cat_key(cat: str) -> str:
    s = _nz(cat)
    if "운동전" in s: return "전"
    if "운동중" in s: return "중"
    if "운동후" in s: return "후"
    if any(k in s for k in ["근력","저항","웨이트","strength","resistance"]): return "근력"
    if any(k in s for k in ["균형","밸런스","balance","stability"]):           return "균형"
    if any(k in s for k in ["스트레칭","유연","rom","flex","flexibility"]):     return "스트레칭"
    return s

STOP_ALIASES = {"운동","주의사항","exercise","note","주의"}

def category_matches(item_cat: str, target_cat: str, AUTO) -> bool:
    if not item_cat or not target_cat: return False
    if _cat_key(item_cat) == _cat_key(target_cat): return True
    for alias in AUTO.get(target_cat, {}).get("aliases", []):
        a = _nz(alias)
        if len(a)>=3 and a not in STOP_ALIASES and a in _nz(item_cat):
            return True
    return False

def _rerank_by_category(results, cat_hint, AUTO):
    if not cat_hint: return results
    same, other = [], []
    for r in results:
        cat = _get_item_category(r.get("meta", {}) or {})
        (same if category_matches(cat, cat_hint, AUTO) else other).append(r)
    return same + other

def _count_cat(results, cat_hint, AUTO):
    if not cat_hint: return 0
    return sum(1 for r in results if category_matches(_get_item_category(r.get("meta", {}) or {}), cat_hint, AUTO))

# ====== 공개 검색 API ======
def search(retr, query: str, top_k: int = 3) -> List[Dict]:
    """카테고리 인지형 검색. (app_demo/answer_demo에서 공용 호출)"""
    index, model, items = _normalize_triplet(retr)
    AUTO = build_auto_categories(items, model)

    # 1차 검색
    res = _faiss_search(index, model, items, query, top_k)
    cat_hint = infer_category_auto(query, model, AUTO)

    # 정렬/재정렬
    res = sorted(res, key=lambda x: x["score"], reverse=True)
    res = _rerank_by_category(res, cat_hint, AUTO)

    # 오버페치
    if cat_hint and _count_cat(res, cat_hint, AUTO) == 0:
        big = _faiss_search(index, model, items, query, max(top_k*10,30))
        big = sorted(big, key=lambda x:x["score"], reverse=True)
        big = _rerank_by_category(big, cat_hint, AUTO)
        if _count_cat(big, cat_hint, AUTO)>0:
            return big[:top_k]

    # 쿼리 확장
    if cat_hint and _count_cat(res, cat_hint, AUTO) == 0:
        aliases = " ".join(AUTO.get(cat_hint,{}).get("aliases",[])[:8])
        q2 = f"{query} {aliases}"
        big = _faiss_search(index, model, items, q2, max(top_k*10,30))
        big = sorted(big, key=lambda x:x["score"], reverse=True)
        big = _rerank_by_category(big, cat_hint, AUTO)
        if _count_cat(big, cat_hint, AUTO)>0:
            return big[:top_k]

    return res
