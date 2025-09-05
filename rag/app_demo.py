# 파일명: rag/app_demo.py
# 목적:
#   - 03_search_demo.py 동적 로드
#   - get_retriever() 자동 해석
#   - [자동 카테고리] items에서 실제 카테고리 수집 → 프로토타입/별칭 자동 학습
#   - 카테고리 인지형 검색(재정렬 + 오버페치 + 쿼리 확장)
#   - 가중치 기반 스코어 보정 + 강제 카테고리 필터
#   - 전/중/후 혼선 방지: 핵심 키 비교 + 안전 alias 매칭 + STOP_ALIASES
# 실행: (.venv) streamlit run rag/app_demo.py

from __future__ import annotations

import sys
from pathlib import Path
import importlib.util
from typing import Any, Dict, List, Tuple

import re
import numpy as np
import streamlit as st

# ──────────────────────────────────────────────────────────────────────────────
# 0) importlib 로더
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_search_demo_module() -> Any:
    here = Path(__file__).resolve().parent
    module_path = here / "03_search_demo.py"
    if not module_path.exists():
        raise FileNotFoundError(f"동적 로드 대상 파일을 찾을 수 없습니다: {module_path}")

    spec = importlib.util.spec_from_file_location("search_demo", module_path)
    if spec is None or spec.loader is None:
        raise ImportError("importlib이 spec 로드를 실패했습니다 (spec/loader None).")

    search_demo = importlib.util.module_from_spec(spec)
    sys.modules["search_demo"] = search_demo
    spec.loader.exec_module(search_demo)
    return search_demo

# ──────────────────────────────────────────────────────────────────────────────
# 1) Retriever 생성(캐시)
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=True)
def get_retriever_cached():
    sd = load_search_demo_module()
    if not hasattr(sd, "get_retriever"):
        raise AttributeError("search_demo 모듈에 get_retriever 함수가 없습니다.")
    retr = sd.get_retriever()
    return retr, sd

# ──────────────────────────────────────────────────────────────────────────────
# 2) 타입 헬퍼 + (index, model, items) 정규화
# ──────────────────────────────────────────────────────────────────────────────
def _looks_like_faiss_index(obj: Any) -> bool:
    return hasattr(obj, "search") and callable(getattr(obj, "search"))

def _looks_like_st_model(obj: Any) -> bool:
    return hasattr(obj, "encode") and callable(getattr(obj, "encode"))

def _looks_like_items(obj: Any) -> bool:
    return isinstance(obj, list) and (len(obj) == 0 or isinstance(obj[0], dict))

def normalize_retriever_triplet(retr: Any) -> Tuple[Any, Any, List[Dict]]:
    if isinstance(retr, dict):
        idx = retr.get("index") or retr.get("faiss") or retr.get("faiss_index")
        mdl = retr.get("model") or retr.get("encoder") or retr.get("st_model")
        itm = retr.get("items") or retr.get("docs") or retr.get("metas")
        if idx is not None and mdl is not None and itm is not None:
            return idx, mdl, itm
    if hasattr(retr, "index") and hasattr(retr, "model") and hasattr(retr, "items"):
        return getattr(retr, "index"), getattr(retr, "model"), getattr(retr, "items")
    if isinstance(retr, (tuple, list)):
        idx = mdl = itm = None
        for x in retr:
            if idx is None and _looks_like_faiss_index(x):
                idx = x; continue
            if mdl is None and _looks_like_st_model(x):
                mdl = x; continue
            if itm is None and _looks_like_items(x):
                itm = x; continue
        if idx is not None and mdl is not None and itm is not None:
            return idx, mdl, itm
    raise TypeError("retriever를 (index, model, items)로 정규화할 수 없습니다.")

# ──────────────────────────────────────────────────────────────────────────────
# 3) Dense 검색 (FAISS)
# ──────────────────────────────────────────────────────────────────────────────
def run_query_faiss(index: Any, model: Any, items: List[Dict], query: str, top_k: int) -> List[Dict]:
    q_emb = model.encode([query], normalize_embeddings=True)
    if hasattr(q_emb, "astype"):
        q_emb = q_emb.astype("float32")
    else:
        q_emb = np.array(q_emb, dtype="float32")
    D, I = index.search(q_emb, top_k)
    scores = D[0].tolist()
    idxs = I[0].tolist()
    results: List[Dict] = []
    for rank, (i, sc) in enumerate(zip(idxs, scores), start=1):
        if i < 0 or i >= len(items):
            continue
        meta = items[i] or {}
        text = meta.get("text") or meta.get("sentence") or meta.get("content") or str(meta)
        results.append({"rank": rank, "text": text, "meta": meta, "score": float(sc)})
    return results

# ──────────────────────────────────────────────────────────────────────────────
# 4) [자동 카테고리] 모델 구축
# ──────────────────────────────────────────────────────────────────────────────
def _nz(s: str) -> str:
    return re.sub(r"\s+", "", str(s or "")).lower()

def _get_item_category(meta: Dict) -> str:
    return meta.get("category") or meta.get("Category") or meta.get("섹션") or meta.get("section") or ""

def _top_keywords(texts: List[str], topn: int = 12) -> List[str]:
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        vect = TfidfVectorizer(
            analyzer="word",
            token_pattern=r"(?u)\b\w+\b",
            ngram_range=(1, 2),
            min_df=1,
            max_features=5000,
        )
        X = vect.fit_transform(texts)
        scores = np.asarray(X.sum(axis=0)).ravel()
        feats = np.array(vect.get_feature_names_out())
        order = np.argsort(scores)[::-1]
        return [feats[i] for i in order[:topn].tolist()]
    except Exception:
        from collections import Counter
        toks = []
        for t in texts:
            toks.extend([w for w in re.split(r"[,\s/;]+", t) if w])
        cnt = Counter(toks)
        return [w for w, _ in cnt.most_common(topn)]

@st.cache_resource(show_spinner=False)
def build_auto_categories(items: List[Dict], _model: Any) -> Dict[str, Dict[str, Any]]:
    """
    items에서 카테고리별 프로토타입/별칭 자동 생성
    _model: SentenceTransformer (Streamlit 캐시 해시 제외용)
    """
    cat_to_texts: Dict[str, List[str]] = {}
    for it in items:
        meta = it or {}
        cat = _get_item_category(meta)
        text = meta.get("text") or meta.get("sentence") or meta.get("content") or ""
        if not text or not cat:
            continue
        cat_to_texts.setdefault(cat, []).append(str(text))

    auto: Dict[str, Dict[str, Any]] = {}
    if not cat_to_texts:
        return auto

    for cat, texts in cat_to_texts.items():
        sample = texts[:200]
        emb = _model.encode(sample, normalize_embeddings=True)
        if isinstance(emb, list):
            emb = np.array(emb, dtype="float32")
        proto = emb.mean(axis=0)

        aliases = _top_keywords(sample, topn=12)
        alias_emb = _model.encode(aliases, normalize_embeddings=True)
        alias_emb = np.array(alias_emb, dtype="float32")

        auto[cat] = {
            "prototype": proto.astype("float32"),
            "aliases": aliases,
            "alias_emb": alias_emb,
        }
    return auto

def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))

def infer_category_auto(query: str, model: Any, AUTO_CATS: Dict[str, Dict[str, Any]],
                        proto_w: float = 1.0, alias_w: float = 0.8,
                        th: float = 0.35) -> str | None:
    if not AUTO_CATS:
        return None
    q_emb = model.encode([query], normalize_embeddings=True)
    q = np.array(q_emb[0], dtype="float32")
    best_cat, best_score = None, -1.0
    for cat, info in AUTO_CATS.items():
        proto = info["prototype"]
        alias_emb = info["alias_emb"]
        s_proto = _cos(q, proto)
        s_alias = float(np.max(alias_emb @ q)) if alias_emb.size else 0.0
        s = proto_w * s_proto + alias_w * s_alias
        if s > best_score:
            best_score, best_cat = s, cat
    return best_cat if best_score >= th else None

# ── 핵심 키 기반 비교 + 안전 alias 매칭 ────────────────────────────────────────
def _cat_key(cat: str) -> str:
    s = _nz(cat)
    if "운동전" in s: return "전"
    if "운동중" in s: return "중"
    if "운동후" in s: return "후"
    if any(k in s for k in ["근력", "저항", "웨이트", "strength", "resistance"]): return "근력"
    if any(k in s for k in ["균형", "밸런스", "balance", "stability"]):           return "균형"
    if any(k in s for k in ["스트레칭", "유연", "rom", "flex", "flexibility"]):     return "스트레칭"
    return s

STOP_ALIASES = {"운동", "주의사항", "exercise", "note", "주의"}

def category_matches_auto(item_cat: str, target_cat: str,
                          AUTO_CATS: Dict[str, Dict[str, Any]]) -> bool:
    if not item_cat or not target_cat:
        return False
    # 1) 핵심 키 일치 우선
    if _cat_key(item_cat) == _cat_key(target_cat):
        return True
    # 2) 안전 alias 부분일치(3글자 이상 + 금지어 제외)
    for alias in AUTO_CATS.get(target_cat, {}).get("aliases", []):
        a = _nz(alias)
        if len(a) >= 3 and a not in STOP_ALIASES and a in _nz(item_cat):
            return True
    return False

# ──────────────────────────────────────────────────────────────────────────────
# 5) 카테고리 인지형 검색 파이프라인
# ──────────────────────────────────────────────────────────────────────────────
def rerank_by_category_auto(results: List[Dict], cat_hint: str | None,
                            AUTO_CATS: Dict[str, Dict[str, Any]]) -> List[Dict]:
    if not cat_hint:
        return results
    same, other = [], []
    for r in results:
        meta = r.get("meta", {}) or {}
        item_cat = _get_item_category(meta)
        (same if category_matches_auto(item_cat, cat_hint, AUTO_CATS) else other).append(r)
    return same + other

def count_matches_auto(results: List[Dict], cat_hint: str | None,
                       AUTO_CATS: Dict[str, Dict[str, Any]]) -> int:
    if not cat_hint:
        return 0
    return sum(1 for r in results
               if category_matches_auto(_get_item_category(r.get("meta", {}) or {}), cat_hint, AUTO_CATS))

def run_category_aware_search(sd: Any, retr: Any, query: str, top_k: int,
                              AUTO_CATS: Dict[str, Dict[str, Any]]) -> List[Dict]:
    index = model = items = None
    if hasattr(sd, "search"):
        try:
            results = sd.search(retr, query, top_k=top_k)
        except TypeError:
            results = sd.search(retr, query)
        try:
            index, model, items = normalize_retriever_triplet(retr)
        except Exception:
            pass
    elif callable(retr):
        try:
            results = retr(query, top_k=top_k)
        except TypeError:
            results = retr(query)
        try:
            index, model, items = normalize_retriever_triplet(retr)
        except Exception:
            pass
    else:
        index, model, items = normalize_retriever_triplet(retr)
        results = run_query_faiss(index, model, items, query, top_k)

    cat_hint = infer_category_auto(query, model, AUTO_CATS)

    results = sorted(results, key=lambda x: x.get("score", 0.0), reverse=True)
    results = rerank_by_category_auto(results, cat_hint, AUTO_CATS)

    if cat_hint and count_matches_auto(results, cat_hint, AUTO_CATS) == 0 and index is not None:
        over_k = max(top_k * 10, 30)
        big = run_query_faiss(index, model, items, query, over_k)
        big = sorted(big, key=lambda x: x.get("score", 0.0), reverse=True)
        big = rerank_by_category_auto(big, cat_hint, AUTO_CATS)
        if count_matches_auto(big, cat_hint, AUTO_CATS) > 0:
            results = big[:top_k]

    if cat_hint and count_matches_auto(results, cat_hint, AUTO_CATS) == 0 and index is not None:
        aliases = " ".join(AUTO_CATS.get(cat_hint, {}).get("aliases", [])[:8])
        q_exp = f"{query} {aliases}"
        big = run_query_faiss(index, model, items, q_exp, max(top_k * 10, 30))
        big = sorted(big, key=lambda x: x.get("score", 0.0), reverse=True)
        big = rerank_by_category_auto(big, cat_hint, AUTO_CATS)
        if count_matches_auto(big, cat_hint, AUTO_CATS) > 0:
            results = big[:top_k]

    return results

# ──────────────────────────────────────────────────────────────────────────────
# 6) 스코어 보정 유틸(카테고리/키워드 가중치)
# ──────────────────────────────────────────────────────────────────────────────
def normalize_korean(s: str) -> str:
    return re.sub(r"\s+", "", str(s)).lower()

def keyword_overlap_ratio(query: str, keywords) -> float:
    if not keywords:
        return 0.0
    if isinstance(keywords, str):
        kws = [k.strip() for k in re.split(r"[,\s/;]+", keywords) if k.strip()]
    else:
        kws = [str(k).strip() for k in keywords if str(k).strip()]
    if not kws:
        return 0.0
    q_tokens = [t for t in re.split(r"[,\s/;]+", query) if t]
    inter = len(set(map(normalize_korean, q_tokens)) & set(map(normalize_korean, kws)))
    return inter / max(len(set(kws)), 1)

def boosted_sort(results: List[Dict], query: str, cat_hint: str | None,
                 w_cat: float = 0.08, w_kw: float = 0.05,
                 AUTO_CATS: Dict[str, Dict[str, Any]] | None = None) -> List[Dict]:
    AUTO_CATS = AUTO_CATS or {}
    ranked = []
    for r in results:
        base = float(r.get("score", 0.0))
        meta = r.get("meta", {}) or {}
        cat = _get_item_category(meta)
        kw = meta.get("keywords") or meta.get("Keywords")
        cat_bonus = (w_cat if (cat_hint and category_matches_auto(cat, cat_hint, AUTO_CATS)) else 0.0)
        kw_bonus = w_kw * keyword_overlap_ratio(query, kw)
        new_score = base + cat_bonus + kw_bonus
        r2 = dict(r)
        r2["adj_score"] = new_score
        r2["cat_bonus"] = cat_bonus
        r2["kw_bonus"] = kw_bonus
        ranked.append(r2)
    ranked.sort(key=lambda x: x.get("adj_score", 0.0), reverse=True)
    return ranked

# ──────────────────────────────────────────────────────────────────────────────
# 7) UI 유틸
# ──────────────────────────────────────────────────────────────────────────────
def score_badge(score: float) -> str:
    if score >= 0.9:
        return "🟢"
    if score >= 0.8:
        return "🟡"
    return "🟠"

def extract_category(meta: Dict) -> str:
    return _get_item_category(meta)

def extract_tag(meta: Dict, key: str) -> str:
    return meta.get(key) or meta.get(key.lower()) or meta.get(key.capitalize()) or ""

# ──────────────────────────────────────────────────────────────────────────────
# 8) Streamlit App
# ──────────────────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(page_title="RAG App Demo", page_icon="🔎", layout="wide")
    st.title("🔎 RAG 검색 데모")

    with st.sidebar:
        st.header("검색 옵션")
        query = st.text_input("질의(Query)", value="운동 후 주의사항은?")
        top_k = st.slider("Top-K", 1, 10, 3)
        threshold = st.slider("Score 최소값(필터)", 0.0, 1.0, 0.80, 0.01)
        st.markdown("### 가중치")
        w_cat = st.slider("카테고리 가중치 (w_cat)", 0.0, 0.20, 0.08, 0.01)
        w_kw  = st.slider("키워드 가중치 (w_kw)",  0.0, 0.20, 0.05, 0.01)
        strict = st.checkbox("해당 카테고리만 표시(강제 필터)", value=False)

    try:
        retr, sd = get_retriever_cached()
        index, model, items = normalize_retriever_triplet(retr)
        AUTO_CATS = build_auto_categories(items, model)  # _model 인자명으로 캐시 해시 회피
    except Exception as e:
        st.error(f"리트리버/카테고리 모델 로드 실패: {e}")
        st.stop()

    if st.button("🔍 검색 실행", use_container_width=True):
        with st.spinner("검색 중..."):
            try:
                results = run_category_aware_search(sd, retr, query, top_k, AUTO_CATS)
            except Exception as e:
                st.error(f"검색 실패: {e}")
                st.stop()

        if not results:
            st.warning("검색 결과가 없습니다.")
            st.stop()

        cat_hint = infer_category_auto(query, model, AUTO_CATS)
        if cat_hint:
            st.caption(f"🔎 자동 추론 카테고리: **{cat_hint}**")

        if strict and cat_hint:
            results = [
                r for r in results
                if category_matches_auto(extract_category(r.get("meta", {}) or {}), cat_hint, AUTO_CATS)
            ]

        results = [r for r in results if float(r.get("score", 0.0)) >= threshold]
        results = boosted_sort(results, query, cat_hint, w_cat=w_cat, w_kw=w_kw, AUTO_CATS=AUTO_CATS)

        st.subheader("검색 결과")
        for r in results:
            text = r.get("text", "").strip()
            meta = r.get("meta", {}) or {}
            base = float(r.get("score", 0.0))
            adj  = float(r.get("adj_score", base))
            cat = extract_category(meta)
            sev = extract_tag(meta, "Severity")
            intent = extract_tag(meta, "Intent")
            keywords = meta.get("keywords") or meta.get("Keywords") or []
            cb = r.get("cat_bonus", 0.0); kb = r.get("kw_bonus", 0.0)

            with st.container(border=True):
                cols = st.columns([6, 1])
                with cols[0]:
                    st.markdown(f"**[{cat}]** {text}" if cat else text)
                    if sev or intent or keywords:
                        smalls = []
                        if sev:    smalls.append(f"**Severity:** {sev}")
                        if intent: smalls.append(f"**Intent:** {intent}")
                        if keywords:
                            kw = ", ".join(map(str, keywords)) if isinstance(keywords, list) else str(keywords)
                            smalls.append(f"**Keywords:** {kw}")
                        st.caption(" · ".join(smalls))
                with cols[1]:
                    st.metric(label="Score", value=f"{score_badge(base)} {base:.3f}",
                              delta=f"adj {adj:.3f}")
                    st.caption(f"cat+{cb:.3f} / kw+{kb:.3f}")

        st.subheader("임시 답안 (Draft)")
        draft_answer = " ".join([r.get("text", "").strip() for r in results[:2] if r.get("text")])
        if draft_answer:
            st.write(draft_answer)
        else:
            st.info("상위 문맥을 기반으로 생성할 임시 답안이 없습니다.")

if __name__ == "__main__":
    main()
