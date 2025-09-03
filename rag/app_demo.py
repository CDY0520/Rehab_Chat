# 파일명: rag/app_demo.py
# 목적:
#   - 숫자로 시작하는 모듈(03_search_demo.py)을 importlib으로 동적 로드
#   - get_retriever() 반환형을 자동 해석(dict/object/tuple/list/callable)
#   - 카테고리 인지형 검색: 재정렬 + 오버페치 + 쿼리 확장
#   - 가중치 기반 스코어 보정(카테고리/키워드) + 강제 카테고리 필터 토글
#   - Streamlit UI에서 점수/메타(Category/Severity/Intent/Keywords)까지 표시
#
# 실행:
#   (.venv) streamlit run rag/app_demo.py

from __future__ import annotations

# ──────────────────────────────────────────────────────────────────────────────
# 0) 공용 임포트
# ──────────────────────────────────────────────────────────────────────────────
import sys
from pathlib import Path
import importlib.util
from typing import Any, Dict, List, Tuple

import re
import streamlit as st


# ──────────────────────────────────────────────────────────────────────────────
# 1) importlib 로더: '03_search_demo.py' 동적 로드
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_search_demo_module() -> Any:
    """
    같은 폴더의 '03_search_demo.py'를 안전하게 동적 임포트한다.
    (모듈명을 'search_demo'로 등록)
    """
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
# 2) Retriever 생성(캐시)
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=True)
def get_retriever_cached():
    """
    search_demo.get_retriever()를 호출하고, 모듈 핸들(sd)과 함께 반환한다.
    """
    sd = load_search_demo_module()
    if not hasattr(sd, "get_retriever"):
        raise AttributeError("search_demo 모듈에 get_retriever 함수가 없습니다.")
    return sd.get_retriever(), sd


# ──────────────────────────────────────────────────────────────────────────────
# 3) 타입 헬퍼 + (index, model, items) 정규화
# ──────────────────────────────────────────────────────────────────────────────
def _looks_like_faiss_index(obj: Any) -> bool:
    return hasattr(obj, "search") and callable(getattr(obj, "search"))

def _looks_like_st_model(obj: Any) -> bool:
    return hasattr(obj, "encode") and callable(getattr(obj, "encode"))

def _looks_like_items(obj: Any) -> bool:
    return isinstance(obj, list) and (len(obj) == 0 or isinstance(obj[0], dict))

def normalize_retriever_triplet(retr: Any) -> Tuple[Any, Any, List[Dict]]:
    """
    다양한 반환형(retr)을 (index, model, items) 표준 튜플로 정규화한다.
    - 지원: dict / 객체속성형 / tuple|list
    """
    # dict
    if isinstance(retr, dict):
        idx = retr.get("index") or retr.get("faiss") or retr.get("faiss_index")
        mdl = retr.get("model") or retr.get("encoder") or retr.get("st_model")
        itm = retr.get("items") or retr.get("docs") or retr.get("metas")
        if idx is not None and mdl is not None and itm is not None:
            return idx, mdl, itm

    # 객체 속성형
    if hasattr(retr, "index") and hasattr(retr, "model") and hasattr(retr, "items"):
        return getattr(retr, "index"), getattr(retr, "model"), getattr(retr, "items")

    # tuple/list
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
# 4) FAISS 검색 (공용)
# ──────────────────────────────────────────────────────────────────────────────
def run_query_faiss(index: Any, model: Any, items: List[Dict], query: str, top_k: int) -> List[Dict]:
    """
    FAISS + SentenceTransformer 기반 단순 유사도 검색.
    """
    import numpy as np  # streamlit 캐시 내 import 안전

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
# 5) 카테고리 인지 유틸(추론/재정렬/확장/매칭 카운트)
# ──────────────────────────────────────────────────────────────────────────────
CATEGORY_ALIASES = {
    "운동 전 주의사항": ["운동 전", "운동전", "before exercise", "pre-exercise"],
    "운동 중 주의사항": ["운동 중", "운동중", "during exercise", "중간"],
    "운동 후 주의사항": ["운동 후", "운동후", "after exercise", "post-exercise"],
    "근력운동 주의사항": ["근력운동", "저항운동", "웨이트", "strength", "resistance"],
}

def infer_category_from_query(query: str) -> str | None:
    """
    자연어 쿼리에서 카테고리 의도를 추론한다(간단한 별칭 매칭).
    """
    q = query.lower().replace(" ", "")
    for cat, aliases in CATEGORY_ALIASES.items():
        for a in aliases:
            if a.replace(" ", "") in q:
                return cat
    return None

def get_meta_category(meta: Dict) -> str:
    """
    items 메타에서 category 키를 안전하게 꺼낸다.
    """
    return meta.get("category") or meta.get("Category") or meta.get("섹션") or meta.get("section") or ""

def rerank_by_category(results: List[Dict], cat_hint: str | None) -> List[Dict]:
    """
    카테고리 힌트가 있으면 동일 카테고리를 상단으로 끌어올리고,
    나머지는 기존 순서를 유지한다.
    """
    if not cat_hint:
        return results
    same, other = [], []
    for r in results:
        (same if get_meta_category(r.get("meta", {}) or {}) == cat_hint else other).append(r)
    return same + other

def count_matches(results: List[Dict], cat_hint: str | None) -> int:
    """
    결과 중 카테고리 일치 개수 세기.
    """
    if not cat_hint:
        return 0
    return sum(1 for r in results if get_meta_category(r.get("meta", {}) or {}) == cat_hint)

def build_expanded_query(query: str, cat_hint: str) -> str:
    """
    카테고리 별칭을 쿼리에 덧붙여 확장(쿼리 확장).
    """
    aliases = CATEGORY_ALIASES.get(cat_hint, [])
    return query + " " + " ".join(aliases)


# ──────────────────────────────────────────────────────────────────────────────
# 6) 스코어 보정 유틸(카테고리/키워드 가중치)
# ──────────────────────────────────────────────────────────────────────────────
def normalize_korean(s: str) -> str:
    return re.sub(r"\s+", "", str(s)).lower()

def keyword_overlap_ratio(query: str, keywords) -> float:
    """
    질문과 아이템 keywords 간 토큰 겹침 비율(0~1). 간단 버전.
    """
    if not keywords:
        return 0.0
    if isinstance(keywords, str):
        kws = [k.strip() for k in re.split(r"[,\s/;]+", keywords) if k.strip()]
    else:
        kws = [str(k).strip() for k in keywords if str(k).strip()]
    if not kws:
        return 0.0
    q_tokens = [t for t in re.split(r"[,\s/;]+", query) if t]
    if not q_tokens:
        return 0.0
    inter = len(set(map(normalize_korean, q_tokens)) & set(map(normalize_korean, kws)))
    return inter / max(len(set(kws)), 1)

def boosted_sort(results: List[Dict], query: str, cat_hint: str | None,
                 w_cat: float = 0.08, w_kw: float = 0.05) -> List[Dict]:
    """
    base_score + (w_cat if 카테고리 일치 else 0) + w_kw * keyword_overlap
    로 '보정 점수(adj_score)'를 부여해 재정렬한다.
    """
    ranked = []
    for r in results:
        base = float(r.get("score", 0.0))
        meta = r.get("meta", {}) or {}
        cat = get_meta_category(meta)
        kw = meta.get("keywords") or meta.get("Keywords")
        cat_bonus = (w_cat if (cat_hint and cat == cat_hint) else 0.0)
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
# 7) 검색 실행(유연 + 카테고리 인지형)
#    - A: sd.search(retr, query, top_k) 우선
#    - B: retriever가 callable
#    - C: (index, model, items) 직접 FAISS
#    - 이후: 재정렬 → (매칭 0개면) 오버페치 → (그래도 0개면) 쿼리 확장
# ──────────────────────────────────────────────────────────────────────────────
def run_category_aware_search(sd: Any, retr: Any, query: str, top_k: int) -> List[Dict]:
    """
    카테고리 인지형 검색의 전체 파이프라인.
    """
    # (선택) 쿼리 태깅 부스팅: 카테고리 힌트를 앞에 추가
    cat_hint0 = infer_category_from_query(query)
    if cat_hint0:
        query_for_dense = f"{cat_hint0}: " + query
    else:
        query_for_dense = query

    # 1) 1차 검색
    index = model = items = None
    if hasattr(sd, "search"):
        try:
            results = sd.search(retr, query_for_dense, top_k=top_k)
        except TypeError:
            results = sd.search(retr, query_for_dense)
        try:
            index, model, items = normalize_retriever_triplet(retr)
        except Exception:
            pass
    elif callable(retr):
        try:
            results = retr(query_for_dense, top_k=top_k)
        except TypeError:
            results = retr(query_for_dense)
        try:
            index, model, items = normalize_retriever_triplet(retr)
        except Exception:
            pass
    else:
        index, model, items = normalize_retriever_triplet(retr)
        results = run_query_faiss(index, model, items, query_for_dense, top_k)

    # 2) 정렬 + 카테고리 재정렬
    results = sorted(results, key=lambda x: x.get("score", 0.0), reverse=True)
    results = rerank_by_category(results, cat_hint0)

    # 3) 오버페치 → 재정렬
    if cat_hint0 and count_matches(results, cat_hint0) == 0 and index is not None:
        over_k = max(top_k * 10, 30)
        big = run_query_faiss(index, model, items, query_for_dense, over_k)
        big = sorted(big, key=lambda x: x.get("score", 0.0), reverse=True)
        big = rerank_by_category(big, cat_hint0)
        if count_matches(big, cat_hint0) > 0:
            results = big[:top_k]

    # 4) 쿼리 확장 → 오버페치
    if cat_hint0 and count_matches(results, cat_hint0) == 0 and index is not None:
        q_exp = build_expanded_query(query_for_dense, cat_hint0)
        big = run_query_faiss(index, model, items, q_exp, max(top_k * 10, 30))
        big = sorted(big, key=lambda x: x.get("score", 0.0), reverse=True)
        big = rerank_by_category(big, cat_hint0)
        if count_matches(big, cat_hint0) > 0:
            results = big[:top_k]

    return results


# ──────────────────────────────────────────────────────────────────────────────
# 8) UI 유틸(점수 뱃지/메타 추출)
# ──────────────────────────────────────────────────────────────────────────────
def score_badge(score: float) -> str:
    if score >= 0.9:
        return "🟢"
    if score >= 0.8:
        return "🟡"
    return "🟠"

def extract_category(meta: Dict) -> str:
    return get_meta_category(meta)

def extract_tag(meta: Dict, key: str) -> str:
    return meta.get(key) or meta.get(key.lower()) or meta.get(key.capitalize()) or ""


# ──────────────────────────────────────────────────────────────────────────────
# 9) Streamlit App
# ──────────────────────────────────────────────────────────────────────────────
def main():
    # 페이지 설정
    st.set_page_config(page_title="RAG App Demo", page_icon="🔎", layout="wide")
    st.title("🔎 RAG 검색 데모 (카테고리 인지형)")
    st.caption("`03_search_demo.py`를 동적 로드하여 인덱스를 활용합니다.")

    # 사이드바: 검색 옵션
    with st.sidebar:
        st.header("검색 옵션")
        query = st.text_input("질의(Query)", value="근력운동 주의사항 알려줘")
        top_k = st.slider("Top-K", 1, 10, 3)
        threshold = st.slider("Score 최소값(필터)", 0.0, 1.0, 0.80, 0.01)
        st.markdown("---")
        st.caption("점수는 코사인 유사도(정규화 내적)이며 0~1 범위입니다.")

        st.markdown("### 카테고리/키워드 가중치")
        w_cat = st.slider("카테고리 가중치 (w_cat)", 0.0, 0.20, 0.08, 0.01)
        w_kw  = st.slider("키워드 가중치 (w_kw)",  0.0, 0.20, 0.05, 0.01)
        strict = st.checkbox("해당 카테고리만 표시(강제 필터)", value=False)

    # 리트리버 로드
    try:
        retr, sd = get_retriever_cached()
    except Exception as e:
        st.error(f"리트리버 로드 실패: {e}")
        st.stop()

    # 검색 버튼
    if st.button("🔍 검색 실행", use_container_width=True):
        with st.spinner("검색 중..."):
            try:
                results = run_category_aware_search(sd, retr, query, top_k)
            except Exception as e:
                st.error(f"검색 실패: {e}")
                st.stop()

        if not results:
            st.warning("검색 결과가 없습니다.")
            st.stop()

        # 점수 필터(기본 score 기준) → 보정 정렬 → (선택) 강제 카테고리 필터
        cat_hint = infer_category_from_query(query)

        # (선택) 강제 카테고리 필터: 먼저 필터링하면 보정과정이 더 깔끔
        if strict and cat_hint:
            results = [r for r in results if extract_category(r.get("meta", {}) or {}) == cat_hint]

        # 점수 임계치(기본 score 기준) 적용
        results = [r for r in results if float(r.get("score", 0.0)) >= threshold]

        # 보정 정렬(카테고리/키워드 가중치)
        results = boosted_sort(results, query, cat_hint, w_cat=w_cat, w_kw=w_kw)

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
                    if cat:
                        st.markdown(f"**[{cat}]** {text}")
                    else:
                        st.markdown(text)
                    if sev or intent or keywords:
                        smalls = []
                        if sev:
                            smalls.append(f"**Severity:** {sev}")
                        if intent:
                            smalls.append(f"**Intent:** {intent}")
                        if keywords:
                            kw = ", ".join(map(str, keywords)) if isinstance(keywords, list) else str(keywords)
                            smalls.append(f"**Keywords:** {kw}")
                        st.caption(" · ".join(smalls))
                with cols[1]:
                    st.metric(label="Score", value=f"{score_badge(base)} {base:.3f}",
                              delta=f"adj {adj:.3f}")
                    st.caption(f"cat+{cb:.3f} / kw+{kb:.3f}")

        # Draft Answer: 상위 2개 합치기
        st.subheader("임시 답안 (Draft)")
        draft_answer = " ".join([r.get("text", "").strip() for r in results[:2] if r.get("text")])
        if draft_answer:
            st.write(draft_answer)
        else:
            st.info("상위 문맥을 기반으로 생성할 임시 답안이 없습니다.")


# ──────────────────────────────────────────────────────────────────────────────
# 10) 엔트리포인트
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
