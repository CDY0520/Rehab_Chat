# app/app_streamlit.py
# ─────────────────────────────────────────────────────────────────────────────
# 재활 RAG 미니앱 (Streamlit)
# - 질의 입력 → FAISS 검색 → 규칙 기반 답변 생성(+ 선택적으로 LLM 요약)
# - 검색 결과 카드, 출처(원본 페이지) 표시, 필터/슬라이더 제공
# - 기존 03_build_faiss.py가 만든 data/vectorstore/{faiss.index, meta.pkl} 사용
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any
import os
import pickle
import time

import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss

# (선택) OpenAI가 설치되어 있고 API 키가 있으면 LLM 요약 모드 활성화
try:
    from openai import OpenAI
    HAS_OPENAI = True
except Exception:
    HAS_OPENAI = False

# ─────────────────────────────────────────────────────────────────────────────
# 경로/리소스 설정
# ─────────────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]           # 프로젝트 루트
VEC_DIR = ROOT / "data" / "vectorstore"              # 인덱스/메타가 저장된 경로
FAISS_INDEX_PATH = VEC_DIR / "faiss.index"
META_PATH        = VEC_DIR / "meta.pkl"

# ─────────────────────────────────────────────────────────────────────────────
# 인덱스/메타/모델 로드 (캐시)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_index_meta_model():
    if not FAISS_INDEX_PATH.exists() or not META_PATH.exists():
        raise FileNotFoundError("벡터 인덱스가 없습니다. 먼저 03_build_faiss.py를 실행하세요.")
    index = faiss.read_index(str(FAISS_INDEX_PATH))
    with open(META_PATH, "rb") as f:
        meta = pickle.load(f)
    model = SentenceTransformer(meta["model_name"])
    return index, meta, model

# ─────────────────────────────────────────────────────────────────────────────
# 검색 함수: 메타 필터 → FAISS 질의 → 상위 K 문장 반환
# ─────────────────────────────────────────────────────────────────────────────
def search(q: str, top_k: int = 5,
           category: str | None = None,
           severity: str | None = None,
           intent: str | None = None) -> List[Dict[str, Any]]:
    index, meta, model = load_index_meta_model()

    # (1) 메타 기반 후보 필터링(간단)
    candidates = list(range(len(meta["records"])))
    if category:
        candidates = [i for i in candidates if meta["records"][i].get("category") == category]
    if severity:
        candidates = [i for i in candidates if meta["records"][i].get("severity") == severity]
    if intent:
        candidates = [i for i in candidates if meta["records"][i].get("intent") == intent]
    if not candidates:
        return []

    # (2) 쿼리 임베딩 → 유사도 검색
    q_emb = model.encode([q], normalize_embeddings=True).astype("float32")
    D, I = index.search(q_emb, k=min(top_k*10, len(meta["texts"])))  # 넉넉히 뽑은 뒤 후보 매칭

    # (3) 후보만 살리고, 중복 문장 제거
    hits = []
    seen = set()
    for dist, idx in zip(D[0], I[0]):
        if idx not in candidates:
            continue
        rec = meta["records"][idx]
        text = " ".join(rec["canonical_ko"].split())
        if text in seen:
            continue
        seen.add(text)
        hits.append({
            "score": float(dist),
            "id": rec["id"],
            "text": rec["canonical_ko"],
            "category": rec.get("category"),
            "severity": rec.get("severity"),
            "intent": rec.get("intent"),
            "source_title": rec.get("source_title"),
            "source_page": rec.get("source_page"),
            "source_orig_page": rec.get("source_orig_page"),
        })
        if len(hits) >= top_k:
            break
    return hits

# ─────────────────────────────────────────────────────────────────────────────
# 우선순위 정렬: High > Medium > Low → 점수 내림차순
# ─────────────────────────────────────────────────────────────────────────────
def prioritize(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    sev_rank = {"High": 0, "Medium": 1, "Low": 2}
    return sorted(results, key=lambda r: (sev_rank.get(r.get("severity", "Low"), 3), -r["score"]))

# ─────────────────────────────────────────────────────────────────────────────
# 출처 포맷: 문서명 + 원본 페이지(없으면 합본 페이지) + id
# ─────────────────────────────────────────────────────────────────────────────
def format_sources(results: List[Dict[str, Any]]) -> str:
    lines, seen = [], set()
    for r in results:
        page = r.get("source_orig_page") or r.get("source_page")
        line = f"- {r['source_title']} (p.{page}, {r['id']})"
        if line not in seen:
            seen.add(line)
            lines.append(line)
    return "\n".join(lines)

# ─────────────────────────────────────────────────────────────────────────────
# 규칙 기반 답변(기본값): 안전도/의도에 따른 아이콘/문구 구성
# ─────────────────────────────────────────────────────────────────────────────
def rule_based_answer(question: str, results: List[Dict[str, Any]]) -> str:
    if not results:
        return "관련 근거를 찾지 못했어요. 질문을 더 구체적으로 바꿔보거나 다른 표현을 시도해 주세요."
    ordered = prioritize(results)
    high_exists = any(r.get("severity") == "High" for r in ordered)

    bullets = []
    for r in ordered:
        sev, tone = r.get("severity"), r.get("intent")
        icon = "•"
        if sev == "High": icon = "🚨"
        elif sev == "Medium": icon = "⚠️"
        elif tone == "권장": icon = "✅"
        bullets.append(f"{icon} {r['text']}")

    header = "아래 근거를 바탕으로 정리했어요."
    if high_exists:
        header = "🚨 안전 우선 안내: 위험 신호가 있어 **먼저 안전 지침**을 따르세요."

    src = format_sources(ordered)

    return f"""{header}

Q. {question}

{chr(10).join(bullets)}

출처:
{src}
"""

# ─────────────────────────────────────────────────────────────────────────────
# (선택) LLM 요약 답변: OPENAI_API_KEY 있을 때만 사용
# ─────────────────────────────────────────────────────────────────────────────
def llm_answer(question: str, results: List[Dict[str, Any]]) -> str:
    if not results or not HAS_OPENAI or not os.getenv("OPENAI_API_KEY"):
        return rule_based_answer(question, results)

    client = OpenAI()
    ctx = "\n".join(
        [f"- ({r['severity']}/{r['intent']}) {r['text']} [출처: {r['source_title']} p.{r.get('source_orig_page') or r.get('source_page')}]"
         for r in prioritize(results)]
    )

    prompt = f"""너는 뇌졸중 재활 운동 안전 가이드 챗봇이야.
사용자 질문: {question}
다음 근거를 안전도 순으로 요약해, 금지/주의/권장 우선순위로 안내하고 마지막에 출처를 나열해.
근거:
{ctx}
출력 형식:
- 핵심 지침 3~6줄 (금지/주의 먼저, 권장은 마지막)
- '출처:' 아래에 문서명과 '원본 페이지 번호'를 줄바꿈으로 표기
"""
    chat = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return chat.choices[0].message.content.strip()

# ─────────────────────────────────────────────────────────────────────────────
# UI 구성 (Streamlit)
# ─────────────────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(page_title="재활 RAG 데모", page_icon="🦵", layout="centered")
    st.title("🦵 재활 안전 RAG 데모")
    st.caption("질문 → 검색 → 안전 우선 답변 + 출처(원본 페이지)")

    # 사이드바: 검색 옵션
    with st.sidebar:
        st.header("검색 옵션")
        top_k = st.slider("Top-K (가져올 근거 수)", 3, 10, 5, 1)
        st.divider()
        st.subheader("필터(선택)")
        category = st.selectbox("카테고리", ["(전체)", "건강상태/증상", "사전측정", "복장/환경", "안전/낙상", "식사/약물"])
        severity = st.selectbox("중요도", ["(전체)", "High", "Medium", "Low"])
        intent   = st.selectbox("의도", ["(전체)", "금지", "주의", "권장", "정보"])
        st.divider()
        use_llm = st.toggle("LLM 요약 사용 (OPENAI_API_KEY 필요)", value=False)
        if use_llm and (not HAS_OPENAI or not os.getenv("OPENAI_API_KEY")):
            st.info("환경변수 OPENAI_API_KEY가 설정되어 있지 않아 규칙기반으로 동작합니다.", icon="ℹ️")

    # 질의 입력
    q = st.text_input("질문을 입력하세요", placeholder="예) 두통이 있으면 운동 가능해?", value="")
    run = st.button("검색 및 답변 생성")

    # 인덱스/모델 로드 상태 표시
    try:
        index, meta, model = load_index_meta_model()
        st.caption(f"임베딩 모델: `{meta['model_name']}` | 문장 수: {len(meta['records'])}")
    except Exception as e:
        st.error(str(e))
        st.stop()

    if run and q.strip():
        # 선택값 처리
        cat_f = None if category == "(전체)" else category
        sev_f = None if severity == "(전체)" else severity
        int_f = None if intent   == "(전체)" else intent

        with st.spinner("검색 중..."):
            results = search(q.strip(), top_k=top_k, category=cat_f, severity=sev_f, intent=int_f)

        if not results:
            st.warning("관련 근거를 찾지 못했어요. 질문을 바꿔보거나 필터를 해제해보세요.")
            return

        # 안전 최우선 경고 배너
        if any(r.get("severity") == "High" for r in results):
            st.error("🚨 High 수준 위험 신호가 포함되어 있습니다. 아래 지침을 우선적으로 따르세요.", icon="🚨")

        # 답변 생성
        with st.spinner("답변 구성 중..."):
            answer = llm_answer(q.strip(), results) if use_llm else rule_based_answer(q.strip(), results)

        st.subheader("🧠 답변")
        st.write(answer)

        # 검색 결과 카드
        st.subheader("🔎 근거 (Top-K)")
        for i, r in enumerate(prioritize(results), 1):
            # 색상/아이콘
            sev = r.get("severity")
            tone = r.get("intent")
            icon = "•"
            color = "#e5e7eb"
            if sev == "High":
                icon, color = "🚨", "#fee2e2"   # 붉은 톤
            elif sev == "Medium":
                icon, color = "⚠️", "#fff7ed"  # 주황 톤
            elif tone == "권장":
                icon, color = "✅", "#ecfdf5"  # 초록 톤

            page = r.get("source_orig_page") or r.get("source_page")
            with st.expander(f"{i}. {icon} {r['text']}"):
                st.markdown(
                    f"""
                    <div style="background:{color}; padding:10px; border-radius:10px;">
                    <b>카테고리:</b> {r.get('category', '-')}&nbsp;&nbsp;|&nbsp;&nbsp;
                    <b>중요도:</b> {r.get('severity', '-')}&nbsp;&nbsp;|&nbsp;&nbsp;
                    <b>의도:</b> {r.get('intent', '-')}<br/>
                    <b>출처:</b> {r['source_title']} (p.{page}, {r['id']})<br/>
                    <b>유사도 점수:</b> {r['score']:.3f}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    # 푸터
    st.markdown("---")
    st.caption("© 재활 프로젝트 · Streamlit RAG 데모")

if __name__ == "__main__":
    main()
