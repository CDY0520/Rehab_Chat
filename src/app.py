"""
파일명: src/app.py
기능: Streamlit UI로 RAG 챗봇을 실행한다.
     - 사이드바: (필수) 카테고리 필터(=data/docs의 PDF 제목 정리본), Top-K, 디버그
     - 메인: 질문 → 선택한 카테고리 범위에서만 검색 → 깔끔한 섹션형 답변 표시

블록 구성
 0) 라이브러리 임포트
 1) 카테고리 로더(load_categories_from_docs)
 2) UI 및 검색 → 답변 표시
 3) 면책 고지

주의
 - ingest.py에서 category를 문서 단위로 고정했으므로, 검색 필터가 정확히 동작한다.
 - 실행: streamlit run src/app.py
"""
# 0) 라이브러리 임포트 -------------------------------------------------
from __future__ import annotations
from typing import Optional, List, Dict
from pathlib import Path
import re
import streamlit as st

# 패키지/직접 실행 모두 대응
try:
    from src.retrieve import Retriever
    from src.compose import compose_answer
except ImportError:
    from retrieve import Retriever
    from compose import compose_answer

# 1) 카테고리 로더 -----------------------------------------------------
def _clean_title(file_stem: str) -> str:
    """ingest.py와 동일 규칙으로 문서 제목 정리."""
    s = re.sub(r"^\d+[_\-\s]*", "", file_stem)
    s = s.replace("_", " ").strip()
    return s

def load_categories_from_docs() -> list[str]:
    """data/docs 안의 PDF 목록을 읽어 카테고리(문서명) 리스트 반환."""
    ROOT = Path(__file__).resolve().parents[1]
    docs_dir = ROOT / "data" / "docs"
    pdfs = sorted(docs_dir.glob("*.pdf"))
    cats_raw = [_clean_title(p.stem) for p in pdfs]
    # 중복 제거(순서 보존)
    seen, out = set(), []
    for c in cats_raw:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out

# 2) UI ---------------------------------------------------------------
st.set_page_config(page_title="재활 홈운동 관리 RAG 챗봇", layout="wide")
st.title("재활 홈운동 관리 챗봇")
st.caption("국립재활원 자료 기반: 운동처방, 주의사항, 생활지도 등에 대한 정보를 제공 합니다.")

with st.sidebar:
    st.header("⚙️ 옵션")
    categories = load_categories_from_docs()  # 항상 문서 6개만
    category_label = st.selectbox(
        "카테고리 필터",
        options=["(전체)"] + categories,
        index=0,
        help="먼저 카테고리를 고르면 해당 문서 범위에서만 검색합니다."
    )
    DEFAULT_TOP_K = 6
    top_k = DEFAULT_TOP_K
    show_debug = st.checkbox("디버그: 검색 히트 미리보기", value=False)

with st.form("qa_form", clear_on_submit=False):
    query = st.text_area(
        "질문을 입력하세요",
        value="운동 중 주의사항 알려줘",
        height=100,
        placeholder="예) 운동 후 주의사항은 무엇인가요?"
    )
    submitted = st.form_submit_button("검색 및 답변 생성")

if submitted:
    if not query.strip():
        st.warning("질문을 입력해 주세요.")
        st.stop()

    selected_category: Optional[str] = None if category_label == "(전체)" else category_label

    try:
        r = Retriever()
    except FileNotFoundError as e:
        st.error(f"인덱스/매핑 파일을 찾지 못했습니다.\n{e}\n\n먼저 `python src/index_build.py`를 실행해 주세요.")
        st.stop()

    hits: List[Dict] = r.search(query, k=top_k, category=selected_category)
    if not hits:
        scope = "전체" if selected_category is None else f"'{selected_category}'"
        st.info(f"{scope} 범위에서 검색 결과가 없습니다. 질문을 조금 더 구체적으로 입력해 보세요.")
    else:
        answer = compose_answer(query, hits)
        st.markdown(answer)

        if show_debug:
            st.divider()
            st.subheader("🔎 검색 히트 (디버그)")
            from textwrap import shorten
            for i, h in enumerate(hits, 1):
                m = h.get("meta", {}) or {}
                with st.expander(
                    f"[{i}] score={h['score']:.3f} | p.{m.get('page','?')} | {m.get('category','?')} | {m.get('section_title', m.get('source_title',''))}",
                    expanded=False
                ):
                    st.write("**미리보기**")
                    st.write(shorten(h.get("text", "").replace("\n", " "), width=400, placeholder=" ..."))
                    st.write("**메타**")
                    st.json(m)

# 3) 면책 고지 ---------------------------------------------------------
st.divider()
st.info(
    "본 서비스는 일반적 건강정보이며, **개인 맞춤 의료 상담을 대체하지 않습니다.** "
    "증상이 악화되거나 의심될 경우 전문의·치료사의 평가를 받으세요."
)
