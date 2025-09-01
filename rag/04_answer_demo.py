"""
파일: rag/04_answer_demo.py
목적: 질의 → FAISS 검색 → 규칙 기반(템플릿) 답변 생성 (+ 출처 표기, 안전 경고)
옵션: OPENAI_API_KEY가 설정되어 있으면 LLM 요약 모드로도 답변 가능

요구:
  - sentence-transformers, faiss-cpu (이미 이전 단계에서 설치)
  - (선택) openai==1.*  (키가 있을 때만)
실행:
  python rag/04_answer_demo.py
"""

from pathlib import Path
import os
import pickle
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
import faiss

try:
    from openai import OpenAI
    HAS_OPENAI = True
except Exception:
    HAS_OPENAI = False

ROOT = Path(__file__).resolve().parents[1]
VEC_DIR = ROOT / "data" / "vectorstore"
FAISS_INDEX_PATH = VEC_DIR / "faiss.index"
META_PATH        = VEC_DIR / "meta.pkl"

# -----------------------------
# 공통 유틸
# -----------------------------
def load_index_meta_model():
    index = faiss.read_index(str(FAISS_INDEX_PATH))
    with open(META_PATH, "rb") as f:
        meta = pickle.load(f)
    model = SentenceTransformer(meta["model_name"])
    return index, meta, model

def search(q: str, top_k: int = 5,
           category: str | None = None,
           severity: str | None = None,
           intent: str | None = None) -> List[Dict[str, Any]]:
    """03_search_demo.py와 동일한 로직(필터 + 검색)"""
    index, meta, model = load_index_meta_model()
    candidates = list(range(len(meta["records"])))
    if category:
        candidates = [i for i in candidates if meta["records"][i].get("category") == category]
    if severity:
        candidates = [i for i in candidates if meta["records"][i].get("severity") == severity]
    if intent:
        candidates = [i for i in candidates if meta["records"][i].get("intent") == intent]
    if not candidates:
        return []

    q_emb = model.encode([q], normalize_embeddings=True).astype("float32")
    D, I = index.search(q_emb, k=min(top_k*10, len(meta["texts"])))

    hits = []
    seen_text = set()
    for dist, idx in zip(D[0], I[0]):
        if idx not in candidates:
            continue
        rec = meta["records"][idx]
        text = rec["canonical_ko"].strip()
        # 중복 문장 제거
        key = " ".join(text.split())
        if key in seen_text:
            continue
        seen_text.add(key)
        hits.append({
            "score": float(dist),
            "id": rec["id"],
            "text": text,
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

# -----------------------------
# 규칙 기반(템플릿) 답변 생성
# -----------------------------
def prioritize(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """High > Medium > Low 우선, 그 다음 점수 내림차순"""
    sev_rank = {"High": 0, "Medium": 1, "Low": 2}
    return sorted(results, key=lambda r: (sev_rank.get(r.get("severity","Low"), 3), -r["score"]))

def format_sources(results: List[Dict[str, Any]]) -> str:
    lines = []
    for r in results:
        page = r.get("source_orig_page") or r.get("source_page")
        lines.append(f"- {r['source_title']} (p.{page}, {r['id']})")
    # 중복 제거
    uniq = []
    seen = set()
    for L in lines:
        if L not in seen:
            seen.add(L)
            uniq.append(L)
    return "\n".join(uniq)

def rule_based_answer(question: str, results: List[Dict[str, Any]]) -> str:
    if not results:
        return "관련 근거를 찾지 못했어요. 질문을 조금 더 구체적으로 바꿔보거나 다른 표현을 시도해 주세요."

    ordered = prioritize(results)
    high_exists = any(r.get("severity") == "High" for r in ordered)

    bullets = []
    for r in ordered:
        tone = r.get("intent")
        sev = r.get("severity")
        prefix = "•"
        if sev == "High":
            prefix = "🚨"
        elif sev == "Medium":
            prefix = "⚠️"
        elif tone == "권장":
            prefix = "✅"

        bullets.append(f"{prefix} {r['text']}")

    src = format_sources(ordered)

    header = "아래 근거를 바탕으로 정리했어요.\n"
    if high_exists:
        header = "🚨 안전 우선 안내\n위험 신호가 포함되어 있어 **먼저 안전 지침**을 따르세요.\n\n"

    answer = (
        f"{header}"
        f"Q. {question}\n\n"
        + "\n".join(bullets)
        + "\n\n"
        + "출처:\n"
        + src
    )
    return answer

# -----------------------------
# (선택) LLM 요약 답변
# -----------------------------
def llm_answer(question: str, results: List[Dict[str, Any]]) -> str:
    """OPENAI_API_KEY가 있을 때만 사용. 규칙기반 대비 자연스러움 강화."""
    if not results or not HAS_OPENAI or not os.getenv("OPENAI_API_KEY"):
        return rule_based_answer(question, results)

    client = OpenAI()
    ctx = "\n".join([f"- ({r['severity']}/{r['intent']}) {r['text']} [출처: {r['source_title']} p.{r.get('source_orig_page') or r.get('source_page')}]"
                     for r in prioritize(results)])

    prompt = f"""너는 뇌졸중 재활 운동 안전 가이드 챗봇이야.
사용자 질문: {question}
다음 근거를 안전도 순으로 요약해, 금지/주의/권장 우선순위로 안내하고 마지막에 출처를 나열해.
근거:
{ctx}
출력 형식:
- 핵심 지침 3~6줄 (금지/주의 먼저, 권장은 마지막)
- '출처:' 아래에 문서명과 원본 페이지 번호를 줄바꿈으로 표기
"""
    chat = client.chat.completions.create(
        model="gpt-4o-mini",  # 또는 가능한 채팅 모델
        messages=[{"role":"user","content":prompt}],
        temperature=0.2,
    )
    return chat.choices[0].message.content.strip()

# -----------------------------
# 데모 실행
# -----------------------------
if __name__ == "__main__":
    queries = [
        "식사 직후 운동해도 돼?",
        "운동 전 혈압 확인해야 해?",
        "두통이 있으면 운동 가능해?",
        "균형이 불안정할 때 어떻게 해야 해?"
    ]
    use_llm = bool(os.getenv("OPENAI_API_KEY")) and HAS_OPENAI

    for q in queries:
        results = search(q, top_k=5)
        ans = llm_answer(q, results) if use_llm else rule_based_answer(q, results)
        print("="*80)
        print(ans)
        print("="*80)
