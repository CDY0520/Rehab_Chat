"""
파일명: src/compose.py
기능: 검색된 청크에서 '의도에 맞는 섹션만' 깔끔하게 요약하여 답변 생성.
     - 의도 감지(주의/방법/생활)로 섹션 제한
     - 섹션 일치: 같은 section_title 그룹 중 최고점 섹션만 사용
     - 제목/머릿말 줄 및 기관명 노이즈 제거
     - 원문 불릿 기호(•, ▪, ●, -, ※ 등) 자동 제거
     - 본문 불릿은 하이픈(-)만 사용, 섹션 제목에만 아이콘 표시
     - 출처 표기: source_title(머릿말) + page_label(꼬릿말 페이지) 우선

블록 구성
 0) 설정/임포트
 1) 의도 감지
 2) 섹션 그룹핑/선택
 3) 불릿 추출 유틸(제목/머릿말/불릿기호 노이즈 제거)
 4) 섹션별 불릿 생성(본문은 아이콘 제거)
 5) 출처 블록(머릿말 + 꼬릿말 페이지)
 6) compose_answer

주의
 - 이 모듈은 RAG 검색 결과(hits: List[Dict])를 입력으로 받아 규칙 기반으로 요약한다.
 - page_label이 없을 경우 내부 page 번호로 자동 폴백한다.
"""

# 0) 설정/임포트 -------------------------------------------------------
from __future__ import annotations
from typing import List, Dict, Iterable, Tuple
import re

# 의도별 키워드
MAX_PER_SECTION = {"caution": 6, "howto": 5, "lifestyle": 3}
KW_CAUT = ("주의", "유의", "안전", "금지", "중단", "보류", "위험", "경고")
KW_HOW  = ("방법", "자세", "실시", "반복", "세트", "강도", "속도", "호흡", "운동법")
KW_LIFE = ("생활", "휴식", "수면", "수분", "물", "식사", "영양", "준비물", "환경", "정리", "온수", "냉수")

# 섹션 제목에만 사용할 아이콘
HEADER_ICON = {"caution": "⚠️", "howto": "🏃", "lifestyle": "💡"}

# 문장/불릿 후보 분할용 정규식
BULLET_SPLIT_RE = re.compile(r"(?:\n|[①-⑳]|[0-9]+\s*[\.\)]|•\s+|▪\s+|-{1,2}\s+)")

# 제목/머릿말 노이즈(불릿에서 제거)
TITLE_NOISE = (
    "National Rehabilitation Center",
    "국립재활원",
    "운동 전 주의사항",
    "운동 중 주의사항",
    "운동 후 주의사항",
    "혼자서도 할 수 있는 운동가이드",
)

# 원문 앞 불릿/기호 제거용 (•, ▪, ●, ○, -, ※, … 등)
LEADING_BULLET_RE = re.compile(
    r"^[\u2022\u2023\u25CF\u25CB\u25C9\u25E6\u2219\u2043\u30FB"
    r"\u25B6\u25B8\u25BA\u25C6\u25A0\u25AA\u25AB"
    r"\-\–\—\·\•\▪\●\○\◦\▶\▷\▸\※\★\◆\◇\s]+"
)

def _strip_leading_bullets(s: str) -> str:
    return LEADING_BULLET_RE.sub("", s).strip()

# 1) 의도 감지 ---------------------------------------------------------
def detect_intent(query: str) -> str:
    q = query.lower()
    if any(k in q for k in KW_CAUT):
        return "caution"
    if any(k in q for k in KW_LIFE):
        return "lifestyle"
    if any(k in q for k in KW_HOW) or "방법" in q or "어떻게" in q:
        return "howto"
    return "mixed"

# 2) 섹션 그룹핑/선택 ---------------------------------------------------
def group_by_section(hits: List[Dict]) -> Dict[str, List[Dict]]:
    groups = {}
    for h in hits:
        m = h.get("meta", {}) or {}
        sec = (m.get("section_title") or m.get("source_title") or "").strip()
        if not sec:
            sec = "(섹션 없음)"
        groups.setdefault(sec, []).append(h)
    return groups

def pick_best_section(groups: Dict[str, List[Dict]], prefer_keywords: Tuple[str, ...]) -> List[Dict]:
    """섹션별 평균 점수 + 키워드 보너스로 최상 섹션 하나 선택."""
    candidates = []
    for sec, lst in groups.items():
        score = sum(h.get("score", 0.0) for h in lst) / max(1, len(lst))
        bias = 0.5 if any(k in sec for k in prefer_keywords) else 0.0
        candidates.append((score + bias, sec, lst))
    candidates.sort(reverse=True, key=lambda x: x[0])
    return candidates[0][2] if candidates else []

# 3) 불릿 추출 유틸 ----------------------------------------------------
def _split_candidates(text: str) -> List[str]:
    """불릿/번호/줄바꿈 기준으로 문장 후보 분할 + 노이즈/과도한 길이 제거."""
    parts = [p.strip() for p in BULLET_SPLIT_RE.split(text) if p and p.strip()]
    out = []
    for p in parts:
        # 제목/머릿말 노이즈 제거
        if p in TITLE_NOISE:
            continue
        # 앞에 붙은 원문 불릿기호 제거 (•, ▪, -, …)
        p = _strip_leading_bullets(p)
        # 영어 기관명 같은 짧은 줄 제거
        if re.fullmatch(r"[A-Za-z\s]+", p) and len(p.split()) <= 5:
            continue
        # 문장 다듬기
        s = re.split(r"[.!?]\s+", p)[0]
        s = re.sub(r"\s{2,}", " ", s)
        s = re.sub(r"^(그리고|또는|그러나|다만)\s+", "", s)
        # 길이 제한
        if 8 <= len(s) <= 120:
            out.append(s)
    return out

def _dedupe_keep_order(items: Iterable[str]) -> List[str]:
    seen, res = set(), []
    for it in items:
        k = it.lower()
        if k not in seen:
            seen.add(k)
            res.append(it)
    return res

def _filter_by_keywords(cands: Iterable[str], kws: Iterable[str]) -> List[str]:
    return [c for c in cands if any(k in c for k in kws)]

# 4) 섹션별 불릿 생성(본문은 아이콘 제거) -------------------------------
def bullets_from_hits(hits: List[Dict], section: str) -> List[str]:
    """섹션 의도에 맞는 불릿만 추려 반환. 본문 불릿은 하이픈(-)만 사용."""
    if section == "caution":
        kws, limit = KW_CAUT, MAX_PER_SECTION["caution"]
    elif section == "howto":
        kws, limit = KW_HOW, MAX_PER_SECTION["howto"]
    else:
        kws, limit = KW_LIFE, MAX_PER_SECTION["lifestyle"]

    pool: List[str] = []
    for h in hits:
        # 불릿 후보 뽑기
        cands = _split_candidates(h.get("text", ""))
        # 이 히트의 source_title(머릿말)과 완전 동일한 줄은 제거
        stitle = (h.get("meta", {}) or {}).get("source_title", "")
        if stitle:
            cands = [c for c in cands if c != stitle]
        pool += cands

    # 의도 키워드 우선
    picked = _filter_by_keywords(pool, kws)
    if len(picked) < max(2, limit // 2):
        picked = (picked + pool[:limit])[:limit]

    picked = _dedupe_keep_order(picked)[:limit]
    # 본문 불릿에는 아이콘 X -> 하이픈으로만
    return [f"- {b}" for b in picked]

# 5) 출처 블록(머릿말 + 꼬릿말 페이지) ----------------------------------
def sources_block(hits: List[Dict]) -> List[str]:
    """
    출처 형식: source_title(머릿말) + page_label(꼬릿말 페이지) 우선.
    page_label 없으면 내부 page로 폴백.
    """
    srcs = []
    for h in hits[:5]:
        m = h.get("meta", {}) or {}
        header = m.get("source_title") or m.get("source_file") or m.get("category") or ""
        p = m.get("page_label") or m.get("page", "?")
        if header:
            srcs.append(f"- {header} (p.{p})")
    return _dedupe_keep_order(srcs)

# 6) compose_answer ----------------------------------------------------
def compose_answer(query: str, hits: List[Dict]) -> str:
    """검색 결과(hits)로부터 질문 의도에 맞는 깔끔한 답변을 조립."""
    # 안전 보강 블록 로딩
    try:
        from .safety import build_safety_block
    except Exception:
        from safety import build_safety_block

    if not hits:
        return "관련 자료를 찾지 못했어요.\n\n### 안전 보강(자동)\n" + build_safety_block(query)

    # 1) 의도 결정
    intent = detect_intent(query)

    # 2) 섹션 묶기 → 의도 키워드 가중으로 최상 섹션만 선택
    groups = group_by_section(hits)
    if intent == "caution":
        best_hits = pick_best_section(groups, KW_CAUT)
    elif intent == "howto":
        best_hits = pick_best_section(groups, KW_HOW)
    elif intent == "lifestyle":
        best_hits = pick_best_section(groups, KW_LIFE)
    else:  # mixed
        best_hits = pick_best_section(groups, tuple())

    # 3) 섹션별 구성 (제목에만 아이콘 표시)
    parts: List[str] = []

    if intent == "caution":
        parts.append(f"### {HEADER_ICON['caution']} 주의/준비 사항")
        parts += bullets_from_hits(best_hits, "caution")
        # 보조로 생활 1~2개만
        ls = bullets_from_hits(best_hits, "lifestyle")[:2]
        if ls:
            parts.append(f"\n### {HEADER_ICON['lifestyle']} 생활 지도")
            parts += ls

    elif intent == "howto":
        parts.append(f"### {HEADER_ICON['howto']} 운동 방법")
        parts += bullets_from_hits(best_hits, "howto")

    elif intent == "lifestyle":
        parts.append(f"### {HEADER_ICON['lifestyle']} 생활 지도")
        parts += bullets_from_hits(best_hits, "lifestyle")

    else:  # 혼합
        parts.append("### 요약")
        cands = _split_candidates(best_hits[0].get("text", "")) if best_hits else []
        parts += [f"- {c}" for c in cands[:2]]
        parts.append(f"\n### {HEADER_ICON['caution']} 주의/준비 사항")
        parts += bullets_from_hits(best_hits, "caution")

    # 4) 안전/출처
    parts.append("\n### 안전 보강(자동)")
    parts.append(build_safety_block(query))

    sb = sources_block(best_hits if best_hits else hits)
    if sb:
        parts.append("\n### 출처")
        parts += sb

    return "\n".join(parts)
