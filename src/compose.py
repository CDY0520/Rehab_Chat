"""
파일명: src/compose.py
기능: 검색 청크에서 '의도에 맞는 섹션만' 깔끔하게 요약하여 답변 생성.
     - 의도 감지: 주의/방법/생활 키워드로 섹션 가드
     - 섹션 일치: 같은 section_title끼리 그룹핑 → 최고점 섹션만 사용
     - 핵심 불릿만 필터(번호/불릿/짧은 문장), 중복 제거

블록 구성
 0) 임포트/설정
 1) 의도 감지
 2) 섹션 그룹핑/선택
 3) 불릿 추출 유틸
 4) 섹션별 불릿 생성
 5) compose_answer
"""
from __future__ import annotations
from typing import List, Dict, Iterable, Tuple
import re

# --------- 0) 설정 ----------
MAX_PER_SECTION = {"caution": 6, "howto": 5, "lifestyle": 3}
KW_CAUT = ("주의", "유의", "안전", "금지", "중단", "보류", "위험", "경고")
KW_HOW  = ("방법", "자세", "실시", "반복", "세트", "강도", "속도", "호흡", "운동법")
KW_LIFE = ("생활", "휴식", "수면", "수분", "물", "식사", "영양", "준비물", "환경", "정리", "온수", "냉수")

BULLET_SPLIT_RE = re.compile(r"(?:\n|[①-⑳]|[0-9]+\s*[\.\)]|•\s+|▪\s+|-{1,2}\s+)")

# --------- 1) 의도 감지 ----------
def detect_intent(query: str) -> str:
    q = query.lower()
    if any(k in q for k in KW_CAUT):
        return "caution"
    if any(k in q for k in KW_LIFE):
        return "lifestyle"
    if any(k in q for k in KW_HOW) or "방법" in q or "어떻게" in q:
        return "howto"
    return "mixed"

# --------- 2) 섹션 그룹핑/선택 ----------
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
    # 키워드 포함 섹션 우선
    candidates = []
    for sec, lst in groups.items():
        score = sum(h["score"] for h in lst) / max(1, len(lst))
        bias = 0.5 if any(k in sec for k in prefer_keywords) else 0.0
        candidates.append((score + bias, sec, lst))
    candidates.sort(reverse=True, key=lambda x: x[0])
    return candidates[0][2] if candidates else []

# --------- 3) 불릿 추출 유틸 ----------
def _split_candidates(text: str) -> List[str]:
    parts = [p.strip() for p in BULLET_SPLIT_RE.split(text) if p and p.strip()]
    out = []
    for p in parts:
        s = re.split(r"[.!?]\s+", p)[0]
        s = re.sub(r"\s{2,}", " ", s)
        s = re.sub(r"^(그리고|또는|그러나|다만)\s+", "", s)
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
    out = []
    for c in cands:
        if any(k in c for k in kws):
            out.append(c)
    return out

# --------- 4) 섹션별 불릿 생성 ----------
def bullets_from_hits(hits: List[Dict], section: str) -> List[str]:
    if section == "caution":
        kws, icon, limit = KW_CAUT, "⚠️", MAX_PER_SECTION["caution"]
    elif section == "howto":
        kws, icon, limit = KW_HOW, "🏃", MAX_PER_SECTION["howto"]
    else:
        kws, icon, limit = KW_LIFE, "💡", MAX_PER_SECTION["lifestyle"]

    pool: List[str] = []
    for h in hits:
        pool += _split_candidates(h.get("text", ""))

    # 우선 키워드 기반으로 추림 → 부족하면 앞부분 보충
    picked = _filter_by_keywords(pool, kws)
    if len(picked) < max(2, limit // 2):
        picked = (picked + pool[:limit])[:limit]

    picked = _dedupe_keep_order(picked)[:limit]
    return [f"- {icon} {b}" for b in picked]

def sources_block(hits: List[Dict]) -> List[str]:
    srcs = []
    for h in hits[:5]:
        m = h.get("meta", {}) or {}
        title = m.get("source_title") or m.get("category") or m.get("file_name", "")
        sec   = m.get("section_title") or ""
        page  = m.get("page", "?")
        label = f"{title}" + (f" · {sec}" if sec and sec not in title else "")
        srcs.append(f"- {label} (p.{page})")
    return _dedupe_keep_order(srcs)

# --------- 5) compose_answer ----------
def compose_answer(query: str, hits: List[Dict]) -> str:
    # 안전 블록 임포트
    try:
        from .safety import build_safety_block
    except Exception:
        from safety import build_safety_block

    if not hits:
        return "관련 자료를 찾지 못했어요.\n\n### 안전 보강(자동)\n" + build_safety_block(query)

    # 1) 의도 결정
    intent = detect_intent(query)

    # 2) 섹션 묶고, 의도에 맞는 섹션 키워드 가중으로 최상 섹션만 선택
    groups = group_by_section(hits)
    if intent == "caution":
        best_hits = pick_best_section(groups, KW_CAUT)
    elif intent == "howto":
        best_hits = pick_best_section(groups, KW_HOW)
    elif intent == "lifestyle":
        best_hits = pick_best_section(groups, KW_LIFE)
    else:  # mixed
        best_hits = pick_best_section(groups, tuple())

    # 3) 섹션별 불릿 생성 (의도에 따라 필요한 섹션만)
    parts: List[str] = []

    if intent == "caution":
        parts.append("### ⚠️ 주의/준비 사항")
        parts += bullets_from_hits(best_hits, "caution")
        # 보조로 생활 1~2개만 붙임
        ls = bullets_from_hits(best_hits, "lifestyle")[:2]
        if ls:
            parts.append("\n### 💡 생활 지도")
            parts += ls

    elif intent == "howto":
        parts.append("### 🏃 운동 방법")
        parts += bullets_from_hits(best_hits, "howto")

    elif intent == "lifestyle":
        parts.append("### 💡 생활 지도")
        parts += bullets_from_hits(best_hits, "lifestyle")

    else:  # mixed
        parts.append("### 요약")
        # 혼합일 때만 짧은 요약 2개
        cands = _split_candidates(best_hits[0].get("text", "")) if best_hits else []
        parts += [f"- {c}" for c in cands[:2]]
        parts.append("\n### ⚠️ 주의/준비 사항")
        parts += bullets_from_hits(best_hits, "caution")

    # 4) 안전/출처
    parts.append("\n### 안전 보강(자동)")
    parts.append(build_safety_block(query))

    sb = sources_block(best_hits if best_hits else hits)
    if sb:
        parts.append("\n### 출처")
        parts += sb

    return "\n".join(parts)
