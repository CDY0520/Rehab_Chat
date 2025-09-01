# -*- coding: utf-8 -*-
"""
파일명: rag/02b_precautions_from_combined.py
용도: 수동 합본 '운동 주의사항.pdf'에서
     - 상단 헤더(원문 제목) 감지
     - 하단 풋터의 숫자 라벨을 읽어 '원본 페이지' 저장
     - 본문(헤더/풋터 제외) 추출 → 문장 규칙 분리
     - category/severity/intent/keywords 자동 태깅
     - 레코드마다 source_title / source_page(합본) / source_orig_page(원본) 기록
출력: data/processed/precautions.jsonl, precautions_preview.csv

요구:
    pip install pdfplumber==0.11.0
"""

from pathlib import Path
import pdfplumber
import unicodedata
import re
import json
import csv
from typing import List, Dict, Optional

# ───────────────────────────── 경로 ─────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
PDF_PATH = ROOT / "data" / "docs" / "운동 주의사항.pdf"
OUT_JSONL = ROOT / "data" / "processed" / "precautions.jsonl"
OUT_CSV   = ROOT / "data" / "processed" / "precautions_preview.csv"
OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)

# ───────────────────── 제목 후보(합본 헤더에 넣은 원문명) ─────────────────────
KNOWN_TITLES = [
    "뇌졸중 장애인의 건강생활 가이드",
    "뇌졸중 장애인을 위한 복합형 재활체육프로그램 가이드북",
    "운동 가이드·운동처방 가이드",
]

# ───────────────────────── 크롭 파라미터 ─────────────────────────
HEADER_READ_HEIGHT = 120  # 상단 영역(제목 감지)
BODY_HEADER_CROP   = 80   # 본문 추출 시 상단 제거
BODY_FOOTER_CROP   = 60   # 본문 추출 시 하단 제거
FOOTER_READ_HEIGHT = 90   # 하단 영역(원본 페이지 라벨 감지)

# ───────────────────────── 유틸 ─────────────────────────
def nfkc(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "")
    s = s.replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{2,}", "\n", s)
    return s.strip()

# ───────────────── 제목 감지(합본 헤더) ─────────────────
def read_header_title(page) -> str:
    x0, top, x1, bottom = page.bbox
    bbox = (x0, top, x1, top + HEADER_READ_HEIGHT)
    area = page.crop(bbox)           # pdfplumber 0.11.x: 컨텍스트X
    txt = nfkc(area.extract_text() or "")
    if not txt:
        return ""
    hits = [t for t in KNOWN_TITLES if t in txt]
    if hits:
        return sorted(hits, key=len, reverse=True)[0]
    if re.search(r"(가이드|재활|운동|국립재활원)", txt):
        return txt.split("\n")[0].strip()[:80]
    return ""

# ──────────────── 원본 페이지 라벨 감지(풋터 숫자) ────────────────
def read_footer_page_label(page) -> Optional[int]:
    """
    페이지 하단(풋터)에서 숫자 라벨 추출.
    예: '16', '국립재활원 16', '16 National Rehabilitation Center' 등에서 16만 추출
    """
    x0, top, x1, bottom = page.bbox
    bbox = (x0, bottom - FOOTER_READ_HEIGHT, x1, bottom)
    area = page.crop(bbox)
    txt = nfkc(area.extract_text() or "")
    if not txt:
        return None
    nums = re.findall(r"\b(\d{1,3})\b", txt)
    if not nums:
        return None
    # 뒤쪽/오른쪽에 가까운 숫자가 실제 페이지일 가능성이 높음 → 뒤에서부터 확인
    for n in nums[::-1]:
        try:
            val = int(n)
            if 1 <= val <= 1000:
                return val
        except Exception:
            pass
    return None

# ───────────────────── 본문 추출(헤더/풋터 제외) ─────────────────────
def read_body_text(page) -> str:
    x0, top, x1, bottom = page.bbox
    bbox = (x0, top + BODY_HEADER_CROP, x1, bottom - BODY_FOOTER_CROP)
    area = page.crop(bbox)
    txt = nfkc(area.extract_text() or "")
    # 노이즈 제거(기관명/페이지번호 텍스트 등)
    txt = re.sub(r"\b(국립재활원|National Rehabilitation Center)\b", "", txt, flags=re.I)
    txt = re.sub(r"\bPage\s*\d+\b|\b페이지\s*\d+\b", "", txt, flags=re.I)
    return nfkc(txt)

# ───────────────────── 규칙 분리(불릿/번호) ─────────────────────
def split_rules(s: str) -> List[str]:
    bullets = {
        "①": "1.", "②": "2.", "③": "3.", "④": "4.", "⑤": "5.", "⑥": "6.",
        "•": "-", "·": "-", "‣": "-", "▪": "-", "▶": "-", "▸": "-", "■": "-"
    }
    for k, v in bullets.items():
        s = s.replace(k, v)

    rules = []
    for raw in s.split("\n"):
        line = raw.strip()
        if not line:
            continue

        parts = re.split(r"(?:(?<=^)|\s)(\d+\.)\s*", line)
        if len(parts) > 1:
            cur = ""
            for p in parts:
                if re.fullmatch(r"\d+\.", p or ""):
                    if cur.strip():
                        rules.append(cur.strip())
                    cur = ""
                else:
                    cur += (" " + (p or ""))
            if cur.strip():
                rules.append(cur.strip())
        else:
            if line.startswith(("-", "–", "—")):
                rules.append(line.lstrip("-–— ").strip())
            elif len(line) >= 8:
                rules.append(line)

    uniq, seen = [], set()
    for r in rules:
        k = re.sub(r"\s+", " ", r)
        if k not in seen:
            uniq.append(k)
            seen.add(k)
    return uniq

# ───────────────────── 태깅 규칙 ─────────────────────
CATS = [
    ("건강상태/증상", ["발열","감기","두통","어지럼","혈압","서맥","흉통","호흡곤란"]),
    ("사전측정",     ["혈압","심박","맥박","산소포화도"]),
    ("복장/환경",     ["옷","의복","소매","바지","신발","바닥","공간","미끄러"]),
    ("안전/낙상",     ["지지대","보조","균형","낙상","부축","보행","손잡이","보호자"]),
    ("식사/약물",     ["식사","식후","공복","저혈당","당뇨","약"]),
]

def guess_category(t: str) -> str:
    for cat, kws in CATS:
        if any(kw in t for kw in kws):
            return cat
    return "운동 주의사항"

def guess_severity(t: str) -> str:
    if any(k in t for k in ["낙상","심한 어지럼","흉통","실신","의식저하","호흡곤란","고혈압","저혈압","출혈"]):
        return "High"
    if any(k in t for k in ["혈압","지지대","보호자","균형","서맥","심박","산소포화도"]):
        return "Medium"
    return "Low"

def extract_intent(t: str) -> str:
    if re.search(r"(하지 마|금지|중단|멈추)", t): return "금지"
    if re.search(r"(확인|주의|점검|체크|모니터)", t): return "주의"
    if re.search(r"(권장|착용|사용|준비|정리|입으|이용)", t): return "권장"
    return "정보"

def extract_keywords(t: str) -> list:
    base = ["발열","감기","두통","혈압","서맥","어지럼","지지대","균형",
            "낙상","식사","식후","공복","보호자","신발","미끄럼","심박","산소포화도"]
    return [kw for kw in base if kw in t]

# ───────────────────── 메인 ─────────────────────
def main():
    if not PDF_PATH.exists():
        raise FileNotFoundError(PDF_PATH)

    records: List[Dict] = []
    with pdfplumber.open(str(PDF_PATH)) as pdf:
        current_title = ""

        for i, page in enumerate(pdf.pages, start=1):
            # 1) 합본 헤더에서 원문 제목 감지
            title = read_header_title(page) or current_title
            current_title = title or current_title

            # 2) 풋터에서 '원본 페이지' 라벨 읽기
            orig_page = read_footer_page_label(page)  # 실패 시 None

            # 3) 본문 추출
            body = read_body_text(page)
            if not body:
                continue

            # 4) 규칙 분리 → 레코드 생성
            for rule in split_rules(body):
                rec = {
                    "id": f"PR-{len(records)+1:03d}",
                    "canonical_ko": rule,
                    "category": guess_category(rule),
                    "severity": guess_severity(rule),
                    "intent": extract_intent(rule),
                    "keywords": extract_keywords(rule),
                    "source_title": current_title or "운동 주의사항(합본)",
                    "source_file": "운동 주의사항.pdf",
                    "source_page": i,                 # 합본 페이지(참고)
                    "source_orig_page": orig_page,     # ✅ 원본 페이지(풋터 인식)
                }
                records.append(rec)

    if not records:
        print("[WARN] 추출된 규칙이 없습니다. 크롭 파라미터/제목 후보를 점검하세요.")
        return

    with OUT_JSONL.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        w.writeheader()
        for r in records:
            w.writerow(r)

    print(f"[OK] 규칙 {len(records)}개 저장")
    print(f" - JSONL: {OUT_JSONL}")
    print(f" - CSV  : {OUT_CSV}")

if __name__ == "__main__":
    main()
