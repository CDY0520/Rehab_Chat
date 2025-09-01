"""
파일명: src/ingest.py
기능: PDF 본문을 추출해 의미 단위 '청크'로 분할하고 JSONL로 저장한다.
     - 카테고리(category) = 문서 파일명(정리본) 고정
     - 메타 저장: source_file(문서명), source_title(머릿말), section_title(섹션),
                 page(PyMuPDF 인덱스), page_label(꼬릿말 실제 페이지)
     - 목차 스킵, 개행 정리, 슬라이딩 청크(기본 400자 / overlap 50자)

블록 구성
 0) 라이브러리 임포트
 1) 설정 로드(config_rag) + 기본값
 2) 유틸(파일명 정리/머릿말&섹션 추출/꼬릿말 페이지 추출/전처리/청크화)
 3) 핵심: PDF → 청크 리스트
 4) 저장/엔트리포인트

주의
 - 텍스트 기반 PDF 가정(스캔본은 OCR 필요).
 - source_title은 페이지 상단 '실제 머릿말'만 추출되도록 기관명/로고성 문구는 무시한다.
"""
# 0) 라이브러리 임포트 -------------------------------------------------
from __future__ import annotations
import re, json
from pathlib import Path
from typing import List, Dict, Iterable
import fitz  # PyMuPDF

# 1) 설정 로드(config_rag) + 기본값 -----------------------------------
try:
    from src.config_rag import (
        DOCS_DIR, PROCESSED_DIR,
        CHUNK_WINDOW, CHUNK_OVERLAP, MIN_CHARS,
        SKIP_TOC_IF_CONTAINS, STRIP_MULTIPLE_NEWLINES
    )
except Exception:
    ROOT = Path(__file__).resolve().parents[1]
    DOCS_DIR = ROOT / "data" / "docs"
    PROCESSED_DIR = ROOT / "data" / "processed"
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    CHUNK_WINDOW, CHUNK_OVERLAP, MIN_CHARS = 400, 50, 100
    SKIP_TOC_IF_CONTAINS, STRIP_MULTIPLE_NEWLINES = ["목차"], True

# 2) 유틸 --------------------------------------------------------------
HEADER_IGNORE = (
    "국립재활원",
    "National Rehabilitation Center",
    "NRC", "copyright", "logo"
)

def _clean_title(stem: str) -> str:
    """문서 제목 표기를 위해 파일명 정리(앞 숫자/구분자 제거, 언더스코어→공백)."""
    s = re.sub(r"^\d+[_\-\s]*", "", stem)
    return s.replace("_", " ").strip()

def is_toc_page(text: str) -> bool:
    """'목차' 포함 + 줄의 30% 이상이 점선/페이지 패턴이면 목차로 간주."""
    if not text or not any(tok in text for tok in SKIP_TOC_IF_CONTAINS):
        return False
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    toc_like = sum(bool(re.search(r"(?:\.\.+\s*\d+|\s\d{1,3}$)", l)) for l in lines)
    return (toc_like / max(1, len(lines))) >= 0.3

def extract_header_title(text: str, fallback: str) -> str:
    """
    페이지 상단 3~5줄 스캔해 '머릿말'을 추출.
    - 기관명/로고성 고정 문구는 무시
    - 3~60자, 한글/영문 포함한 제목 느낌 우선
    """
    if not text:
        return fallback
    head = [l.strip() for l in text.splitlines()[:5] if l.strip()]
    for l in head:
        if any(ign in l for ign in HEADER_IGNORE):
            continue
        if 3 <= len(l) <= 60 and re.search(r"[가-힣A-Za-z]", l):
            return l
    return fallback

def extract_section_title(page_text: str, source_title: str) -> str:
    """
    페이지 본문 첫 의미 줄을 소제목으로 추정.
    - 너무 길거나 없으면 source_title 재사용
    """
    lines = [l.strip() for l in page_text.splitlines() if l.strip()]
    body_lines = lines[1:] if len(lines) > 1 else lines
    for l in body_lines[:5]:
        if 3 <= len(l) <= 60:
            return l
    return source_title

def extract_page_label(page_text: str) -> str | None:
    """
    꼬릿말 실제 페이지 번호 추출:
    - 마지막 3~5줄에서 숫자 토큰을 찾아 가장 마지막 숫자 사용
      (예: '12', '- 12 -', '12/86' → 12)
    """
    lines = [l.strip() for l in page_text.splitlines() if l.strip()]
    tail = lines[-5:] if len(lines) >= 5 else lines
    cands = []
    for l in tail:
        nums = re.findall(r"(?:^|\D)(\d{1,3})(?:\D|$)", l)
        cands.extend(nums)
    return cands[-1] if cands else None

def normalize_text(text: str) -> str:
    """연속 개행 정리 등 간단 전처리."""
    return re.sub(r"\n{3,}", "\n\n", text).strip() if STRIP_MULTIPLE_NEWLINES else text.strip()

def paragraphs(text: str) -> List[str]:
    """두 줄 이상 개행 기준 문단 분리."""
    return [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]

def sliding_chunks(paras: Iterable[str], window=CHUNK_WINDOW, overlap=CHUNK_OVERLAP) -> List[str]:
    """문단을 누적하면서 길이가 window를 넘으면 overlap 만큼 물려 자른다."""
    chunks, cur = [], ""
    for para in paras:
        cur = para if not cur else (cur + "\n" + para)
        while len(cur) > window:
            chunks.append(cur[:window])
            cur = cur[window - overlap:]
    if cur:
        chunks.append(cur)
    return [c for c in chunks if len(c) >= MIN_CHARS]

# 3) 핵심: PDF → 청크 --------------------------------------------------
def pdf_to_chunks(pdf_path: Path) -> List[Dict]:
    doc = fitz.open(pdf_path)
    chunks_all: List[Dict] = []
    doc_title = _clean_title(pdf_path.stem)  # 문서 단위 카테고리/출처 표기명

    for pno in range(len(doc)):
        page = doc[pno]
        text = page.get_text("text")
        if not text:
            continue
        if is_toc_page(text):
            continue

        page_text = normalize_text(text)
        source_title  = extract_header_title(page_text, fallback=doc_title)  # 머릿말
        section_title = extract_section_title(page_text, source_title)       # 섹션
        page_label    = extract_page_label(page_text)                        # 꼬릿말 페이지

        # 머릿말 1줄 제거 후 본문에서 청크 생성
        body_lines = page_text.splitlines()
        body = "\n".join(body_lines[1:]) if len(body_lines) > 1 else page_text
        paras = paragraphs(body)
        chunks = sliding_chunks(paras)

        for ch in chunks:
            chunks_all.append({
                "text": ch,
                "meta": {
                    "file_name": pdf_path.name,
                    "source_file": doc_title,        # 문서명(정리본)
                    "source_title": source_title,    # 머릿말
                    "section_title": section_title,  # 섹션(없으면 머릿말 동일)
                    "page": pno + 1,                 # 내부 페이지(1부터)
                    "page_label": page_label,        # 꼬릿말 실제 페이지
                    "category": doc_title            # 문서 단위 검색 필터
                }
            })
    return chunks_all

# 4) 저장/엔트리포인트 -------------------------------------------------
def save_jsonl(chunks: List[Dict], out_path: Path):
    with out_path.open("w", encoding="utf-8") as f:
        for obj in chunks:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def main():
    pdfs = sorted((DOCS_DIR).glob("*.pdf"))
    if not pdfs:
        print(f"[안내] PDF가 없습니다: {DOCS_DIR}")
        return
    total = 0
    for pdf in pdfs:
        chunks = pdf_to_chunks(pdf)
        out = PROCESSED_DIR / f"{pdf.stem}.jsonl"
        save_jsonl(chunks, out)
        total += len(chunks)
        print(f" - {pdf.name}: {len(chunks)} chunks → {out.name}")
    print(f"✅ 완료: {len(pdfs)}개 파일, 총 {total}개 청크 생성")

if __name__ == "__main__":
    main()
