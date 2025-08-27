"""
파일명: src/ingest.py
기능: PDF 본문을 추출하여 의미 단위 '청크'로 분할하고 JSONL로 저장한다.
     - 카테고리(category) = PDF 파일명(확장자 제외) 정리본  ← 문서 단위로 고정
     - 메타: file_name, source_title(머릿말), section_title(섹션), page
     - 목차 페이지 스킵, 개행 정리
     - 더 촘촘한 청크(기본 400자 / overlap 50자)

블록 구성
 0) 라이브러리 임포트
 1) 설정 로드(config_rag) + 기본값
 2) 유틸(파일명 정리/목차 판별/제목 추출/전처리/슬라이딩 청크)
 3) 핵심: PDF → 청크 리스트
 4) 저장: JSONL
 5) main()
주의
 - 텍스트 기반 PDF를 가정(스캔 PDF는 OCR 필요).
"""
# 0) 라이브러리 임포트
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
    # 단독 실행 대비
    ROOT = Path(__file__).resolve().parents[1]
    DOCS_DIR = ROOT / "data" / "docs"
    PROCESSED_DIR = ROOT / "data" / "processed"
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    CHUNK_WINDOW = 400
    CHUNK_OVERLAP = 50
    MIN_CHARS = 100
    SKIP_TOC_IF_CONTAINS = ["목차"]
    STRIP_MULTIPLE_NEWLINES = True

# 2) 유틸 -------------------------------------------------------------
def _clean_title(file_stem: str) -> str:
    """문서 제목 표시용으로 파일명을 정리(앞 숫자/구분자 제거, 언더스코어→공백)."""
    s = re.sub(r"^\d+[_\-\s]*", "", file_stem)  # 예: '09_', '100-' 제거
    s = s.replace("_", " ").strip()
    return s

def is_toc_page(text: str) -> bool:
    """텍스트에 '목차'가 있고, 줄의 30% 이상이 번호/점선 패턴이면 목차로 간주."""
    if not text:
        return False
    if not any(tok in text for tok in SKIP_TOC_IF_CONTAINS):
        return False
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    toc_like = sum(bool(re.search(r"(?:\.\.+\s*\d+|\s\d{1,3}$)", l)) for l in lines)
    return (toc_like / max(1, len(lines))) >= 0.3

def extract_header_title(text: str, fallback: str) -> str:
    """페이지 상단 1~3줄에서 머릿말 제목 후보를 추출, 없으면 fallback."""
    if not text:
        return fallback
    head = [l.strip() for l in text.splitlines()[:3] if l.strip()]
    for l in head:
        if re.search(r"(가이드|재활|뇌졸중|프로그램|건강|운동|주의|생활|유의)", l):
            return l[:120]
    return fallback

def extract_section_title(page_text: str, source_title: str) -> str:
    """본문 첫 의미 줄을 섹션 제목으로 추정(너무 길거나 없으면 source_title 재사용)."""
    lines = [l.strip() for l in page_text.splitlines() if l.strip()]
    body_lines = lines[1:] if len(lines) > 1 else lines
    for l in body_lines[:5]:
        if 3 <= len(l) <= 60:
            return l
    return source_title

def normalize_text(text: str) -> str:
    if STRIP_MULTIPLE_NEWLINES:
        text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def paragraphs(text: str) -> List[str]:
    return [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]

def sliding_chunks(paras: Iterable[str], window=CHUNK_WINDOW, overlap=CHUNK_OVERLAP) -> List[str]:
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
    # ★ 문서 단위 카테고리 고정 (파일명 정리본)
    doc_category = _clean_title(pdf_path.stem)

    for pno in range(len(doc)):
        page = doc[pno]
        text = page.get_text("text")
        if not text:
            continue
        if is_toc_page(text):
            continue

        page_text = normalize_text(text)
        source_title  = extract_header_title(page_text, fallback=doc_category)  # 머릿말(참고)
        section_title = extract_section_title(page_text, source_title)          # 섹션(참고)

        # 머릿말(첫 줄) 제거 후 본문 청크화
        body_lines = page_text.splitlines()
        body = "\n".join(body_lines[1:]) if len(body_lines) > 1 else page_text
        paras = paragraphs(body)
        chunks = sliding_chunks(paras)

        for ch in chunks:
            chunks_all.append({
                "text": ch,
                "meta": {
                    "file_name": pdf_path.name,
                    "source_title": source_title,
                    "section_title": section_title,
                    "page": pno + 1,
                    "category": doc_category,   # ★ 문서(파일) 단위로 고정
                }
            })
    return chunks_all

# 4) 저장 --------------------------------------------------------------
def save_jsonl(chunks: List[Dict], out_path: Path):
    with out_path.open("w", encoding="utf-8") as f:
        for obj in chunks:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

# 5) main --------------------------------------------------------------
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
