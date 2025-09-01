# rag/02c_apply_manual_edits.py
# 기능: 사용자가 엑셀에서 수정한 precautions_preview.csv를 읽어
# 동일 스키마의 최종 precautions.jsonl로 덮어쓰기

from pathlib import Path
import csv, json

ROOT = Path(__file__).resolve().parents[1]
CSV_IN  = ROOT / "data" / "processed" / "precautions_preview.csv"   # 수정본
JSON_OUT = ROOT / "data" / "processed" / "precautions.jsonl"        # 최종본

def main():
    if not CSV_IN.exists():
        raise FileNotFoundError(CSV_IN)
    rows = []
    with CSV_IN.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            # 불필요한 공백 정리
            row = {k: (v.strip() if isinstance(v, str) else v) for k, v in row.items()}
            rows.append(row)
    with JSON_OUT.open("w", encoding="utf-8") as f:
        for rec in rows:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[OK] 반영 완료 → {JSON_OUT}")

if __name__ == "__main__":
    main()
