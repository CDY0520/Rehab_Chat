"""
파일명: safety.py
작성자: Dayeon Choi
프로젝트: Rehab_Chat (RAG 기반 지식 챗봇)
설명:
    - 사용자 입력을 분석하여 안전 가이드라인을 적용하는 모듈
    - 주요 기능:
        * 의료 관련 질문 시 면책 문구 추가
        * 자해/위험 관련 질문 시 위기 안내 메시지 추가
작성일: 2025-08-27
"""

# 0) 라이브러리 임포트
from __future__ import annotations
import re

_MEDICAL_PATTERNS = [
    r"약 ?용량", r"복용법", r"처방", r"금기", r"부작용", r"임신", r"소아", r"용량조절",
    r"혈압", r"혈당", r"항응고", r"와파린", r"NOAC", r"출혈", r"간기능", r"신장기능",
]

_SELF_HARM = [r"자해", r"죽고싶", r"극단적 선택", r"자살"]

def build_safety_block(user_query: str) -> str:
    """질문을 보고 필요 시 안전/면책 문구를 반환. 없으면 빈 문자열."""
    q = user_query.lower()

    if any(re.search(p, user_query) for p in _SELF_HARM):
        return ("⚠️ 위기 징후가 보입니다. 주변의 도움을 요청하고, 즉시 지역 긴급 상담/의료기관에 연락하세요.\n"
                "한국 생명존중 핫라인 1393, 정신건강 위기 상담 1577-0199 등을 이용할 수 있어요.")

    if any(re.search(p, user_query) for p in _MEDICAL_PATTERNS):
        return ("의료 관련 정보는 교육/참고용이에요. 개인의 임상 판단과 기관의 가이드라인을 우선해 주세요.\n"
                "필요 시 전문가와 상의해 맞춤 결정을 내리시길 권장합니다.")

    return ""