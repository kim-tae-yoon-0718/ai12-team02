"""
문자열 정규화 — 기관명·사업명을 **비교할 때만** 쓰는 키를 만든다.

원칙 (2026-09-02 문서 특정 정책 확정):
- 원본 표시 문자열은 절대 바꾸지 않는다. 여기서 만든 키는 비교 전용이다.
- 기관명은 법인격 표기 차이만 정리한다. `협회`·`연구원`·`진흥원`처럼 기관의
  실제 핵심 이름에 해당하는 부분은 절대 지우지 않는다.
- 사업명은 유니코드 정규화·공백·문장부호/괄호 표기 차이만 정리한다.
  단어를 임의로 빼거나, 유사도·임베딩으로 "비슷한 사업"을 고르지 않는다.
"""
from __future__ import annotations

import re
import unicodedata

# 법인격 표기 — 비교 시에만 제거한다.
# NFKC를 먼저 적용하므로 ㈜ → "(주)", 전각 괄호 "（사）" → "(사)"로 이미 접힌 상태다.
_LEGAL_ENTITY_MARKERS = (
    "주식회사",
    "사단법인",
    "재단법인",
    "(주)",
    "(사)",
    "(재)",
)

# 기관 핵심 이름 — 실수로 지우면 안 되는 표현(회귀 테스트용 화이트리스트)
PROTECTED_ORG_TOKENS = (
    "협회", "연구원", "진흥원", "재단", "공사", "공단", "센터", "위원회",
    "대학교", "교육청", "연구소", "조합", "은행", "병원", "박물관",
)


def _strip_punctuation_and_space(text: str) -> str:
    """유니코드 문장부호(P*)·기호성 구분자·모든 공백을 제거한다."""
    out = []
    for ch in text:
        if ch.isspace():
            continue
        cat = unicodedata.category(ch)
        if cat.startswith("P"):          # 괄호·하이픈·마침표·따옴표 등
            continue
        if ch in "·ㆍ~∼―─–—/\\|＋+￦₩$":  # 자주 섞이는 구분·화폐 기호
            continue
        out.append(ch)
    return "".join(out)


def normalize_display(text: str | None) -> str:
    """표시·저장용 정규화 — NFC + 앞뒤 공백 제거 + 연속 공백 1칸.
    문자를 지우지 않는다(내용 보존)."""
    if not text:
        return ""
    s = unicodedata.normalize("NFC", str(text))
    return re.sub(r"\s+", " ", s).strip()


# 등록명 끝에 붙는 괄호 주석 — "(용역)", "(협상)(긴급)", "(총체 및 1차)" 처럼
# 절차·회차를 적어 둔 부분이다. 지금 정규화는 괄호만 지우고 **안의 글자는 남겨서**
# 키에 그대로 붙는다("한국철도공사용역", "…고도화사업협상긴급"). 그래서 사람이 부르는
# 이름("한국철도공사", "봉화군 재난통합관리시스템 고도화 사업")과 매칭되지 않는다.
# 아래는 그 주석을 **통째로 떼어낸** 보조 키를 만든다(원본 표시는 건드리지 않는다).
_BRACKETED_RE = re.compile(r"[（(\[「【][^）)\]」】]*[）)\]」】]")


def strip_bracketed(text: str | None) -> str:
    """괄호로 묶인 주석을 통째로 제거한다(원본 표시용 문자열은 바꾸지 않는다)."""
    if not text:
        return ""
    s = unicodedata.normalize("NFKC", str(text))
    prev = None
    while prev != s:                       # 중첩·연속 괄호까지 모두 제거
        prev = s
        s = _BRACKETED_RE.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip()


def normalize_org_key_unbracketed(text: str | None) -> str:
    """괄호 주석을 뗀 기관명 비교 키. 원래 키와 같으면 빈 문자열."""
    stripped = strip_bracketed(text)
    key = normalize_org_key(stripped)
    return "" if key == normalize_org_key(text) else key


def normalize_project_key_unbracketed(text: str | None) -> str:
    """괄호 주석을 뗀 사업명 비교 키. 원래 키와 같으면 빈 문자열."""
    stripped = strip_bracketed(text)
    key = normalize_project_key(stripped)
    return "" if key == normalize_project_key(text) else key


def normalize_org_key(text: str | None) -> str:
    """기관명 비교 키.

    적용: NFKC(㈜·전각괄호 접기) → 법인격 표기 제거 → 문장부호·공백 제거 → casefold.
    미적용: 기관의 핵심 이름(협회·연구원·진흥원 등)은 그대로 둔다.

    >>> normalize_org_key("(사)벤처기업협회") == normalize_org_key("벤처기업협회")
    True
    >>> normalize_org_key("사단법인 보험개발원") == normalize_org_key("(사)보험개발원")
    True
    >>> "협회" in normalize_org_key("(사)벤처기업협회")
    True
    """
    if not text:
        return ""
    s = unicodedata.normalize("NFKC", str(text)).strip()
    for marker in _LEGAL_ENTITY_MARKERS:
        s = s.replace(marker, "")
    return _strip_punctuation_and_space(s).casefold()


def normalize_project_key(text: str | None) -> str:
    """사업명 비교 키 — NFKC → 문장부호·괄호·공백 제거 → casefold.
    단어를 빼거나 바꾸지 않는다."""
    if not text:
        return ""
    s = unicodedata.normalize("NFKC", str(text)).strip()
    return _strip_punctuation_and_space(s).casefold()


def strip_leading_org_from_project(project_name: str, buyer_org: str) -> str:
    """사업명 앞에 발주기관명이 그대로 덧붙어 있으면 그 접두부만 떼어낸 나머지를
    돌려준다(원본은 보존, 반환값은 비교용 보조 표현).

    공식 identity_v2에는 `project_name`이 "서민금융진흥원 서민금융 채팅 상담시스템
    구축"처럼 기관명을 접두로 포함하는 행이 있다. 사용자는 보통 기관명을 빼고
    "서민금융 채팅 상담시스템 구축 사업"이라고만 부른다. 이건 사업명 단어를
    임의로 삭제하는 게 아니라, **같은 행의 buyer_org 값과 글자 그대로 겹치는
    접두부**만 제거하는 결정적 연산이라 재현 가능하고 데이터에 근거한다.

    기관명 접두가 없으면 빈 문자열을 반환한다(추가 후보 없음).
    """
    proj_key = normalize_project_key(project_name)
    org_key = normalize_org_key(buyer_org)
    if not proj_key or not org_key:
        return ""
    # 법인격 표기가 제거된 org_key 기준으로 접두 비교하기 위해 사업명도 같은
    # 방식(법인격 제거)으로 한 번 더 접어서 비교한다.
    proj_key_legalless = normalize_org_key(project_name)
    if proj_key_legalless.startswith(org_key) and len(proj_key_legalless) > len(org_key):
        return proj_key_legalless[len(org_key):]
    if proj_key.startswith(org_key) and len(proj_key) > len(org_key):
        return proj_key[len(org_key):]
    return ""
