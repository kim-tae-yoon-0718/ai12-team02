"""
grader.normalize — 3-7 문자열 매칭 채점의 함정 처리

근거:
  3-7  비교 전 정규화 / 표기 변형 정답 처리 / 부분 일치 규칙
  1-14 금액(원·천원·백만원 혼용)·날짜 표기 실측 [대기 ← 박예진]
  1-17 유니코드 정규화 — ★코퍼스와 채점기가 **같은 규칙**을 써야 한다
  2-8  복수 정답 / 허용 표기 목록 [대기 ← 임현진]

경고 (3-7):
  정규화 없이 비교하면 맞은 답을 틀렸다고 채점하는 **조용한 오류**가 생긴다.
  금액·날짜는 이 프로젝트의 핵심 필드라 여기서 새는 점수가 크다.

[대기] ORG_ALIASES 는 1-18 실측(박예진)이 나오면 채운다.
       지금은 구조만 두고 비워 둔다 — 추측으로 채우면 실측을 덮어쓴다.
"""

from __future__ import annotations

import re
import unicodedata

# 1-17과 반드시 동일해야 하는 값. 다르면 검색·매칭이 조용히 어긋난다.
UNICODE_FORM = "NFC"

# [대기 ← 1-18] 기관명 약칭/정식명 사전. 실측 후 채운다.
ORG_ALIASES: dict[str, str] = {}

_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[·・,()\[\]{}<>「」『』\"'`~!?:;／/\\|+*^=_-]")


def to_halfwidth(s: str) -> str:
    return unicodedata.normalize("NFKC", s)


def nfc(s: str) -> str:
    return unicodedata.normalize(UNICODE_FORM, s)


def normalize_text(s, drop_punct: bool = True, lower: bool = True) -> str:
    """공백·대소문자·전각/반각·문장부호·유니코드 정규화."""
    if s is None:
        return ""
    s = nfc(to_halfwidth(str(s))).strip()
    if lower:
        s = s.lower()
    if drop_punct:
        s = _PUNCT.sub(" ", s)
    s = _WS.sub(" ", s).strip()
    return ORG_ALIASES.get(s, s)


# ------------------------------------------------------------------ 금액

_HANGUL_DIGIT = {"영": 0, "일": 1, "이": 2, "삼": 3, "사": 4, "오": 5,
                 "육": 6, "칠": 7, "팔": 8, "구": 9}
_BIG = {"조": 10**12, "억": 10**8, "만": 10**4}
_SMALL = {"천": 1000, "백": 100, "십": 10}

# 금액처럼 보이는 구간. '5천만원', '500,000,000원', '5억', '오억원' 등
_AMOUNT_SPAN = re.compile(
    r"(?:[0-9][0-9,]*(?:\.[0-9]+)?|[영일이삼사오육칠팔구]+)"
    r"\s*(?:조|억|천|백|십|만)*\s*원?")


def parse_amount(s) -> int | None:
    """한국어 금액 표기를 '원 단위 정수'로 정규화.

    5억 / 500,000,000원 / 500백만원 / 오억 / 5천만원 / 3억5천만원 → 정수
    실패하면 None (조용히 0을 반환하지 않는다 — 0은 유효한 값이다).

    ★ 3-10 테스트가 잡은 버그: 단순 정규식 곱셈으로는 '5천만'을 5,000으로 읽는다.
      금액 단위만 틀린 답을 정답 처리하는 것이 이 도메인에서 가장 비싼 채점 오류다.
      그래서 자리수 누산 방식으로 다시 썼다.
    """
    if s is None:
        return None
    t = to_halfwidth(str(s)).replace(",", "").strip()
    t = re.sub(r"(이상|이하|초과|미만|내외|정도|약|원)", "", t)
    for k, v in _HANGUL_DIGIT.items():
        t = t.replace(k, str(v))
    t = _WS.sub("", t)
    if not t or not re.search(r"\d", t):
        return None

    total = 0.0
    current = 0.0
    num: float | None = None
    i = 0
    seen_unit = False
    while i < len(t):
        m = re.match(r"\d+(?:\.\d+)?", t[i:])
        if m:
            num = float(m.group(0))
            i += len(m.group(0))
            continue
        ch = t[i]
        if ch in _SMALL:
            current += (num if num is not None else 1) * _SMALL[ch]
            num = None
            seen_unit = True
        elif ch in _BIG:
            current += (num if num is not None else 0)
            total += (current if current else 1) * _BIG[ch]
            current = 0.0
            num = None
            seen_unit = True
        else:
            return None          # 금액 표기가 아니다
        i += 1
    total += current + (num if num is not None else 0)
    if not seen_unit and total == 0:
        return None
    return int(round(total))


def extract_amounts(text: str) -> list[tuple[int, str]]:
    """문장 안에서 금액 후보를 (값, 원문구간) 으로 뽑는다.
    ★'정답 + 환각 덧붙임'을 가려내려면 값만이 아니라 **어디서 나왔는지**가 필요하다."""
    out = []
    for m in _AMOUNT_SPAN.finditer(to_halfwidth(str(text or ""))):
        span = m.group(0).strip()
        if not span or not re.search(r"[0-9영일이삼사오육칠팔구]", span):
            continue
        v = parse_amount(span)
        if v is not None:
            out.append((v, span))
    return out


# ------------------------------------------------------------------ 날짜

_DATE_PATTERNS = [
    re.compile(r"(?P<y>\d{4})\s*[-./년]\s*(?P<m>\d{1,2})\s*[-./월]\s*(?P<d>\d{1,2})"),
    re.compile(r"(?P<y>\d{4})\s*[-./년]\s*(?P<m>\d{1,2})\s*월?$"),
]


def parse_date(s) -> str | None:
    """날짜 표기를 ISO(YYYY-MM-DD)로. 일자가 없으면 YYYY-MM 까지."""
    if s is None:
        return None
    t = to_halfwidth(str(s)).strip()
    m = _DATE_PATTERNS[0].search(t)
    if m:
        return f"{int(m.group('y')):04d}-{int(m.group('m')):02d}-{int(m.group('d')):02d}"
    m = _DATE_PATTERNS[1].search(t)
    if m:
        return f"{int(m.group('y')):04d}-{int(m.group('m')):02d}"
    return None


def _date_span(text: str) -> tuple[str, str] | None:
    t = to_halfwidth(str(text or ""))
    m = _DATE_PATTERNS[0].search(t)
    if not m:
        return None
    return (f"{int(m.group('y')):04d}-{int(m.group('m')):02d}-{int(m.group('d')):02d}",
            m.group(0))


# ------------------------------------------------------------------ 매칭

# 값 뒤에 남는 군더더기 허용 길이. 이 이상 남으면 '정답 + 덧붙임'으로 본다.
RESIDUAL_LIMIT = 20


def _residual_ok(text: str, span: str, limit: int = RESIDUAL_LIMIT) -> bool:
    rest = to_halfwidth(str(text)).replace(span, " ", 1)
    rest = normalize_text(rest)
    return len(rest) <= limit


def canonical(value, kind: str = "auto") -> str:
    """비교용 정규형. kind: auto|amount|date|text"""
    if value is None:
        return ""
    if kind in ("auto", "date"):
        d = parse_date(value)
        if d is not None:
            return f"DATE:{d}"
    if kind in ("auto", "amount"):
        a = parse_amount(value)
        if a is not None:
            return f"AMT:{a}"
    return normalize_text(value)


def match_short(gold, pred, accept=None, kind: str = "auto",
                allow_partial: bool = False,
                residual_limit: int = RESIDUAL_LIMIT) -> tuple[bool, str]:
    """단답 채점. (맞았는가, 판정 근거) 를 함께 돌려준다 — 문항별 결과에 판정 이유로 저장한다.

    ★ 문장 안에서 금액·날짜를 뽑아 비교하면 그 자체가 사실상 '부분 일치'가 되어,
      '정답 + 근거에 없는 내용 덧붙임' 답변이 만점을 받는다.
      ⇒ 값이 맞아도 **뒤에 남은 군더더기가 길면** 통과시키지 않는다(allow_partial=False 기준).
    """
    accept = accept or []
    golds = [gold] + list(accept)
    pred_s = str(pred if pred is not None else "")

    # 0) 정확 일치 — 정답 자체가 서술형(날짜/금액을 포함한 문장)이어도 pred 가 정답 그대로면 통과.
    #    (이 검사를 뒤에 두면 "정답 == pred" 인데 verbose 패널티로 0점 나는 버그가 생긴다)
    np = normalize_text(pred_s)
    for g in golds:
        ng = normalize_text(g)
        if ng and ng == np:
            return True, "normalized_exact"

    # 1) 날짜
    g_date = parse_date(gold)
    if g_date:
        ds = _date_span(pred_s)
        if ds and ds[0] == g_date:
            if allow_partial or _residual_ok(pred_s, ds[1], residual_limit):
                return True, "date"
            return False, "date_match_but_verbose(정답+덧붙임 의심)"

    # 2) 금액
    g_amt = parse_amount(gold) if not g_date else None
    if g_amt is not None:
        for val, span in extract_amounts(pred_s):
            if val == g_amt:
                if allow_partial or _residual_ok(pred_s, span, residual_limit):
                    return True, "amount"
                return False, "amount_match_but_verbose(정답+덧붙임 의심)"

    # 3) 문자열 (정확 일치는 위 0)에서 이미 처리)
    if allow_partial:
        for g in golds:
            ng = normalize_text(g)
            if ng and ng in np:
                return True, "partial_contains(주의: 환각 덧붙임도 통과)"
    return False, "no_match"


def match_location(gold, pred, precision: str = "section") -> bool:
    """근거/좌표 일치. precision: document | section | ref_no (세분 정도가 올라가는 순서).

    ref_no 정밀도에서 (임현진 09-01):
      - gold 에 field 가 있으면(비교형 원소) 문서+절만 맞아도 인정 — field 그룹핑은 상위(grade_*)에서
      - 양쪽에 line 정보가 있으면 ref_no 대신 line 범위 포함으로 대조
        (block_index 가 source_type 별로 세져 "heading 0" 이 문서에 여럿 → line 이 명확)
    """
    if precision != "ref_no":
        return gold.key(precision) == pred.key(precision)

    if gold.key("section") != pred.key("section"):
        return False
    g_line = getattr(gold, "line", None)
    p_line = getattr(pred, "line", None)
    if g_line is not None and p_line is not None:
        p_end = getattr(pred, "line_end", None) or p_line
        return min(p_line, p_end) <= g_line <= max(p_line, p_end)
    return gold.key("ref_no")[-1] == pred.key("ref_no")[-1]
