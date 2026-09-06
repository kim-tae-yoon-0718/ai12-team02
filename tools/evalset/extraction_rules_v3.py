#!/usr/bin/env python3
#@title 1-12-2 추출 규칙 — 12개 필드의 항목명·동의어·판정 기준
#@markdown 파일시스템을 모르는 **순수 규칙 모음**이다. 합성 입력으로 시험할 수 있어야 하므로
#@markdown 여기에는 파일을 읽는 코드를 두지 않는다.
#@markdown
#@markdown 항목명 목록의 출발점은 30건 기준 자료(`rfp_30_fields_audit_merged.csv`)의
#@markdown `항목명 문자열 그대로` 열이다. 새 표현이 나오면 **필드를 늘리지 않고**
#@markdown 기존 12개 중 하나에 연결하고, 후보 목록에만 따로 기록한다.

from __future__ import annotations

import re
import unicodedata

# ══════════════════════════════════════════════════════════════════
# 0. 필드 — 정확히 12개. 절대 늘리지 않는다.
# ══════════════════════════════════════════════════════════════════

FIELDS = [
    "사업 개요", "사업분야", "공고일", "사업기간", "예산",
    "참가 자격(면허·실적)", "지역제한", "컨소시엄 요건", "평가 배점",
    "제출 방식", "필수 제출 서류", "과업 범위",
]
FIELD_SET = frozenset(FIELDS)

# 허용 상태값 — 필드 값, 부재, 비공개, 정상 최종 안내 2종, 대기 2종
STATUSES = ("value_present", "field_absent", "not_disclosed",
            "conflict", "external_reference", "extraction_failed", "review_required")

# 출처 유형 — 이 5개만 쓴다
SOURCE_TYPES = ("metadata_csv", "front_matter", "body_table",
                "body_sentence", "attachment_form")


def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s or "")


def _pat(*words: str) -> re.Pattern:
    """항목명 정규식을 만든다.

    한글 항목명은 자간이 벌어진 형태(`사 업 비`)로도 자주 나오므로,
    글자 사이에 공백이 끼어도 잡히게 만든다(1-12-1 전처리가 allow-list 항목명은
    이미 붙여 놓았지만, 목록에 없던 표현은 벌어진 채 남아 있다)."""
    alts = []
    for w in words:
        alts.append(r"\s*".join(re.escape(ch) for ch in w))
    return re.compile("(?:" + "|".join(alts) + ")")


# ══════════════════════════════════════════════════════════════════
# 1. 필드별 항목명(동의어) — 30건 기준 자료에서 확인된 표현이 출발점
# ══════════════════════════════════════════════════════════════════

FIELD_ALIASES: dict[str, list[str]] = {
    "사업 개요": ["사업개요", "사업 개요", "용역개요", "용역 개요", "사업의 개요",
                "추진개요", "과업개요", "제안개요", "구축개요", "사업명", "과업명", "용역명"],
    "사업분야": ["사업분야", "사업 분야", "용역분야", "과업분야"],
    "공고일": ["공고일", "공고일자", "입찰공고일", "입찰공고일자", "공고기간",
             "공고 및 접수기간", "게시일", "공고게시일"],
    "사업기간": ["사업기간", "용역기간", "과업기간", "계약기간", "수행기간",
               "이행기간", "개발기간", "사업 기간", "기간"],
    "예산": ["사업예산", "소요예산", "사업비", "과업비", "사업금액", "용역금액", "예산액",
           "총사업비", "추정가격", "기초금액", "설계금액", "집행 한도액", "집행한도액",
           "배정예산", "예산", "계약금액", "입찰금액", "총 사업비"],
    "참가 자격(면허·실적)": ["입찰참가자격", "입찰 참가자격", "참가자격", "응찰자격",
                      "제안자격", "참여자격", "입찰참가 자격", "자격요건",
                      "입찰참가자격 요건", "제안(입찰)참가자격",
                      "입찰 참가 자격 조건", "입찰 참가자격 및 입찰방식",
                      "입찰 자격 요건"],
    "지역제한": ["지역제한", "지역 제한", "소재지 제한", "지역의무", "지역업체",
              "본사 소재지", "지역 요건",
              # 실제 문서는 '법인등기부상 본점이 …에 있는 업체' 처럼 쓴다
              "법인등기부상 본점", "본점 소재지", "주된 영업소", "관내 업체",
              "지역 의무공동도급"],
    "컨소시엄 요건": ["컨소시엄", "공동수급", "공동이행", "분담이행", "공동계약",
                 "공동도급", "단독참여", "컨소시엄 구성"],
    "평가 배점": ["평가배점", "배점한도", "평가비율", "평가 비율",
               "전체평가비율", "전체 평가비율", "평가방식",
               "제안서 평가항목 및 배점", "제안서 평가항목 및 배점한도",
               "평가항목 및 배점", "평가항목 및 배점한도",
               "평가항목 및 비율", "평가기준 및 방법", "배점기준",
               "종합평가점수", "기술·가격평가 비중", "기술ㆍ가격평가 비중",
               "제안서 평가", "제안서평가", "기술평가",
               "제안서 평가 방법",
               "기술성 평가", "평가기준", "심사표", "평가항목", "제안평가",
               "평가방법", "정성적 평가", "정량적 평가"],
    # 입찰·낙찰·계약 방식은 제안서를 '어떻게 제출하는가'와 다른 필드다.
    "제출 방식": ["제출방법", "제출 방법", "제안서 제출", "제안서 제출방법",
               "제출방식", "접수방법", "제출처", "제출장소", "제출기한 및 방법",
               "제출방법 및 유의사항", "제안서 접수"],
    "필수 제출 서류": ["제출서류", "제출 서류", "구비서류", "첨부서류", "제출문서",
                 "제출서류 명세", "필수 제출서류", "제안서 제출서류",
                 "제출서류 및 자료", "제출서류목록", "붙임서류",
                 "제출할 도서의 품목, 매체, 수량",
                 # 목록을 '제출자료'·'제출 목록'으로 부르는 문서가 있다
                 "제출자료", "제출 자료", "제출목록", "구비 서류"],
    "과업 범위": ["과업범위", "과업의 범위", "사업범위", "용역범위", "주요 과업내용",
               "과업의 범위 및 내용",
               "주요 사업내용", "주요 사업범위", "사업 주요내용", "추진목표", "구축범위",
               "사업 추진 방향 및 규모", "사업 최종목표 시스템 및 사업 범위",
               "과업내용", "사업내용", "용역내용", "과업 및 범위",
               "제안요청내용", "추진내용"],
}
# 공식 코퍼스의 제목/표 왼쪽 칸을 별칭 검색보다 먼저 훑어 확인한 진짜 신규
# 표현이다. 아래 표현을 FIELD_ALIASES에 채택한 뒤에도 신규 표현 검토 파일에서
# 발견 이력을 재현할 수 있도록 직전 기준선에서만 제외한다.
SOURCE_REVIEW_ADOPTED_ALIASES = {
    "사업 개요": frozenset({"과업명", "용역명"}),
    "제출 방식": frozenset({"제안서 접수"}),
    "필수 제출 서류": frozenset({"입찰 등록 서류"}),
    "과업 범위": frozenset({"구축범위"}),
}
# 실제 추출에는 입찰 등록 서류도 즉시 사용한다.
FIELD_ALIASES["필수 제출 서류"].append("입찰 등록 서류")
# 규칙으로 확정한 표현 집합(새 표현 판별의 기준선)
BASELINE_EXPRESSIONS = {f: frozenset(nfc(a) for a in v) for f, v in FIELD_ALIASES.items()}


def expression_key(text: str) -> str:
    """공백·자간·문장부호를 제거한 표현 비교 키."""
    return re.sub(r"[^0-9A-Za-z가-힣]", "", nfc(text)).casefold()


BASELINE_EXPRESSION_KEYS = {
    f: frozenset(expression_key(a) for a in values)
    for f, values in FIELD_ALIASES.items()
}
DISCOVERY_BASELINE_EXPRESSION_KEYS = {
    f: frozenset(k for k in BASELINE_EXPRESSION_KEYS[f]
                 if k not in {expression_key(a) for a in SOURCE_REVIEW_ADOPTED_ALIASES.get(f, ())})
    for f in FIELDS
}
# 실제 신규 표현은 공백·자간·문장부호를 제거한 키로도 기존 목록에 없었던
# 세 표현뿐이다. 기존 동의어의 표기 변형은 발견 수에 다시 포함하지 않는다.
DISCOVERED_EXPRESSION_KEYS = {
    field: frozenset(expression_key(a) for a in aliases)
    for field, aliases in SOURCE_REVIEW_ADOPTED_ALIASES.items()
}

ALIAS_RE = {f: _pat(*sorted(set(v), key=len, reverse=True)) for f, v in FIELD_ALIASES.items()}


# ══════════════════════════════════════════════════════════════════
# 2. 값으로 쓰면 안 되는 문맥
# ══════════════════════════════════════════════════════════════════

# 제안업체 자신에 관한 서식 — 발주 사업의 값이 아니다
PROPOSER_CONTEXT = _pat(
    "일반현황", "회사개요", "업체현황", "제안업체", "참여업체", "수급인", "신청업체",
    "회사 일반현황", "업체 일반현황", "소프트웨어사업자", "참여인력", "전문분야",
    "실적증명", "사업실적", "유사용역", "경력사항",
    # 제안사가 자기 과거 실적을 적는 표. 발주 사업의 값이 아니다.
    # '연번'·'지분율'은 평가표 등 일반 표에도 흔해서 넣지 않는다(표 머리글 판정으로 거른다).
    "수행실적", "획득실적", "이행완료", "수급형태", "증명서용도", "실적 증명용")

# 실제 값이 아니라 다른 문서를 보라는 안내
# 실제 값이 아니라 다른 문서를 보라는 안내. '참조'와 '참고'를 함께 본다.
REFER_ONLY = re.compile(
    r"((입찰\s*)?공고(문|서)?[\s\"'’”」】]*(?:\s*(?:에|에서))?\s*(참조|참고|따름|의함|"
    r"정한\s*바(?:에\s*따름)?)|공고\s*(참조|참고|따름|의함)|별도\s*공고|"
    r"입찰\s*공고[^\n]{0,24}(?:참조|참고|따름|의함)|"
    r"(붙임|별첨|별지)\s*(?:자료\s*)?(참조|참고|에\s*의함)|"
    r"입찰참가신청서[^\n]{0,40}붙임\s*자료에\s*의함|"
    r"추후\s*(공지|통보|안내)|해당\s*없음\s*\(?참조|"
    r"본\s*제안요청서[^\n]{0,80}(?:참조|참고))")

# ``external_reference``는 단순히 '참고'라는 낱말이 있는 경우가 아니라,
# 현재 문서 대신 확인해야 할 자료 이름과 참조 동작이 함께 명시된 경우에만 쓴다.
EXTERNAL_REFERENCE_TARGET = re.compile(
    r"(?:입찰\s*)?공고(?:문|서)?|"
    r"붙임(?:\s*제?\s*\d+\s*호)?|"
    r"별첨(?:\s*제?\s*\d+\s*호)?|"
    r"별지(?:\s*제?\s*\d+\s*호)?")
EXTERNAL_REFERENCE_ACTION = re.compile(
    r"(?:참조|참고|확인|따름|의함|정한\s*바(?:에\s*따름)?)")


def external_reference_target(text: str) -> str:
    """명시적으로 확인하라고 한 외부 자료명을 돌려준다."""
    source = nfc(text or "")
    for match in EXTERNAL_REFERENCE_TARGET.finditer(source):
        tail = source[match.end():match.end() + 60]
        if EXTERNAL_REFERENCE_ACTION.search(tail):
            return re.sub(r"\s+", " ", match.group(0)).strip()
    return ""

# 문서가 그 필드의 값 자체를 밝히지 않겠다고 한 경우.
# "포함되지 않"·"미포함" 같은 일반 문구는 본문 아무 데나 나오므로 단독으로 쓰지 않는다.
# 반드시 대상어(예산·금액·배점 등)와 붙어 있을 때만 비공개로 본다.
_ND_SUBJECT = r"(사업\s*예산|사업\s*비|예산|사업\s*금액|용역\s*금액|금액|가격|배점|평가\s*배점|추정\s*가격|기초\s*금액)"
_ND_VERB = r"(비\s*공개|미\s*공개|공개하지\s*않|공개되지\s*않|공개\s*불가|명시하지\s*않|" \
           r"기재하지\s*않|포함되지\s*않|포함하지\s*않|미\s*포함|밝히지\s*않)"
NOT_DISCLOSED = re.compile(
    _ND_SUBJECT + r"[^\n]{0,12}(은|는|이|가|을|를)?\s*[^\n]{0,12}" + _ND_VERB)

# 제안사가 채울 빈칸 표시
BLANK_MARK = re.compile(r"^[\s\|:：·\-–—_○◯〇\*\.]*$")


def looks_blank(text: str) -> bool:
    """값이 실제로 비어 있는가(빈 서식 칸·기호만 있는 경우 포함)."""
    return not text or bool(BLANK_MARK.match(text.strip()))


# ══════════════════════════════════════════════════════════════════
# 3. 예산 값 해석 — 지어내지 않는 것이 핵심
# ══════════════════════════════════════════════════════════════════

# 금액 표기: 1,234,000원 / 49,500천원 / 3억원 / 5.5억
AMOUNT_RE = re.compile(
    r"(?P<num>\d[\d,\.]*)\s*(?P<unit>백\s*만\s*원|천\s*원|만\s*원|억\s*원|억|원)")
VAT_RE = _pat("부가세 포함", "부가가치세 포함", "부가세 별도", "부가가치세 별도",
              "VAT 포함", "VAT 별도", "부가세포함", "부가세별도")
LIMIT_RE = _pat("이내", "이하", "상한", "한도", "예정가격", "미만", "초과 불가")

UNIT_FACTOR = {"원": 1, "천원": 1_000, "만원": 10_000, "백만원": 1_000_000,
               "억원": 100_000_000, "억": 100_000_000}


def parse_amounts(text: str) -> list[dict]:
    """문장에서 금액 표기를 뽑는다.

    **명시된 단위만** 원 단위로 바꾼다(`49,500천원` → 49,500,000원).
    부가세를 더하거나 빼지 않고, `이내` 를 확정 금액으로 바꾸지 않으며,
    구성 금액만 있을 때 합계를 만들어 넣지 않는다."""
    out = []
    for m in AMOUNT_RE.finditer(text or ""):
        raw_num, unit = m.group("num"), re.sub(r"\s+", "", m.group("unit"))
        krw = None
        digits = raw_num.replace(",", "")
        if digits.count(".") <= 1 and digits.replace(".", "").isdigit():
            try:
                val = float(digits)
                exact = val * UNIT_FACTOR[unit]
                # 소수점이 남으면 정수로 단정하지 않는다(원문만 보존)
                if abs(exact - round(exact)) < 1e-9:
                    krw = int(round(exact))
            except (ValueError, OverflowError):
                krw = None
        out.append({"raw": m.group(0), "unit": unit, "krw": krw})
    return out


def budget_value(text: str, label: str = "") -> dict:
    """예산 필드의 값 구조. 문서에 없는 정보는 절대 채우지 않는다."""
    amounts = parse_amounts(text)
    vat = VAT_RE.search(text or "")
    limit = LIMIT_RE.search(text or "")
    return {
        "item_label": nfc(label).strip(),          # 실제 항목명 (사업예산 / 집행 한도액 …)
        "amount_raw": nfc(text).strip(),           # 원문 표현 그대로
        "amounts": amounts,                        # 구성 금액들 (합계는 만들지 않는다)
        "krw": (amounts[0]["krw"] if len(amounts) == 1 else None),
        "vat_text": vat.group(0) if vat else "",   # 부가세 표현 원문
        "limit_text": limit.group(0) if limit else "",   # 이내·상한 등 제한 표현
    }


# ══════════════════════════════════════════════════════════════════
# 4. 개인정보 가림 — 결과물에 실명·연락처를 남기지 않는다
# ══════════════════════════════════════════════════════════════════

PII_RULES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9\-]+(?:\.[A-Za-z0-9\-]+)+"), "[이메일 가림]"),
    (re.compile(r"(?<![\d\-])0\d{1,2}\s*[\-\)]\s*\d{3,4}\s*[\-]\s*\d{4}(?![\d\-])"), "[전화번호 가림]"),
    (re.compile(r"(?<![\d\-])01\d\s*[\-]\s*\d{3,4}\s*[\-]\s*\d{4}(?![\d\-])"), "[전화번호 가림]"),
    (re.compile(r"(?<![Ll])(?<![\d\-])1[5-9]\d{2}\s*[\-]\s*\d{4}(?![\d\-])"), "[전화번호 가림]"),
    (re.compile(r"(?<!\d)\d{6}\s*-\s*[1-4]\d{6}(?!\d)"), "[주민등록번호 가림]"),
    # 담당자 라벨 뒤의 사람 이름
    (re.compile(r"(?P<lab>(?:담당자|책임자|문의처|성명|담당관|실무담당)\s*[:：]\s*)"
                r"(?P<val>[가-힣]{2,4})(?![가-힣])"), r"\g<lab>[이름 가림]"),
    # 직위가 붙은 이름
    (re.compile(r"(?<![가-힣])(?P<val>[가-힣]{2,4})\s+(?P<tail>주무관|사무관|과장|팀장|대리|"
                r"부장|차장|실장|주임|연구원|담당관|사원|본부장|센터장|소장|원장|국장|처장|이사)"),
     r"[이름 가림] \g<tail>"),
]


def redact(text: str) -> str:
    """발췌문에서 개인정보 후보를 가린다(기관명은 그대로 둔다)."""
    out = nfc(text)
    for pat, repl in PII_RULES:
        if "이름" in repl:
            # 기관명은 사람 이름이 아니므로 가리지 않는다
            out = pat.sub(lambda m: m.group(0) if _is_org(m.group(0)) else
                          m.expand(repl), out)
        else:
            out = pat.sub(repl, out)
    return out


# 기관명 목록. 값을 여기에 적어 넣지 않고 공식 등록부 파일명에서 뽑아 채운다.
# '기초과학연구원', '광주연구원' 처럼 기관명 끝의 '연구원·원장·소장'이
# 직위로 보여 사람 이름으로 잘못 걸리는 것을 막는다.
_ORG_NAMES: frozenset = frozenset()


def set_org_allowlist(names) -> int:
    """기관명 목록을 정한다. 반드시 실제 자료(등록부 파일명)에서 뽑아 넘길 것."""
    global _ORG_NAMES
    _ORG_NAMES = frozenset(n for n in (nfc(x).strip() for x in names) if len(n) >= 2)
    return len(_ORG_NAMES)


def _is_org(fragment: str) -> bool:
    """이 조각이 기관명인가(사람 이름이 아닌가).

    '기초과학연구원'처럼 기관명 안에 들어 있는 경우와,
    '광주연구원장'처럼 기관명 뒤에 직위가 붙은 경우를 모두 기관으로 본다.
    """
    return any(fragment in org or org in fragment for org in _ORG_NAMES)


def pii_hits(text: str) -> int:
    """남아 있는 개인정보 후보 수(값은 돌려주지 않는다)."""
    total = 0
    for pat, label in PII_RULES:
        for m in pat.finditer(text or ""):
            if m.groupdict().get("val") == "가림":
                continue          # 이미 [이름 가림] 처리된 표식
            if "이름" in label and _is_org(m.group(0)):
                continue          # 기관명은 개인정보가 아니다
            total += 1
    return total
