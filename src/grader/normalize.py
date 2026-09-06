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
# ★한국어 조판 따옴표(‘ ’ “ ”)와 화살괄호(《 》 〈 〉)도 문장부호다. 곧은 따옴표만
#   지우면 원문 그대로 옮긴 답변이 '‘경영정보시스템’의' 처럼 남아 매칭에서 빠진다.
# ★가운뎃점 계열은 문서마다 다른 코드포인트를 쓴다 — 같은 뜻인데 정규화 결과가
#   갈리면 원문을 그대로 인용한 답변이 오답이 된다.
#   실측: '시․공간'(U+2024) → '시.공간' / '시·공간'(U+00B7) → '시 공간' 으로 어긋났다.
# 문서마다 다른 코드포인트를 쓰는 가운뎃점 계열 — 뜻은 같다.
_MIDDOT = re.compile(r"[․‧‥∙⋅•・]")

_PUNCT = re.compile(r"[·・․‧‥∙⋅•,()\[\]{}<>「」『』《》〈〉‘’“”\"'`~!?:;／/\\|+*^=_-]")

# 로마숫자 유니코드 문자 → 아라비아 (NFKC 가 "Ⅳ"→"IV" 로 분해하기 전에 바꾼다).
# "Ⅳ. 별지서식" 같은 절 제목이 "4장" 과 안 맞는 문제 방지 (회귀 테스트 대상).
_ROMAN = {chr(0x2160 + i): str(i + 1) for i in range(12)}          # Ⅰ..Ⅻ (대문자)
_ROMAN.update({chr(0x2170 + i): str(i + 1) for i in range(12)})    # ⅰ..ⅻ (소문자)
# 뜻이 확실히 같은 단위 표기만 통일(팀장 2-8). 앞부분만 같다고 인정하는 방식은 쓰지 않는다.
_UNIT_ALIAS = [(re.compile(r"\s*퍼센트|\s*프로|\s*percent", re.I), "%")]
# 짧은 정답 뒤에 붙는 한국어 종결 조사/서술 어미 (문장 끝에서만 뗀다).
_JOSA_TAIL = re.compile(r"(은|는|이|가|을|를|과|와|의|에|에서|으로|로|입니다|이다|임|다|요)$")

# ══════════════════════════════════ 응답 유형 분류 (채점기 피드백 2026-09-03) ═══════
# ★규칙 기반 패턴 매칭이다 — 진짜 의미 이해(LLM)가 아니다. "이 문장이 되묻기/거절/
#   부재응답처럼 보이는가"를 판정하는 보조 신호일 뿐, response.abstained(bool, 팀장
#   2-4 확정)를 대체하거나 텍스트로 abstained 값을 추론하는 데 쓰지 않는다.
#   용도: (a) 문항이 원래 모호해서(unspecified_type) 되묻기가 정답인 경우를
#         '불필요한 거절'로 잘못 세지 않게 grade_abstention 에서 참고,
#         (b) grade_short_answer 에서 gold 자체가 되묻는 문장일 때 표현이 달라도
#         인정하는 데 참고. 새 필드를 요구하지 않는다 — 기존 response.answer 텍스트만 본다.
_CLARIFICATION_MARKERS = re.compile(
    r"(어떤\s*[가-힣]{0,6}(사업|공고|문서|건)|말씀하시는|말씀하신|알려주시겠|알려주세요|"
    r"알려주시면|알려주실|특정할\s*수\s*없|여러\s*(건|개)[의]?\s*(사업|공고)?|"
    r"어느\s*(사업|공고|문서)|사업명을|무엇을\s*찾으시는|정확히\s*어떤)")
_NOT_FOUND_MARKERS = re.compile(
    r"(확인(할|이|되지)\s*(수\s*없|어렵|않)|찾을\s*수\s*없|(존재하지|나와\s*있지)\s*않|"
    r"자료(가|만으로는)\s*(없|부족)|정보가\s*없|명시(되어|돼)?\s*있지\s*않|기재(되어|돼)?\s*있지\s*않)")
_REFUSAL_MARKERS = re.compile(
    r"(답변\s*(을)?\s*(드리기|하기)?\s*(가)?\s*(곤란|어렵|불가)|말씀드리기\s*(곤란|어렵)|"
    r"제공(해\s*드릴|할)\s*수\s*없|알려\s*드릴\s*수\s*없|권한이\s*없)")


# ══════════════════════════ 자연어 '없음' 판정 (확정 정책 2026-09-04) ══════════════
# ★"문서에 그 항목이 없다"고 **단정한** 답과 "확인하지 못했다"는 답은 다른 주장이다.
#   전자는 field_absent 정답과 같은 뜻이고, 후자는 모른다는 뜻이다. 둘을 합치면
#   "확인할 수 없음"이라고만 해도 없음 문항에서 만점이 나온다.
#   그래서 확인 실패 신호를 **먼저** 보고, 그것이 없을 때만 부재 단정을 인정한다 —
#   "없다고 확인할 수 없습니다" 처럼 부정이 겹친 문장이 그 순서 때문에 걸러진다.
ABSENCE_CONFIRMED = "ABSENCE_CONFIRMED"     # 항목이 없다고 단정
ABSENCE_NOT_VERIFIED = "NOT_VERIFIED"       # 확인·판단 실패(모름)
ABSENCE_OTHER = "OTHER"                     # 부재 진술이 아님

# ① 확인 실패·불확실·부재의 부정(뒤집기) — **원문 전체**에서 가장 먼저 검사한다.
_NOT_VERIFIED_MARKERS = re.compile(
    r"(확인(할|이|하기|은|도|되지|이\s*되지)?\s*(수\s*없|어렵|어려[움우]|불가|않|안\s*됨|안\s*되)|"
    r"찾(지\s*못|을\s*수\s*없|아볼\s*수\s*없)|알\s*수\s*없|알지\s*못|"
    r"자료(가|는|만으로는|로는)?\s*(부족|없어|불충분)|정보(가|는)?\s*(부족|불충분)|"
    r"판단(할|이|하기)?\s*(수\s*없|어렵|어려[움우]|불가)|답(변)?(을)?\s*(할|드릴)\s*수\s*없|"
    r"불러오지\s*못|읽지\s*못|생성하지\s*못|접근(할|이)?\s*(수\s*없|불가)|오류|불확실|"
    # 검색·응답·결과의 부재는 '항목 부재'가 아니다
    r"(검색|조회)\s*결과(가|는)?\s*없|응답(이|은)?\s*없|결과(가|는)?\s*없|"
    # 부재 주장을 부정·유보하는 표현 — '없다'가 들어 있어도 부재 단정이 아니다
    r"없(지는|지도|진)\s*않|없는\s*(것|건)(은|이)?\s*아니|없다는\s*(의미|뜻|말)(은|는|이)?\s*아니|"
    r"없(다고|는지|음을|음으로|다고는|다\s*고)\s*(단정|확인|판단|보기|말하기|볼|할|말할|단정할|확인할|판단할)(할|이|하기|은|는|도)?\s*(수(는|도)?\s*)?(없|어렵|어려[움우]|불가)|"
    r"없을\s*수(도|는)?\s*있|없을지(도)?\s*모)")
# ② 문서 항목 부재의 **단정** — 좁게. 단순히 '없'이 들어 있다고 확정하지 않는다.
_ABSENCE_CONFIRMED_MARKERS = re.compile(
    r"(^없음$|^없다$|^없습니다$|^해당\s*(사항\s*)?없음$|^해당\s*없음$|"
    # '<항목명> 없음' 꼴 — 항목명이 검색·응답·결과·답변이면 제외(위 ①에서 이미 걸러짐)
    r"^[가-힣A-Za-z0-9·/():\-\s]{1,30}?(제한|조건|요건|항목|규정|사항|기준|서류|정보|배점|예산|기간|방식)\s*(이|은|는|도|:)?\s*없(음|다|습니다)$|"
    r"(명시|기재|언급|규정|적혀|나와|포함|제시|기술|작성)(되어|돼|되어\s*)?\s*있지\s*않|미기재|기재\s*(되지|돼\s*있지)\s*않|"
    r"(원문|문서|공고문|본문)에\s*(는\s*)?(해당\s*|관련\s*|그\s*)?(항목|내용|조건|기재|언급)(이|은|가|도)?\s*(자체가\s*)?(없|존재하지\s*않)|"
    r"(해당\s*)?(항목|내용)\s*자체(가|는)?\s*(없|존재하지\s*않)|"
    r"없다고\s*(확인|판단|명시|기재|답)(했|하였|됨|됩니다|합니다|드립니다)|"
    r"(존재하지|있지)\s*않(음|습니다|다|는다)$)")
# ③ 좁은 예외 — "원문에 해당 항목 자체가 없다" 본문 + 괄호는 '현실의 값·조건이 없다고 단정할 수
#    없다'는 해석 유보일 때만, 그 괄호를 부정 신호로 세지 않는다. 검색·확인 실패 괄호는 예외 아님.
_FIELD_ABSENT_MAIN = re.compile(r"(원문|문서|공고문|본문)에\s*(해당\s*)?(항목|내용)\s*자체(가|는)?\s*(없|존재하지\s*않)")
_INTERPRETIVE_CAVEAT = re.compile(
    r"^(실제로?\s*|현실(에서|의)?\s*)?(값|제한|조건|요건|정책)(이|은|가)?\s*없다는\s*(뜻|의미)(으로|은|는|이)?\s*(단정|확인|판단|해석)(할|하기|은)?\s*(수\s*)?(없|어렵)")


def classify_absence(text) -> str:
    """답변이 '항목이 없다'는 **단정**인지, '확인 실패·불확실·부정'인지, 둘 다 아닌지.

    ★순서가 정책이다. 원문 **전체**(괄호 포함)에서 확인 실패·불확실·부재의 부정을
      먼저 찾고, 그것이 없을 때만 좁은 부재 단정 패턴을 본다. 괄호를 먼저 지우고
      바깥의 '없음'으로 확정하면 "없음(확인할 수 없음)"이 부재로 둔갑한다(실측).
    ★유일한 예외: 본문이 "원문에 해당 항목 자체가 없다"를 명시하고 괄호가 '현실의
      값·조건이 없다고 단정할 수 없다'는 해석 유보일 때 — 그 괄호만 부정 신호에서 뺀다.
      검색 실패·확인 실패를 말하는 괄호는 예외가 아니다.
    """
    t = normalize_text(str(text or ""), drop_punct=False)
    t = re.sub(r"[.。!?\s]+$", "", t).strip()
    if not t:
        return ABSENCE_OTHER
    parens = re.findall(r"\(([^)]*)\)", t)
    main = re.sub(r"\([^)]*\)", " ", t).strip()
    scan = t
    if parens and _FIELD_ABSENT_MAIN.search(main) and all(
            _INTERPRETIVE_CAVEAT.search(pc.strip()) for pc in parens):
        scan = main                              # 해석 유보 괄호만 제외(좁은 예외)
    if _NOT_VERIFIED_MARKERS.search(scan):
        return ABSENCE_NOT_VERIFIED
    if _ABSENCE_CONFIRMED_MARKERS.search(scan):
        return ABSENCE_CONFIRMED
    return ABSENCE_OTHER


# ---------------------------------------------------------------------------
# field_absent 전용 판정 — "문서에 그 항목이 안 적혀 있다" ≠ "실제로 그런 조건이 없다"
# ---------------------------------------------------------------------------
# 추출표 status=field_absent 는 **문서에 항목이 기재돼 있지 않다**는 뜻뿐이다.
# 거기서 "그러니 지역 제한이 없다 / 공동수급이 허용된다 / 예산이 0원이다"까지
# 나아간 답은 원문에 없는 사실을 새로 만든 것이므로 정답이 아니다.
FIELD_ABSENT_STATED = "FIELD_ABSENT_STATED"       # 문서에 항목이 없다고 분명히 말함
FIELD_ABSENT_INFERRED = "FIELD_ABSENT_INFERRED"   # 항목 부재에서 실제 조건·금액까지 단정
FIELD_ABSENT_NOT_VERIFIED = "FIELD_ABSENT_NOT_VERIFIED"   # 확인·검색 실패
FIELD_ABSENT_OTHER = "FIELD_ABSENT_OTHER"         # 위 어느 것도 아님(맨 '없음' 포함)

# ① 문서·원문을 가리키는 말 — 주어가 '문서'여야 항목 부재 진술이다.
_DOC_SUBJECT = r"(?:원문|문서|공고문|공고|본문|제안요청서|자료|해당\s*문서|이\s*문서)"
# ② '적혀 있지 않다'는 서술
_NOT_WRITTEN = (r"(?:명시|기재|기술|언급|규정|표기|적혀|적시|나와|나타나|포함|제시|안내)"
                r"(?:되어|돼|되어\s*)?\s*있지\s*않|미기재|미표기|"
                r"(?:명시|기재|기술|언급|규정|표기|제시)\s*(?:되지|돼\s*있지|하지)\s*않|"
                r"(?:적혀|나와|들어)\s*있지\s*않|(?:항목|내용|조건|규정|기재|언급)"
                r"(?:\s*자체)?(?:이|가|은|는|도)?\s*(?:없|존재하지\s*않|보이지\s*않)")
_FIELD_ABSENT_STATED_RE = re.compile(
    rf"{_DOC_SUBJECT}[^.]{{0,40}}?(?:{_NOT_WRITTEN})"
    rf"|(?:{_NOT_WRITTEN})[^.]{{0,20}}?{_DOC_SUBJECT}"
    rf"|(?:해당\s*)?(?:항목|필드)(?:이|가|은|는)?\s*미기재"
    rf"|(?:항목|내용)\s*자체(?:가|는)?\s*(?:없|존재하지\s*않)")
# ③ 항목 부재 → 실제 제도·조건·금액 단정으로 **넘어간** 표현
_INFERENCE_BRIDGE = r"(?:으므로|이므로|므로|따라서|그래서|때문에|덕분에|이니|니까|기에|이라서|라서|해서|고\s*보면)"
_REAL_WORLD_CLAIM = (r"(?:실제(?:로)?\s*[^.]{0,20}?없|제한(?:이|은)?\s*없|조건(?:이|은)?\s*없|"
                     r"요건(?:이|은)?\s*없|허용|가능합니다|가능하다|참여\s*가능|"
                     r"전국|누구(?:나|든)|0\s*원|영\s*원|없는\s*것으로\s*볼|없다고\s*보|"
                     r"제한\s*없음|존재하지\s*않습니다|존재하지\s*않는다)")
_FIELD_ABSENT_INFERRED_RE = re.compile(
    rf"(?:{_NOT_WRITTEN}|미기재|없)[^.]{{0,25}}?{_INFERENCE_BRIDGE}[^.]{{0,40}}?{_REAL_WORLD_CLAIM}")
# ④ 문서 언급 없이 실체가 없다고만 말한 답 — 'X 없음', '제출 서류가 존재하지 않습니다'
_BARE_ABSENCE_RE = re.compile(
    r"^[^.]{0,30}?(?:이|가|은|는|도)?\s*(?:없음|없습니다|없다|없어요|존재하지\s*않(?:습니다|는다|음|다))\s*$")


def classify_field_absent_answer(text) -> str:
    """field_absent 문항의 답변 성격. 확인 실패 → 부재 추론 → 문서 항목 부재 순.

    순서가 정책이다.
      1) 확인·검색 실패는 무엇보다 먼저 걸러낸다(기존 classify_absence 와 같은 신호).
      2) '항목이 없으니 실제로도 없다/허용된다/0원이다' 는 추론이므로 오답이다.
         — 문서 항목 부재를 정확히 말한 앞부분이 있어도 결론이 원문 밖이면 오답이다.
      3) 그 다음에야 '문서에 항목이 적혀 있지 않다'는 진술을 인정한다.
    """
    t = normalize_text(str(text or ""), drop_punct=False)
    t = re.sub(r"[.。!?\s]+$", "", t).strip()
    if not t:
        return FIELD_ABSENT_OTHER
    if classify_absence(t) == ABSENCE_NOT_VERIFIED:
        return FIELD_ABSENT_NOT_VERIFIED
    if _FIELD_ABSENT_INFERRED_RE.search(t):
        return FIELD_ABSENT_INFERRED
    if _FIELD_ABSENT_STATED_RE.search(t):
        return FIELD_ABSENT_STATED
    if _BARE_ABSENCE_RE.search(t):
        return FIELD_ABSENT_OTHER            # '없음' · '지역 제한 없음' 은 부족하다
    return FIELD_ABSENT_OTHER


def classify_response_kind(text) -> str:
    """response.answer 텍스트가 어떤 성격인지 규칙 기반으로 분류한다.

    반환값: "CLARIFICATION" | "ABSTENTION" | "REFUSAL" | "ANSWER"
    ★"IRRELEVANT"(질문과 무관한 답)는 여기서 판정하지 않는다 — 이 함수는 답변
      텍스트만 보고, 질문과의 관련성 판정은 질문 의미 이해가 필요해 규칙 매칭으로
      안전하게 못 만든다(의미비교 모델 영역, 팀장 2-3 "분리해도 됨"과 동일 이유).
      필요하면 이 함수 반환값에 IRRELEVANT 를 별도로 얹을 수 있게 문자열 리터럴로
      열어둔다(호출부에서 우선순위만 조정하면 됨).
    ★순서 중요: CLARIFICATION → ABSTENTION(부재) → REFUSAL. "확인할 수 없다"류가
      "말씀하시는"류보다 뒤에 오면 되묻기 문장 속 부정 표현에 잘못 걸릴 수 있어
      되묻기를 먼저 본다.
    """
    t = normalize_text(str(text or ""), drop_punct=False)
    if not t:
        return "ABSTENTION"
    if _CLARIFICATION_MARKERS.search(t):
        return "CLARIFICATION"
    if _NOT_FOUND_MARKERS.search(t):
        return "ABSTENTION"
    if _REFUSAL_MARKERS.search(t):
        return "REFUSAL"
    return "ANSWER"


def is_document_not_found_response(text) -> bool:
    """'문서에서 확인할 수 없습니다' 류의 자연어 부재 응답인지."""
    return classify_response_kind(text) == "ABSTENTION"


def to_halfwidth(s: str) -> str:
    # 로마숫자는 NFKC 가 "Ⅳ"→"IV" 로 분해하기 전에 아라비아로 바꾼다.
    for r, a in _ROMAN.items():
        if r in s:
            s = s.replace(r, a)
    return unicodedata.normalize("NFKC", s)


def nfc(s: str) -> str:
    return unicodedata.normalize(UNICODE_FORM, s)


def normalize_text(s, drop_punct: bool = True, lower: bool = True) -> str:
    """공백·대소문자·전각/반각·문장부호·유니코드 정규화.
    ★뜻이 바뀌는 정규화는 하지 않는다 — '없음'/'제한 없음'/'N/A' 를 하나로 묶지 않는다(팀장 2-5)."""
    if s is None:
        return ""
    # ★가운뎃점 계열을 **정규화 앞단에서** 하나로 모은다. nfc/to_halfwidth 를 먼저
    #   태우면 U+2024(․)가 마침표로 바뀌어 버리는데, 마침표는 금액·날짜 때문에
    #   일부러 안 지운다. 그러면 '시․공간'과 '시·공간'이 서로 다른 값이 된다.
    s = _MIDDOT.sub("·", str(s))
    s = nfc(to_halfwidth(s)).strip()
    if lower:
        s = s.lower()
    for rx, rep in _UNIT_ALIAS:
        s = rx.sub(rep, s)
    if drop_punct:
        s = _PUNCT.sub(" ", s)
    s = _WS.sub(" ", s).strip()
    return ORG_ALIASES.get(s, s)


# ------------------------------------------------------------------ 금액

_HANGUL_DIGIT = {"영": 0, "일": 1, "이": 2, "삼": 3, "사": 4, "오": 5,
                 "육": 6, "칠": 7, "팔": 8, "구": 9}
_BIG = {"조": 10**12, "억": 10**8, "만": 10**4}
_SMALL = {"천": 1000, "백": 100, "십": 10}

# 금액처럼 보이는 구간. '5천만원', '500,000,000원', '5억', '오억원', '천원' 등
_AMOUNT_SPAN = re.compile(
    r"(?:[0-9][0-9,]*(?:\.[0-9]+)?|[영일이삼사오육칠팔구]+)"
    r"\s*(?:조|억|천만|천|백만|백|십만|십|만)*\s*원?")

# ★뜻을 바꾸는 조건 — 삭제하지 않고 따로 읽어 함께 비교한다(팀장 2-1).
#   팀장이 명시한 것만: 이내 / 이상 / 초과 / 미만 (+ 자연스러운 짝 이하). '한도·까지' 같은
#   유사 표현은 넣지 않는다 — 뜻이 확실히 같은 경우만 제한적으로.
_AMOUNT_CONDS = [
    (re.compile(r"이내"), "이내"),
    (re.compile(r"이상"), "이상"),
    (re.compile(r"이하"), "이하"),
    (re.compile(r"초과"), "초과"),
    (re.compile(r"미만"), "미만"),
    (re.compile(r"(?:부가\s*가?\s*치?\s*세|vat|세액|세금)\s*포함", re.I), "vat_included"),
    (re.compile(r"(?:부가\s*가?\s*치?\s*세|vat|세액|세금)\s*별도", re.I), "vat_excluded"),
]
# 뜻을 안 바꾸는 잡음 — 안전하게 제거
_AMOUNT_NOISE = re.compile(r"내외|정도|약|금\s*|원|\(.*?\)|\s")


def amount_conditions(s) -> frozenset[str]:
    """금액 문자열에서 뜻을 바꾸는 조건 토큰을 뽑는다 ('이내', 'vat_included' 등)."""
    t = to_halfwidth(str(s or "")).lower()
    out = set()
    for rx, tag in _AMOUNT_CONDS:
        if rx.search(t):
            out.add(tag)
    return frozenset(out)


def parse_amount(s) -> int | None:
    """한국어 금액 표기를 '원 단위 정수'로 정규화 (조건은 무시 — amount_conditions 로 별도).

    5억 / 500,000,000원 / 500백만원 / 오억 / 5천만원 / 3억5천만원 / 248,796천원 / 금 5억원(부가세 포함) → 정수
    실패하면 None (조용히 0을 반환하지 않는다 — 0은 유효한 값이다).
    """
    if s is None:
        return None
    t = to_halfwidth(str(s)).replace(",", "")
    t = re.sub(r"이상|이하|초과|미만|이내", "", t)
    t = _AMOUNT_NOISE.sub("", t).lower().replace("vat", "")
    for k, v in _HANGUL_DIGIT.items():
        t = t.replace(k, str(v))
    # '248,796천원' → '248796천': 누산 루프에서 천(_SMALL=1000)이 앞 숫자에 곱해진다.
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


_TIME_RE = re.compile(
    r"(?P<ampm>오전|오후|am|pm)?\s*(?P<h>\d{1,2})\s*[:시]\s*(?P<mi>\d{2})?", re.I)


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


def parse_time(s) -> str | None:
    """'16:00' / '16시' / '오후 4시' / '오전 4시 30분' → 'HH:MM'(24시간제). 날짜 문자열에
    시각이 섞여 있어도 뽑는다.
    ★정답에 시각이 있으면 시각까지 비교한다(팀장 2-7). 없으면 날짜만.
    ★'오전 4시'와 '오후 4시'는 12시간 차이 나는 다른 시각이다 — 오전/오후 표기를 무시하고
      시(時)만 뽑으면 둘을 같다고 오판한다(채점기 피드백 회귀). 12시간제 규칙(오전 12시=00:00,
      오후 12시=12:00)으로 24시간제로 환산한다."""
    if s is None:
        return None
    t = to_halfwidth(str(s))
    # 날짜의 '월/일' 숫자를 시각으로 오인하지 않도록, 날짜 구간을 지운 뒤 찾는다
    for rx in _DATE_PATTERNS:
        t = rx.sub(" ", t)
    m = _TIME_RE.search(t)
    if not m:
        return None
    h = int(m.group("h"))
    ampm = (m.group("ampm") or "").lower()
    if ampm in ("오후", "pm"):
        if h != 12:
            h += 12
    elif ampm in ("오전", "am"):
        if h == 12:
            h = 0
    return f"{h % 24:02d}:{int(m.group('mi') or 0):02d}"


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


def _residual(text: str, span: str) -> str:
    """값 구간을 지우고 남은 정규화 문자열."""
    rest = to_halfwidth(str(text)).replace(span, " ", 1)
    return normalize_text(rest)


def _residual_ok(text: str, span: str, limit: int = RESIDUAL_LIMIT) -> bool:
    return len(_residual(text, span)) <= limit


# 문장 맨 끝의 종결 부호. _PUNCT 는 마침표를 일부러 안 지운다(금액 "1.5억"·날짜
# "2024.05.20" 이 깨지므로) — 그래서 **맨 끝** 부호만 여기서 따로 뗀다.
# 이게 없으면 "계약일로부터 6개월입니다" 는 정답인데 "…6개월입니다." 는 오답이 된다.
_SENT_END = re.compile(r"[.。!?]+$")


def _strip_josa(s: str) -> str:
    """짧은 정답 끝에 붙은 한국어 조사/어미와 문장 종결 부호를 뗀다
    (팀장 2-8 — prefix 매치 아님).

    ★빈 문자열까지 깎지 않는다. 전부 조사로만 이루어진 짧은 값("이", "가", "임")은
      끝까지 깎으면 서로 다른 답이 모두 ""가 되어 일치해 버린다 —
      실제로 match_short("이", "가") 가 정답으로 처리됐다.
    """
    original = s
    prev = None
    while prev != s:
        prev = s
        stripped = _JOSA_TAIL.sub("", _SENT_END.sub("", s).strip()).strip()
        if not stripped:
            break            # 더 깎으면 빈 값이 된다 — 여기서 멈춘다
        s = stripped
    return s or original


# 추출 테이블 원문에 흔한 "번호/기호 + 항목명 + 콜론" 라벨 접두. 뒤에 오는 콜론까지 최대
# 20자 이내인, 좁게 잡은 패턴만 인정한다 — 아무 콜론이나 라벨로 오인해 값의 일부를
# 잘라내면 안 된다(채점기 피드백: "나. 사업기간 : 계약일로부터 6개월" vs "계약일로부터 6개월").
_LABEL_PREFIX = re.compile(r"^\s*(?:[가-힣]{1,3}|[0-9]{1,2}|[①-⑮ⅰ-ⅹ])[.\)]\s*[^:：\n]{1,20}[:：]\s*")


def strip_label_prefix(s: str) -> str | None:
    """'나. 사업기간 : 계약일로부터 6개월' → '계약일로부터 6개월'.
    라벨(번호+항목명+콜론)이 없으면 None — 원문을 훼손하지 않는다. 이 함수는 정답 쪽에서
    "라벨 생략도 정답으로 인정" 하기 위한 추가 후보를 만드는 용도로만 쓴다(치환이 아니라
    accept 후보 추가) — 답변 쪽 문자열은 건드리지 않는다."""
    m = _LABEL_PREFIX.match(str(s or ""))
    if m and m.end() < len(s):
        core = s[m.end():].strip()
        return core or None
    return None


# 목록 항목 앞의 순번 표시(①, 1., 가. 등) — 콜론 없이 그냥 항목 앞에 붙는다. strip_label_prefix
# 와 다른 패턴(라벨+콜론이 아니라 순서 번호만). 실제 모델(이태민 answer_pipeline.py
# _ENUM_PREFIX_RE, 2026-09-04)이 "원문에서 딸려온 문서 서식 흔적 — 값의 일부가 아니다"라고
# 이미 확정한 것과 동일 패턴을 채점기 쪽에도 반영 — 모델이 항목 1개는 번호를 지우고
# 다른 항목은 안 지운 채 낼 때(실 데이터에서 확인됨) 목록 항목 매칭이 깨지는 걸 막는다.
_ENUM_PREFIX = re.compile(r"^\s*(?:[가-힣]\s*[.)]|\(?\d{1,2}\s*[.)]|[①-⑳]|[ⅰ-ⅹ]\s*[.)])\s*")


def strip_enum_prefix(s: str) -> str:
    """'① 입찰참가신청서 1부' → '입찰참가신청서 1부'. 순번 표시가 없으면 원문 그대로."""
    t = str(s or "")
    m = _ENUM_PREFIX.match(t)
    return t[m.end():].strip() if m else t


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


# 외화 단위 — 정답이 원화일 때 같은 숫자에 이 단위가 붙으면 다른 값이다.
_FOREIGN_CURRENCY = re.compile(r"(달러|USD|US\$|\$|유로|EUR|€|엔화|JPY|¥|위안|CNY|파운드|GBP|£)", re.I)


def _amount_has_foreign_unit(text: str, amount: int) -> bool:
    """text 안에서 amount 와 같은 아라비아 숫자 구간에 **바로 붙은 단위**가 외화인가.

    ★창을 넓게 잡으면 "…493원($5,198,901…)" 처럼 원화 뒤에 괄호로 환율이 따라오는
      정상 표기가 외화로 오판된다(실측). 숫자 직후의 첫 단위 토큰만 본다.
    """
    for m in re.finditer(r"[0-9][0-9,]*(?:\.[0-9]+)?", text):
        digits = m.group(0).replace(",", "")
        try:
            if int(float(digits)) != amount:
                continue
        except ValueError:
            continue
        unit = re.match(r"\s*([가-힣A-Za-z]+|[$€¥£])", text[m.end():])
        if unit and _FOREIGN_CURRENCY.fullmatch(unit.group(1)):
            return True
        head = re.search(r"([$€¥£]|USD|US\$)\s*$", text[max(0, m.start() - 4):m.start()], re.I)
        if head:
            return True
    return False


_PLAUSIBLE_AMOUNT = re.compile(
    r"[0-9][0-9,]*(?:\.[0-9]+)?\s*(?:조|억|천만|천|백만|백|십만|십|만)*\s*(?:원|달러|\$|USD|유로|€|엔|¥|위안)")


def _sourced_amount_values(text: str) -> set:
    """근거 없는 덧붙임 검사에 쓸 '금액다운' 값만 뽑는다.

    ★extract_amounts 는 한글 숫자('사업'의 '사'=4)까지 잡는다. 그것을 '추가 금액'으로
      세면 "해당 사업의 예산은 …" 같은 정상 답이 오답이 된다. 통화 단위가 붙은
      아라비아 숫자만 금액으로 본다.
    """
    out = set()
    for m in _PLAUSIBLE_AMOUNT.finditer(text):
        v = parse_amount(m.group(0))
        if v is not None:
            out.add(v)
    return out


def match_short(gold, pred, accept=None, kind: str = "auto",
                allow_partial: bool = False,
                residual_limit: int = RESIDUAL_LIMIT) -> tuple[bool, str]:
    """단답 채점. (맞았는가, 판정 근거) 를 함께 돌려준다 — 문항별 결과에 판정 이유로 저장한다.

    ★ 문장 안에서 금액·날짜를 뽑아 비교하면 그 자체가 사실상 '부분 일치'가 되어,
      '정답 + 근거에 없는 내용 덧붙임' 답변이 만점을 받는다.
      ⇒ 값이 맞아도 **뒤에 남은 군더더기가 길면** 통과시키지 않는다(allow_partial=False 기준).
    """
    accept = accept or []
    golds = [g for g in ([gold] + list(accept)) if g is not None and str(g) != ""]
    pred_s = str(pred if pred is not None else "")

    # 0) 정확 일치 — 정답 자체가 서술형이어도 pred 가 정답 그대로면 통과.
    #    한국어 띄어쓰기(3-7)와 끝 조사(팀장 2-8)는 표기 변형이라 제거 후에도 대조한다.
    #    ★prefix 매치는 하지 않는다 — 조사를 뗀 '전체'가 같아야 한다.
    np = normalize_text(pred_s)
    np_forms = {np, np.replace(" ", ""), _strip_josa(np.replace(" ", ""))}
    for g in golds:
        ng = normalize_text(g)
        if not ng:
            continue
        if ng in np_forms or ng.replace(" ", "") in np_forms or _strip_josa(ng.replace(" ", "")) in np_forms:
            return True, "normalized_exact"

    # 1) 날짜 (+ 시각) — 정답에 시각이 있으면 시각까지 비교한다(팀장 2-7)
    g_date = parse_date(gold)
    if g_date:
        g_time = parse_time(gold)
        ds = _date_span(pred_s)
        if ds and ds[0] == g_date:
            p_time = parse_time(pred_s)
            if g_time and p_time != g_time:
                return False, f"date_ok_time_mismatch(정답 {g_time} vs 답변 {p_time})"
            span = ds[1] + (f" {p_time}" if p_time else "")
            if allow_partial or _residual_ok(pred_s, ds[1], residual_limit):
                return True, "datetime" if g_time else "date"
            return False, f"date_match_but_verbose(남은 부분 '{_residual(pred_s, ds[1])}')"

    # 2) 금액 — 숫자와 조건('이내','vat_included' 등)을 따로 읽어 함께 비교한다(팀장 2-1)
    g_amt = parse_amount(gold) if not g_date else None
    if g_amt is not None:
        g_conds = amount_conditions(gold)
        # ★[2026-09-04] 통화가 다르면 숫자가 같아도 다른 값이다 — "6,758,571,493달러"가
        #   원화 정답에 맞는다고 처리됐다(실측). 정답이 원화인데 답변의 그 숫자에
        #   외화 단위가 붙어 있으면 불일치.
        #   판정은 **그 숫자에 붙은 단위** 기준이다 — 허용 답 목록에 환율 설명($5,198,901)이
        #   있다고 해서 "6,758,571,493달러"까지 허용되면 안 된다.
        if _amount_has_foreign_unit(pred_s, g_amt) and not any(
                _amount_has_foreign_unit(str(g), g_amt) for g in golds):
            return False, "amount_currency_mismatch(정답은 원화인데 답변은 외화 단위)"
        # ★출처에 없는 **추가 금액**은 근거 없는 덧붙임이다. 허용 답(정답·accept 목록)
        #   어디에도 없는 금액이 답변에 더 있으면 핵심 값이 맞아도 오답 — "…이며 별도
        #   예산 3억원이 추가됩니다"가 정답 처리됐다(실측). 임의 숫자 전부를 허용하는
        #   넓은 예외를 두지 않는다: 연결된 추출표 값에 있는 숫자(환율 설명 등)만 허용.
        allowed_amts = set().union(*(_sourced_amount_values(str(g)) for g in golds)) | {g_amt}
        extra = sorted(_sourced_amount_values(pred_s) - allowed_amts)
        for val, span in extract_amounts(pred_s):
            if val != g_amt:
                continue
            p_conds = amount_conditions(pred_s)
            if p_conds != g_conds:
                return False, f"amount_ok_cond_mismatch(정답 {sorted(g_conds)} vs 답변 {sorted(p_conds)})"
            if not (allow_partial or _residual_ok(pred_s, span, residual_limit)):
                return False, f"amount_match_but_verbose(남은 부분 '{_residual(pred_s, span)}')"
            if extra:
                # 잔여가 짧아 통과할 뻔한 답이라도, 허용 답 어디에도 없는 금액을 더 말했다면
                # 근거 없는 덧붙임이다 — "…이며 별도 예산 3억원이 추가됩니다"(실측 1.0).
                return False, f"amount_extra_unsourced(허용 답에 없는 추가 금액 {extra})"
            return True, "amount"
        # 값 자체가 안 맞았거나 조건 불일치
        p_amts = [v for v, _ in extract_amounts(pred_s)]
        if p_amts:
            return False, f"amount_mismatch(정답 {g_amt} vs 답변 {p_amts})"

    # 3) 표현 차이 허용(채점기 피드백: 항목번호/제목 생략·조사·존댓말·어순·군더더기 삭제는
    #    의미가 같으면 정답) — gold 의 핵심 어절이 pred 에 빠짐없이 들어있는지만 본다.
    #    ★진짜 의미 이해(LLM)가 아니라 정규화된 어절 포함 관계다 — gold 에 있는 어절이
    #      pred 에서 하나라도 빠지면 오답으로 남는다(정보 누락 = 오답, 팀장 §6-3단계와 동일 원칙).
    #      날짜·금액은 전용 규칙(1·2단계)이 이미 최종 판정을 냈으므로 여기서 되살리지 않는다.
    #      너무 긴 gold(문단급, 60자 초과)는 체크포인트/요약형 채점 영역이라 여기서 안 다룬다.
    if g_date is None and g_amt is None:
        for g in golds:
            core = strip_label_prefix(g) or g
            if len(str(core)) > 60:
                continue
            gtoks = [w for w in normalize_text(core).split() if w]
            if not gtoks:
                continue
            # ★한국어 조사/서술어미는 앞 단어에 공백 없이 붙는다("6개월입니다") — 어절
            #   전체를 그대로 비교하면 어미 하나 때문에 놓친다. 어절마다 어미를 뗀 형태도
            #   같이 넣어서 비교한다.
            raw_ptoks = np.split()
            ptoks = set(raw_ptoks) | {_strip_josa(w) for w in raw_ptoks}
            if all(w in ptoks for w in gtoks):
                return True, "content_equivalent(핵심 어절 전부 포함, 표현만 다름)"

    # 4) 문자열 부분일치 (allow_partial 켰을 때만 — 환각 덧붙임도 통과시키는 걸 알고 쓰는 모드)
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
