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
# ⚠️ 2026-09-03 정정 — 괄호를 두 종류로 나눈다.
#
# 예전에는 여는 괄호 [（(\[「【] 와 닫는 괄호 [）)\]」】] 를 한 묶음으로 보고 그 사이를
# 통째로 지웠다. 그래서 **제목을 감싸는 인용 괄호**까지 내용째 사라졌다.
#   "(긴급)「2024년도 차세대 응급의료 상황관리시스템 구축」 위탁용역"
#     → 보조 키가 "위탁용역" 네 글자만 남음
# 이 짧은 키는 여러 사업에 나올 수 있는 일반 표현이라, 전혀 다른 이름의 질문까지
# 그 문서 하나로 확정시켰다(실제 재현).
#
#   ① 제목 인용 괄호 「」『』【】《》〈〉 — **껍데기만** 벗기고 안의 제목은 보존한다.
#      (공식 100문서에서 발주기관명에는 이 괄호가 한 건도 없다 — 사업 제목 전용 표기)
#   ② 절차·회차 주석 ( ) [ ] — "(긴급)", "(협상)(긴급)", "[협상에 의한 계약]", "(2차)"처럼
#      제목 앞뒤에 붙는 진짜 주석이므로 통째로 제거한다(기존 동작 유지).
_TITLE_QUOTE_CHARS = "「」『』【】《》〈〉"
_ANNOTATION_RE = re.compile(r"[（(\[［][^）)\]］]*[）)\]］]")


def strip_bracketed(text: str | None) -> str:
    """괄호 주석을 제거한다(원본 표시용 문자열은 바꾸지 않는다).

    제목을 감싸는 인용 괄호는 **내용을 보존**하고 껍데기만 벗긴다.
    """
    if not text:
        return ""
    s = unicodedata.normalize("NFKC", str(text))
    s = s.translate({ord(c): " " for c in _TITLE_QUOTE_CHARS})   # ① 제목 보존
    prev = None
    while prev != s:                       # ② 중첩·연속 주석까지 모두 제거
        prev = s
        s = _ANNOTATION_RE.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip()


# ---------------------------------------------------------------------------
# 일반 표현만으로 이루어진 보조 키 걸러내기
# ---------------------------------------------------------------------------
# "위탁용역"·"구축사업"·"기능개선"처럼 여러 문서에 나올 수 있는 표현만 남은 키는
# 문서를 한 건으로 특정하지 못한다. 아래는 **닫힌 목록**이며 유사도·임계값을 쓰지
# 않는다 — 키를 이 낱말들로만 전부 쪼갤 수 있으면 일반 표현으로 본다.
# ⚠️ 원래(공식 사업명 전체) 키에는 적용하지 않는다. 주석을 떼며 짧아진 **보조 키**에만 쓴다.
# ⚠️ 목록은 **조달 상투어**로만 좁힌다. "시스템·관리·통합·구축·개선" 같은 분야 낱말을
#    넣으면 "통합관리시스템 구축"처럼 실제로 문서를 특정하는 이름까지 일반 표현으로
#    잘못 판정한다(기존 회귀 테스트에서 실제로 걸렸다). 여기 있는 낱말만으로 키가
#    통째로 쪼개질 때만 일반 표현이다.
_GENERIC_PROJECT_WORDS = (
    # 복합 상투어(먼저 매칭되도록 길이순 정렬해 사용)
    "유지보수용역", "유지관리용역", "고도화사업", "위탁용역", "구축사업", "개선사업",
    "용역사업", "위탁운영",
    # 단일 상투어
    "재공고", "용역", "위탁", "수탁", "사업", "공사", "구매", "임차", "대여", "납품",
    "제안", "입찰", "공고", "계약", "선정", "긴급", "협상", "기타", "관련", "및", "등",
)
_GENERIC_SORTED = tuple(sorted(_GENERIC_PROJECT_WORDS, key=len, reverse=True))


def is_generic_project_key(key: str | None) -> bool:
    """키가 일반 표현 낱말만으로 전부 쪼개지는가(=사업을 특정하지 못하는가)."""
    if not key:
        return False
    rest = key
    while rest:
        for word in _GENERIC_SORTED:
            if rest.startswith(word):
                rest = rest[len(word):]
                break
        else:
            return False
    return True


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


# 회차 표기의 서수 접두 "제" — "제2차 사업"과 "2차 사업"은 같은 회차를 가리킨다.
# ⚠️ 숫자+차 **바로 앞**의 "제"만 접는다. "제주"·"제안"·"제2공장"은 건드리지 않는다.
_ORDINAL_PREFIX_RE = re.compile(r"제\s*(?=\d{1,2}\s*차)")


def normalize_project_key(text: str | None) -> str:
    """사업명 비교 키 — NFKC → 회차 서수 접두 정리 → 문장부호·괄호·공백 제거 → casefold.
    단어를 빼거나 바꾸지 않는다(회차 숫자는 그대로 남는다)."""
    if not text:
        return ""
    s = unicodedata.normalize("NFKC", str(text)).strip()
    s = _ORDINAL_PREFIX_RE.sub("", s)
    return _strip_punctuation_and_space(s).casefold()


# ---------------------------------------------------------------------------
# 사업 회차 — 사업을 **구분하는** 정보라 주석처럼 지우면 안 된다
# ---------------------------------------------------------------------------
# "…기능개선 사업(2차)"와 "…기능개선 사업"은 서로 다른 사업이다. 괄호를 주석으로만
# 보면 두 이름이 같아져, 사용자가 (7차)라고 물어도 회차가 없는 문서로 연결된다
# (실제 재현). 그래서 이름 비교와 별개로 **회차 숫자만** 따로 뽑아 대조한다.
#
# ⚠️ 일반 숫자를 회차로 오인하지 않도록 뒤따르는 글자를 본다.
#   - 뒤가 한글이 아니면(공백·괄호·문장부호·끝) 회차로 인정: "1차 구축", "사업(7차)"
#   - 뒤가 한글이면 사업·용역·공고로 이어질 때만 인정: "2차사업"
#   그래서 "3차원·2차전지·1차년도·1차산업·2차로·1차선·3차방정식"은 회차가 아니다.
#   "2개·7일·2억원·문단 2"처럼 '차'가 없는 표현도 당연히 회차가 아니다.
#
# ⚠️ 공고의 재공고 차수(identity_v2의 notice_round)와 같은 값으로 취급하지 않는다.
#   여기서 보는 것은 **공식 사업명에 적힌 사업 회차**뿐이다.
_PROJECT_ROUND_RE = re.compile(
    r"(?:제\s*)?(\d{1,2})\s*차(?![가-힣])"
    r"|(?:제\s*)?(\d{1,2})\s*차(?=(?:사업|용역|공고))"
)


def project_rounds(text: str | None) -> set[int]:
    """**공식 사업명**에 적힌 사업 회차 숫자들. 없으면 빈 집합.

    입력이 이미 사업명이라는 전제로 이름 안의 "N차"를 읽는다.
    사용자 질문에는 쓰지 말 것 — 질문에는 공고 차수·회의 차수처럼 사업과 무관한
    "N차"가 섞이므로 `question_project_rounds()`를 써야 한다.

    >>> sorted(project_rounds("기능개선 사업(2차)"))
    [2]
    >>> project_rounds("3차원 설계 용역") == set()
    True
    """
    if not text:
        return set()
    return {int(a or b) for a, b in _PROJECT_ROUND_RE.findall(str(text))}


# ---------------------------------------------------------------------------
# 질문에서 읽는 사업 회차 — 사업을 가리키는 문맥일 때만
# ---------------------------------------------------------------------------
# ⚠️ 질문 전체에서 "N차"를 그냥 찾으면 공고 차수·대화 순서까지 사업 회차로 읽는다.
#    "2차 공고인 사업", "재공고 2차 사업", "2차 회의에서 본 사업", "앞서 말한 2차 질문"이
#    전부 {2}가 되어, 공식 사업명에 그 회차가 없다는 이유로 **정상 문서가 후보에서
#    사라졌다**(실제 재현). 그래서 질문 쪽은 두 가지 문맥에서만 회차를 읽는다.
#
#   ① 괄호 안에 회차만 적힌 경우 — "기능개선 사업(2차)", "용역(총체 및 1차)"
#   ② 회차 바로 뒤에 **사업을 가리키는 낱말**이 오는 경우 — "제2차 사업", "2차사업",
#      "1차 구축 용역", "2차 고도화"
#
#    그리고 공고·회의·질문처럼 사업이 아닌 것을 세는 낱말이 앞뒤에 붙으면 제외한다.
#    ⚠️ "공고"는 사업 회차의 허용 낱말이 아니다. identity_v2의 notice_round(공고 차수)와도
#    합치지 않는다 — 이 함수는 사업명 회차만 다룬다.
_ROUND_CONTEXT_LIST = ("사업", "용역", "구축", "고도화", "개선",
                       "추진", "도입", "운영")
_ROUND_CONTEXT_WORDS = "|".join(_ROUND_CONTEXT_LIST)
_ROUND_BLOCK_WORDS = (
    "공고", "재공고", "입찰공고", "정정공고", "변경공고", "사전공고", "긴급공고",
    "회의", "질문", "답변", "회신", "검토", "평가", "심사", "통보", "안내", "요청",
)
_ROUND_IN_PAREN_RE = re.compile(
    r"[(（]\s*(?P<inner>[^)）\n]*?)(?:제\s*)?(?P<num>\d{1,2})\s*차\s*[)）]")
# 괄호 **바로 앞/뒤의 가장 가까운 낱말** — 조사·구두점은 건너뛴다.
# ⚠️ "앞뒤 10~20자 안에 금지 낱말이 있으면 제외"하는 식으로 넓게 보지 않는다.
#    실제 사업명에 "평가"·"안내" 같은 낱말이 들어 있을 수 있기 때문이다.
_WORD_BEFORE_RE = re.compile(r"([가-힣A-Za-z0-9]+)[^가-힣A-Za-z0-9]*$")
_WORD_AFTER_RE = re.compile(r"^[^가-힣A-Za-z0-9]*([가-힣A-Za-z0-9]+)")


def _nearest_word_before(text: str, pos: int) -> str:
    m = _WORD_BEFORE_RE.search(text[:pos])
    return m.group(1) if m else ""


def _nearest_word_after(text: str, pos: int) -> str:
    m = _WORD_AFTER_RE.search(text[pos:])
    return m.group(1) if m else ""
_ROUND_WITH_CONTEXT_RE = re.compile(
    rf"(?P<before>.{{0,12}}?)(?:제\s*)?(?P<num>\d{{1,2}})\s*차\s*(?:{_ROUND_CONTEXT_WORDS})")


def _has_block_word(text: str) -> bool:
    return any(w in text for w in _ROUND_BLOCK_WORDS)


def question_project_rounds(question: str | None) -> set[int]:
    """사용자 질문에서 **사업 회차**만 읽는다(공고 차수·대화 순서는 제외).

    >>> sorted(question_project_rounds("기능개선 사업(2차)의 예산은?"))
    [2]
    >>> question_project_rounds("2차 공고인 사업의 예산은?") == set()
    True
    >>> question_project_rounds("재공고 2차 사업의 예산은?") == set()
    True
    """
    if not question:
        return set()
    text = str(question)
    found: set[int] = set()
    for m in _ROUND_IN_PAREN_RE.finditer(text):
        if _has_block_word(m.group("inner")):
            continue                       # "(공고 2차)" 같은 표기는 사업 회차가 아니다
        # ⚠️ "(N차)"만 보고 무조건 회차로 쓰지 않는다. 괄호 **바로 앞뒤의 가장 가까운
        #    낱말**을 확인한다. 앞 낱말은 조사가 뒤에 붙으므로 끝을, 뒤 낱말은 조사가
        #    뒤따르므로 앞을 본다("사업(2차)" / "(2차) 회의에서").
        before = _nearest_word_before(text, m.start())
        after = _nearest_word_after(text, m.end())
        before_project = any(before.endswith(w) for w in _ROUND_CONTEXT_LIST)
        after_project = any(after.startswith(w) for w in _ROUND_CONTEXT_LIST)
        before_block = any(before.endswith(w) for w in _ROUND_BLOCK_WORDS)
        after_block = any(after.startswith(w) for w in _ROUND_BLOCK_WORDS)
        # 괄호 앞 낱말은 `(N차)`가 직접 수식하는 대상이므로 뒤보다 먼저 본다.
        # 예: `공고(2차) 사업`은 공고 차수이고, `사업(2차) 회의`는 사업 회차다.
        # 앞으로 판단할 수 없을 때만 뒤 낱말을 본다. 앞·뒤의 사업 신호를
        # OR로 묶으면 `공고(2차) 사업`이 다시 사업 회차로 되므로 순서가 계약의 일부다.
        if before_project:
            found.add(int(m.group("num")))          # 사업(2차) 회의
        elif before_block:
            continue                                 # 공고(2차) 사업
        elif after_project:
            found.add(int(m.group("num")))          # (2차) 사업
        elif after_block:
            continue                                 # (2차) 공고
        else:
            continue                                 # 애매하면 쓰지 않는다(보수적)
    for m in _ROUND_WITH_CONTEXT_RE.finditer(text):
        if _has_block_word(m.group("before")):
            continue                       # "재공고 2차 사업"은 공고 차수다
        found.add(int(m.group("num")))
    return found


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


# ---------------------------------------------------------------------------
# 보조 비교 키 ① — 낱말 사이의 소유격 조사 "의" 하나만 접기
# ---------------------------------------------------------------------------
# 공식 사업명이 "AI 기반**의** 영상감시 시스템"인데 사람은 "AI 기반 영상감시
# 시스템"이라고 부른다. 이 한 글자 차이 때문에 부분문자열 비교가 통째로 실패한다.
#
# ⚠️ 허용 범위를 아주 좁게 고정한다 — 임베딩·편집거리·유사도는 쓰지 않는다.
#   ① 낱말 **끝**에 붙어 있고            (앞이 공백이 아닌 글자)
#   ② 뒤에 공백과 다른 낱말이 이어질 때  (낱말과 낱말 **사이**의 조사)
#   만 지운다. "의정부"·"의료"처럼 낱말 안에 든 "의"는 뒤가 공백이 아니므로
#   절대 지워지지 않는다. 낱말 하나가 통째로 "의"인 경우도 앞이 공백이라 남는다.
# ⚠️ 2026-09-03 적대적 점검 반영 — 위치 규칙만으로는 "의로 **끝나는** 낱말"을
#    지키지 못했다. 예전 규칙은 뒤가 공백이기만 하면 잘라서 "심의 위원회"를
#    "심 위원회"로, "착수 회의 일정"을 "착수 회 일정"으로 망가뜨렸다.
#    그래서 조건을 하나 더 둔다.
#      ⓐ "의"를 뗀 나머지가 **두 글자 이상**일 것.
#         한국어에서 조사가 아닌 "…의"는 대부분 2음절 한자어다
#         (심의·회의·협의·동의·정의·주의·논의·문의·건의·임의·강의·편의·예의·
#          상의·발의·결의·제의·의의·수의·거의). 이들은 나머지가 한 글자라
#         자동으로 보호된다.
#
#    ⚠️ 낱말 목록(화이트리스트)은 **일부러 쓰지 않는다**. "합의"를 목록에 넣으면
#       "통합의 관리시스템"(통합+의)까지 막히고, "회의"를 넣으면 "위원회의 결정"
#       (위원회+의)이 막힌다 — 목록은 진짜 조사를 놓치는 반대 방향 오류를 만든다.
#       남는 위험은 "국무회의 자료"처럼 3음절 이상이면서 조사가 아닌 경우인데,
#       그 결과는 **보조 키가 하나 더 생기는 것**뿐이다. 원래 키는 그대로 남고,
#       보조 키가 두 문서에 걸리면 임의로 고르지 않고 후보를 남겨 되묻는다.
# 낱말 끝의 "의" + 뒤에 공백과 다른 낱말이 이어질 때만 후보로 본다.
_POSSESSIVE_PARTICLE_TOKEN_RE = re.compile(r"\S+의(?=\s+\S)")


def _fold_token(match: "re.Match[str]") -> str:
    token = match.group(0)
    stem = token[:-1]
    if len(stem) < 2:          # ⓐ 2음절 한자어(심의·회의·주의·정의…) 보호
        return token
    return stem


def fold_possessive_particles(text: str | None) -> str:
    """낱말 사이에 붙은 소유격 조사 "의"만 접는다(원본 표시는 바꾸지 않는다).

    >>> fold_possessive_particles("AI 기반의 영상감시 시스템")
    'AI 기반 영상감시 시스템'
    >>> fold_possessive_particles("의정부시 의료원 통합 시스템")
    '의정부시 의료원 통합 시스템'
    >>> fold_possessive_particles("제안서 평가 심의 위원회 일정")
    '제안서 평가 심의 위원회 일정'
    >>> fold_possessive_particles("통합의 관리시스템 구축")   # 진짜 조사는 접는다
    '통합 관리시스템 구축'
    """
    if not text:
        return ""
    return _POSSESSIVE_PARTICLE_TOKEN_RE.sub(_fold_token, str(text))


def normalize_project_key_particle_folded(text: str | None) -> str:
    """조사 "의"만 접은 사업명 보조 키. 원래 키와 같으면 빈 문자열(추가 안 함)."""
    if not text:
        return ""
    folded = normalize_project_key(fold_possessive_particles(text))
    return "" if folded == normalize_project_key(text) else folded


# ---------------------------------------------------------------------------
# 보조 비교 키 ② — 공식 파일명의 짧은 사업명
# ---------------------------------------------------------------------------
# identity_v2의 source_filename_nfc는 `발주기관명_짧은 사업명.md` 형태다.
# 파일명 길이 제한 때문에 사업명 뒤쪽이 잘려 있는데, 사람이 부르는 이름은 오히려
# 이 짧은 쪽에 가깝다("관산근린공원 … 회원 통합운영" ⊂ 정식명 "… 관리 시스템 구축").
#
# ⚠️ 파일명 앞부분이 **같은 행의 buyer_org와 글자 그대로 일치할 때만** 쓴다.
#    다른 기관명이 앞에 붙어 있으면 그 파일명은 이 행의 이름표가 아니므로
#    별칭으로 인정하지 않는다(추측으로 이름을 만들지 않는다).
_FILENAME_SUFFIXES = (".md",)


def filename_project_alias(source_filename_nfc: str | None,
                           buyer_org: str | None,
                           project_name: str | None = None) -> str:
    """`발주기관명_짧은 사업명.md`에서 짧은 사업명만 떼어낸다.

    기관 접두가 정확히 일치하지 않으면 빈 문자열(별칭 없음)을 돌려준다.

    ⚠️ 2026-09-03 적대적 점검 반영 — 파일명은 길이 제한 때문에 사업명을 **낱말
    중간에서** 자른다("… 자동분석 시스", "… 민속아카이브 자"). 이 조각을 그대로
    부분문자열 키로 쓰면 "자동분석 시스코", "민속아카이브 자원봉사자"처럼 전혀
    다른 뒷말에도 붙는다. 그래서 `project_name`을 함께 받으면 잘린 낱말을
    **정식 사업명 기준으로 낱말 끝까지 늘린다**(자르는 게 아니라 늘린다 —
    "2025 구미아시아육상경" → "2025 구미아시아육상경기선수권대회").
    결과는 항상 공식 사업명의 낱말 경계에서 끝나는 접두부다.
    """
    filename = normalize_display(source_filename_nfc)
    org = normalize_display(buyer_org)
    if not filename or not org:
        return ""
    prefix = org + "_"
    if not filename.startswith(prefix):
        return ""
    alias = filename[len(prefix):]
    for suffix in _FILENAME_SUFFIXES:
        if alias.lower().endswith(suffix):
            alias = alias[: -len(suffix)]
            break
    alias = alias.strip()
    return _extend_to_word_end(alias, project_name)


def _extend_to_word_end(alias: str, project_name: str | None) -> str:
    """별칭이 정식 사업명을 낱말 중간에서 자른 것이면 그 낱말 끝까지 늘린다."""
    if not alias or not project_name:
        return alias
    full = normalize_display(project_name)
    if not full.startswith(alias):
        # 파일명이 정식명의 접두부가 아니면(문자 치환 등) 손대지 않는다.
        return alias
    end = len(alias)
    if end >= len(full) or full[end].isspace():
        return alias                      # 이미 낱말 경계에서 끝난다
    while end < len(full) and not full[end].isspace():
        end += 1
    return full[:end].strip()
