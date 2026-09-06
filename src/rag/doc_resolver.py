"""
문서 특정(document resolution) — 2026-09-02 확정 정책.

우선순위
  1) 질문에 명시된 정확한 문서 ID (RFP-000001)
  2) 이전 대화에서 확정된 active_document_id (후속 질문일 때)
  3) 정규화한 발주기관명 + 사업명이 **둘 다** 일치
  4) 정규화한 발주기관명만 일치
  5) 정규화한 사업명만 일치하며 후보가 정확히 1개

모호한 경우
  - 후보 0개  → 찾을 수 없음(method="none")
  - 후보 1개  → 그 문서 선택
  - 후보 2개+ → 임의로 첫 문서를 고르지 않고 candidates를 채워 되묻기

⚠️ 우선순위 2의 적용 범위: "후속 질문일 때만 이어진다"(세션 정책 5-6)와
   충돌하지 않도록, active_document_id는 (a) 질문에 지시 표현("거기", "그 사업")이
   있거나 (b) 이름 기반 단서가 아예 없을 때만 쓴다. 질문이 명시적으로 다른
   기관·사업을 부르고 있는데 직전 문서로 덮어쓰지 않는다.

⚠️ LLM·임베딩으로 "비슷해 보이는 사업"을 자동 선택하지 않는다. 여기서 쓰는
   건 결정적 문자열 비교뿐이다.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from identity_metadata import IdentityIndex
from text_normalize import (
    fold_possessive_particles, normalize_org_key, normalize_project_key,
    project_rounds, question_project_rounds, strip_leading_org_from_project,
)

# 여섯 자리 뒤에 숫자·영문 꼬리가 붙은 잘못된 번호를 기존 문서로 잘라 읽지 않는다.
_DOC_ID_RE = re.compile(r"(?<![A-Za-z0-9_])RFP-\d{6}(?![A-Za-z0-9_])")
_DOC_ID_TOKEN_RE = re.compile(r"(?<![A-Za-z0-9_])RFP-[A-Za-z0-9_]*")

# 지시 표현(4-14) — 후속 질문 신호
ANAPHORA_MARKERS = (
    "그 사업", "이 사업", "해당 사업", "그 문서", "이 문서", "위 사업", "동 사업",
    "그거", "이거", "거기", "거기서", "그곳", "그 공고", "이 공고", "그 건",
    # 2026-09-03 보완 — 라우터의 옛 제외 목록에는 있었는데 여기엔 빠져 있던 표현.
    # 두 목록이 어긋나면 "해당 공고의 예산 알려줘"가 선별형으로 새어나간다.
    "해당 공고", "해당 문서", "해당 건", "위 공고", "위 문서", "이 건",
)

# 기관명처럼 생긴 표현 — 존재하지 않는 기관 판별용(유사도 임계값 대신 구조화 데이터로 판단).
# ⚠️ 뒤따르는 조사·기관 지시어까지 확인한다. 접미사만 보면 "제안요청서"(→청),
#    "우주정거장 데이터센터 구축사업"(→센터)처럼 기관이 아닌 표현이 기관명으로
#    오탐된다(실제 재현 확인). 기관을 부르는 문맥일 때만 후보로 삼는다.
_ORG_TAIL = (
    r"(?=\s*(?:에서|에게|께|의|이|가|은|는|에|과|와|랑|이랑|도|만|부터|"
    r"사업|공고|프로젝트|발주|측|쪽|[,.·!?]|$))"
)
_ORG_SUFFIX_RE = re.compile(
    r"[가-힣]{2,20}(?:재단|공사|공단|진흥원|연구원|연구소|협회|협의회|위원회|"
    r"교육청|시청|도청|청|처|센터|대학교|공제회|조합|은행|병원|박물관)"
    + _ORG_TAIL
)

# 기관을 가리키는 **일반 명사** — 고유 기관명이 아니므로 미등록 기관으로 세지 않는다.
# ⚠️ 등록부에 같은 이름의 기관이 실제로 있으면 아래 known 검사에서 먼저 인정되므로,
#    이 목록 때문에 실재 기관을 놓치지 않는다. "그 사업의 발주처와 예산 알려줘"처럼
#    일반 명사 하나로 후속 질문의 활성 문서 연결이 끊기던 문제를 막는다.
_GENERIC_ORG_WORDS = frozenset({
    "발주처", "발주청", "수요처", "계약처", "주관처", "담당처", "주관청", "관할청",
})

# 비교 대상을 나누는 접속 표현. 기관·사업명 충돌 검사와 미확인 대상 검사가
# 같은 경계를 쓰도록 한 곳에 둔다.
_COMPARE_CLAUSE_SPLIT_RE = re.compile(
    r"[,，;；]|(?:그리고|및)\s+|(?:과|와|이랑|랑)(?=\s)")

RESOLVE_EXPLICIT = "explicit"
RESOLVE_ACTIVE = "active_document"
RESOLVE_ORG_AND_PROJECT = "org_and_project"
RESOLVE_ORG = "org"
RESOLVE_PROJECT = "project"
RESOLVE_NONE = "none"
RESOLVE_INVALID_EXPLICIT = "invalid_explicit"
RESOLVE_UNKNOWN_EXPLICIT = "unknown_explicit"
RESOLVE_NAME_CONFLICT = "name_conflict"


# ⚠️ 2026-09-03 수정: 단순 부분문자열 검사는 조사에 걸려 오탐이 났다.
# "제출 방식**이 공고**문 자체엔 …" 이 "이 공고"(지시 표현)로 잡혀서, 여러 문서를
# 고르는 질문이 특정 문서 질문으로 새어나갔다(실제 재현). 지시어는 낱말 앞에서만
# 지시 표현이므로 앞이 문장 시작·공백·구두점일 때만 인정한다.
_ANAPHORA_RE = re.compile(
    r"(?:^|[\s,.!?;:()\[\]{}<>\"'·、，。/])("
    + "|".join(re.escape(m) for m in ANAPHORA_MARKERS) + r")"
)


def has_anaphora(question: str) -> bool:
    return bool(_ANAPHORA_RE.search(question))


# 문서명을 찾는 것과 문서명을 부르는 질문인지는 다르다. 미등록 제목을 찾지
# 못했을 때 전체 선별이나 이전 문서로 범위를 바꾸지 않기 위한 보수적인 신호다.
# 쉼표뿐 아니라 "용역의 예산", "구축 사업 예산"처럼 조사·공백 경계도 인정한다.
_TITLE_HEAD_RE = re.compile(
    r"(?:사업|용역|공고|프로젝트|시스템|플랫폼)"
    r"(?=(?:의|은|는|에|에서|\s|[,，:：?？]|$))"
)
_TITLE_QUERY_RE = re.compile(
    r"예산|사업비|사업\s*기간|지역\s*제한|제출|서류|공동\s*수급|컨소시엄|"
    r"참가\s*자격|면허|실적|배점|과업|마감|공고일|개요|사업\s*분야|"
    r"얼마|어떻게|무엇|뭐|알려|설명|요약|비교"
)
_TITLE_CONDITION_RE = re.compile(
    r"기재|명시|미기재|비공개|판단|공개된|적힌|적혀|되어|돼\s*있|"
    r"있는|없는|있고|없고|있어|없어|이상|이하|미만|초과|가능|불가|"
    r"허용|금지|필수|필요|해야|이어야|인\s+것|인\s+공고|"
    r"[가-힣]+(?:는데|인데|해서|아서|어서|하므로|이므로|으므로|라서|니까|하려고|라면|다면|"
    r"하며|이고|하고|지만|이면|이어서|여서|이라)(?:\s|$)"
    r"|(?:우리|저희|당사)\s+"
)
_TITLE_GENERIC_WORDS = frozenset({
    "그", "이", "해당", "위", "동", "어떤", "전체", "모든", "각", "여러",
    "그런", "이런", "저런", "그러한", "이러한", "에서", "낸", "발주한", "나온",
    "다른", "나머지", "사업", "용역", "공고", "문서", "제안요청서", "입찰",
    "공모", "프로젝트", "시스템", "플랫폼", "구축", "고도화", "개발", "운영",
    "예산", "사업비", "기간", "사업기간", "지역", "제한", "지역제한", "제출",
    "방식", "제출방식", "서류", "목록", "리스트", "내용", "조건", "관련",
    "대한", "대한것", "대한건", "대한내용", "개요", "정보", "평가", "배점",
    "참가", "자격", "공동수급", "컨소시엄", "공고일", "마감일", "과업", "범위",
    "주식회사", "사단법인", "재단법인",
})


def _specific_title_prefixes(question: str, *,
                             require_query_tail: bool = True) -> list[str]:
    """질문에서 이름처럼 부른 명사형 제목 부분을 모은다.

    `require_query_tail=False`는 이미 비교 대상 하나로 잘라낸 절을 볼 때만 쓴다.
    그 절은 "무엇을 묻는가"가 뒤쪽 절에 있어서 제목 뒤에 질문 표현이 없다.
    """
    prefixes: list[str] = []
    for match in _TITLE_HEAD_RE.finditer(question):
        tail = question[match.end():]
        if require_query_tail and not _TITLE_QUERY_RE.search(tail):
            continue
        # 새 문장/제목 앞 문맥을 잘라 이름 부분만 판단한다. "그 사업 말고 새
        # 제목"처럼 명시적인 대상 변경은 뒤쪽 이름을 검사한다.
        prefix = re.split(r"[,，:：;；\n]|(?:말고|대신)\s*", question[:match.start()])[-1]
        prefix = prefix.strip(" \t\"'“”‘’()（）[]")
        if not prefix or _TITLE_CONDITION_RE.search(prefix):
            continue
        words = re.findall(r"[가-힣A-Za-z0-9]+", prefix)
        if not words or len(words) > 18:
            continue
        # "그 사업의 전체 사업비"/"제출 서류 목록 공고"는 이름이 아니다.
        # 공백 없는 제목도 두 글자 이상의 고유한 부분이 있으면 보호한다.
        meaningful = [word for word in words
                      if len(word) >= 2 and word not in _TITLE_GENERIC_WORDS]
        if meaningful:
            prefixes.append(prefix)
    return prefixes


def looks_like_specific_document(question: str) -> bool:
    """미등록 제목도 문서 질문으로 보호한다. 후보 문서를 추측하는 함수는 아니다.

    명사형 제목 + 사업/용역 등 + 해당 내용 질문을 보되, 조건절을 제목으로
    오인하지 않는다. 실제 전체/복수 범위의 우선순위는 라우터가 판단한다.
    """
    return bool(_specific_title_prefixes(question))


def _project_terms_beyond_org(question: str, org_keys: list[str]) -> list[str]:
    """기관명 자체를 제목 단서로 세지 않고, 따로 부른 사업 제목의 낱말을 얻는다.

    "가온재단 사업 예산"은 기관의 사업을 묻는 기존 문법이다. 반대로
    "가온재단 새 관측망 구축 사업의 예산"은 기관이 같아도 기존 다른 사업으로
    대신 답할 수 없다. 비교 키의 글자를 공백·구두점 차이만 허용해 제거한다.
    """
    remainder = unicodedata.normalize("NFKC", question)
    for key in sorted(org_keys, key=len, reverse=True):
        pattern = r"[\W_]*".join(re.escape(char) for char in key)
        remainder = re.sub(pattern, " ", remainder, flags=re.IGNORECASE)
    prefixes = _specific_title_prefixes(remainder)
    if not prefixes:
        return []
    # 내부의 '시스템/플랫폼'에서 잘린 짧은 부분만 보면 공통 단어 하나로 다른
    # 사업을 승인할 수 있다. 가장 긴 제목 부분의 중요 낱말을 **모두** 대조한다.
    words = re.findall(r"[가-힣A-Za-z0-9]+", max(prefixes, key=len))
    terms: list[str] = []
    for word in words:
        word = re.sub(r"(?:의|에서)$", "", word)
        if len(word) >= 2 and word not in _TITLE_GENERIC_WORDS:
            terms.append(normalize_project_key(word))
    return terms


# ---------------------------------------------------------------------------
# 기관 별칭·사업명 핵심 낱말 매칭 (2026-09-04 §9)
# ---------------------------------------------------------------------------
# 문제: 공식 발주기관명이 "KOICA 전자조달" 인데 사용자는 "KOICA" 라고만 부른다.
#       공식 사업명은 아주 길고("우즈베키스탄 열린 의정활동 … PMC 용역") 사용자는
#       수식어를 빼고 "우즈베키스탄 국회 방송시스템 구축 사업" 이라고 부른다.
#       기존 매칭은 등록 키 **전체**가 질문 안에 연속으로 들어 있어야 해서 둘 다 놓쳤다.
#
# 해결: 둘 다 **공식 원자료(identity_v2)에서 결정적으로 파생**한 정보만 쓴다.
#   · 기관 별칭 = 공식 buyer_org 를 낱말로 쪼갠 것 중, 일반어가 아니고 **기관 하나에만**
#     나타나는 낱말. "KOICA" 는 되고 "전자조달"·"서울특별시" 처럼 여러 기관에 걸치거나
#     일반어인 낱말은 별칭이 되지 않는다.
#   · 사업명 핵심 낱말 = 질문이 부른 제목의 낱말이 공식 제목 안에 **같은 순서로 모두**
#     들어 있는가. 편집거리·임베딩 유사도는 쓰지 않는다.
# 확정 조건: 핵심 낱말이 2개 이상이고, 그렇게 걸린 문서가 **정확히 1건**일 때만.

# 기관명을 이루지만 그 자체로는 기관을 가리키지 못하는 낱말(별칭에서 뺀다).
_ORG_ALIAS_STOPWORDS = frozenset({
    "전자조달", "조달", "구매", "계약", "사업단", "사업소", "본부", "지사", "지역본부",
    "지방", "사무소", "센터", "본사", "지원단", "관리단", "추진단", "위원회", "재단",
    "공사", "공단", "협회", "진흥원", "연구원", "연구소", "대학교", "학교", "병원",
    "시청", "도청", "교육청", "주식회사", "사단법인", "재단법인", "korea", "한국",
})
_MIN_ALIAS_LEN = 3          # 두 글자 약어는 다른 이름 조각과 부딪히기 쉬워 쓰지 않는다
_MIN_TITLE_TERMS = 2        # 핵심 낱말이 하나뿐이면 일반어 하나로 문서를 고를 수 있다
# 예외: 기관을 명시해 범위가 이미 좁혀졌고, 그 하나의 낱말이 흔한 낱말이 아니라
#       6자 이상의 복합명사일 때만 한 낱말로도 확정을 허용한다.
#       ("국민연금공단 이러닝시스템 운영 용역" — 공식명은 '2024년 이러닝시스템 운영 용역')
#       기관 단서가 없으면 이 예외를 쓰지 않는다("방송시스템 구축 사업"은 계속 확정 금지).
_MIN_SINGLE_TERM_LEN = 6


def _alias_tokens(display: str) -> list[str]:
    """공식 표기 하나에서 별칭 후보 낱말을 뽑는다(공백·구두점 기준, 결정적)."""
    text = unicodedata.normalize("NFKC", display or "")
    out: list[str] = []
    for word in re.findall(r"[가-힣A-Za-z0-9]+", text):
        key = normalize_org_key(word)
        if not key or len(key) < _MIN_ALIAS_LEN:
            continue
        if key in _ORG_ALIAS_STOPWORDS:
            continue
        if key not in out:
            out.append(key)
    return out


def _title_word_keys(index: IdentityIndex) -> set[str]:
    """공식 **사업명**에 등장하는 낱말 키 전체.

    사업 제목에도 쓰이는 낱말("입찰공고", "의료원", "2025")은 기관을 가리키는 별칭이
    될 수 없다 — 그런 낱말을 별칭으로 쓰면 정식 사업명을 그대로 적은 질문이
    엉뚱한 기관으로 끌려가 이름 충돌로 막힌다(전수 점검에서 재현).
    """
    words: set[str] = set()
    for doc_id in index.document_ids():
        rec = index.get(doc_id)
        if rec is None:
            continue
        for name in (rec.project_name, getattr(rec, "filename_project_alias", "") or ""):
            for word in re.findall(r"[가-힣A-Za-z0-9]+",
                                   unicodedata.normalize("NFKC", name)):
                key = normalize_org_key(word)
                if key:
                    words.add(key)
    return words


def org_alias_map(index: IdentityIndex) -> dict[str, str]:
    """별칭 낱말 → 공식 기관 키. 공식 identity_v2 의 buyer_org 에서만 만든다.

    남기는 조건(모두 만족해야 한다)
      ① 한 기관에만 나타나는 낱말        — 여러 기관이 공유하면 기관을 못 가른다
      ② 공식 사업명에는 쓰이지 않는 낱말 — 제목 낱말은 기관 이름이 아니다
      ③ 숫자만으로 이뤄지지 않은 낱말    — "2025" 같은 연도는 기관이 아니다
    임의 별칭표를 코드에 적지 않는다.
    """
    cached = getattr(index, "_org_alias_map", None)
    if cached is not None:
        return cached
    title_words = _title_word_keys(index)
    owners: dict[str, set[str]] = {}
    for org_key in index.by_org_key:
        display = index.org_display_by_key.get(org_key, org_key)
        for token in _alias_tokens(display):
            if token == org_key or token in title_words or token.isdigit():
                continue
            owners.setdefault(token, set()).add(org_key)
    alias = {tok: next(iter(orgs)) for tok, orgs in owners.items() if len(orgs) == 1}
    try:
        object.__setattr__(index, "_org_alias_map", alias)
    except Exception:
        pass
    return alias


def matched_org_alias_tokens(question: str, index: IdentityIndex) -> list[str]:
    """질문 안에 실제로 적힌 기관 별칭 낱말(예: "koica")."""
    qkey = normalize_org_key(question)
    if not qkey:
        return []
    return sorted({token for token in org_alias_map(index) if token in qkey})


def match_org_alias_keys(question: str, index: IdentityIndex) -> list[str]:
    """질문에 **공식 기관명의 고유 낱말**이 적혀 있으면 그 기관 키를 돌려준다."""
    alias = org_alias_map(index)
    hits = [alias[token] for token in matched_org_alias_tokens(question, index)]
    return _drop_subsumed(sorted(set(hits)))


def _question_title_terms(question: str, org_keys: list[str]) -> list[str]:
    """질문이 부른 제목의 핵심 낱말을 **질문에 적힌 순서대로** 뽑는다."""
    remainder = unicodedata.normalize("NFKC", question)
    for key in sorted({k for k in org_keys if k}, key=len, reverse=True):
        remainder = re.sub(_key_pattern(key), " ", remainder, flags=re.IGNORECASE)
    prefixes = _specific_title_prefixes(remainder)
    if not prefixes:
        prefixes = _specific_title_prefixes(remainder, require_query_tail=False)
    if not prefixes:
        return []
    longest = max(prefixes, key=len)
    terms: list[str] = []
    for word in re.findall(r"[가-힣A-Za-z0-9]+", longest):
        word = re.sub(r"(?:의|에서|은|는|이|가|을|를)$", "", word)
        if len(word) < 2 or word in _TITLE_GENERIC_WORDS:
            continue
        key = normalize_project_key(word)
        if key and key not in terms:
            terms.append(key)
    return terms


def _terms_in_order(terms: list[str], name: str) -> bool:
    """핵심 낱말이 공식 제목 안에 **같은 순서로 모두** 나오는가(부분열 검사)."""
    key = normalize_project_key(name)
    if not key:
        return False
    pos = 0
    for term in terms:
        found = key.find(term, pos)
        if found < 0:
            return False
        pos = found + len(term)
    return True


def _official_names(rec) -> list[str]:
    names = [rec.project_name]
    alias = getattr(rec, "filename_project_alias", "")
    if alias:
        names.append(alias)
    return [n for n in names if n]


def match_docs_by_title_terms(question: str, index: IdentityIndex,
                              scope: list[str] | None = None,
                              org_keys: list[str] | None = None) -> tuple[list[str], list[str]]:
    """줄여 부른 사업명으로 문서를 찾는다. (후보 문서, 사용한 핵심 낱말).

    ★확정은 호출측이 한다 — 여기서는 조건에 맞는 후보를 **전부** 돌려준다.
      후보가 둘 이상이면 호출측이 되묻는다(임의로 하나를 고르지 않는다).
    """
    terms = _question_title_terms(question, org_keys or [])
    enough = (len(terms) >= _MIN_TITLE_TERMS
              or (len(terms) == 1 and scope is not None
                  and len(terms[0]) >= _MIN_SINGLE_TERM_LEN))
    if not enough:
        return [], terms
    candidates = scope if scope is not None else index.document_ids()
    hits = []
    for doc_id in candidates:
        rec = index.get(doc_id)
        if rec is None:
            continue
        if any(_terms_in_order(terms, name) for name in _official_names(rec)):
            hits.append(doc_id)
    return hits, terms

def detect_document_id(question: str) -> str | None:
    m = _DOC_ID_RE.search(question)
    return m.group(0) if m else None


def detect_document_ids(question: str) -> list[str]:
    seen: list[str] = []
    for m in _DOC_ID_RE.finditer(question):
        if m.group(0) not in seen:
            seen.append(m.group(0))
    return seen


def detect_invalid_document_ids(question: str) -> list[str]:
    """번호 형식이 틀리면 문서 이름이나 이전 대화로 몰래 대체하지 않게 표시한다."""
    return list(dict.fromkeys(
        match.group(0) for match in _DOC_ID_TOKEN_RE.finditer(question)
        if not re.fullmatch(r"RFP-\d{6}", match.group(0))
    ))


def _drop_subsumed(keys: list[str]) -> list[str]:
    """서로 포함 관계인 매칭 키는 긴 쪽만 남긴다
    ("서울특별시" vs "서울특별시교육청" 오매칭 방지)."""
    return [k for k in keys if not any(other != k and k in other for other in keys)]


def _key_pattern(key: str) -> str:
    """정규화 키를 원문에서 찾기 위한 패턴 — 공백·구두점 차이만 허용한다."""
    return r"[\W_]*".join(re.escape(char) for char in key)


def _matched_org_spans(question: str, org_keys: list[str]) -> list[tuple[int, int]]:
    """질문 원문에서 등록 기관명이 실제로 적힌 구간."""
    spans: list[tuple[int, int]] = []
    for key in org_keys:
        if not key:
            continue
        for match in re.finditer(_key_pattern(key), question, flags=re.IGNORECASE):
            spans.append((match.start(), match.end()))
    return spans


def _matched_project_spans(question: str, index: IdentityIndex) -> list[tuple[int, int]]:
    """질문 원문에서 **공식 사업명**이 실제로 적힌 구간.

    ⚠️ 공식 사업명 안에 기관처럼 생긴 낱말이 들어 있는 경우가 있다
    ("IP-NAVI 해외지식재산센터 사업관리 시스템 기능개선"의 '해외지식재산센터').
    이건 사용자가 따로 부른 기관이 아니라 **사업 제목의 일부**다. 그런데도
    미등록 기관으로 세면 정확한 공식 사업명을 그대로 적은 질문까지 차단된다(실제 재현).
    여기서 얻은 구간 안에 들어 있는 표현은 미등록 기관 후보에서 뺀다.
    구간 밖에서 따로 부른 미등록 기관은 그대로 걸러낸다.
    """
    spans: list[tuple[int, int]] = []
    for key in match_project_keys(question, index):
        if not key:
            continue
        for match in re.finditer(_key_pattern(key), question, flags=re.IGNORECASE):
            spans.append((match.start(), match.end()))
    return spans


def match_org_keys(question: str, index: IdentityIndex) -> list[str]:
    """질문 안에 정규화 기준으로 등장하는 기관명 키(최장 매칭만)."""
    qkey = normalize_org_key(question)
    if not qkey:
        return []
    matched = [k for k in index.by_org_key if k and k in qkey]
    return _drop_subsumed(matched)


def _project_key_tables(index: IdentityIndex) -> list[dict[str, list[str]]]:
    """사업명 비교에 쓰는 색인 네 벌. 원래 키 색인이 항상 먼저다."""
    return [
        index.by_project_key,                     # ① 공식 사업명 그대로
        index.by_project_core_key,                # ② 기관 접두부만 뗀 형태
        getattr(index, "by_project_alias_key", {}),   # ③ 공식 파일명의 짧은 사업명
        getattr(index, "by_project_folded_key", {}),  # ④ 조사 "의"만 접은 보조 키
    ]


def match_project_keys(question: str, index: IdentityIndex) -> list[str]:
    """질문 안에 등장하는 사업명 키(최장 매칭만).

    네 형태를 본다 — ① 사업명 전체 ② 사업명에서 발주기관 접두부만 뗀 형태
    ③ 공식 파일명(`발주기관명_짧은 사업명.md`)의 짧은 사업명
    ④ 낱말 사이의 조사 "의" 하나만 접은 보조 키.

    ②③④는 전부 **결정적 문자열 변환**이다 — 임베딩·편집거리·유사도는 쓰지
    않는다. ③은 파일명 앞 기관명이 같은 행 buyer_org와 정확히 일치할 때만
    만들어지고, ④는 질문과 등록명 **양쪽에 같은 규칙**을 적용해 비교한다.
    ①의 원래 키는 그대로 남아 있어 기존 매칭 결과가 나빠지지 않는다.
    """
    qkey = normalize_project_key(question)
    if not qkey:
        return []
    matched = [k for k in index.by_project_key if k and k in qkey]
    matched += [k for k in index.by_project_core_key if k and k in qkey]
    matched += [k for k in getattr(index, "by_project_alias_key", {}) if k and k in qkey]
    # 보조 키는 질문 쪽도 같은 규칙으로 접은 뒤에만 대조한다.
    qkey_folded = normalize_project_key(fold_possessive_particles(question))
    matched += [k for k in getattr(index, "by_project_folded_key", {})
                if k and k in qkey_folded]
    return _drop_subsumed(sorted(set(matched)))


def _docs_for_project_keys(keys: list[str], index: IdentityIndex) -> list[str]:
    docs: list[str] = []
    for k in keys:
        for table in _project_key_tables(index):
            for d in table.get(k, []):
                if d not in docs:
                    docs.append(d)
    return docs


def _docs_for_primary_project_keys(keys: list[str], index: IdentityIndex) -> list[str]:
    """원래(공식 사업명) 키로 매칭된 문서만. 보조 키로만 걸린 문서는 뺀다."""
    docs: list[str] = []
    for k in keys:
        for table in (index.by_project_key, index.by_project_core_key):
            for d in table.get(k, []):
                if d not in docs:
                    docs.append(d)
    return docs


def _all_title_terms_beyond_org(question: str, org_keys: list[str]) -> list[str]:
    """질문이 부른 제목 조각 **전부**의 중요 낱말을 모은다.

    `_project_terms_beyond_org`는 가장 긴 제목 조각 하나만 본다. 사업명에 쉼표가
    들어가면("… 전산 및 시스템, 홈페이지 유지·보수") 제목이 쉼표에서 잘려,
    뒤쪽 조각에 있는 다른 핵심어("유지관리")를 놓친다. 별칭 전용 매칭을 검사할
    때는 조각을 하나도 버리지 않는다.
    """
    remainder = unicodedata.normalize("NFKC", question)
    for key in sorted(org_keys, key=len, reverse=True):
        pattern = r"[\W_]*".join(re.escape(char) for char in key)
        remainder = re.sub(pattern, " ", remainder, flags=re.IGNORECASE)
    terms: list[str] = []
    for prefix in _specific_title_prefixes(remainder):
        for word in re.findall(r"[가-힣A-Za-z0-9]+", prefix):
            word = re.sub(r"(?:의|에서)$", "", word)
            if len(word) >= 2 and word not in _TITLE_GENERIC_WORDS:
                key = normalize_project_key(word)
                if key and key not in terms:
                    terms.append(key)
    return terms


def _round_matches(question_rounds: set[int], doc_id: str,
                   index: IdentityIndex) -> bool:
    """질문이 말한 사업 회차가 그 문서의 **공식 사업명 회차**와 같은가.

    질문에 회차가 있는데 문서 이름에 회차가 없으면 **일치로 보지 않는다** —
    "(7차)"를 물었는데 회차가 없는 사업으로 답하면 안 되기 때문이다.
    """
    rec = index.get(doc_id)
    if rec is None:
        return False
    return bool(question_rounds & project_rounds(rec.project_name))


def _filter_by_round(docs: list[str], question_rounds: set[int],
                     index: IdentityIndex) -> list[str]:
    """질문에 회차가 명시됐을 때만 회차가 같은 후보를 남긴다."""
    if not question_rounds:
        return docs
    return [d for d in docs if _round_matches(question_rounds, d, index)]


def _title_terms_match_official(question: str, doc_id: str,
                                index: IdentityIndex) -> bool:
    """질문이 부른 제목의 낱말이 그 문서의 공식 사업명 안에 전부 있는가.

    ⚠️ 파일명 별칭은 사업명 **앞부분**만 담고 있어서, 뒤쪽 낱말이 전혀 다른
    이름도 별칭 부분만 겹치면 매칭된다("… 회원 통합운영 관리 **장비** 구축").
    보조 키로만 걸린 문서에는 기존 기관 경로에서 쓰던 것과 같은 대조를 적용해,
    질문이 부른 제목의 낱말이 공식 사업명에 실제로 있는지 확인한다.
    (유사도가 아니라 글자 그대로의 포함 검사다.)
    """
    terms = _all_title_terms_beyond_org(question, match_org_keys(question, index))
    if not terms:
        return True
    rec = index.get(doc_id)
    if rec is None:
        return True
    official = {normalize_project_key(rec.project_name),
                normalize_project_key(fold_possessive_particles(rec.project_name))}
    alias = rec.filename_project_alias
    if alias:
        official.add(normalize_project_key(alias))
        official.add(normalize_project_key(fold_possessive_particles(alias)))
    return any(all(term in name for term in terms) for name in official if name)


def _docs_for_org_keys(keys: list[str], index: IdentityIndex) -> list[str]:
    docs: list[str] = []
    for k in keys:
        for d in index.by_org_key.get(k, []):
            if d not in docs:
                docs.append(d)
    return docs


def detect_unknown_orgs(question: str, index: IdentityIndex) -> list[str]:
    """질문에서 기관명처럼 생긴 표현 중 identity_v2에 없는 것을 **전부** 반환.

    알려진 기관과 알 수 없는 기관이 함께 있어도 알 수 없는 쪽을 누락하지 않는다.
    """
    matches = list(_ORG_SUFFIX_RE.finditer(question))
    if not matches:
        return []
    known = index.known_org_keys()
    # "새가온재단"에 "가온재단"이 포함돼 있다는 이유로 같은 기관이라고 보지 않는다.
    # 다만 공백 없는 "가온재단과별빛공사"는 등록 기관들의 연결 표현으로 인정한다.
    known_alternation = "|".join(re.escape(key) for key in sorted(known, key=len, reverse=True) if key)
    known_composition = (re.compile(
        rf"(?:(?:과|와|및|그리고|이랑|랑))?(?:{known_alternation})"
        rf"(?:(?:과|와|및|그리고|이랑|랑)(?:{known_alternation}))*"
    ) if known_alternation else None)
    # 질문이 **정식 기관명을 그대로 적은 구간**. 접미사 규칙은 공백을 넘지 못해
    # "서울특별시 여성가족재단"에서 "여성가족재단"만 잘라내는데, 그 조각은 새
    # 기관이 아니라 같은 이름의 일부다. 구간 안에 들어 있을 때만 인정하므로
    # 등록 기관명을 품은 더 긴 미등록 이름("새가온재단")은 구간 밖이라 그대로
    # 미등록으로 남는다. 이름 경계를 보므로 "서울특별시 여성가족재단이랑 부산
    # 여성가족재단"의 두 번째는 여전히 미등록으로 잡힌다.
    known_spans = _matched_org_spans(question, match_org_keys(question, index))
    # 공식 사업명 구간도 '이미 확인된 이름'이다 — 그 안의 낱말은 새 기관이 아니다.
    known_spans += _matched_project_spans(question, index)
    unknown: list[str] = []
    for match in matches:
        cand = match.group(0)
        ckey = normalize_org_key(cand)
        if not ckey:
            continue
        if ckey in known or bool(known_composition and known_composition.fullmatch(ckey)):
            continue
        if ckey in _GENERIC_ORG_WORDS:
            continue
        if any(start <= match.start() and match.end() <= end
               for start, end in known_spans):
            continue
        if cand not in unknown:
            unknown.append(cand)
    return unknown


@dataclass
class DocumentResolution:
    document_id: str | None
    method: str
    candidates: list[str] = field(default_factory=list)
    candidate_labels: list[str] = field(default_factory=list)
    unknown_orgs: list[str] = field(default_factory=list)
    matched_org_keys: list[str] = field(default_factory=list)
    matched_project_keys: list[str] = field(default_factory=list)

    @property
    def is_ambiguous(self) -> bool:
        return self.document_id is None and len(self.candidates) > 1


def _label(index: IdentityIndex, doc_id: str) -> str:
    rec = index.get(doc_id)
    if rec is None:
        return doc_id
    return f"{doc_id} ({rec.buyer_org} / {rec.project_name})"


def resolve_document(
    question: str,
    index: IdentityIndex | None = None,
    active_document_id: str | None = None,
) -> DocumentResolution:
    """확정된 우선순위대로 문서를 특정한다. 세션 상태는 호출측이 갱신한다."""
    explicit_ids = detect_document_ids(question)
    if detect_invalid_document_ids(question):
        return DocumentResolution(None, RESOLVE_INVALID_EXPLICIT)
    if explicit_ids:
        # 명시한 두 문서를 한 문서로 줄이지 않는다. 비교가 아니면 호출측이 되묻는다.
        if index is not None and any(index.get(doc_id) is None for doc_id in explicit_ids):
            return DocumentResolution(None, RESOLVE_UNKNOWN_EXPLICIT)
        if len(explicit_ids) > 1:
            return DocumentResolution(
                None, RESOLVE_EXPLICIT, candidates=explicit_ids,
                candidate_labels=[_label(index, d) if index is not None else d
                                  for d in explicit_ids],
            )
        return DocumentResolution(explicit_ids[0], RESOLVE_EXPLICIT)

    names_new_title = looks_like_specific_document(question)
    if index is None:
        if active_document_id and has_anaphora(question) and not names_new_title:
            return DocumentResolution(active_document_id, RESOLVE_ACTIVE)
        return DocumentResolution(None, RESOLVE_NONE)

    unknown_orgs = detect_unknown_orgs(question, index)
    if unknown_orgs:
        # 미등록 긴 기관명 안의 등록된 짧은 이름으로 문서를 대신 찾지 않는다.
        return DocumentResolution(None, RESOLVE_NONE, unknown_orgs=unknown_orgs)

    org_keys = match_org_keys(question, index)
    # ★[2026-09-04 §9] 정식 기관명 전체가 안 적혀 있어도, 공식 기관명의 **고유 낱말**
    #   ("KOICA 전자조달" 의 KOICA)만 적은 질문은 그 기관을 부른 것이다. 별칭은
    #   identity_v2 의 buyer_org 에서 결정적으로 파생하며 한 기관에만 나타나는
    #   낱말만 인정한다. 정식 이름이 이미 걸렸으면 별칭은 보지 않는다(기존 동작 유지).
    alias_org_keys: list[str] = []
    if not org_keys:
        alias_org_keys = match_org_alias_keys(question, index)
        org_keys = list(alias_org_keys)
    project_keys = match_project_keys(question, index)
    org_docs = _docs_for_org_keys(org_keys, index)
    project_docs = _docs_for_project_keys(project_keys, index)
    # ★사업명을 줄여 불러 등록 키가 통째로 걸리지 않을 때만, 핵심 낱말이 공식 제목
    #   안에 같은 순서로 모두 있는 문서를 찾는다. 기관 단서가 있으면 **그 기관 안에서만**
    #   찾는다(다른 기관의 비슷한 이름으로 새지 않게).
    title_terms: list[str] = []
    title_docs: list[str] = []
    if not project_docs and names_new_title:
        scope = org_docs if org_docs else None
        # 질문에 적힌 별칭 낱말("KOICA")도 기관 단서이므로 제목 낱말에서 뺀다 —
        # 안 빼면 기관 이름이 사업명 핵심 낱말로 섞여 공식 제목과 어긋난다.
        title_docs, title_terms = match_docs_by_title_terms(
            question, index, scope=scope,
            org_keys=org_keys + matched_org_alias_tokens(question, index))
        if title_docs:
            project_docs = title_docs
    # 사업 회차가 질문에 명시됐으면 회차가 같은 문서만 남긴다. 회차는 사업을
    # 구분하는 정보이므로 다른 회차·회차 없는 문서로 대신 답하지 않는다.
    # 질문 쪽은 **사업을 가리키는 문맥**의 회차만 읽는다(공고 차수·대화 순서 제외).
    question_rounds = question_project_rounds(question)
    if question_rounds:
        org_docs = _filter_by_round(org_docs, question_rounds, index)
        project_docs = _filter_by_round(project_docs, question_rounds, index)
    # 보조 키(파일명 별칭·조사 접기)로만 걸린 문서는 제목 낱말까지 대조한다.
    # 원래 공식 사업명 키로 걸린 문서는 기존 그대로 둔다(동작 변화 없음).
    primary_docs = _docs_for_primary_project_keys(project_keys, index)
    # ★핵심 낱말 경로로 찾은 문서는 이미 공식 제목과 **순서까지** 대조했다. 아래의
    #   보조 키 검증(부분문자열 포함)을 다시 걸면 같은 확인을 더 약한 규칙으로
    #   되풀이해 정상 매칭을 지운다 — 그래서 이 경로에는 적용하지 않는다.
    if title_docs:
        pass
    elif len(project_docs) != len(primary_docs):
        kept = [d for d in project_docs
                if d in primary_docs
                or _title_terms_match_official(question, d, index)]
        # ⚠️ 이 대조는 후보를 **줄이기만** 한다. 줄인 결과로 여러 후보가 한 건이 되면
        #    "확신 있는 답"이 만들어지는데, 그건 근거가 늘어서가 아니라 근거를 뺐기
        #    때문이다. 실제로 그 한 건이 아주 짧은 꼬리말 키로 걸린 엉뚱한 문서일 수
        #    있다(적대적 점검에서 재현). 그런 경우에는 걸러내기 전 후보를 그대로 두어
        #    기존처럼 되묻는다 — 임의 확정보다 되묻기가 안전하다.
        if len(project_docs) > 1 and len(kept) == 1:
            pass                      # 후보를 줄이지 않는다(되묻기 유지)
        else:
            project_docs = kept

    # 지시어가 앞에 있어도 새 기관·사업명을 명시했다면 새 이름이 우선이다.
    # "그 사업 말고 별빛공사"나 미등록 기관을 직전 문서로 대신 답하지 않는다.
    if (active_document_id and has_anaphora(question) and not names_new_title
            and not org_keys and not project_keys and not unknown_orgs):
        return DocumentResolution(active_document_id, RESOLVE_ACTIVE)

    def _finish(docs: list[str], method: str) -> DocumentResolution:
        if len(docs) == 1:
            return DocumentResolution(
                docs[0], method, unknown_orgs=unknown_orgs,
                matched_org_keys=org_keys, matched_project_keys=project_keys,
            )
        return DocumentResolution(
            None, method, candidates=docs,
            candidate_labels=[_label(index, d) for d in docs],
            unknown_orgs=unknown_orgs,
            matched_org_keys=org_keys, matched_project_keys=project_keys,
        )

    # 우선순위 3 — 기관명 + 사업명 둘 다 일치
    both = [d for d in org_docs if d in project_docs]
    if both:
        return _finish(both, RESOLVE_ORG_AND_PROJECT)

    # ★기관이 **별칭 낱말로만** 걸렸고 사업명은 공식 키로 정확히 걸렸다면, 더 강한 쪽
    #   (정식 사업명)을 따른다. 별칭은 보조 단서라 이름 충돌로 막을 근거가 못 된다.
    if alias_org_keys and org_docs and project_docs and not set(org_docs) & set(project_docs):
        if _docs_for_primary_project_keys(project_keys, index):
            org_docs, org_keys = [], []
    # 기관과 사업명이 각각 존재해도 서로 다른 문서라면 둘 중 하나를 임의로 고르지 않는다.
    if org_docs and project_docs:
        return DocumentResolution(
            None, RESOLVE_NAME_CONFLICT, unknown_orgs=unknown_orgs,
            matched_org_keys=org_keys, matched_project_keys=project_keys,
        )

    # 기관은 찾아도 사용자가 따로 부른 새 제목이 안 맞으면, 그 기관의 유일한
    # 다른 사업을 대신 답하지 않는다. 기관만 부른 질문은 기존 우선순위를 유지한다.
    # ★[2026-09-04 §9] 예전에는 looks_like_specific_document(질문 전체)가 참일 때만 이
    #   보호가 걸렸다. 그 함수는 기관명이 제목 앞에 붙어 있으면 제목을 못 알아봐서
    #   "기초과학연구원 화성기지 관제시스템 구축 사업 예산"(없는 사업)이 그 기관의
    #   유일한 문서로 확정됐다(전수 점검에서 재현). 이제 기관명을 **먼저 지운 뒤**
    #   남은 제목 낱말이 있으면 그 낱말이 공식 제목과 맞는지 항상 확인한다.
    if org_docs and not project_docs and len(org_keys) == 1 and not title_docs:
        terms = _project_terms_beyond_org(question, org_keys)
        compatible = any(
            all(term in normalize_project_key(index.get(doc_id).project_name) for term in terms)
            for doc_id in org_docs if index.get(doc_id) is not None
        )
        if terms and not compatible:
            return DocumentResolution(
                None, RESOLVE_NONE, unknown_orgs=unknown_orgs,
                matched_org_keys=org_keys, matched_project_keys=project_keys,
            )

    # 우선순위 4 — 기관명만 일치
    if org_docs:
        return _finish(org_docs, RESOLVE_ORG)

    # 우선순위 5 — 사업명만 일치하고 후보가 정확히 1개
    if len(project_docs) == 1:
        return _finish(project_docs, RESOLVE_PROJECT)
    if len(project_docs) > 1:
        return _finish(project_docs, RESOLVE_PROJECT)

    # 아무 이름 단서도 없으면, 마지막으로 직전 활성 문서를 쓴다(지시 표현이
    # 없더라도 "다른 문서를 부르고 있지 않다"는 게 확인된 상태)
    if active_document_id and not names_new_title and not unknown_orgs:
        return DocumentResolution(
            active_document_id, RESOLVE_ACTIVE, unknown_orgs=unknown_orgs
        )

    return DocumentResolution(None, RESOLVE_NONE, unknown_orgs=unknown_orgs)


def resolve_documents_for_compare(
    question: str,
    index: IdentityIndex | None = None,
) -> tuple[list[str], list[str]]:
    """비교형 — 질문에 등장한 문서를 **전부** 찾는다. (문서 ID 목록, 미확인 기관).

    명시적 ID를 우선 쓰고, 부족하면 기관명/사업명으로 하나로 특정되는 문서를
    덧붙인다(여러 건에 매칭되는 이름은 모호해서 건너뛴다 — 임의로 안 고름).
    """
    doc_ids = detect_document_ids(question)
    if index is None:
        return doc_ids, []

    unknown_orgs = detect_unknown_orgs(question, index)

    # 기관+사업 조합으로 문서 하나가 확정되는 경우를 먼저 채운다
    for org_key in match_org_keys(question, index):
        org_docs = index.by_org_key.get(org_key, [])
        if len(org_docs) == 1:
            if org_docs[0] not in doc_ids:
                doc_ids.append(org_docs[0])
            continue
        # 기관에 문서가 여러 개면 사업명으로 좁혀본다
        project_docs = _docs_for_project_keys(match_project_keys(question, index), index)
        narrowed = [d for d in org_docs if d in project_docs]
        if len(narrowed) == 1 and narrowed[0] not in doc_ids:
            doc_ids.append(narrowed[0])

    # 기관명 없이 사업명만 나온 문서도 하나로 확정되면 포함
    for pkey in match_project_keys(question, index):
        pdocs = _docs_for_project_keys([pkey], index)
        if len(pdocs) == 1 and pdocs[0] not in doc_ids:
            doc_ids.append(pdocs[0])

    return doc_ids, unknown_orgs


def _strip_known_names(text: str, explicit_ids: list[str], keys: list[str]) -> str:
    """확인된 문서 ID·기관명·사업명을 지우고 남은 표현만 돌려준다."""
    remainder = unicodedata.normalize("NFKC", text)
    for doc_id in explicit_ids:
        remainder = remainder.replace(doc_id, " ")
    for key in sorted({k for k in keys if k}, key=len, reverse=True):
        remainder = re.sub(_key_pattern(key), " ", remainder, flags=re.IGNORECASE)
    return remainder


def _document_title_keys(doc_id: str, index: IdentityIndex) -> list[str]:
    """문서의 **공식** 사업명 비교 키.

    원형 + 기관 접두부를 뗀 형태 + 공식 파일명의 짧은 사업명 + 조사만 접은 형태.
    전부 공식 자료에서 결정적으로 만든 이름이라 "닮은 이름"을 승인하지 않는다.
    """
    rec = index.get(doc_id)
    if rec is None:
        return []
    keys = [normalize_project_key(rec.project_name),
            strip_leading_org_from_project(rec.project_name, rec.buyer_org)]
    for name in rec.project_names_for_keys():
        keys.append(normalize_project_key(name))
        keys.append(normalize_project_key(fold_possessive_particles(name)))
    return [k for k in dict.fromkeys(keys) if k]


def unexplained_comparison_targets(
    question: str, index: IdentityIndex, resolved_doc_ids: list[str],
) -> list[str]:
    """비교 대상 중 **확정된 문서로 설명되지 않는** 이름을 모은다.

    비교 절 하나씩 보고, 확인된 문서 ID·기관명·사업명을 지운 뒤 남은 제목을
    그 절이 지목한 문서의 공식 사업명과 대조한다.

    - 확정된 문서의 공식 사업명을 **연속된 일부만 줄여 부른 표현**은 새 대상이
      아니다("지능정보화전략계획(ISP)" ⊂ "서울특별시교육청 지능정보화전략계획(ISP)
      수립(2차) 사업 용역"). 이름을 임의로 닮은 정도로 재는 게 아니라 공식
      사업명 안에 글자 그대로 이어져 있어야 한다.
    - 반대로 공식 사업명에 없는 제목은 문서가 몇 건 확정됐든 새 대상이다.
      기관에 문서가 하나뿐이라는 이유로 전혀 다른 사업명을 인정하지 않는다.
    """
    explicit_ids = detect_document_ids(question)
    org_keys = match_org_keys(question, index)
    project_keys = match_project_keys(question, index)
    clauses = _COMPARE_CLAUSE_SPLIT_RE.split(question)
    # 절이 하나뿐이면 대상을 나눌 수 없으므로 기존처럼 질문 전체를 본다.
    single_clause = len([c for c in clauses if c.strip()]) < 2
    unexplained: list[str] = []
    for clause in clauses:
        if not clause.strip():
            continue
        clause_key_org = normalize_org_key(clause)
        clause_key_project = normalize_project_key(clause)
        clause_orgs = [k for k in org_keys if k in clause_key_org]
        clause_projects = [k for k in project_keys if k in clause_key_project]
        remainder = _strip_known_names(clause, explicit_ids, clause_orgs + clause_projects)
        titles = _specific_title_prefixes(remainder, require_query_tail=single_clause)
        if not titles:
            continue
        # 이 절이 직접 지목해 확정된 문서를 먼저 대조하고, 그런 문서가 없을
        # 때만 확정된 전체와 대조한다(대상과 문서의 연결을 유지).
        scoped = [d for d in (_docs_for_org_keys(clause_orgs, index)
                              + _docs_for_project_keys(clause_projects, index))
                  if d in resolved_doc_ids]
        official = [k for doc_id in (scoped or list(resolved_doc_ids))
                    for k in _document_title_keys(doc_id, index)]
        for title in titles:
            title_key = normalize_project_key(title)
            if title_key and any(title_key in official_key for official_key in official):
                continue
            if title not in unexplained:
                unexplained.append(title)
    return unexplained


def comparison_scope_issue(
    question: str,
    index: IdentityIndex | None = None,
    resolved_doc_ids: list[str] | None = None,
) -> str | None:
    """비교 대상 일부를 못 찾았는데 찾은 문서끼리만 비교하는 일을 차단한다.

    기존 비교 문서 목록 함수의 반환 형식은 유지한다. 실행부는 이 사유가 있을 때
    비교표를 만들기 전에 되묻고, 문서 ID를 명시하도록 안내한다.

    `resolved_doc_ids`는 `resolve_documents_for_compare()`가 실제로 확정한
    문서다. 차단 판단이 확정 결과를 보지 못해 정상 비교까지 막던 문제 때문에
    실행부에서 넘겨받는다(생략하면 여기서 같은 함수로 구한다).
    """
    invalid_ids = detect_invalid_document_ids(question)
    if invalid_ids:
        return "문서 ID 형식을 확인해주세요: " + ", ".join(invalid_ids)
    explicit_ids = detect_document_ids(question)
    if index is None:
        return None
    unknown_ids = [doc_id for doc_id in explicit_ids if index.get(doc_id) is None]
    if unknown_ids:
        return "공식 목록에서 확인되지 않은 문서 ID: " + ", ".join(unknown_ids)
    unknown_orgs = detect_unknown_orgs(question, index)
    if unknown_orgs:
        return "공식 목록에서 확인되지 않은 기관: " + ", ".join(unknown_orgs)

    org_keys = match_org_keys(question, index)
    project_keys = match_project_keys(question, index)
    project_docs = _docs_for_project_keys(project_keys, index)
    org_docs = _docs_for_org_keys(org_keys, index)
    for org_key in org_keys:
        candidates = list(dict.fromkeys(index.by_org_key.get(org_key, [])))
        if len(candidates) > 1:
            narrowed = [doc_id for doc_id in candidates if doc_id in project_docs]
            if len(narrowed) != 1:
                return "한 기관에 비교 후보 문서가 여러 건 있습니다. 사업명 또는 문서 ID로 대상을 지정해주세요."
    for project_key in project_keys:
        candidates = _docs_for_project_keys([project_key], index)
        if len(candidates) > 1:
            narrowed = [doc_id for doc_id in candidates if doc_id in org_docs]
            if len(narrowed) != 1:
                return "같은 사업명에 문서가 여러 건 있습니다. 발주기관 또는 문서 ID로 대상을 지정해주세요."

    # "기관 A의 사업 B"를 서로 다른 문서 두 개로 늘려 비교하지 않는다.
    clauses = _COMPARE_CLAUSE_SPLIT_RE.split(question)
    for clause in clauses:
        clause_orgs = match_org_keys(clause, index)
        clause_projects = match_project_keys(clause, index)
        if len(clause_orgs) == 1 and clause_projects:
            if not set(_docs_for_org_keys(clause_orgs, index)).intersection(
                    _docs_for_project_keys(clause_projects, index)):
                return "기관명과 사업명이 서로 다른 문서를 가리킵니다. 비교할 문서 ID를 확인해주세요."

    # 확인된 이름을 지운 뒤에도 고유한 사업 제목이 남으면 미확인 비교 대상이다.
    # 단, 확정된 문서의 공식 사업명을 줄여 부른 표현은 새 대상이 아니다.
    # 이름 일부를 임의로 추측해서 승인하지는 않는다.
    resolved = (resolved_doc_ids if resolved_doc_ids is not None
                else resolve_documents_for_compare(question, index)[0])
    if unexplained_comparison_targets(question, index, resolved):
        return "이름으로 확인되지 않은 비교 대상이 있습니다. 모든 대상의 문서 ID를 알려주세요."
    return None
