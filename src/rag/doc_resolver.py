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
    normalize_org_key, normalize_project_key, strip_leading_org_from_project,
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


def match_org_keys(question: str, index: IdentityIndex) -> list[str]:
    """질문 안에 정규화 기준으로 등장하는 기관명 키(최장 매칭만)."""
    qkey = normalize_org_key(question)
    if not qkey:
        return []
    matched = [k for k in index.by_org_key if k and k in qkey]
    return _drop_subsumed(matched)


def match_project_keys(question: str, index: IdentityIndex) -> list[str]:
    """질문 안에 등장하는 사업명 키(최장 매칭만).

    두 형태를 본다 — ① 사업명 전체 ② 사업명에서 발주기관 접두부만 뗀 형태.
    ②는 임의 단어 삭제가 아니라 같은 행 buyer_org와 글자 그대로 겹치는
    접두부만 제거한 결정적 변환이다(text_normalize.strip_leading_org_from_project).
    """
    qkey = normalize_project_key(question)
    if not qkey:
        return []
    matched = [k for k in index.by_project_key if k and k in qkey]
    matched += [k for k in index.by_project_core_key if k and k in qkey]
    return _drop_subsumed(sorted(set(matched)))


def _docs_for_project_keys(keys: list[str], index: IdentityIndex) -> list[str]:
    docs: list[str] = []
    for k in keys:
        for d in index.by_project_key.get(k, []) + index.by_project_core_key.get(k, []):
            if d not in docs:
                docs.append(d)
    return docs


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
    project_keys = match_project_keys(question, index)
    org_docs = _docs_for_org_keys(org_keys, index)
    project_docs = _docs_for_project_keys(project_keys, index)

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

    # 기관과 사업명이 각각 존재해도 서로 다른 문서라면 둘 중 하나를 임의로 고르지 않는다.
    if org_docs and project_docs:
        return DocumentResolution(
            None, RESOLVE_NAME_CONFLICT, unknown_orgs=unknown_orgs,
            matched_org_keys=org_keys, matched_project_keys=project_keys,
        )

    # 기관은 찾아도 사용자가 따로 부른 새 제목이 안 맞으면, 그 기관의 유일한
    # 다른 사업을 대신 답하지 않는다. 기관만 부른 질문은 기존 우선순위를 유지한다.
    if org_docs and not project_docs and names_new_title and len(org_keys) == 1:
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
    """문서의 **공식** 사업명 비교 키(원형 + 기관 접두부를 뗀 형태)."""
    rec = index.get(doc_id)
    if rec is None:
        return []
    keys = [normalize_project_key(rec.project_name),
            strip_leading_org_from_project(rec.project_name, rec.buyer_org)]
    return [k for k in keys if k]


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
