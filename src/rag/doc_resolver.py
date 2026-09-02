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
from dataclasses import dataclass, field

from identity_metadata import IdentityIndex
from text_normalize import normalize_org_key, normalize_project_key

_DOC_ID_RE = re.compile(r"RFP-\d{6}")

# 지시 표현(4-14) — 후속 질문 신호
ANAPHORA_MARKERS = (
    "그 사업", "이 사업", "해당 사업", "그 문서", "이 문서", "위 사업", "동 사업",
    "그거", "이거", "거기", "거기서", "그곳", "그 공고", "이 공고", "그 건",
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

RESOLVE_EXPLICIT = "explicit"
RESOLVE_ACTIVE = "active_document"
RESOLVE_ORG_AND_PROJECT = "org_and_project"
RESOLVE_ORG = "org"
RESOLVE_PROJECT = "project"
RESOLVE_NONE = "none"


def has_anaphora(question: str) -> bool:
    return any(m in question for m in ANAPHORA_MARKERS)


def detect_document_id(question: str) -> str | None:
    m = _DOC_ID_RE.search(question)
    return m.group(0) if m else None


def detect_document_ids(question: str) -> list[str]:
    seen: list[str] = []
    for m in _DOC_ID_RE.finditer(question):
        if m.group(0) not in seen:
            seen.append(m.group(0))
    return seen


def _drop_subsumed(keys: list[str]) -> list[str]:
    """서로 포함 관계인 매칭 키는 긴 쪽만 남긴다
    ("서울특별시" vs "서울특별시교육청" 오매칭 방지)."""
    return [k for k in keys if not any(other != k and k in other for other in keys)]


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
    candidates = _ORG_SUFFIX_RE.findall(question)
    if not candidates:
        return []
    known = index.known_org_keys()
    unknown: list[str] = []
    for cand in candidates:
        ckey = normalize_org_key(cand)
        if not ckey:
            continue
        is_known = any(ckey in k or k in ckey for k in known)
        if not is_known and cand not in unknown:
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
    explicit = detect_document_id(question)
    if explicit:
        return DocumentResolution(explicit, RESOLVE_EXPLICIT)

    if index is None:
        if active_document_id and has_anaphora(question):
            return DocumentResolution(active_document_id, RESOLVE_ACTIVE)
        return DocumentResolution(None, RESOLVE_NONE)

    unknown_orgs = detect_unknown_orgs(question, index)

    # 우선순위 2 — 후속 질문(지시 표현)일 때만 직전 활성 문서를 이어받는다
    if active_document_id and has_anaphora(question):
        return DocumentResolution(
            active_document_id, RESOLVE_ACTIVE, unknown_orgs=unknown_orgs
        )

    org_keys = match_org_keys(question, index)
    project_keys = match_project_keys(question, index)
    org_docs = _docs_for_org_keys(org_keys, index)
    project_docs = _docs_for_project_keys(project_keys, index)

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
    if active_document_id:
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
