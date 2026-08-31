from __future__ import annotations

import re
from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator


TaskType = Literal["selection", "extraction", "qa"]

# 스키마 v0.2 (임현진, 2026-08-28) — answer_type 값 통일.
#   document_set : 선별형 정답(문서 ID 배열은 answer_raw 에 담긴다)
#   value        : 단일 값 (구 short_answer)
#   list         : 항목 배열 (완전일치)
#   summary      : 요약형 — 체크포인트 배열을 answer_raw 에 담는다(별도 checkpoints 필드 없음)
#   comparison   : 비교표
#   unanswerable : 답 없는 질문 — 기권 사유 문자열을 answer_raw 에 담는다(별도 필드 없음)
AnswerType = Literal["document_set", "value", "list", "summary", "comparison", "unanswerable"]

# task_type 별 허용 answer_type (v0.2 스키마 표). check_evalset_integrity 가 강제한다.
ALLOWED_ANSWER_TYPES: dict[str, set[str]] = {
    "selection": {"document_set"},
    "extraction": {"value", "list"},
    "qa": {"value", "summary", "comparison", "unanswerable"},
}

UnspecifiedType = Literal[
    "abbreviation",
    "org_only",
    "time_reference",
    "ambiguous_match",
]
ScenarioType = Literal[
    "workflow_chain",
    "anaphora",
    "condition_add",
    "selection_to_extraction",
]
AnswerSource = Literal["table", "verified", "metadata"]
FieldTag = Literal["critical", "major", "minor"]  # 임현진 FIELD_SPEC (1-5 등급표)

# 임현진 확정: 시간 의존 문항의 기준 시점은 상수. now 금지(평가셋이 저절로 틀려짐).
REFERENCE_TIME = "2024-06-01"

# 1-12-2 / 3-2-1 세 상태. answer_source=table 인 선별형/추출형 문항의 순환 판별에 쓴다.
ExtractState = Literal["value", "absent", "failed"]

# 3-2 검색 실패 두 종류 (D11·D2). 이 구분이 없으면 reranker가 실제로 필요한지 알 수 없다.
FailureKind = Literal["none", "recall_failure", "rank_failure"]

# 2-3 / 4-12-1: "항목 자체 없음"과 "코퍼스에 없는 사업"은 다른 상태다.
AbstentionKind = Literal[
    "ok",  # 기권해야 하는데 기권함 / 안 해도 되는데 안 함
    "hallucination",  # 기권해야 하는데 답을 지어냄
    "over_refusal",  # 답이 있는데 기권함
    "critical_abstain",  # field_tag=critical 문항의 기권 — 정상 동작 후보 (1-5)
]


def as_id_list(value: Any) -> list[str]:
    """document_id / intermediate_answer 는 v0.2에서 문자열 또는 배열 둘 다 허용한다
    (EXT-007: "DOC-091" / SEL-001: ["DOC-014","DOC-036"]). 채점·정합성 로직이
    항상 리스트로 다루도록 정규화한다."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    return [str(v) for v in value]


# 분할 표의 조각 표기 `(2/3)`.
_PART_OF_RE = re.compile(r"\s*\(\d+\s*/\s*\d+\)\s*$")


def _strip_part(ref: str) -> str:
    """분할 표의 `표 7 (2/3)` → `표 7`.

    ★팀 확정 좌표 형식 (2026-08-31, 박예진 파싱 ↔ 김하루 채점 ↔ 임현진 평가셋):
      좌표 단위는 `ref_no = "표 7"` 로 통일한다. part/of 는 **포함하지 않는다** —
      어느 조각에 답이 있든 "표 7 에 있다"로 채점한다. 박예진 청크는 처음부터 이 방향,
      임현진 평가셋도 part/of 제외로 통일(박예진 안내). 옛 데이터에 `(N/M)` 이 남아
      있어도 매칭 단계에서 여기서 떼므로 안전하다.
    """
    return _PART_OF_RE.sub("", str(ref or "")).strip()


class Location(BaseModel):
    model_config = ConfigDict(extra="allow")

    document: str
    section: str
    ref_no: str

    def key(self, precision: Literal["document", "section", "ref_no"] = "section") -> tuple:
        if precision == "document":
            return (self.document,)
        if precision == "section":
            return (self.document, self.section)
        return (self.document, self.section, _strip_part(self.ref_no))

    @classmethod
    def from_chunk(cls, document_id: str, section_path: list[str] | None,
                   location_label: str | None) -> "Location":
        """박예진 청크(C-3) 좌표를 grader 의 {document, section, ref_no} 로 변환한다.

        section_path : ["2. 사업개요", "제18조(평가배점)"]  (마지막 원소가 절)
        location_label: "제18조(평가배점) · 표 7 (2/3)"  또는 "... · 문단 1-4"
        → ref_no 는 ` · ` 뒤 부분에서 (part/of) 를 뗀 것.
        """
        sp = [s for s in (section_path or []) if s]
        section = sp[-1] if sp else ""
        label = str(location_label or "")
        ref_no = _strip_part(label.rsplit(" · ", 1)[-1]) if " · " in label else ""
        if not section and " · " in label:
            section = label.rsplit(" · ", 1)[0].strip()
        return cls(document=document_id, section=section, ref_no=ref_no)


class EvaluationItem(BaseModel):
    """
    평가셋 한 문항 — **스키마 v0.2**(임현진, 2026-08-28 확정본, 15개 필드).

    v0.1 → v0.2 제거 필드와 이 코드의 대응:
      - document_unspecified / time_dependent / conversational: 제거됨. 각각
        unspecified_type / reference_time / scenario_type 의 존재 여부로 파생하는
        computed_field 로 남겨 grade_doc_selection·check_evalset_integrity 가 그대로
        쓴다(v0.2 정정: 파생 근거가 intermediate_answer/active_document_id 가 아니라
        해당 "타입 필드"로 바뀌었다).
      - difficulty: 2-4 재확정으로 폐기.
      - schema_version: 문항 레벨에서 빠지고 evalset/v1/VERSION.txt(파일 단위)로 이동
        (grader.versioning.read_schema_version).
      - checkpoints: 별도 필드 아님 — 요약형 체크포인트 배열은 answer_raw 에 담는다.
      - unanswerable_reason: 별도 필드 아님 — answer_type=unanswerable 이고 기권 사유
        문자열을 answer_raw 에 담는다(null 금지).
    """

    model_config = ConfigDict(extra="forbid")

    # ── 팀 결정 (2026-08-31) ──────────────────────────────────────────────
    # C. field_absent 를 새 스키마 필드로 신설하지 않는다. 부재를 표현하는 방식은
    #    쓰임에 따라 갈린다:
    #      · 답 없음 문제로 쓸 때  → answer_type=unanswerable, answer_raw 에 부재 문구
    #      · 도메인상 유의미한 답일 때 → task_type=extraction, answer_type=value,
    #        answer_raw 에 확정 문구("지역제한 없음" 등)
    #    conflict 셀은 해당 조건의 정답으로 쓰지 않고 문항에서 제외한다(v1 미포함) —
    #    그래서 채점기에 conflict/외부참조용 별도 분기가 없다. (B-2 추출 테이블의 5상태는
    #    3-2-1 grade_extraction_audit 의 감사 어휘일 뿐 문항 필드가 아니다.)
    # D. 복수 질문·정답이 필요할 때 라벨 배열 구조를 신설하지 않는다. 의미가 다른 값마다
    #    각각 별도 문항으로 분리한다(과업수행기간 문항 1개 + 사업개요 사업기간 문항 1개).
    #    각 문항은 독립적으로 채점되며 채점기 변경은 없다 — answer_raw 에 원문 표현 그대로.
    # ─────────────────────────────────────────────────────────────────────
    id: str
    question: str
    task_type: TaskType
    answer_type: AnswerType
    # v0.2: extraction/qa 는 단일 문서 ID(문자열), 배열도 허용(하위호환). 선별형은 이 필드
    # 대신 answer_raw 에 문서 ID 배열을 담는다.
    document_id: str | list[str] | None = None
    unspecified_type: UnspecifiedType | None = None
    intermediate_answer: str | list[str] | None = None
    active_document_id: str | None = None
    scenario_type: ScenarioType | None = None
    reference_time: str | None = None
    field_tag: FieldTag | None = None
    answer_source: AnswerSource | None = None
    # v0.2: any. document_set=문서ID배열 / value=값 / list=항목배열 /
    #   summary=체크포인트배열 / comparison=비교표 / unanswerable=사유 문자열
    answer_raw: Any | None = None
    answer_normalized: Any | None = None
    location: Location | None = None

    @model_validator(mode="before")
    @classmethod
    def _drop_meta_keys(cls, data: Any) -> Any:
        """`_` 로 시작하는 주석 키(예: practice_items.jsonl 의 `_source_note`)는
        스키마 필드가 아니므로 조용히 떼어낸다 — 그래야 practice 세트가 로드된다.
        ★최종셋 CI 검사에서 `_` 키를 FAIL 로 잡는 것은 raw JSON 단계
        (validation.check_meta_keys)에서 별도로 한다. 나머지 오타성 extra 는
        extra='forbid' 가 그대로 걸러낸다."""
        if isinstance(data, dict):
            return {k: v for k, v in data.items() if not str(k).startswith("_")}
        return data

    @computed_field  # type: ignore[misc]
    @property
    def document_unspecified(self) -> bool:
        """v0.2: 문서 미특정 여부 — unspecified_type 존재로 파생(2-2-2)."""
        return self.unspecified_type is not None

    @computed_field  # type: ignore[misc]
    @property
    def time_dependent(self) -> bool:
        """v0.2: 시간 의존 여부 — reference_time 존재로 파생(2-6-2)."""
        return self.reference_time is not None

    @computed_field  # type: ignore[misc]
    @property
    def conversational(self) -> bool:
        """v0.2: 후속 질문 여부 — scenario_type 존재로 파생(2-5)."""
        return self.scenario_type is not None

    @property
    def checkpoints(self) -> list[str]:
        """요약형(summary) 체크포인트 — v0.2에서 answer_raw 에 담긴 배열.
        별도 스키마 필드가 아니라 answer_raw 뷰다(김하루 채점기 호환)."""
        return list(self.answer_raw) if isinstance(self.answer_raw, list) else []

    @property
    def unanswerable_reason(self) -> str | None:
        """기권 사유 — v0.2에서 answer_type=unanswerable 일 때 answer_raw 문자열.
        별도 스키마 필드가 아니라 answer_raw 뷰다(김하루 채점기 호환)."""
        if self.answer_type != "unanswerable":
            return None
        return self.answer_raw if isinstance(self.answer_raw, str) else None


def _chunk_location_before(data: Any) -> Any:
    """박예진 청크 스키마(C-3)를 그대로 받으면 location 을 합성한다.
    이태민 검색 출력이 `{document_id, section_path, location_label, ...}` 형태로 오면
    grader 는 location({document, section, ref_no}) 이 필요하다."""
    if not isinstance(data, dict) or data.get("location") is not None:
        return data
    doc = data.get("document_id")
    if doc and ("section_path" in data or "location_label" in data):
        data = dict(data)
        data["location"] = Location.from_chunk(
            doc, data.get("section_path"), data.get("location_label")
        ).model_dump()
    return data


class ContextChunk(BaseModel):
    model_config = ConfigDict(extra="allow")

    document_id: str | None = None
    text: str = ""  # 박예진 청크는 `content`/`search_text` 로 오므로 필수 아님
    location: Location | None = None

    @model_validator(mode="before")
    @classmethod
    def _derive_location(cls, data: Any) -> Any:
        data = _chunk_location_before(data)
        if isinstance(data, dict) and not data.get("text"):
            data = dict(data)
            data["text"] = data.get("search_text") or data.get("content") or ""
        return data


class RetrievedItem(BaseModel):
    """3-2 검색 후보 하나. retrieval_k / reranker_k 단계 결과를 같은 형태로 받는다."""

    model_config = ConfigDict(extra="allow")

    document_id: str
    location: Location | None = None
    score: float | None = None

    @model_validator(mode="before")
    @classmethod
    def _derive_location(cls, data: Any) -> Any:
        return _chunk_location_before(data)


class ModelResponse(BaseModel):
    """
    4번(이태민) 시스템 출력. retrieval_k / reranker_k / context_k 를 분리해서 받는다 —
    최종 k가 정해지지 않았어도 각 단계를 따로 채점할 수 있어야 한다(4-9-3).
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    answer: str
    # list/comparison 답변의 구조화 형태 — list[str] 또는 {document_id: {field: value}}
    structured_answer: Any | None = None
    contexts: list[ContextChunk] = Field(default_factory=list)  # 실제 LLM 입력(context_k)
    retrieved: list[RetrievedItem] = Field(default_factory=list)  # retrieval_k 후보 전체
    reranked: list[RetrievedItem] = Field(default_factory=list)  # reranker_k 이후
    citations: list[Location] = Field(default_factory=list)  # 3-4-3 출처 좌표
    selected_document_ids: list[str] = Field(default_factory=list)  # 선별형 결과
    active_document_id: str | None = None  # 대화형 상태 (2-5)
    abstained: bool = False
    # 시스템이 기권 사유를 자유 텍스트로 함께 낼 수 있음(v0.2: 평가셋은 answer_raw 문자열).
    unanswerable_reason: str | None = None
    route: str | None = None  # 4-10-1 분기 결과
    failure: str | None = None  # 4-5: 오류로 인한 0점 vs 오답 0점 구분
    latency_ms: float | None = None
    cost_usd: float = 0.0


class JudgeRequest(BaseModel):
    evaluation: EvaluationItem
    response: ModelResponse


class JudgeMetricResult(BaseModel):
    model_config = ConfigDict(extra="allow")

    metric: str
    raw: Any
    parsed: dict[str, Any] | None = None
    verdict: bool | None = None
    score: float | None = None
    reason: str | None = None


class FormatStatus(BaseModel):
    """3-3 형식 계약. 내용과 독립적인 축 — 내용이 맞아도 여기서 FAIL이면 최종 FAIL."""

    passed: bool
    violations: list[str] = Field(default_factory=list)


class TaskScore(BaseModel):
    """selection / extraction(+list) / qa / comparison 채점 결과. task_scoring.py 산출물."""

    model_config = ConfigDict(extra="allow")

    kind: str
    score: float
    detail: dict[str, Any] = Field(default_factory=dict)


class RetrievalDiagnostic(BaseModel):
    model_config = ConfigDict(extra="allow")

    applicable: bool
    stage: str | None = None  # retrieval_k | reranker_k | context_k
    recall_at_k: float | None = None
    recall_all: bool | None = None
    precision_at_k: float | None = None
    mrr: float | None = None
    failure_kind: FailureKind | None = None
    reason: str | None = None


class CitationDiagnostic(BaseModel):
    """3-4-3 출처 좌표 채점 결과. citation 미표기는 check_format의 FAIL 판정과
    무관하다 — 여기서만 score/matched로 반영한다."""

    model_config = ConfigDict(extra="allow")

    applicable: bool
    matched: bool | None = None
    score: float | None = None
    n_citations: int | None = None
    precision_unit: str | None = None
    reason: str | None = None


class AbstentionResult(BaseModel):
    model_config = ConfigDict(extra="allow")

    should_abstain: bool
    abstained: bool
    abstention_kind: AbstentionKind


# provenance 자산의 "값 미상" 표기. ★누락(키 자체가 없음)과 미상은 다르다 —
# 누락은 조용한 버그(3-17이 그 축을 '안 바뀜'으로 오독), 미상은 기록된 사실이다.
UNKNOWN = "UNKNOWN"

# 팀 확정 6-자산 provenance. 필드명·순서는 팀 실험 인프라 규약 `base.yaml` §① "재료 버전
# 6칸 (필드명 고정 — 절대 개명 금지)" 과 **정확히 일치**한다. 값 규약도 동일: `v1`/`v2` …
# regression.CHANGE_AXES / config.ProvenanceDefaults / runner 가 이 목록을 단일 출처로 따른다.
#   corpus     : 원문 코퍼스
#   preprocess : 전처리 파이프라인(파싱·정제·청킹)
#   table      : 구조화 추출 테이블(1-12-2)
#   index      : 검색 인덱스(임베딩·색인 — 평가 대상 RAG 검색계, 4번)
#   evalset    : 평가셋(Ground Truth)
#   scorer     : 채점기(이 저장소). 값은 `v1` 등 — 심판 프롬프트 세부 버전은 별도로
#                manifest.judge_prompt_versions 에 남긴다(회귀 3-17 축은 이 필드 하나).
PROVENANCE_ASSETS: tuple[tuple[str, str], ...] = (
    ("corpus", "코퍼스"),
    ("preprocess", "전처리"),
    ("table", "추출 테이블"),
    ("index", "검색 인덱스"),
    ("evalset", "평가셋"),
    ("scorer", "채점기"),
)
PROVENANCE_FIELDS: tuple[str, ...] = tuple(k for k, _ in PROVENANCE_ASSETS)


class Provenance(BaseModel):
    """3-11 / 3-13-1 / 3-17: 평가 결과 하나가 '어떤 자산 조합'에서 나왔는지 **6자산**으로 못박는다.

    ★설계 원칙
      - 6개 필드가 **항상** 결과에 존재한다(EvaluationResult.provenance 는 기본값이
        빈 Provenance 라 절대 None 이 아니다). 값이 없으면 필드를 빼지 않고 "UNKNOWN".
      - 6축 전부 str(`v1` 규약). 회귀 원인 분리(regression.attribute_change)가 축마다
        다른 비교 규칙을 쓰지 않도록.
      - 축 목록·순서·필드명은 models.PROVENANCE_ASSETS 한 곳에서만 정의한다
        (= base.yaml §① 6칸).
    """

    model_config = ConfigDict(extra="forbid")

    corpus: str = UNKNOWN      # ① 원문 코퍼스
    preprocess: str = UNKNOWN  # ② 전처리 파이프라인(파싱·정제·청킹)
    table: str = UNKNOWN       # ③ 구조화 추출 테이블(1-12-2)
    index: str = UNKNOWN       # ④ 검색 인덱스(평가 대상 RAG 검색계, 4번)
    evalset: str = UNKNOWN     # ⑤ 평가셋(Ground Truth)
    scorer: str = UNKNOWN      # ⑥ 채점기(이 저장소 코드 + 심판 프롬프트 묶음)


class EvaluationResult(BaseModel):
    id: str
    schema_version: str
    mode: str
    provider: str
    judge_results: list[JudgeMetricResult] = Field(default_factory=list)
    task_score: TaskScore | None = None
    format_status: FormatStatus | None = None
    retrieval: list[RetrievalDiagnostic] = Field(default_factory=list)
    citation: CitationDiagnostic | None = None
    abstention: AbstentionResult | None = None
    # 3-3 최종 판정: PASS / FAIL-content / FAIL-format / FAIL-format+content
    final_status: str | None = None
    # ★항상 존재한다(6축 전부). 명시하지 않아도 빈 Provenance(=6축 모두 "UNKNOWN")로 채워진다 —
    # per_item/report 에서 provenance 축이 조용히 사라지는 것을 구조적으로 막는다.
    provenance: Provenance = Field(default_factory=Provenance)
    runtime_ms: float = 0.0
    cache_hit: bool = False
    error: str | None = None
