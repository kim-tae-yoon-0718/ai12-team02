from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import os
import yaml
from dotenv import load_dotenv


@dataclass(frozen=True)
class JudgeConfig:
    names: tuple[str, ...]
    prompt_files: dict[str, Path]
    metrics: tuple[str, ...]
    # judge_faithfulness처럼 "문항 전체"를 한 번 채점하는 judge. judge_checkpoint /
    # judge_list_item 은 여기 넣지 않는다 — 그 둘은 task_scoring 안에서 항목 단위로
    # 여러 번 호출되는 매처(matcher)라서 문항당 1회 metric 결과와 다르다.
    whole_item_metrics: tuple[str, ...] = ()
    model: str = "stub"
    family: str = "none"
    tier: str = "dev"
    verification_path: str | None = None
    min_agreement: float = 0.80


@dataclass(frozen=True)
class RuntimeConfig:
    concurrency: int
    timeout_seconds: int
    temperature: float


@dataclass(frozen=True)
class CacheConfig:
    enabled: bool
    directory: Path


@dataclass(frozen=True)
class RetrievalConfig:
    """4-9-3: retrieval_k / reranker_k / context_k 는 서로 다른 단계다.
    ★최종 k는 미확정 — 이 셋을 하나의 값으로 합치지 않는다."""
    retrieval_k: int = 20
    reranker_k: int = 10
    context_k: int = 5
    precision: str = "ref_no"  # document | section | ref_no (2-9 확정 단위)
    # 이태민 top_k=5 결과를 채점 때 여러 k로 잘라 별도 기록 (base.yaml top_k 주석, 김하루 협의)
    eval_k: tuple[int, ...] = (3, 5)


@dataclass(frozen=True)
class GradingConfig:
    allow_partial: bool = False
    docset_partial_credit: bool = True
    miss_weight: float = 0.7
    require_table_format: bool = True
    grade_citations: bool = True


@dataclass(frozen=True)
class GateConfig:
    """[대기] 아래 임계값들은 baseline 실측 전 임시값이다. 실측 없이 최종 threshold로
    쓰지 않는다 — task_scoring.combine_status가 부분점수 태스크는 PENDING_THRESHOLD로
    남기는 이유가 이것이다."""
    severity_gate: dict[str, float] = field(default_factory=lambda: {"critical": 0.85})
    column_severity: dict[str, str] = field(default_factory=dict)  # [대기 ← 박예진 1-12-1]
    task_quota: dict[str, int] = field(default_factory=dict)
    data_thresholds: dict[str, float] = field(default_factory=dict)  # [대기 ← 박예진]
    # 2026-08-27 박예진 확정 — 게이트(실행을 막음)와 성격이 다른 "경고"(실행을 안 막고
    # 사람이 보고 넘김) 기준선. data_thresholds와 절대 합치지 않는다 — 합치면 경고성
    # 신호 하나로 실행이 멈추는 사고가 난다.
    data_warn_thresholds: dict[str, float] = field(default_factory=dict)
    doc_org: dict[str, str] = field(default_factory=dict)
    # 2026-08-27 임현진 2-16 확정: 통합 점수 "산출 방식"은 확정(가중치는 1-2 업무빈도·
    # 위험에서 끌어옴, 태스크별 3개 점수와 항상 나란히 표시). 다만 1-2의 실제 가중치
    # 숫자는 아직 전달되지 않아 기본값은 비워 둔다 — 비어 있으면 diagnostics.main_metrics가
    # integrated_score를 None으로 남기고 그 이유를 note에 남긴다.
    task_weight: dict[str, float] = field(default_factory=dict)
    # 2026-08-28 재확인 — field_tag 재정의: "부분점수 인정 여부"가 아니라 "오답
    # 심각도(critical/major/minor) 표시 + 3-3-2 ①가중 평균 채점 입력"으로 쓴다.
    # (한때 "가중치를 감으로 정하면 근거가 없다"고 기각했던 ①방식을 재도입하기로
    # 팀이 재확인함 — diagnostics.severity_weighted_score 참고.) task_weight와
    # 동일한 원칙: 실제 숫자(예: {critical: 3, major: 2, minor: 1})가 올 때까지
    # 비워 두고, 비어 있으면 계산하지 않는다.
    field_tag_weight: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class ProvenanceDefaults:
    """팀 확정 6-자산 provenance의 설정 기본값(models.PROVENANCE_ASSETS 순서·이름과 일치).

    corpus/preprocess/table/index/evalset/scorer 는 CLI 인자 > 이 기본값 순으로 채워진다.
    필드명·값 규약은 base.yaml §① "재료 버전 6칸" 과 동일(`v1`/`v2` …).
    """
    corpus: str = "UNKNOWN"
    preprocess: str = "UNKNOWN"
    table: str = "UNKNOWN"
    index: str = "UNKNOWN"
    evalset: str = "UNKNOWN"
    scorer: str = "v1"  # 채점기 버전 — 김하루가 채점기 확정/변경 시 올린다(base.yaml §①)


@dataclass(frozen=True)
class GraderConfig:
    schema_version: str
    judge: JudgeConfig
    runtime: RuntimeConfig
    cache: CacheConfig
    retrieval: RetrievalConfig
    grading: GradingConfig
    gate: GateConfig
    provenance: ProvenanceDefaults
    out_dir: str = "artifacts"
    use_stub_judge: bool = True  # ★dev/CI 값싼 검사 전용. tier=final 판정에는 쓰지 않는다.
    ci_subset_size: int = 20
    dev_subset_size: int = 60
    enforce_gates: bool = True


def _judge_prompt_path(entry: dict, base_dir: Path) -> Path:
    p = Path(entry["prompt_file"])
    return p if p.is_absolute() else (base_dir / p)


# 팀 실험 인프라 규약 config/base.yaml §① "재료 버전 6칸".
_BASE_YAML_CANDIDATES = ("config/base.yaml", "base.yaml", "../config/base.yaml")
_ASSET_KEYS = ("corpus", "preprocess", "table", "index", "evalset", "scorer")


def read_base_yaml_assets(path: str | Path | None = None) -> dict[str, str]:
    """team 규약 config/base.yaml §① 6칸을 읽는다. 없으면 빈 dict.
    grader 는 이 값을 configs/default.yaml 의 provenance 블록보다 우선한다(단일 출처)."""
    paths = [Path(path)] if path else [Path(p) for p in _BASE_YAML_CANDIDATES]
    for p in paths:
        if not p.exists():
            continue
        raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        return {k: str(raw[k]) for k in _ASSET_KEYS
                if isinstance(raw.get(k), (str, int)) and str(raw[k]).lower() not in ("null", "none", "todo", "")}
    return {}


def _deep_merge(base: dict, overlay: dict) -> dict:
    """overlay 의 키만 base 위에 덮는다(중첩 dict 는 재귀). 실험별 config 규약."""
    out = dict(base)
    for k, v in overlay.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | Path = "configs/default.yaml",
                overlay: str | Path | None = None) -> GraderConfig:
    """configs/default.yaml 을 읽고, overlay(configs/ci.yaml 등)가 있으면 바뀐 줄만 덮는다."""
    load_dotenv()
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if overlay is not None:
        ov = yaml.safe_load(Path(overlay).read_text(encoding="utf-8")) or {}
        raw = _deep_merge(raw, ov)
    base_dir = path.resolve().parent.parent  # configs/default.yaml -> 프로젝트 루트

    judge_raw = raw["judge"]
    judge_entries = judge_raw["judges"]

    retrieval_raw = raw.get("retrieval", {})
    grading_raw = raw.get("grading", {})
    gate_raw = raw.get("gate", {})
    provenance_raw = raw.get("provenance", {})

    return GraderConfig(
        schema_version=raw["schema_version"],
        judge=JudgeConfig(
            names=tuple(x["name"] for x in judge_entries),
            prompt_files={x["name"]: _judge_prompt_path(x, base_dir) for x in judge_entries},
            metrics=tuple(judge_raw.get("metrics", [])),
            whole_item_metrics=tuple(judge_raw.get("whole_item_metrics", [])),
            model=judge_raw.get("model", "stub"),
            family=judge_raw.get("family", "none"),
            tier=judge_raw.get("tier", "dev"),
            verification_path=judge_raw.get("verification_path"),
            min_agreement=float(judge_raw.get("min_agreement", 0.80)),
        ),
        runtime=RuntimeConfig(
            concurrency=int(os.getenv("GRADER_CONCURRENCY", raw["runtime"]["concurrency"])),
            timeout_seconds=int(os.getenv("GRADER_TIMEOUT_SECONDS", raw["runtime"]["timeout_seconds"])),
            temperature=float(os.getenv("GRADER_TEMPERATURE", raw["runtime"]["temperature"])),
        ),
        cache=CacheConfig(
            enabled=bool(raw["cache"]["enabled"]),
            directory=Path(raw["cache"]["directory"]),
        ),
        retrieval=RetrievalConfig(
            retrieval_k=int(retrieval_raw.get("retrieval_k", 20)),
            reranker_k=int(retrieval_raw.get("reranker_k", 10)),
            context_k=int(retrieval_raw.get("context_k", 5)),
            precision=retrieval_raw.get("precision", "ref_no"),
            eval_k=tuple(int(k) for k in retrieval_raw.get("eval_k", (3, 5))),
        ),
        grading=GradingConfig(
            allow_partial=bool(grading_raw.get("allow_partial", False)),
            docset_partial_credit=bool(grading_raw.get("docset_partial_credit", True)),
            miss_weight=float(grading_raw.get("miss_weight", 0.7)),
            require_table_format=bool(grading_raw.get("require_table_format", True)),
            grade_citations=bool(grading_raw.get("grade_citations", True)),
        ),
        gate=GateConfig(
            severity_gate=gate_raw.get("severity_gate", {"critical": 0.85}),
            column_severity=gate_raw.get("column_severity", {}),
            task_quota=gate_raw.get("task_quota", {}),
            data_thresholds=gate_raw.get("data_thresholds", {}),
            data_warn_thresholds=gate_raw.get("data_warn_thresholds", {}),
            doc_org=gate_raw.get("doc_org", {}),
            task_weight=gate_raw.get("task_weight", {}),
            field_tag_weight=gate_raw.get("field_tag_weight", {}),
        ),
        provenance=ProvenanceDefaults(
            corpus=provenance_raw.get("corpus", "UNKNOWN"),
            preprocess=provenance_raw.get("preprocess", "UNKNOWN"),
            table=provenance_raw.get("table", "UNKNOWN"),
            index=provenance_raw.get("index", "UNKNOWN"),
            evalset=provenance_raw.get("evalset", "UNKNOWN"),
            scorer=provenance_raw.get("scorer", "v1"),
        ),
        out_dir=raw.get("out_dir", "artifacts"),
        use_stub_judge=bool(raw.get("use_stub_judge", True)),
        ci_subset_size=int(raw.get("ci_subset_size", 20)),
        dev_subset_size=int(raw.get("dev_subset_size", 60)),
        enforce_gates=bool(raw.get("enforce_gates", True)),
    )
