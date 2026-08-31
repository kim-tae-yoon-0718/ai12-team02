"""
grader.contamination — 실험 오염 방지 (4-9, 김태윤 팀장 합의)

Judge를 바꿔서 점수가 오르거나, 평가셋 표현에 맞춰 채점 prompt를 고쳐 점수를
올리는 것은 **시스템 개선으로 인정하지 않는다.** 이 모듈은 그 원칙을 실험 기록에
강제로 남기게 한다 — 기록이 없으면 "무엇이 개선됐다고 주장하는지"조차 알 수 없다.

regression.attribute_change(3-17)가 "무엇이 바뀌었는가"를 가른다면,
이 모듈은 "그 변경이 채점기 자체를 유리하게 만든 것은 아닌가"를 가른다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .regression import attribute_change

# 팀 확정 6-자산 축 중 "채점기 자신"에 해당하는 축(provenance.scorer = 채점기 코드 +
# 심판 프롬프트 묶음). 나머지 5축(corpus·preprocess·table·index·evalset)은 "실제
# 데이터/시스템 변경"이라 오염이 아니다.
GRADER_AXIS = "채점기"


@dataclass
class ExperimentRecord:
    """실험을 시작하기 전에 선언해야 하는 것.

    target_metrics    : 이 실험이 올리려는 지표 (예: ["extraction.overall_accuracy"])
    protected_metrics : 이 실험이 건드리면 안 되는 지표 (예: ["field_tag.critical"])
    expected_direction: 지표별 기대 방향 {"metric": "up"|"down"|"flat"} — 3-16과 연결
    """

    target_metrics: list[str] = field(default_factory=list)
    protected_metrics: list[str] = field(default_factory=list)
    expected_direction: dict[str, str] = field(default_factory=dict)
    note: str = ""

    def validate(self) -> list[str]:
        errs = []
        if not self.target_metrics:
            errs.append("target_metrics 없음 — 이 실험이 무엇을 올리려는지 선언되지 않음")
        overlap = set(self.target_metrics) & set(self.protected_metrics)
        if overlap:
            errs.append(f"target_metrics/protected_metrics 가 겹침: {sorted(overlap)}")
        return errs


def check_contamination(record: ExperimentRecord, prev_provenance: dict, cur_provenance: dict,
                        baseline: dict, current: dict) -> dict:
    """채점기 자체가 유리해지는 변경과 실제 시스템 개선을 분리한다.

    ★규칙: 데이터/시스템 5축(corpus·preprocess·table·index·evalset)은 그대로인데
      provenance.scorer(채점기 코드 + 심판 프롬프트) 축만 바뀌었고, 그 상태에서
      target_metrics 가 올랐다면 **오염 의심**으로 표시한다. 이 경우 실험을 "개선"으로
      보고하는 것을 REJECT 한다.
    """
    cause = attribute_change(prev_provenance, cur_provenance)
    grader_only_changed = bool(cause["axes"]) and all(axis == GRADER_AXIS for axis in cause["axes"])

    suspicious_findings = []
    for m in record.target_metrics:
        base, cur = baseline.get(m), current.get(m)
        if base is None or cur is None:
            continue
        if cur > base and grader_only_changed:
            suspicious_findings.append({
                "metric": m, "baseline": base, "current": cur,
                "message": (
                    f"★{m} 상승이 채점기/심판 프롬프트 변경과만 겹친다 — 시스템 개선으로 "
                    "인정하지 않는다(4-9). 시스템 변경 없이 이 실험을 '개선'으로 보고하지 말 것."
                ),
            })

    protected_violations = []
    for m in record.protected_metrics:
        base, cur = baseline.get(m), current.get(m)
        if base is None or cur is None:
            continue
        want = record.expected_direction.get(m, "flat")
        if want == "up":
            broke = cur < base            # 올라야 하는데 내려감
        elif want == "down":
            broke = cur > base            # 내려가야 하는데 올라감
        else:                             # "flat": 보호 지표는 하락만 위반으로 본다
            broke = cur < base            # (이 도메인의 걱정은 회귀 — 상승은 위반이 아님)
        if broke:
            protected_violations.append({"metric": m, "baseline": base, "current": cur,
                                         "expected_direction": want})

    declaration_errors = record.validate()
    verdict = "REJECT" if (declaration_errors or (suspicious_findings and grader_only_changed)) else "OK"

    return {
        "declared": {"target_metrics": record.target_metrics,
                     "protected_metrics": record.protected_metrics,
                     "expected_direction": record.expected_direction},
        "declaration_errors": declaration_errors,
        "cause_attribution": cause,
        "grader_only_changed": grader_only_changed,
        "suspicious_findings": suspicious_findings,
        "protected_metric_violations": protected_violations,
        "verdict": verdict,
        "note": record.note,
    }
