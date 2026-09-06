from grader.diagnostics import ExperimentRecord, check_contamination


def test_declaration_without_target_metrics_is_rejected():
    record = ExperimentRecord()
    out = check_contamination(record, {}, {}, {}, {})
    assert out["verdict"] == "REJECT"
    assert out["declaration_errors"]


def test_grader_only_change_with_rising_target_metric_is_suspicious():
    """Judge/평가셋 표현에 맞춘 prompt 변경으로 점수가 오르면 자동으로
    '시스템 개선'으로 간주하지 않는다(4-9)."""
    record = ExperimentRecord(target_metrics=["extraction.overall_accuracy"])
    prev = {"scorer": "s1", "corpus": "c1"}
    cur = {"scorer": "s2", "corpus": "c1"}  # 채점기(코드/심판 프롬프트)만 바뀜
    baseline = {"extraction.overall_accuracy": 0.70}
    current = {"extraction.overall_accuracy": 0.85}
    out = check_contamination(record, prev, cur, baseline, current)
    assert out["grader_only_changed"] is True
    assert out["suspicious_findings"]
    assert out["verdict"] == "REJECT"


def test_system_change_with_rising_target_metric_is_ok():
    record = ExperimentRecord(target_metrics=["extraction.overall_accuracy"])
    prev = {"scorer": "s1", "index": "i1"}
    cur = {"scorer": "s1", "index": "i2"}  # 실제 RAG 검색 인덱스가 바뀜
    baseline = {"extraction.overall_accuracy": 0.70}
    current = {"extraction.overall_accuracy": 0.85}
    out = check_contamination(record, prev, cur, baseline, current)
    assert out["grader_only_changed"] is False
    assert not out["suspicious_findings"]
    assert out["verdict"] == "OK"


def test_protected_metric_violation_is_reported():
    record = ExperimentRecord(target_metrics=["overall_score"], protected_metrics=["field_tag.critical"])
    out = check_contamination(record, {}, {}, {"field_tag.critical": 0.9, "overall_score": 0.6},
                              {"field_tag.critical": 0.5, "overall_score": 0.7})
    assert out["protected_metric_violations"]


def test_protected_metric_expected_down_is_not_violated_by_a_drop():
    """expected_direction='down'이면 하락은 기대한 방향 — 위반이 아니다.
    (예전엔 flat/down을 한 조건으로 묶어 하락을 무조건 위반으로 표시했다.)"""
    record = ExperimentRecord(
        target_metrics=["overall_score"],
        protected_metrics=["over_refusal_rate"],
        expected_direction={"over_refusal_rate": "down"},
    )
    out = check_contamination(record, {}, {}, {"over_refusal_rate": 0.20, "overall_score": 0.6},
                              {"over_refusal_rate": 0.10, "overall_score": 0.6})
    assert out["protected_metric_violations"] == []
    # 반대로 올라가면(내려가야 하는데) 위반으로 잡힌다
    out2 = check_contamination(record, {}, {}, {"over_refusal_rate": 0.10, "overall_score": 0.6},
                               {"over_refusal_rate": 0.25, "overall_score": 0.6})
    assert out2["protected_metric_violations"]
