"""field_tag severity weighting — 팀 확정 critical : major : minor = 5 : 3 : 2.

★critical 의 '기권 관련 중요 문항' 역할(critical_abstain)과 'severity 가중치' 역할은
서로 **별개 축**이다 — 하나가 다른 하나를 대체하지 않는다.
"""

from grader.config import load_config
from grader.diagnostics import full_report, severity_weighted_score
from grader.models import EvaluationItem, ModelResponse
from grader.task_scoring import grade_abstention


def test_config_carries_5_3_2_weights():
    cfg = load_config("config/grader.yaml")
    assert cfg.gate.field_tag_weight == {"critical": 5, "major": 3, "minor": 2}


def test_weighted_score_uses_5_3_2_from_config():
    cfg = load_config("config/grader.yaml")
    rows = [
        {"field_tag": "critical", "score": 0.0},   # 치명 오답
        {"field_tag": "major", "score": 1.0},
        {"field_tag": "minor", "score": 1.0},
    ]
    score, note = severity_weighted_score(rows, cfg.gate.field_tag_weight)
    # (0*5 + 1*3 + 1*2) / (5+3+2) = 5/10 = 0.5
    assert score == 0.5
    assert "gate.field_tag_weight" in note


def test_full_report_applies_config_weights():
    cfg = load_config("config/grader.yaml")
    rows = [
        {"task_type": "extraction", "field_tag": "critical", "score": 0.0,
         "final_status": "FAIL-content", "abstention": {"abstention_kind": "ok"}},
        {"task_type": "extraction", "field_tag": "minor", "score": 1.0,
         "final_status": "PASS", "abstention": {"abstention_kind": "ok"}},
    ]
    rep = full_report(rows, field_tag_weight=cfg.gate.field_tag_weight)
    # (0*5 + 1*2) / (5+2) = 2/7
    assert rep["severity_weighted_score"] == round(2 / 7, 4)
    # ②③ 축(분리 집계/게이트)은 그대로 나란히 유지된다
    assert "by_field_tag" in rep["severity"]


def test_critical_abstain_axis_is_independent_of_weighting():
    """critical 문항이 기권하면 critical_abstain(정상 동작 후보)로 잡히고,
    이건 severity 가중치와 무관하게 별도로 기록된다."""
    it = EvaluationItem(id="C", question="q", task_type="extraction",
                        answer_type="value", answer_raw="5억원", field_tag="critical")
    r = grade_abstention(it, ModelResponse(id="C", answer="", abstained=True))
    assert r.abstention_kind == "critical_abstain"  # 기권 축은 그대로 살아있다
