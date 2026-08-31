from grader.regression import (
    RegressionPolicy,
    attribute_change,
    judge_regression,
    measure_variance,
    pp_per_item,
)


def test_variance_needs_two_runs():
    assert measure_variance([{"overall_score": 0.6}])["usable"] is False


def test_regression_without_variance_is_undecidable():
    """★변동 폭 없이 CI를 켜지 말 것 — 판정 자체가 성립하지 않는다."""
    out = judge_regression({"overall_score": 0.70}, {"overall_score": 0.50},
                           {"metrics": {}}, RegressionPolicy())
    assert out["status"] == "PASS"
    assert any("변동 폭 미측정" in s.get("reason", "") for s in out["skipped"])


def test_pp_per_item_matches_임현진_2_17_confirmed_quota():
    """2026-08-27 임현진 2-17 확정: 선별형 4%p(25문항) / 추출형 약 6.7%p(15문항) /
    QA형 10%p(10문항). pp_per_item()은 이미 일반화돼 있으므로 숫자만 확인한다."""
    assert pp_per_item(25) == 4.0
    assert pp_per_item(15) == 6.67
    assert pp_per_item(10) == 10.0


def test_gate_vs_warning():
    var = {"metrics": {"field_tag.critical": {"sd": 0.02}, "overall_score": {"sd": 0.02}}}
    out = judge_regression(
        {"field_tag.critical": 0.95, "overall_score": 0.70},
        {"field_tag.critical": 0.80, "overall_score": 0.65},
        var, RegressionPolicy(), cell_sizes={"field_tag.critical": 20, "overall_score": 60})
    assert out["status"] == "FAIL"
    assert out["gate_failures"]
    assert out["warnings"]


def test_small_cells_are_not_alarmed():
    var = {"metrics": {"field_tag.critical": {"sd": 0.01}}}
    out = judge_regression({"field_tag.critical": 0.95}, {"field_tag.critical": 0.50},
                           var, RegressionPolicy(), cell_sizes={"field_tag.critical": 2})
    assert out["status"] == "PASS"
    assert out["skipped"]


def test_pp_per_item():
    assert pp_per_item(50) == 2.0


def test_attribute_change_detects_mixed_experiment():
    prev = {"corpus": "c1", "index": "i1"}
    cur = {"corpus": "c2", "index": "i2"}
    out = attribute_change(prev, cur)
    assert out["clean_comparison"] is False  # 코퍼스 + 검색 인덱스 = 2축
    assert out["corpus_changed_flag"] is True
    assert out["index_changed_flag"] is True


def test_attribute_change_single_axis_is_clean():
    prev = {"corpus": "c1", "scorer": "g1"}
    cur = {"corpus": "c1", "scorer": "g2"}
    out = attribute_change(prev, cur)
    assert out["clean_comparison"] is True
    assert out["axes"] == ["채점기"]
    assert out["scorer_changed_flag"] is True


def test_attribute_change_compares_all_six_provenance_axes():
    """★회귀 원인 분리는 6-자산 provenance 6축을 전부 diff 해야 한다 —
    manifest.provenance 가 내는 6개 필드와 1:1."""
    from grader.models import PROVENANCE_FIELDS

    prev = {k: f"{k}-A" for k in PROVENANCE_FIELDS}
    cur = {k: f"{k}-B" for k in PROVENANCE_FIELDS}
    out = attribute_change(prev, cur)

    assert set(out["axes_checked"]) == set(PROVENANCE_FIELDS)
    assert len(out["axes_checked"]) == 6
    assert set(out["changed_keys"]) == set(PROVENANCE_FIELDS)  # 6축 전부 감지
    assert out["clean_comparison"] is False


def test_attribute_change_all_unknown_is_no_false_positive():
    """6축이 전부 'UNKNOWN'(미상)이어도 '바뀌었다'로 잡지 않는다."""
    from grader.models import PROVENANCE_FIELDS

    same = {k: "UNKNOWN" for k in PROVENANCE_FIELDS}
    out = attribute_change(same, dict(same))
    assert out["changed"] == []
    assert out["changed_keys"] == []
    assert out["clean_comparison"] is True


def test_absolute_gate_fails_even_without_variance():
    """★절대 기준선은 변동 폭 미측정이어도 적용된다 — 예전엔 skipped 로 새어나갔다."""
    out = judge_regression(
        {"field_tag.critical": 0.95}, {"field_tag.critical": 0.50},
        {"metrics": {}},  # 변동 폭 없음
        RegressionPolicy(),  # absolute_gates = {"field_tag.critical": 0.85}
    )
    assert out["status"] == "FAIL"
    assert any(g["metric"] == "field_tag.critical" for g in out["gate_failures"])
    assert any("절대 기준선" in g.get("reason", "") for g in out["gate_failures"])
