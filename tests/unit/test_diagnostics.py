from grader.diagnostics import (
    abstention_report,
    cell_report,
    diagnose,
    format_report,
    full_report,
    integrated_score,
    render_table,
    severity_report,
    severity_weighted_score,
)


def _rows():
    return [
        {"id": "1", "task_type": "extraction", "field_tag": "critical", "difficulty": "hard",
         "score": 0.0, "final_status": "FAIL-content", "abstention": {"abstention_kind": "ok"}},
        {"id": "2", "task_type": "extraction", "field_tag": "minor", "difficulty": "easy",
         "score": 1.0, "final_status": "PASS", "abstention": {"abstention_kind": "ok"}},
        {"id": "3", "task_type": "selection", "field_tag": "major", "difficulty": "medium",
         "score": 0.5, "final_status": "PENDING_THRESHOLD", "abstention": {"abstention_kind": "hallucination"}},
        {"id": "4", "task_type": "qa", "field_tag": None, "difficulty": "medium",
         "score": 1.0, "final_status": "FAIL-format", "abstention": {"abstention_kind": "ok"}},
    ]


def test_cell_report_warns_small_cells():
    rep = cell_report(_rows(), "task_type", min_cell=2)
    assert rep["qa"]["reliable"] is False
    assert rep["qa"]["warning"]


def test_severity_report_gate_fail_when_critical_below_threshold():
    rep = severity_report(_rows(), gate={"critical": 0.85})
    assert rep["gate"]["critical"]["status"] == "FAIL"


def test_abstention_report_counts_hallucination():
    rep = abstention_report(_rows())
    assert rep["counts"]["hallucination"] == 1


def test_format_report_counts_format_failures():
    rep = format_report(_rows())
    assert rep["format_failure_rate"] == 0.25


def test_full_report_has_main_axes():
    rep = full_report(_rows())
    assert rep["n_items"] == 4
    assert "selection" in rep["main"]["by_task_type"] or "selection" in rep["main"]["primary"]


def test_render_table_mentions_circularity():
    assert "순환 경보" in render_table()


def test_diagnose_flags_circularity():
    values = {"extraction_accuracy": 0.70, "selection_score": 0.95}
    out = diagnose(values)
    assert any("순환 경보" in d["suspect"] for d in out)


# ------------------------------------------------------------------ 2-16 통합 점수


def test_integrated_score_is_none_without_weights():
    """★가중치(1-2) 없이 감으로 채우지 않는다 — 산출 방식 확정과 값 부재는 다르다."""
    by_task = {"selection": {"score": 0.8, "n": 10}, "extraction": {"score": 0.6, "n": 10}}
    score, note = integrated_score(by_task, {})
    assert score is None
    assert "산출 방식은 확정" in note


def test_integrated_score_weighted_average():
    by_task = {
        "selection": {"score": 0.8, "n": 25},
        "extraction": {"score": 0.6, "n": 15},
        "qa": {"score": 1.0, "n": 10},
    }
    score, note = integrated_score(by_task, {"selection": 0.4, "extraction": 0.35, "qa": 0.25})
    # 0.8*0.4 + 0.6*0.35 + 1.0*0.25 = 0.32 + 0.21 + 0.25 = 0.78
    assert score == 0.78
    assert "gate.task_weight" in note


def test_full_report_shows_integrated_score_alongside_task_scores():
    """2-16: 통합 점수는 태스크별 3개 점수를 대체하지 않고 항상 나란히 있어야 한다."""
    rep = full_report(_rows(), task_weight={"extraction": 0.6, "selection": 0.4})
    assert "by_task_type" in rep["main"]
    assert "integrated_score" in rep["main"]
    assert rep["main"]["integrated_score"] is not None
    # 나란히 — 태스크별 점수가 사라지지 않았는지 확인
    assert "extraction" in rep["main"]["by_task_type"]


# ------------------------------------------------------------------ 3-3-2 ① field_tag 가중 평균 (2026-08-28 재확인)


def test_severity_weighted_score_is_none_without_weights():
    """★가중치 없이 감으로 채우지 않는다 — 방식 재도입 확정과 값 부재는 다르다(2-16과 동일 원칙)."""
    score, note = severity_weighted_score(_rows(), {})
    assert score is None
    assert "3-3-2" in note


def test_severity_weighted_score_weighs_critical_more_than_minor():
    """critical(score=0)·minor(score=1)·major(score=0.5) — critical 오류가
    minor보다 가중 평균을 더 크게 끌어내려야 한다."""
    score, note = severity_weighted_score(_rows(), {"critical": 3, "major": 2, "minor": 1})
    # critical(0*3) + major(0.5*2) + minor(1*1) = 0 + 1 + 1 = 2 / (3+2+1=6) = 0.3333
    assert score == 0.3333
    assert "gate.field_tag_weight" in note
    # field_tag=None인 문항(4번)은 제외됐다는 사실이 note에 남아야 한다
    assert "제외된 문항" in note


def test_severity_weighted_score_excludes_items_without_matching_tag_weight():
    """gate.field_tag_weight에 없는 태그의 문항은 가중 평균에서 조용히 빠진다."""
    score, note = severity_weighted_score(_rows(), {"critical": 3})
    # critical(0*3) / 3 = 0.0 — major/minor/None 은 전부 제외
    assert score == 0.0


def test_severity_weighted_score_all_excluded_returns_none():
    score, note = severity_weighted_score(_rows(), {"nonexistent_tag": 5})
    assert score is None
    assert "겹치는 항목이 없다" in note


def test_full_report_shows_severity_weighted_score_alongside_gate():
    """severity_weighted_score가 있어도 severity(②③)를 대체하지 않고 나란히 있어야 한다."""
    rep = full_report(_rows(), gate={"critical": 0.85},
                      field_tag_weight={"critical": 3, "major": 2, "minor": 1})
    assert rep["severity"]["gate"]["critical"]["status"] == "FAIL"  # ②③ 그대로 유지
    assert rep["severity_weighted_score"] == 0.3333  # ①도 함께 계산됨
