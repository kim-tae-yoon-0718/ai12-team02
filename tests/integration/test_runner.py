import json
from dataclasses import replace
from pathlib import Path

from grader.config import load_config
from grader.models import EvaluationItem, ModelResponse
from grader.prompts import PromptRepository
from grader.providers import MockJudgeProvider
from grader.runner import GraderRunner, execute, stratified_subset


def test_runner_with_mock(tmp_path: Path):
    prompt = tmp_path / "judge.txt"
    prompt.write_text(
        "Question: {{question}}\nAnswer: {{answer}}\nGround Truth: {{ground_truth}}",
        encoding="utf-8",
    )

    cfg = load_config("configs/default.yaml")
    # 테스트에서는 실제 provider(mock)가 호출되는 경로를 보기 위해 StubJudge를 끈다.
    cfg = replace(
        cfg,
        use_stub_judge=False,
        judge=replace(
            cfg.judge,
            names=("judge_test",),
            prompt_files={"judge_test": prompt},
            whole_item_metrics=("judge_test",),
            metrics=("correctness",),
            model="test-model",
        ),
    )

    runner = GraderRunner(
        config=cfg,
        provider=MockJudgeProvider(),
        prompt_repo=PromptRepository(cfg.judge.prompt_files),
        corpus="corpus-test-001",
        table="extract-test-001",
    )

    evaluation = EvaluationItem.model_validate(
        {
            "id": "Q0001",
            "question": "예산은 얼마인가?",
            "task_type": "extraction",
            "answer_type": "value",
            "answer_raw": "5억원",
        }
    )
    response = ModelResponse(id="Q0001", answer="5억원", contexts=[])

    result = runner.run_one(evaluation, response, mode="development")
    assert result.id == "Q0001"
    assert result.provider == "test-model"
    assert len(result.judge_results) == 1
    assert result.task_score.score == 1.0
    assert result.final_status == "PASS"


def test_stratified_subset_keeps_every_task_type():
    """무작위로 뽑으면 selection이 0개인 부분집합이 나올 수 있고,
    그러면 CI가 selection 회귀를 영영 못 잡는다."""
    items = [
        EvaluationItem(id=f"s{i}", question="q", task_type="selection", answer_type="document_set",
                      answer_raw=[])
        for i in range(20)
    ] + [
        EvaluationItem(id=f"e{i}", question="q", task_type="extraction", answer_type="value",
                      answer_raw="v")
        for i in range(3)
    ]
    sub = stratified_subset(items, 6)
    task_types = {i.task_type for i in sub}
    assert "extraction" in task_types
    assert "selection" in task_types


def test_execute_checks_mode_smoke(tmp_path: Path):
    """1~3층만 도는 값싼 검사 — data_manifest/extraction_audit 없이도 SKIP으로 통과해야 한다."""
    cfg = load_config("configs/default.yaml")
    cfg = replace(cfg, use_stub_judge=True)
    runner = GraderRunner(
        config=cfg,
        provider=MockJudgeProvider(),
        prompt_repo=PromptRepository(cfg.judge.prompt_files),
        corpus="corpus-test", table="ext-test",
    )
    code, report = execute(
        runner,
        evaluation_set_path="tests/fixtures/evaluation_set.jsonl",
        responses_path=None,
        mode="checks",
        allow_final=False,
        runner_name="pytest",
        out_dir=str(tmp_path),
    )
    assert code == 0
    assert report["layers"][0]["status"] == "SKIP"  # data_manifest 없음
    assert report["layers"][1]["status"] == "PASS"  # 평가셋 스키마 자체는 유효


def _passing_audit(tmp_path: Path) -> str:
    """최종 모드는 독립 원문 표본 대조(--extraction-audit) 필수. 게이트를 통과하는 최소 파일."""
    p = tmp_path / "audit.jsonl"
    p.write_text(json.dumps({
        "document_id": "D1", "column": "budget",
        "gold_state": "value", "gold_value": "5억원",
        "pred_state": "value", "pred_value": "5억원",
    }, ensure_ascii=False) + "\n", encoding="utf-8")
    return str(p)


def test_execute_final_mode_requires_independent_extraction_audit(tmp_path: Path):
    """★순환 방지(5): 최종 평가는 독립 원문 표본 대조 없이 허용되지 않는다."""
    cfg = replace(load_config("configs/default.yaml"), use_stub_judge=True)
    runner = GraderRunner(config=cfg, provider=MockJudgeProvider(),
                          prompt_repo=PromptRepository(cfg.judge.prompt_files),
                          corpus="corpus-test", table="ext-test")
    code, report = execute(
        runner, evaluation_set_path="tests/fixtures/evaluation_set.jsonl",
        responses_path="tests/fixtures/model_responses.jsonl",
        mode="final", allow_final=True, runner_name="pytest",
        out_dir=str(tmp_path),  # extraction_audit_path 주지 않음 → L3 SKIP
    )
    from grader.runner import EXIT_CONFIG
    assert code == EXIT_CONFIG
    assert "표본 대조" in report["message"]


def test_execute_final_mode_rejects_stub_judge(tmp_path: Path):
    """★use_stub_judge=true 로는 tier=final을 실행할 수 없다 — StubJudge는 심판이 아니다.
    (독립 표본 대조는 통과시킨 뒤 심판 가드가 걸리는지 확인)"""
    cfg = load_config("configs/default.yaml")
    cfg = replace(cfg, use_stub_judge=True)
    runner = GraderRunner(
        config=cfg,
        provider=MockJudgeProvider(),
        prompt_repo=PromptRepository(cfg.judge.prompt_files),
        corpus="corpus-test", table="ext-test",
    )
    code, report = execute(
        runner,
        evaluation_set_path="tests/fixtures/evaluation_set.jsonl",
        responses_path="tests/fixtures/model_responses.jsonl",
        mode="final",
        allow_final=True,
        runner_name="pytest",
        extraction_audit_path=_passing_audit(tmp_path),
        out_dir=str(tmp_path),
    )
    from grader.runner import EXIT_CONFIG
    assert code == EXIT_CONFIG
    assert "StubJudge" in report["message"]


def test_layer1_warning_does_not_block_execution(tmp_path: Path):
    """2026-08-27 박예진 확정 — 표 0개 문서는 경고이지 게이트가 아니다. 1층이 FAIL로
    바뀌면 안 되고, 뒤 층(checks 모드면 2~3층)도 계속 돌아야 한다."""
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "doc_count": 10, "load_rate": 1.0, "duplicate_doc_ids": [],
        "corpus_version": "corpus-1", "extraction_table_version": "extract-1",
        "zero_table_doc_count": 3,
    }, ensure_ascii=False), encoding="utf-8")

    cfg = load_config("configs/default.yaml")
    from dataclasses import replace as _replace
    cfg = _replace(cfg, use_stub_judge=True, gate=_replace(
        cfg.gate, data_warn_thresholds={"max_zero_table_docs": 0}))
    runner = GraderRunner(
        config=cfg, provider=MockJudgeProvider(),
        prompt_repo=PromptRepository(cfg.judge.prompt_files),
        corpus="corpus-test", table="ext-test",
    )
    code, report = execute(
        runner,
        evaluation_set_path="tests/fixtures/evaluation_set.jsonl",
        responses_path=None, mode="checks", allow_final=False, runner_name="pytest",
        data_manifest_path=str(manifest), out_dir=str(tmp_path),
    )
    L1 = report["layers"][0]
    assert L1["status"] == "PASS"  # 경고는 status를 FAIL로 바꾸지 않는다
    assert any("표 0개" in w for w in L1["warnings"])
    assert code == 0  # 뒤 층까지 정상 진행


def test_ci_mode_without_practice_set_hard_fails(tmp_path: Path):
    """★임현진 확정 방향 — practice 세트가 없으면 최종셋으로 조용히 폴백하지 않고
    exit 1 로 즉시 실패한다(유출은 경고로 뭉갤 문제가 아님)."""
    cfg = load_config("configs/default.yaml")
    cfg = replace(cfg, use_stub_judge=True)
    runner = GraderRunner(
        config=cfg, provider=MockJudgeProvider(),
        prompt_repo=PromptRepository(cfg.judge.prompt_files),
        corpus="corpus-test", table="ext-test",
    )
    code, report = execute(
        runner,
        evaluation_set_path="tests/fixtures/evaluation_set.jsonl",
        responses_path="tests/fixtures/model_responses.jsonl",
        mode="ci", allow_final=False, runner_name="pytest",
        out_dir=str(tmp_path),
    )
    from grader.runner import EXIT_FAIL
    assert code == EXIT_FAIL
    assert "practice" in report["message"]
    # 4/5층(성능 평가)까지 가지 않았다 — 최종셋이 노출되지 않았다
    assert not any(L.get("layer") in (4, 5) for L in report["layers"])


def test_ci_mode_uses_practice_set_when_provided(tmp_path: Path):
    """practice_set이 있으면 최종 50문항이 아니라 그 파일에서 채점 대상을 가져온다."""
    dev_set = tmp_path / "practice_items.jsonl"
    dev_set.write_text(
        json.dumps({
            "id": "DEV0001", "question": "개발 확인용", "task_type": "extraction",
            "answer_type": "value",
            "answer_raw": "1억원",
        }, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    responses = tmp_path / "dev_responses.jsonl"
    responses.write_text(
        json.dumps({"id": "DEV0001", "answer": "1억원"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    cfg = load_config("configs/default.yaml")
    cfg = replace(cfg, use_stub_judge=True)
    runner = GraderRunner(
        config=cfg, provider=MockJudgeProvider(),
        prompt_repo=PromptRepository(cfg.judge.prompt_files),
        corpus="corpus-test", table="ext-test",
    )
    code, report = execute(
        runner,
        evaluation_set_path="tests/fixtures/evaluation_set.jsonl",  # 최종셋(구조 검증용)
        responses_path=str(responses),
        mode="ci", allow_final=False, runner_name="pytest",
        practice_set_path=str(dev_set),
        out_dir=str(tmp_path / "out"),
    )
    L4 = report["layers"][-1]
    assert L4["layer"] == 4
    per_item_path = tmp_path / "out" / "per_item.jsonl"
    ids = [json.loads(line)["id"] for line in per_item_path.read_text(encoding="utf-8").splitlines()]
    assert ids == ["DEV0001"]  # 최종셋(Q0001)이 아니라 practice_set에서 왔는지 확인


def test_execute_final_mode_without_allow_final_is_blocked(tmp_path: Path):
    cfg = load_config("configs/default.yaml")
    runner = GraderRunner(
        config=cfg, provider=MockJudgeProvider(),
        prompt_repo=PromptRepository(cfg.judge.prompt_files),
        corpus="corpus-test", table="ext-test",
    )
    code, report = execute(
        runner, evaluation_set_path="tests/fixtures/evaluation_set.jsonl",
        responses_path=None, mode="final", allow_final=False, runner_name="pytest",
        out_dir=str(tmp_path),
    )
    from grader.runner import EXIT_CONFIG
    assert code == EXIT_CONFIG
    assert "누수 방지" in report["message"]
