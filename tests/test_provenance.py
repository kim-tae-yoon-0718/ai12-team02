"""팀 확정 6-자산 provenance 체계 (3-11 / 3-13-1 / 3-17).

평가 결과 하나가 '어떤 자산 조합'에서 나왔는지 정확히 6개 필드로 못박는다:
  corpus     원문 코퍼스
  preprocess 전처리 파이프라인(파싱·정제·청킹)
  table      구조화 추출 테이블(1-12-2)
  index      검색 인덱스(평가 대상 RAG 검색계, 4번)
  evalset    평가셋(Ground Truth)
  scorer     채점기(이 저장소 코드 + 심판 프롬프트 묶음)

확인 항목:
  - 6개 필드가 모두 결과(EvaluationResult / report.json / per_item.jsonl)에 존재하는가
  - 정확한 필드명이 맞는가
  - 입력값이 report / per_item 에 자동 복사되는가
  - 값이 UNKNOWN 일 때도 누락되지 않는가
  - 회귀 비교(attribute_change)가 6축을 제대로 비교하는가
"""

import json
from dataclasses import replace
from pathlib import Path

from grader.config import load_config
from grader.models import (
    PROVENANCE_FIELDS,
    EvaluationResult,
    Provenance,
    canonical_scorer_version,
)
from grader.prompts import PromptRepository
from grader.providers import MockJudgeProvider
from grader.regression import CHANGE_AXES, attribute_change
from grader.runner import GraderRunner, execute

EXPECTED_FIELDS = {"corpus", "preprocess", "table", "index", "evalset", "scorer"}


# ------------------------------------------------------------------ 필드명 / 개수

def test_provenance_has_exactly_six_fields_with_exact_names():
    dumped = Provenance().model_dump()
    assert set(dumped) == EXPECTED_FIELDS
    assert len(dumped) == 6
    assert tuple(PROVENANCE_FIELDS) == (
        "corpus", "preprocess", "table", "index", "evalset", "scorer",
    )
    # regression 축 목록도 같은 6개 키와 1:1
    assert [k for k, _ in CHANGE_AXES] == list(PROVENANCE_FIELDS)


def test_evaluation_result_always_carries_full_provenance():
    """provenance 를 명시하지 않아도 6축이 전부 존재한다(None 아님)."""
    res = EvaluationResult(id="Q1", schema_version="v0.2", mode="development", provider="stub")
    assert res.provenance is not None
    dumped = res.model_dump()["provenance"]
    assert set(dumped) == EXPECTED_FIELDS
    # exclude_none 로 덤프해도 6축이 살아남는다(전부 str이라 절대 안 빠진다)
    assert set(res.model_dump(exclude_none=True)["provenance"]) == EXPECTED_FIELDS


def test_unknown_values_are_present_not_missing():
    dumped = Provenance().model_dump()
    assert set(dumped) == EXPECTED_FIELDS  # 키가 빠지지 않는다
    for k in EXPECTED_FIELDS:
        assert dumped[k] == "UNKNOWN"
    assert None not in dumped.values()


def test_canonical_scorer_version_is_deterministic_string():
    assert canonical_scorer_version(None, {}) == "UNKNOWN"
    assert canonical_scorer_version(None, None) == "UNKNOWN"
    s = canonical_scorer_version("grader-0.1",
                                 {"judge_faithfulness": "v1", "judge_checkpoint": "v2"})
    assert s == "grader-0.1|judge_checkpoint:v2|judge_faithfulness:v1"  # 정렬된 결정적 문자열
    assert isinstance(s, str)


# ------------------------------------------------------------------ 입력값 자동 복사

def _runner(**over):
    cfg = load_config("configs/grader.yaml")
    cfg = replace(cfg, use_stub_judge=True, dev_subset_size=10)
    kwargs = dict(
        config=cfg,
        provider=MockJudgeProvider(),
        prompt_repo=PromptRepository(cfg.judge.prompt_files),
    )
    kwargs.update(over)
    return GraderRunner(**kwargs)


def _run_dev(tmp_path: Path, runner: GraderRunner):
    code, report = execute(
        runner,
        evaluation_set_path="data/evaluation/evaluation_set.jsonl",
        responses_path="data/evaluation/model_responses.jsonl",
        mode="development",
        allow_final=False,
        runner_name="pytest",
        out_dir=str(tmp_path / "out"),
    )
    per_item = [
        json.loads(line)
        for line in (tmp_path / "out" / "per_item.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    return report, per_item


_SCORER = "grader-0.1|judge_checkpoint:v1|judge_faithfulness:v1|judge_list_item:v1"


def test_input_versions_are_copied_into_report_and_per_item(tmp_path: Path):
    runner = _runner(
        corpus="corpus-2026-08-28",
        preprocess="kordoc-1.4",
        table="xtab-7916",
        index="faiss-taemin-42",
        evalset="evalset-v0.2",
    )
    report, per_item = _run_dev(tmp_path, runner)

    expected = {
        "corpus": "corpus-2026-08-28",
        "preprocess": "kordoc-1.4",
        "table": "xtab-7916",
        "index": "faiss-taemin-42",
        "evalset": "evalset-v0.2",
        # scorer = 채점기 코드 버전 + 프롬프트 3종 파일명 <name>.v1.md 자동 도출
        "scorer": _SCORER,
    }

    # report.json 의 manifest.provenance — 6축 전부, 입력값 그대로
    assert report["manifest"]["provenance"] == expected

    # per_item.jsonl 의 각 문항 provenance — 6축 전부, 같은 값
    assert per_item, "per_item 이 비어 있으면 안 된다"
    for row in per_item:
        assert set(row["provenance"]) == EXPECTED_FIELDS
        assert row["provenance"] == expected

    # report ↔ per_item 값이 어긋나지 않는다(같은 runner.provenance() 출처)
    assert per_item[0]["provenance"] == report["manifest"]["provenance"]


def test_missing_versions_fall_back_to_unknown_without_dropping_fields(tmp_path: Path):
    runner = _runner()  # 버전 인자 전혀 안 줌 + configs/grader.yaml 도 UNKNOWN
    report, per_item = _run_dev(tmp_path, runner)

    prov = report["manifest"]["provenance"]
    assert set(prov) == EXPECTED_FIELDS  # 6축 전부 존재
    for k in ("corpus", "preprocess", "table", "index", "evalset"):
        assert prov[k] == "UNKNOWN"
    assert prov["scorer"].startswith("grader-0.1|judge_")  # scorer 는 여전히 자동 도출됨

    for row in per_item:
        assert set(row["provenance"]) == EXPECTED_FIELDS
        assert row["provenance"]["index"] == "UNKNOWN"


# ------------------------------------------------------------------ 회귀 6축 비교

def test_regression_attribute_change_compares_all_six_axes_from_real_reports(tmp_path: Path):
    """두 번의 실제 실행 manifest.provenance 를 attribute_change 에 그대로 넣으면
    6축이 모두 비교 대상이 되고, 바뀐 축만 잡힌다."""
    r1, _ = _run_dev(tmp_path / "a", _runner(
        corpus="c1", preprocess="p1", table="x1", index="i1", evalset="e1"))
    r2, _ = _run_dev(tmp_path / "b", _runner(
        corpus="c1", preprocess="p1", table="x1", index="i2", evalset="e1"))  # index 만 바뀜

    out = attribute_change(r1["manifest"]["provenance"], r2["manifest"]["provenance"])

    assert set(out["axes_checked"]) == EXPECTED_FIELDS   # 6축 전부 확인함
    assert out["changed_keys"] == ["index"]              # 바뀐 건 검색 인덱스 하나뿐
    assert out["index_changed_flag"] is True
    assert out["corpus_changed_flag"] is False
    assert out["clean_comparison"] is True               # 축 1개 → 비교 가능


def test_regression_attribute_change_flags_all_six_when_all_differ():
    prev = {k: f"{k}-A" for k in PROVENANCE_FIELDS}
    cur = {k: f"{k}-B" for k in PROVENANCE_FIELDS}
    out = attribute_change(prev, cur)
    assert set(out["changed_keys"]) == EXPECTED_FIELDS
    assert out["clean_comparison"] is False  # 6축 전부 → 비교 불가


def test_regression_attribute_change_all_unknown_is_no_false_positive():
    same = {k: "UNKNOWN" for k in PROVENANCE_FIELDS}
    out = attribute_change(same, dict(same))
    assert out["changed"] == []
    assert out["clean_comparison"] is True
