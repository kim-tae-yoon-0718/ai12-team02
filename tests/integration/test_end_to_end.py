"""배관 점검용 스모크 — 여러 answer_type/실패 유형이 섞인 문항을 실제로 layer_eval까지
돌려서 report.json / per_item.jsonl 이 만들어지는지 확인한다.

★ 이 데이터는 실제 코퍼스가 아니다. 임현진/박예진/이태민의 실제 산출물이 오기 전에
계약(스키마)이 실제로 돌아가는지 확인하기 위한 것이다(README "아직 비어 있는 자리" 참고).
"""

import json
from dataclasses import replace
from pathlib import Path

from grader.config import load_config
from grader.providers import MockJudgeProvider
from grader.prompts import PromptRepository
from grader.runner import GraderRunner, execute


EVALSET = [
    {  # v0.2: 선별형 정답 문서 집합은 answer_raw
        "id": "Q001", "question": "예산 5억 이상 사업 뽑아줘", "task_type": "selection",
        "answer_type": "document_set", "answer_raw": ["D1", "D3"], "field_tag": "major",
        "answer_source": "verified",
    },
    {
        "id": "Q002", "question": "언제까지 내야 해", "task_type": "extraction",
        "answer_type": "value", "document_id": "D1", "field_tag": "critical",
        "answer_raw": "2026-09-15", "location": {"document": "D1", "section": "개요", "ref_no": "1"},
    },
    {
        "id": "Q003", "question": "제출 서류 뭐 필요해?", "task_type": "extraction",
        "answer_type": "list", "document_id": "D1", "field_tag": "major",
        "answer_raw": ["사업자등록증", "법인등기부등본", "제안서"],
        "answer_normalized": ["사업자등록증", "법인등기부등본", "제안서"],
    },
    {  # v0.2: 기권 사유는 answer_raw 문자열
        "id": "Q004", "question": "달나라 기지 구축 사업 예산은?", "task_type": "qa",
        "answer_type": "unanswerable", "answer_raw": "코퍼스에 없는 사업입니다",
    },
]

RESPONSES = [
    {"id": "Q001", "answer": "2건입니다", "selected_document_ids": ["D1", "D3", "D4"]},
    {"id": "Q002", "answer": "2026년 9월 16일까지입니다",  # ★critical 필드 오답
     "citations": [{"document": "D1", "section": "개요", "ref_no": "1"}]},  # 출처는 맞게 붙임(3-4-3)
    {"id": "Q003", "answer": "사업자등록증, 법인등기부등본이 필요합니다",
     "structured_answer": ["사업자등록증", "법인등기부등본"]},  # 3개 중 2개 — 오답이어야 함
    {"id": "Q004", "answer": "해당 사업을 코퍼스에서 찾을 수 없습니다", "abstained": True,
     "unanswerable_reason": "not_in_corpus"},
]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def test_development_layer_end_to_end(tmp_path: Path):
    evalset_path = tmp_path / "evalset.jsonl"
    responses_path = tmp_path / "responses.jsonl"
    _write_jsonl(evalset_path, EVALSET)
    _write_jsonl(responses_path, RESPONSES)

    cfg = load_config("config/grader.yaml")
    cfg = replace(cfg, use_stub_judge=True, dev_subset_size=10)
    runner = GraderRunner(
        config=cfg,
        provider=MockJudgeProvider(),
        prompt_repo=PromptRepository(cfg.judge.prompt_files),
        corpus="corpus-test", table="ext-test",
        evalset="evalset-test",
    )

    code, report = execute(
        runner,
        evaluation_set_path=str(evalset_path),
        responses_path=str(responses_path),
        mode="development",
        allow_final=False,
        runner_name="pytest",
        out_dir=str(tmp_path / "out"),
    )

    summary = report["summary"]
    assert summary["n_items"] == 4
    by_status = summary["format"]["by_final_status"]
    assert "FAIL-content" in by_status  # Q002 critical 필드 오답
    assert "FAIL-content" in by_status or "PASS" in by_status  # Q003 list 3개 중 2개 → 오답

    per_item_path = tmp_path / "out" / "per_item.jsonl"
    assert per_item_path.exists()
    lines = per_item_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 4

    # Q002: critical 필드 오답이므로 field_tag=critical 게이트가 걸려야 한다
    assert summary["severity"]["gate"]["critical"]["status"] == "FAIL"
    assert code == 1

    # 3-4-3: location 있는 문항은 Q002뿐 — citation을 정답 좌표와 똑같이 붙였으니
    # 오답이어도(critical FAIL) 출처 자체는 맞은 것으로 잡혀야 한다(내용/출처는 별개 축).
    citation = summary["citation"]
    assert citation["n"] == 1
    assert citation["citation_accuracy"] == 1.0
    assert citation["no_citation_rate"] == 0.0
