#@title 추출표 정답용 값과 에이전트 전달 계약 회귀 검사
#@markdown 원문은 감사용으로 보존하되 최종 답은 정규화 값을 쓰고, 완성된 표 조회는 추가 검색 없이 끝나는지 검사합니다.
from __future__ import annotations

from types import SimpleNamespace

from answer_pipeline import (
    Answer,
    _deterministic_list_units,
    _stage2_observation,
    answer_stage2_structgpt,
    build_field_evidence,
)
from stage1_plan import Stage1Plan
from stage2_agent import Stage2Decision


def _row(*, raw, normalized, field="필수 제출 서류", status="value_present"):
    return {
        "document_id": "RFP-000001",
        "field_name": field,
        "status": status,
        "answer_raw": raw,
        "answer_normalized": normalized,
        "active": "true",
        "representative_location": None,
        "additional_locations": [],
    }


def test_answer_ready_normalized_value_wins_but_raw_is_kept_for_audit(base_cfg):
    row = _row(
        field="지역제한",
        raw="입찰자는 경상북도 소재하여야 하고 본적도 경상북도여야 한다.",
        normalized="입찰공고일 전날부터 계약체결일까지 본점이 경상북도에 소재",
    )
    evidence = build_field_evidence(
        [row], "RFP-000001", "지역제한", base_cfg)

    assert evidence.answer_text == row["answer_normalized"]
    assert evidence.value_raw == row["answer_raw"]
    assert evidence.structured["answer_raw"] == row["answer_raw"]


def test_raw_value_is_used_only_when_normalized_value_is_missing(base_cfg):
    row = _row(field="예산", raw="금 359백만원", normalized="")
    evidence = build_field_evidence([row], "RFP-000001", "예산", base_cfg)

    assert evidence.answer_text == "359백만원"
    assert not evidence.abstained


def test_normalized_list_wins_over_longer_raw_layout():
    structured = {
        "items": ["총계 100점", "기술능력평가 90점", "가격평가 10점"],
        "answer_raw": "총계 100점\n기술능력평가 90점\n정성 세부항목 22점\n가격평가 10점",
    }

    assert _deterministic_list_units(structured) == [
        "총계 100점", "기술능력평가 90점", "가격평가 10점"]


def test_explicit_raw_hierarchy_keeps_normalized_children_nested():
    structured = {
        "items": [
            "1) 정보시스템 운영 및 유지관리",
            "가) 연구행정시스템",
            "나) 공통관리시스템",
            "2) 연계 데이터 처리",
            "3) 개발·운영 업무",
        ],
        "answer_raw": (
            "1) 정보시스템 운영 및 유지관리\n"
            "가) 연구행정시스템\n나) 공통관리시스템\n"
            "2) 연계 데이터 처리\n3) 개발·운영 업무"
        ),
    }

    assert _deterministic_list_units(structured) == [
        "정보시스템 운영 및 유지관리", "연계 데이터 처리", "개발·운영 업무"]


def test_table_lookup_observation_declares_completion_without_repeating_values():
    plan = SimpleNamespace(
        action="table_lookup", task_type="extract",
        document_scope="specific_documents", document_ids=["RFP-000001"],
        requested_fields=["예산"], condition_logic="AND", conditions=[],
        search_query="", answer_mode="value",
    )
    result = Answer(
        text="정규화된 예산", task_type="extract", route="추출테이블_값조회",
        selected_document_ids=["RFP-000001"],
        condition_query=[{
            "field": "예산", "document_id": "RFP-000001",
            "status": "value_present",
        }],
        citations=[{"document": "RFP-000001", "field": "예산"}],
    )

    observation = _stage2_observation("O1", plan, result)

    assert observation["answer_text"] == "추출표 조회 완료: 1/1개 필드"
    assert observation["structured_result"]["result_status"] == "complete"
    assert observation["evidence"][0]["content"] == "정규화된 예산"
    assert "정규화된 예산" not in str(observation["structured_result"])


class _OnePlanAgent:
    def __init__(self, decision):
        self.decision = decision
        self.calls = 0

    def decide(self, question, observations, **kwargs):
        self.calls += 1
        if self.calls > 1:
            raise AssertionError("완성된 표 조회 뒤 LLM을 다시 호출했습니다.")
        return self.decision


def test_complete_table_list_finishes_without_vector_search_or_second_llm_call(base_cfg):
    plan = Stage1Plan(
        action="table_lookup", task_type="extract",
        document_scope="specific_documents", document_ids=["RFP-000001"],
        requested_fields=["필수 제출 서류"], answer_mode="list",
    )
    decision = Stage2Decision(
        action="table_lookup", task_type="extract",
        document_scope="specific_documents", result_assessment="initial",
        document_ids=["RFP-000001"], requested_fields=["필수 제출 서류"],
        answer_mode="list", tool_plan=plan,
    )
    agent = _OnePlanAgent(decision)
    cfg = dict(base_cfg)
    cfg.update({
        "stage2_max_tool_calls": 3,
        "stage2_dual_interpretation_enabled": False,
        "stage2_plan_verifier_enabled": False,
        "stage2_deterministic_extraction_assembly": True,
    })
    table = [_row(
        raw="가. 원문 서류 A\n나. 원문 서류 B\n다. 원문에만 있는 조판 문구",
        normalized='["서류 A", "서류 B"]',
    )]

    result = answer_stage2_structgpt(
        "자유로운 질문", None, lambda: None,
        lambda: agent, table, cfg,
    )

    assert result.structured_answer == ["서류 A", "서류 B"]
    assert result.execution_plan["tool_calls"] == 1
    assert [item["action"] for item in result.execution_plan["decisions"]] == [
        "table_lookup", "finalize"]
    assert agent.calls == 1


def test_unresolved_table_result_does_not_use_automatic_completion(base_cfg):
    plan = Stage1Plan(
        action="table_lookup", task_type="extract",
        document_scope="specific_documents", document_ids=["RFP-000001"],
        requested_fields=["필수 제출 서류"], answer_mode="list",
    )
    first = Stage2Decision(
        action="table_lookup", task_type="extract",
        document_scope="specific_documents", result_assessment="initial",
        document_ids=["RFP-000001"], requested_fields=["필수 제출 서류"],
        answer_mode="list", tool_plan=plan,
    )
    second = Stage2Decision(
        action="clarify", task_type="extract",
        document_scope="specific_documents", result_assessment="insufficient",
        document_ids=["RFP-000001"], requested_fields=["필수 제출 서류"],
        answer_mode="list", clarification="원문 확인이 필요합니다.",
    )

    class TwoPlanAgent(_OnePlanAgent):
        def decide(self, question, observations, **kwargs):
            self.calls += 1
            return first if self.calls == 1 else second

    agent = TwoPlanAgent(first)
    cfg = dict(base_cfg)
    cfg.update({
        "stage2_max_tool_calls": 3,
        "stage2_dual_interpretation_enabled": False,
        "stage2_plan_verifier_enabled": False,
        "stage2_deterministic_extraction_assembly": True,
    })

    result = answer_stage2_structgpt(
        "자유로운 질문", None, lambda: None, lambda: agent,
        [_row(raw="", normalized="", status="review_required")], cfg,
    )

    assert result.abstained
    assert result.route == "애매_되묻기"
    assert agent.calls == 2
