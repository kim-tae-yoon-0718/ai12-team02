"""평가셋·채점기 통합 2차 회귀 테스트 (2026-09-04).

이번에 고친 결함이 되살아나지 않게 못박는다.
  §3 답 유형과 정답 구조의 일치 / 되묻기 수용 조건
  §4 근거 계약 — 출처가 달라도 같은 근거면 인정, 틀린 근거는 실패
  §5 평가셋 검사기가 실제로 실패시키는지
  §6 시스템 실패 0점 강제 / Stub·Mock 구분 / final 게이트

★최종 50문항의 질문·정답은 이 파일에 넣지 않는다. 전부 합성 입력이다.
"""
import json

import pytest

from checks.check_evalset import (
    check_answer_type_shape, check_consortium_states, check_evidence_shape,
    check_factual_has_evidence, check_retrievable_evidence,
    check_selection_gold_recomputed, check_unspecified_type_consistency,
)
from grader.diagnostics.report import full_report
from grader.models import EvaluationItem, Location, ModelResponse
from grader.normalize import match_short
from grader.retrieval import grade_citation
from grader.runner import _missing_row, faithfulness_axis, score_breakdown
from grader.task_scoring import check_format, score_item

CFG = {"residual_limit": 20}


def _it(**k):
    b = dict(id="X", question="q", task_type="extraction", answer_type="value", answer_raw="x")
    b.update(k)
    return EvaluationItem.model_validate(b)


def _r(**k):
    b = dict(id="X", answer="", abstained=False)
    b.update(k)
    return ModelResponse(**b)


# ── §6-1. 시스템 실패는 정답 문자열이어도 0점 ──────────────────────────

@pytest.mark.parametrize("item_kw,resp_kw", [
    ({}, dict(answer="5억원", failure="timeout")),                     # 정답 문자열 + 실패
    ({}, dict(answer="", failure="api_error")),                        # 빈 답 + 실패
    (dict(answer_type="unanswerable", answer_raw="자료에 없음"),
     dict(answer="", abstained=True, failure="timeout")),              # 기권 문항 + 실패
    ({}, dict(answer=None, failure="crash")),                          # answer=null + 실패
])
def test_system_failure_scores_zero_even_when_answer_matches(item_kw, resp_kw):
    """★실패한 실행이 우연히 정답 문자열을 담고 있어도 시스템이 답을 낸 것이 아니다.

    이걸 정답으로 세면 '실패할수록 점수가 오르는' 구간이 생긴다.
    """
    base = dict(answer_type="value", answer_raw="5억원")
    base.update(item_kw)
    resp = dict(id="X", abstained=False)
    resp.update(resp_kw)
    out = score_item(_it(**base), ModelResponse(**resp), CFG)
    assert out["task_score"].score == 0.0
    assert out["task_score"].kind == "system_failure"
    assert out["final_status"] == "FAIL-system"


def test_system_failure_and_content_zero_are_counted_separately():
    """오답 0점과 시스템 실패 0점은 처방이 다르다 — 한 칸에 합치면 원인을 못 가른다."""
    rows = [
        {"id": "A", "task_type": "qa", "answer_type": "value", "field_tag": None,
         "answer_source": "chunk", "score": 0.0, "final_status": "FAIL-content",
         "abstention": {}, "failure": None, "route": None},
        _missing_row(_it(id="B", task_type="qa")),
        {"id": "C", "task_type": "qa", "answer_type": "value", "field_tag": None,
         "answer_source": "chunk", "score": 1.0, "final_status": "PASS",
         "abstention": {}, "failure": None, "route": None},
    ]
    f = full_report(rows)["failures"]
    assert f["system_error_zero"] == 1 and f["system_error_ids"] == ["B"]
    assert f["content_zero"] == 1 and f["content_zero_ids"] == ["A"]
    assert full_report(rows)["n_items"] == 3      # 분모에서 빠지지 않는다


# ── §6-2. Stub · Mock · 실제 심판 구분 ────────────────────────────────

@pytest.mark.parametrize("kind", ["stub", "mock"])
def test_no_final_overall_without_a_real_judge(kind):
    """문자열 매처나 더미가 낸 숫자를 '최종 점수'로 적으면 안 된다."""
    rep = {"overall_score": 0.87, "main": {"by_task_type": {
        "selection": {"score": 0.9}, "extraction": {"score": 0.8}, "qa": {"score": 0.7}}},
        "faithfulness": {"applicable": False}}
    out = score_breakdown(rep, {"kind": kind})
    assert out["final_overall"] is None
    assert kind in out["final_overall_note"]
    assert out["deterministic"]["by_task_type"] == {"selection": 0.9, "extraction": 0.8}


def test_real_judge_alone_does_not_produce_a_final_overall():
    """★[2026-09-04 정정] '실제 LLM이다'만으로 최종 점수를 내면 안 된다.

    검증되지 않은 실제 LLM 을 붙였을 때 실행은 게이트가 막는데 보고서만
    final_overall 에 숫자를 적던 모순이 있었다. 준비 판정이 통과해야 한다.
    """
    rep = {"overall_score": 0.87, "main": {"by_task_type": {"qa": {"score": 0.7}}},
           "faithfulness": {"applicable": True, "mean_score": 0.9}}
    out = score_breakdown(rep, {"kind": "real_judge", "usable_for_final": False})
    assert out["final_overall"] is None
    assert out["provisional_real_judge"]["applicable"] is True
    assert out["provisional_real_judge"]["value"] == 0.87


def test_ready_real_judge_produces_a_final_overall():
    class _R:
        ready = True
        problems: list = []
        def model_dump(self): return {"ready": True}
    rep = {"overall_score": 0.87, "main": {"by_task_type": {"qa": {"score": 0.7}}},
           "faithfulness": {"applicable": True, "mean_score": 0.9}}
    out = score_breakdown(rep, {"kind": "real_judge", "usable_for_final": True}, _R())
    assert out["final_overall"] == 0.87
    assert out["provisional_real_judge"]["applicable"] is False


def test_stub_marks_list_and_summary_scores_provisional():
    out = score_breakdown({"main": {"by_task_type": {}}}, {"kind": "stub"})
    assert out["provisional_stub"]["applicable"] is True
    assert set(out["provisional_stub"]["affected_answer_types"]) == {"list", "summary"}


@pytest.mark.parametrize("kind", ["stub", "mock"])
def test_faithfulness_is_na_without_a_real_judge(kind):
    axis = faithfulness_axis([], {"kind": kind})
    assert axis["applicable"] is False
    assert "mean_score" not in axis          # 안 돌렸으면 숫자를 만들지 않는다


def test_qa_headline_metric_is_not_called_faithfulness():
    """QA 대표지표는 task_scoring 이 낸 내용 점수다 — 충실성이라고 적으면 안 된다."""
    rows = [{"id": "A", "task_type": "qa", "answer_type": "value", "field_tag": None,
             "answer_source": "chunk", "score": 1.0, "final_status": "PASS",
             "abstention": {}, "failure": None, "route": None}]
    qa = full_report(rows)["main"]["primary"]["qa"]
    assert "충실성" not in qa["metric"]
    assert "내용 점수" in qa["metric"]


# ── §6-3. 실제 심판 준비 게이트 ───────────────────────────────────────

def test_mock_provider_is_not_usable_for_final():
    from grader.judge import Judge, JudgeConfig
    from grader.prompts import PromptRepository
    from grader.providers import MockJudgeProvider

    j = Judge(JudgeConfig(model="m", family="claude", tier="final", temperature=0.0),
              prompt_repo=PromptRepository({}), provider=MockJudgeProvider())
    assert j.is_mock is True
    problems = j.assert_ready(generator_family="gpt", strict=False)
    assert any("더미(mock)" in p for p in problems)
    assert any("사람 채점 대조 기록 없음" in p for p in problems)


# ── §3. 되묻기 수용은 '정답 자체가 되묻기'일 때만 ─────────────────────

def test_clarification_accepted_only_when_gold_is_a_clarification():
    ambiguous = _it(answer_type="value", unspecified_type="ambiguous_match",
                    intermediate_answer=["RFP-000001", "RFP-000002"],
                    answer_raw="사업이 여러 건이라 특정할 수 없습니다. 사업명을 알려주시겠어요?")
    asked_back = _r(answer="어떤 사업을 말씀하시는 건가요?")
    assert score_item(ambiguous, asked_back, CFG)["task_score"].score == 1.0

    # ★해소 가능한 문항(정답이 사실 답변)인데 되묻기만 하면 오답이어야 한다.
    resolvable = _it(answer_type="value", unspecified_type="abbreviation",
                     intermediate_answer="RFP-000003",
                     answer_raw="공동수급체는 5개 이하로 구성")
    assert score_item(resolvable, asked_back, CFG)["task_score"].score == 0.0


def test_checker_flags_unspecified_type_mismatch():
    resolvable = {"id": "EXT-X", "answer_type": "value", "unspecified_type": "abbreviation",
                  "answer_raw": "공동수급체는 5개 이하로 구성"}
    assert any("되묻기가 아님" in e for e in check_unspecified_type_consistency([resolvable]))
    clarify_no_type = {"id": "EXT-Y", "answer_type": "value",
                       "answer_raw": "어떤 사업을 말씀하시는 건가요?"}
    assert any("unspecified_type 이 없음" in e
               for e in check_unspecified_type_consistency([clarify_no_type]))


# ── §3. 답 유형과 정답 구조 ───────────────────────────────────────────

def test_long_value_gold_is_flagged():
    """60자를 넘는 value 정답은 채점기의 의미 매칭 경로 밖이라 축자 일치만 정답이 된다."""
    long_gold = ("입찰공고일 전날부터 입찰일까지 법인등기부상 본점이 해당 경상북도에 소재하여야 하고, "
                 "낙찰자는 계약체결일까지 법인등기부상 본점이 경상북도에 소재하여야 한다")
    assert len(long_gold) > 60
    assert any("자 (>60)" in e for e in check_answer_type_shape(
        [{"id": "EXT-X", "answer_type": "value", "answer_raw": long_gold}]))
    # 실제로 뜻이 같은 재진술이 0점이 되는지 확인 — 이 규칙의 근거다
    paraphrase = "입찰공고일 전날부터 입찰일까지 본점이 경상북도에 있어야 하고 낙찰자는 계약체결일까지 유지해야 한다"
    assert match_short(long_gold, paraphrase)[0] is False


def test_all_josa_values_do_not_match_each_other():
    """★조사만 남는 짧은 값을 끝까지 깎으면 서로 다른 답이 모두 빈 값이 되어 일치한다."""
    assert match_short("이", "가")[0] is False
    assert match_short("임", "요")[0] is False


def test_multiline_value_gold_is_flagged():
    assert any("여러 줄" in e for e in check_answer_type_shape(
        [{"id": "EXT-X", "answer_type": "value", "answer_raw": "가. 첫째\n나. 둘째"}]))


def test_summary_checkpoints_must_be_short():
    long_cp = ["기술원 환경에 최적화된 공정 및 장비 실시간 모니터링 기능 구현을 위한 효율적이고 체계적인 시스템 구축"]
    assert any("원문 문장 수준" in e for e in check_answer_type_shape(
        [{"id": "QA-X", "answer_type": "summary", "answer_raw": long_cp}]))


def test_clarification_item_is_exempt_from_length_rule():
    """되묻기가 정답인 문항은 채점 경로가 달라 길이 규칙 대상이 아니다."""
    item = {"id": "EXT-X", "answer_type": "value", "unspecified_type": "ambiguous_match",
            "answer_raw": "국민연금공단이 발주한 사업이 여러 건이라 어떤 사업을 말씀하시는지 "
                          "특정할 수 없습니다. 사업명을 함께 알려주시겠어요?"}
    assert check_answer_type_shape([item]) == []


# ── §4. 근거 계약 ─────────────────────────────────────────────────────

def _ev_item(evidence):
    return EvaluationItem.model_validate(dict(
        id="X", question="q", task_type="selection", answer_type="document_set",
        answer_raw=["RFP-000012"], evidence=evidence))


def _cite(**k):
    b = dict(document="RFP-000012", section="", ref_no="")
    b.update(k)
    return Location.model_validate(b)


ABSENT_EV = [{"kind": "extraction_table", "document": "RFP-000012", "field": "지역제한",
              "status": "field_absent", "source": "extraction_table_v3"}]


def test_field_absent_evidence_is_gradable_without_a_line():
    """★미기재 근거에는 가리킬 원문 줄이 없다. 없는 좌표를 지어내지 않고도 채점된다."""
    ok = grade_citation(_ev_item(ABSENT_EV),
                        _r(citations=[_cite(field="지역제한", source="extraction_table_v3")]))
    assert ok["applicable"] is True and ok["matched"] is True


def test_wrong_field_or_document_fails_evidence():
    for bad in (_cite(field="예산"), _cite(document="RFP-000099", field="지역제한")):
        out = grade_citation(_ev_item(ABSENT_EV), _r(citations=[bad]))
        assert out["matched"] is False
        assert out["n_wrong_citations"] == 1


def test_same_evidence_via_a_different_path_is_accepted():
    """추출표 근거를 청크 좌표로 인용해도 같은 원문을 가리키면 맞는 근거다."""
    ev = [{"kind": "extraction_table", "document": "RFP-000020", "field": "예산",
           "status": "value_present", "source": "extraction_table_v3",
           "location": {"document": "RFP-000020", "section": "1. 사업 개요",
                        "ref_no": "1. 사업 개요 · 문단 1-46", "line": 42}}]
    out = grade_citation(_ev_item(ev), _r(citations=[Location(
        document="RFP-000020", section="1. 사업 개요",
        ref_no="1. 사업 개요 · 문단 1-46", line=40, line_end=50)]))
    assert out["matched"] is True


def test_identity_evidence_is_graded_and_wrong_field_fails():
    """§4: identity 기반 마감일 근거도 출처 채점 대상이다."""
    ev = [{"kind": "identity", "document": "RFP-000060", "field": "bid_deadline",
           "value": "2024-06-20 14:00", "source": "identity_v2"}]
    ok = grade_citation(_ev_item(ev), _r(citations=[_cite(
        document="RFP-000060", field="bid_deadline", source="identity_v2")]))
    assert ok["applicable"] is True and ok["matched"] is True
    bad = grade_citation(_ev_item(ev), _r(citations=[_cite(
        document="RFP-000060", field="공고일", source="identity_v2")]))
    assert bad["matched"] is False


def test_factual_item_without_evidence_fails_the_checker():
    assert any("근거(location/evidence)가 없음" in e for e in check_factual_has_evidence(
        [{"id": "SEL-X", "task_type": "selection", "answer_type": "document_set",
          "answer_raw": ["RFP-000001"]}]))
    # 의도적인 빈 정답(0건)은 근거를 요구하지 않는다
    assert check_factual_has_evidence(
        [{"id": "SEL-Y", "task_type": "selection", "answer_type": "document_set",
          "answer_raw": []}]) == []


def test_evidence_shape_rejects_unknown_kind_and_missing_keys(tmp_path):
    table = tmp_path / "t.json"
    table.write_text(json.dumps({"rows": [
        {"document_id": "RFP-000012", "field_name": "지역제한", "status": "field_absent",
         "active": "true"}]}), encoding="utf-8")
    ident = tmp_path / "i.csv"
    ident.write_text("document_id,bid_deadline\nRFP-000012,2024-01-01\n", encoding="utf-8-sig")

    bad_kind = [{"id": "X", "evidence": [{"kind": "구전", "document": "RFP-000012"}]}]
    assert any("알 수 없는 근거 종류" in e
               for e in check_evidence_shape(bad_kind, table, ident))
    wrong_status = [{"id": "X", "evidence": [{"kind": "extraction_table",
                     "document": "RFP-000012", "field": "지역제한", "status": "value_present"}]}]
    assert any("근거가 실제 자료와 다름" in e
               for e in check_evidence_shape(wrong_status, table, ident))
    bad_field = [{"id": "X", "evidence": [{"kind": "identity",
                  "document": "RFP-000012", "field": "없는필드"}]}]
    assert any("identity_v2 에 없는 필드" in e
               for e in check_evidence_shape(bad_field, table, ident))


def test_only_non_retrievable_evidence_fails_unless_declared_blocked(tmp_path):
    chunks = tmp_path / "c.jsonl"
    chunks.write_text(json.dumps({
        "document_id": "RFP-000007", "location_label": "목차 · 문단 1",
        "retrieval_eligible": False, "md_line_start": 103, "md_line_end": 149}) + "\n",
        encoding="utf-8")
    item = {"id": "EXT-X", "evidence": [{"kind": "chunk", "document": "RFP-000007",
            "location": {"document": "RFP-000007", "section": "목차",
                         "ref_no": "목차 · 문단 1", "line": 114}}]}
    assert any("검색 불가능한 청크" in e for e in check_retrievable_evidence([item], chunks))
    # 차단 사유를 명시해 escalate 한 항목은 조용한 실패가 아니라 별도 목록으로 나간다
    declared = dict(item, blocked_reason="청킹 오분류 — 담당자 확인 필요")
    assert check_retrievable_evidence([declared], chunks) == []


# ── §5. 선별형 정답 독립 재계산 · 공동수급 상태 ───────────────────────

def _official_fixture(tmp_path, consortium_text, region_status="value_present"):
    reg = tmp_path / "reg.json"
    reg.write_text(json.dumps({"documents": [
        {"document_id": "RFP-000001", "active": True, "retrieval_eligible": True}]}),
        encoding="utf-8")
    ident = tmp_path / "id.csv"
    ident.write_text("document_id,bid_deadline\nRFP-000001,2024-12-31\n", encoding="utf-8-sig")
    table = tmp_path / "tbl.json"
    table.write_text(json.dumps({"rows": [
        {"document_id": "RFP-000001", "field_name": "지역제한", "status": region_status,
         "answer_raw": "부산", "active": "true"},
        {"document_id": "RFP-000001", "field_name": "컨소시엄 요건", "status": "value_present",
         "answer_raw": consortium_text, "active": "true"}]}), encoding="utf-8")
    return reg, ident, table


def test_method_restricted_document_is_rejected_from_a_full_ban_item(tmp_path):
    """★"공동이행방식은 허용하지 않음" 문서를 '공동수급 금지' 정답에 넣으면 실패."""
    reg, ident, table = _official_fixture(
        tmp_path, "본 사업은 공동수급(공동이행방식)을 허용하지 않음")
    item = {"id": "SEL-024", "task_type": "selection", "answer_type": "document_set",
            "answer_raw": ["RFP-000001"]}
    assert any("특정 이행방식만 금지" in e
               for e in check_consortium_states([item], reg, ident, table))
    assert any("정답 집합이 다름" in e
               for e in check_selection_gold_recomputed([item], reg, ident, table))


def test_full_ban_document_is_accepted(tmp_path):
    reg, ident, table = _official_fixture(tmp_path, "본 사업은 공동수급을 불허함")
    item = {"id": "SEL-024", "task_type": "selection", "answer_type": "document_set",
            "answer_raw": ["RFP-000001"]}
    assert check_consortium_states([item], reg, ident, table) == []
    assert check_selection_gold_recomputed([item], reg, ident, table) == []


def test_valid_ids_but_wrong_gold_set_is_caught(tmp_path):
    """등록부에 있는 ID 만 썼어도 집합 자체가 틀리면 실패해야 한다."""
    reg, ident, table = _official_fixture(tmp_path, "본 사업은 공동수급을 불허함")
    empty = {"id": "SEL-024", "task_type": "selection", "answer_type": "document_set",
             "answer_raw": []}
    assert any("누락 ['RFP-000001']" in e
               for e in check_selection_gold_recomputed([empty], reg, ident, table))


# ── 형식 계약 ─────────────────────────────────────────────────────────

def test_missing_abstained_key_is_a_format_error():
    fs = check_format(_it(answer_type="value", answer_raw="x"),
                      ModelResponse(id="X", answer="x"), CFG)
    assert fs.passed is False
    assert any("abstained" in v for v in fs.violations)


def test_trailing_period_does_not_break_a_correct_answer():
    """실제 LLM 답변은 마침표로 끝난다 — 그것 때문에 정답이 오답이 되면 안 된다."""
    gold = "나. 사업기간 : 계약일로부터 6개월"
    assert match_short(gold, "계약일로부터 6개월입니다")[0] is True
    assert match_short(gold, "계약일로부터 6개월입니다.")[0] is True
    # 금액·날짜의 마침표는 그대로 지킨다
    from grader.normalize import parse_amount
    assert parse_amount("1.5억원") == 150_000_000


# ── §6-3. 게이트가 tier 설정으로 우회되지 않는다 ──────────────────────

def test_final_gate_is_not_bypassed_by_setting_tier_to_dev():
    """★예전에는 `strict and tier=='final'` 이라, tier 를 dev 로 두면 최종 실행
    게이트가 통째로 꺼졌다. 설정 한 줄로 검사를 무력화할 수 있으면 게이트가 아니다."""
    from grader.judge import Judge, JudgeConfig, JudgeNotReady
    from grader.prompts import PromptRepository

    class _P:
        name = "openai_compatible"
        def judge(self, prompt): return {"raw_text": "{}"}

    j = Judge(JudgeConfig(model="m", family="claude", tier="dev", temperature=0.0),
              prompt_repo=PromptRepository({}), provider=_P())
    with pytest.raises(JudgeNotReady):
        j.assert_ready(generator_family="gpt", strict=True)
    # 기본 호출(strict 미지정)은 dev 심판을 막지 않고 보고만 한다 — 두 쓰임이 공존한다
    assert j.assert_ready(generator_family="gpt")


def test_judge_prompts_live_in_a_tracked_path():
    """★심판 프롬프트가 .gitignore 대상 경로에 있으면 프롬프트가 바뀌어도
    git_dirty 에 안 잡힌다 — 채점 기준이 조용히 바뀔 수 있다."""
    import subprocess
    from pathlib import Path

    from grader.config import load_config

    repo = Path(__file__).resolve().parents[2]
    cfg = load_config(str(repo / "config/grader.yaml"))
    assert cfg.judge.prompt_files, "심판 프롬프트 경로가 설정에 없다"
    for name, path in cfg.judge.prompt_files.items():
        p = Path(path)
        assert p.exists(), f"{name}: 프롬프트 파일 없음 {p}"
        rel = p.relative_to(repo).as_posix()
        ignored = subprocess.run(["git", "-C", str(repo), "check-ignore", "-q", rel],
                                 capture_output=True).returncode == 0
        assert not ignored, f"{name}: {rel} 이 .gitignore 대상 — 버전관리 밖이다"


# ── §6-1. 응답 누락 문항이 진단 축 분모에서도 빠지지 않는다 ────────────

def test_missing_response_stays_in_retrieval_and_citation_denominators():
    """★주 점수만 0 으로 남기고 검색·출처 분모에서 빼면, 응답을 안 낸 문항이
    검색 재현율과 출처 정확도를 **올려 주는** 결과가 된다."""
    from grader.config import load_config
    from grader.prompts import PromptRepository
    from grader.providers import MockJudgeProvider
    from grader.runner import GraderRunner, grade_all
    from pathlib import Path

    repo = Path(__file__).resolve().parents[2]
    cfg = load_config(str(repo / "config/grader.yaml"))
    runner = GraderRunner(config=cfg, provider=MockJudgeProvider(),
                          prompt_repo=PromptRepository(cfg.judge.prompt_files))
    loc = {"document": "A", "section": "S", "ref_no": "S · 문단 1", "line": 10}
    answered = EvaluationItem.model_validate(dict(
        id="A", question="q", task_type="qa", answer_type="value", answer_raw="x",
        location=loc))
    unanswered = EvaluationItem.model_validate(dict(
        id="B", question="q", task_type="qa", answer_type="value", answer_raw="y",
        location=loc))
    resp = {"A": ModelResponse(id="A", answer="x", abstained=False, route="chunk_search",
                               citations=[Location.model_validate(loc)])}
    out = grade_all(runner, [answered, unanswered], resp, "development")
    assert out["missing_predictions"] == ["B"]
    assert len(out["scored_rows"]) == 2                 # 주 점수 분모
    assert out["citation_agg"]["n"] == 2, "출처 채점 분모에서 누락 문항이 빠졌다"
    assert out["citation_agg"]["citation_accuracy"] == 0.5, (
        "응답을 안 낸 문항이 출처 정확도를 올려 주면 안 된다")
