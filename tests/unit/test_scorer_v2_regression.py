"""scorer v2 회귀 테스트 — 팀장 지시(2026-09-02) + 김하루 감사 보고분.

'데이터·표기가 조금 달라져도 채점 결과가 틀어지는 숨은 가정' 을 고정한다.
★기존 테스트를 느슨하게 만들지 않고, 각 상황을 독립 케이스로 못박는다.
"""
from grader.models import EvaluationItem, ModelResponse
from grader.normalize import amount_conditions, match_short, parse_amount, parse_time, normalize_text
from grader.task_scoring import (
    _item_match, check_format, grade_comparison, grade_list, grade_short_answer, score_item,
)

CFG = {"residual_limit": 20}


def _it(**k):
    b = dict(id="X", question="q", task_type="extraction", answer_type="value", answer_raw="x")
    b.update(k)
    return EvaluationItem.model_validate(b)


def _r(**k):
    b = dict(id="X", answer="", abstained=False)
    b.update(k)
    return ModelResponse(**b)


# ── 2-1. 금액: 원/천원, 이내·이상, 부가세 포함/별도 ────────────────────

def test_amount_won_vs_cheonwon_same():
    assert parse_amount("230,000천원") == parse_amount("230,000,000원") == 230_000_000
    assert match_short("230,000,000원", "230,000천원")[0] is True


def test_amount_condition_이내_differs():
    assert match_short("230,000,000원 이내", "230,000,000원")[0] is False
    assert match_short("230,000,000원 이내", "230,000천원 이내")[0] is True


def test_amount_vat_included_vs_excluded_differ():
    assert amount_conditions("5억원(부가세 포함)") == frozenset({"vat_included"})
    assert amount_conditions("5억원 부가세 별도") == frozenset({"vat_excluded"})
    assert match_short("352,000,000원(부가가치세 포함)", "352,000,000원 부가세 별도")[0] is False
    assert match_short("352,000,000원(부가가치세 포함)", "금 352,000천원 부가세 포함")[0] is True


# ── 2-2. 빈 answer_normalized ─────────────────────────────────────────

def test_empty_answer_normalized_falls_back_to_raw():
    it = _it(answer_type="list", answer_raw=["가", "나", "다"], answer_normalized=[])
    s = grade_list(it, _r(structured_answer=["가", "나", "다"]))
    assert s.score == 1.0


def test_genuinely_empty_list_answer_stays_empty():
    it = _it(answer_type="list", answer_raw=[], answer_normalized=[])
    s = grade_list(it, _r(structured_answer=[]))
    assert s.score == 1.0  # 빈 정답 · 빈 답 → 통과
    s2 = grade_list(it, _r(structured_answer=["가짜"]))
    assert s2.score == 0.0 and s2.detail["n_extra"] == 1


# ── 2-3. 목록형 1:1 대응 (제안서 / 제안서 요약본 중복 오인) ────────────

def test_list_no_double_count_of_one_answer_item():
    it = _it(answer_type="list", answer_normalized=["제안서", "제안서 요약본"])
    s = grade_list(it, _r(structured_answer=["제안서 요약본"]))
    assert s.score == 0.0 and s.detail["missing"] == ["제안서"]


def test_item_match_no_prefix():
    assert _item_match("A", "A등급 확인서") is False        # prefix 매치 금지
    assert _item_match("A", "법인 A 등급") is True           # 토큰 경계
    assert _item_match("제안서", "제안서를 제출") is True      # 조사


# ── 2-4. abstained 누락 = 형식 오류 ──────────────────────────────────

def test_missing_abstained_is_format_error():
    it = _it(task_type="qa", answer_type="value", answer_raw="5억원")
    resp = ModelResponse(id="X", answer="5억원")  # abstained 안 넣음 → None
    out = score_item(it, resp, CFG)
    assert out["format_status"].passed is False
    assert any("abstained" in v for v in out["format_status"].violations)


def test_abstained_not_inferred_from_text():
    it = _it(task_type="qa", answer_type="unanswerable", answer_raw="없는 사업")
    # 본문에 '찾을 수 없습니다' 있어도 abstained=False 면 hallucination
    from grader.task_scoring import grade_abstention
    assert grade_abstention(it, _r(answer="찾을 수 없습니다", abstained=False)).abstention_kind == "hallucination"


# ── 2-5. '없음' 표현은 묶지 않는다 ──────────────────────────────────

def test_absence_expressions_are_not_merged():
    assert match_short("지역 제한 없음", "제한 없음")[0] is False
    assert match_short("지역 제한 없음", "해당 없음")[0] is False
    assert match_short("지역 제한 없음", "확인 불가")[0] is False
    # 단, 같은 문구의 띄어쓰기 차이는 인정
    assert match_short("지역 제한 없음", "지역제한 없음")[0] is True


# ── 2-6. 비교형: 문서 ID 형식 / 항목 키 ─────────────────────────────

def test_comparison_bad_doc_id_recorded_not_dropped():
    it = _it(task_type="qa", answer_type="comparison",
             answer_raw=[{"항목": "예산", "RFP-38": "1억", "RFP-000043": "2억"}])
    s = grade_comparison(it, _r(structured_answer={"RFP-000043": {"예산": "2억"}}))
    assert any("RFP-38" in e for e in s.detail.get("gold_errors", []))
    assert s.detail["cells"] == 1  # RFP-000043 만 채점


def test_comparison_bad_doc_id_in_response_recorded():
    it = _it(task_type="qa", answer_type="comparison",
             answer_raw=[{"항목": "예산", "RFP-000043": "2억"}])
    s = grade_comparison(it, _r(structured_answer={"RFP-43": {"예산": "2억"}, "RFP-000043": {"예산": "2억"}}))
    assert any("RFP-43" in e for e in s.detail.get("pred_errors", []))


def test_comparison_missing_항목_key_recorded():
    it = _it(task_type="qa", answer_type="comparison",
             answer_raw=[{"category": "예산", "RFP-000038": "1억"}])
    s = grade_comparison(it, _r(structured_answer={}))
    assert any("항목" in e for e in s.detail.get("gold_errors", []))


# ── 2-7. 마감일 시간 ────────────────────────────────────────────────

def test_deadline_time_compared_when_gold_has_time():
    assert parse_time("2024-06-24 16:00") == "16:00"
    assert match_short("2024-06-24 16:00", "2024-06-24 18:00")[0] is False
    assert match_short("2024-06-24 16:00", "2024. 6. 24. 16시")[0] is True
    # gold 에 시각 없으면 날짜만
    assert match_short("2024-06-24", "2024년 6월 24일 18:00")[0] is True


# ── 2-8. 퍼센트·기간·조사·유니코드 ─────────────────────────────────

def test_percent_and_duration_notation():
    assert match_short("20%", "20 퍼센트")[0] is True
    assert match_short("5개월", "5 개월")[0] is True
    # 단위·기호 자체가 사라지거나 뜻이 바뀌면 안 된다 (앞부분 일치로 통과 금지)
    assert match_short("20%", "20")[0] is False
    assert match_short("5개월", "5년")[0] is False
    assert "%" in normalize_text("20 퍼센트")


def test_unicode_roman_numeral_preserved():
    assert normalize_text("Ⅳ장") == normalize_text("4장")
    assert normalize_text("Ⅲ. 사업 개요") == normalize_text("3. 사업 개요")
    # NFKC 가 단위 뜻을 바꾸지 않는다: ㎡ 는 면적, ℃ 는 온도로 유지(문자열에 남아 있음)
    assert normalize_text("100㎡").strip() != ""
    assert normalize_text("20℃") != normalize_text("20%")


# ── 2-9. residual_limit 이유 기록 ──────────────────────────────────

def test_residual_reason_recorded():
    it = _it(answer_type="value", answer_raw="5억원")
    long_pad = "5억원이며 유지보수 3년과 교육 프로그램 및 별도 컨설팅 비용이 포함되어 있습니다"
    s = grade_short_answer(it, _r(answer=long_pad), CFG)
    assert s.score == 0.0
    assert "남은 부분" in s.detail["why"]


# ── J. 검색 경로 구분 (팀장 상단 항목) ──────────────────────────────

def test_retrieval_na_for_non_search_route():
    from grader.models import Location
    from grader.retrieval import grade_retrieval, grade_citation
    it = _it(task_type="qa", answer_type="value", answer_raw="x",
             location=Location(document="A", section="S", ref_no="S · 문단 1", line=10))
    resp = _r(answer="x", route="extract_table")
    nsr = frozenset({"extract_table"})
    stages = grade_retrieval(it, resp, 20, 10, 5, non_search_routes=nsr)
    assert all(s["applicable"] is False for s in stages)
    assert grade_citation(it, resp, non_search_routes=nsr)["applicable"] is False
    # 청크 검색 경로면 정상 채점
    resp2 = _r(answer="x", route="chunk_search")
    assert grade_retrieval(it, resp2, 20, 10, 5, non_search_routes=nsr)[0]["applicable"] is True


# ── C4. 존재하지 않는 공식 문서 ID (팀장 회귀 목록) ────────────────

def test_unknown_doc_id_in_ref_intg(tmp_path):
    from checks.check_evalset import check_ref_intg
    p = tmp_path / "ids.json"
    p.write_text('["RFP-000001"]', encoding="utf-8")
    it = _it(task_type="qa", answer_type="value", answer_raw="x",
             location=[{"document": "RFP-999999", "section": "S", "ref_no": "r", "line": 1}])
    errs = check_ref_intg([{"id": "Q", "location": it.model_dump()["location"]}], p)
    assert any("RFP-999999" in e for e in errs)


# ── KNOWN_GROUND_TRUTH_MISMATCH (팀장 3) ──────────────────────────

def test_known_ground_truth_mismatch_excluded_from_aggregate():
    from dataclasses import replace
    from grader.config import load_config
    from grader.judge import StubJudge
    from grader.prompts import PromptRepository
    from grader.runner import GraderRunner, grade_all
    cfg = load_config("config/grader.yaml")
    cfg = replace(cfg, use_stub_judge=True)
    cfg = replace(cfg, grading=replace(cfg.grading, known_ground_truth_mismatch=("Q_BAD",)))
    runner = GraderRunner(config=cfg, provider=StubJudge(),
                          prompt_repo=PromptRepository(cfg.judge.prompt_files))
    items = [
        EvaluationItem.model_validate(dict(id="Q_OK", question="q", task_type="extraction",
                                           answer_type="value", answer_raw="5억원")),
        EvaluationItem.model_validate(dict(id="Q_BAD", question="q", task_type="extraction",
                                           answer_type="value", answer_raw="틀린정답")),
    ]
    resps = {"Q_OK": ModelResponse(id="Q_OK", answer="5억원", abstained=False),
             "Q_BAD": ModelResponse(id="Q_BAD", answer="5억원", abstained=False)}
    g = grade_all(runner, items, resps, "development")
    assert g["known_mismatch_ids"] == ["Q_BAD"]
    assert {r["id"] for r in g["scored_rows"]} == {"Q_OK"}
    bad = next(r for r in g["results_rows"] if r["id"] == "Q_BAD")
    assert bad["final_status"] == "KNOWN_GROUND_TRUTH_MISMATCH"
