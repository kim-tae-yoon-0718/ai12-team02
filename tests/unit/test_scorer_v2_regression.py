"""scorer v2 회귀 테스트 — 팀장 지시(2026-09-02) + 김하루 감사 보고분.

'데이터·표기가 조금 달라져도 채점 결과가 틀어지는 숨은 가정' 을 고정한다.
★기존 테스트를 느슨하게 만들지 않고, 각 상황을 독립 케이스로 못박는다.
"""
from grader.models import EvaluationItem, ModelResponse
from grader.normalize import (
    amount_conditions, match_short, parse_amount, parse_time, normalize_text, strip_label_prefix,
)
from grader.task_scoring import (
    _item_match, check_format, grade_comparison, grade_list, grade_short_answer,
    grade_summary_checkpoint, score_item,
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


# ── K. 채점기 피드백 회귀 (2026-09-03) ────────────────────────────────
# 라벨(번호+항목명) 생략 / AM·PM 시각 구분 / 목록·요약형 이중 계산.
# 의도적으로 다루지 않는 것(별도 기록, 여기서 구현 안 함):
#  - 완전한 의미 기반(semantic) 자유문장 비교 — LLM judge 영역, 팀장 2-3 "분리해도 됨"
#  - abstained 를 CLARIFICATION/REFUSAL/IRRELEVANT 로 세분 — 팀장 2-4(bool 확정) 재논의 필요
#  - '없음'(field extracted as 없음) vs '문서에서 확인할 수 없음' 동일시 — 팀장 2-5 위반 소지,
#    현진·하루 협의 필요(코드로 선반영하지 않음)

def test_label_prefix_omission_accepted():
    it = _it(answer_raw="나. 사업기간 : 계약일로부터 6개월")
    ok, why = match_short(it.answer_raw, "계약일로부터 6개월",
                          accept=[strip_label_prefix(it.answer_raw)])
    assert ok is True and why == "normalized_exact"


def test_label_prefix_omission_does_not_loosen_wrong_value():
    it = _it(answer_raw="나. 사업기간 : 계약일로부터 6개월")
    r = grade_short_answer(it, _r(answer="계약일로부터 12개월"), CFG)
    assert r.score == 0.0


def test_label_prefix_no_false_positive_on_plain_value():
    # 콜론 없는 평범한 값은 그대로 — strip_label_prefix 가 아무것도 잘라내면 안 됨
    assert strip_label_prefix("지역 제한 없음") is None
    assert strip_label_prefix("248,796천원") is None


def test_ampm_time_distinguished():
    assert parse_time("오전 4시") == "04:00"
    assert parse_time("오후 4시") == "16:00"
    assert parse_time("오전 12시") == "00:00"   # 자정
    assert parse_time("오후 12시") == "12:00"   # 정오
    ok, why = match_short("2024-06-24 오전 4시", "2024-06-24 오후 4시")
    assert ok is False and "time_mismatch" in why


def test_list_structured_answer_does_not_double_count_via_free_text():
    """구조화 답변(structured_answer)으로 이미 만족시킨 항목의 원문이 response.answer 에
    그대로 남아있어도, 다른 정답 항목이 거기서 또 만족된 것으로 이중 계산되면 안 된다."""
    it = _it(task_type="extraction", answer_type="list", answer_raw=["제안서", "제안서 요약본"])
    r = grade_list(it, _r(answer="제안서 요약본", structured_answer=["제안서 요약본"]))
    assert r.score == 0.0
    assert r.detail["hit"] == ["제안서 요약본"]
    assert r.detail["missing"] == ["제안서"]


def test_summary_checkpoint_no_double_count():
    """요약형(체크포인트)도 목록형과 동일한 1:1 소진 원칙을 따른다."""
    it = _it(task_type="qa", answer_type="summary", answer_raw=["제안서", "제안서 요약본"])
    r = grade_summary_checkpoint(it, _r(answer="제안서 요약본"))
    assert r.score == 0.5
    assert r.detail["covered"] == ["제안서 요약본"]
    assert r.detail["missing"] == ["제안서"]


# ── L. 채점기 피드백 2차 (2026-09-03) — 표현 차이·되묻기·부재응답 ─────

def test_semantic_equivalence_label_and_honorific_omission():
    """정답: '나. 사업기간 : 계약일로부터 6개월' / 응답: '계약일로부터 6개월' → PASS.
    라벨 생략은 accept 후보, 조사/존댓말 차이는 어절 포함 관계로 흡수한다."""
    it = _it(answer_type="value", answer_raw="나. 사업기간 : 계약일로부터 6개월")
    ok, why = match_short(it.answer_raw, "계약일로부터 6개월")
    assert ok is True
    ok2, _ = match_short(it.answer_raw, "계약일로부터 6개월입니다")  # 존댓말 어미
    assert ok2 is True


def test_semantic_equivalence_does_not_accept_missing_or_wrong_info():
    it = _it(answer_type="value", answer_raw="나. 사업기간 : 계약일로부터 6개월")
    assert match_short(it.answer_raw, "6개월")[0] is False              # 정보 누락(계약일로부터)
    assert match_short(it.answer_raw, "계약일로부터 12개월")[0] is False  # 값 자체가 다름
    assert match_short(it.answer_raw, "계약체결일로부터 6개월")[0] is False  # 핵심 어절 다름


def test_clarification_accepted_for_ambiguous_question():
    """질문: 국민연금공단 사업이 여러 건이라 특정 불가(unspecified_type=ambiguous_match).
    응답: '어떤 사업을 말씀하시는지 사업명을 알려주세요.' → PASS, 오답도 불필요한 거절도 아님."""
    it = _it(task_type="extraction", answer_type="value",
             answer_raw="국민연금공단이 발주한 사업이 여러 건이라 어떤 사업인지 특정할 수 없습니다. "
                        "사업명을 알려주시겠어요?",
             unspecified_type="ambiguous_match")
    out = score_item(it, _r(answer="어떤 사업을 말씀하시는지 사업명을 알려주세요.", abstained=False), CFG)
    assert out["task_score"].score == 1.0
    assert out["abstention"].abstention_kind == "ok"

    # abstained=True 로 답해도(기권으로 표시) '불필요한 거절'로 잘못 세지 않는다
    out2 = score_item(it, _r(answer="여러 사업이 있어 사업명을 알려주세요.", abstained=True), CFG)
    assert out2["abstention"].abstention_kind == "clarification_ok"
    assert out2["abstention"].abstention_kind != "over_refusal"

    # 모호하지 않은 문항에서는 여전히 오답(엉뚱한 되묻기는 정답 처리 안 함)
    plain = _it(task_type="extraction", answer_type="value", answer_raw="500,000,000원")
    out3 = score_item(plain, _r(answer="어떤 사업을 말씀하시는지 알려주세요.", abstained=False), CFG)
    assert out3["task_score"].score == 0.0


def test_document_not_found_toggle_off_by_default():
    it = _it(answer_type="value", answer_raw="없음", answer_source="table")
    resp = _r(answer="문서에서 확인할 수 없습니다.", abstained=True)
    assert grade_short_answer(it, resp, {"accept_natural_absence_phrasing": False}).score == 0.0
    assert grade_short_answer(it, resp, {"accept_natural_absence_phrasing": True}).score == 1.0


def test_ampm_time_fail_case_from_feedback():
    assert match_short("2024-06-24 오전 4시", "2024-06-24 오후 4시")[0] is False


def test_money_won_cheonwon_pass_case_from_feedback():
    assert match_short("10,000원", "10천원")[0] is True


def test_list_counting_one_item_not_counted_as_two():
    it = _it(task_type="extraction", answer_type="list", answer_raw=["서류A", "서류B"])
    r = grade_list(it, _r(answer="서류A", structured_answer=["서류A"]))
    assert r.score == 0.0
    assert r.detail["hit"] == ["서류A"] and r.detail["missing"] == ["서류B"]


# ── M. 실제 모델 계약 확인 (2026-09-04, answer_pipeline.py 팀원 브랜치 fetch) ──
# non_search_routes/clarify_routes 기본값을 이태민 실제 ROUTE_* 상수로 교체.
# 전엔 추측값이라 실제 응답 route 와 하나도 안 맞았다(조용한 오류) — 코드 추적으로 확인.

def test_non_search_routes_match_real_router_constants():
    from grader.config import load_config
    C = load_config("config/grader.yaml")
    real_non_search = {
        "추출테이블_문서선별", "추출테이블_값조회", "추출테이블_비교조립",
        "identity_v2_값조회", "애매_되묻기", "검색불필요_인사응답", "검색불필요_사용법안내",
    }
    assert C.retrieval.non_search_routes == frozenset(real_non_search)
    # 실제 검색을 쓰는 라우트는 제외 목록에 없어야 한다
    assert "chunks검색_LLM답변" not in C.retrieval.non_search_routes
    assert "구조화자료_결합_LLM답변" not in C.retrieval.non_search_routes


def test_clarification_detected_via_real_route_and_structured_answer():
    """실제 모델(_clarify())이 채우는 신호 — route='애매_되묻기' +
    structured_answer.clarification_needed=True. 텍스트 패턴이 안 맞아도 이걸로 판정돼야 한다."""
    it = _it(task_type="extraction", answer_type="value",
             answer_raw="여러 사업이 있어 특정할 수 없습니다.", unspecified_type="ambiguous_match")
    # 텍스트만 봐서는 CLARIFICATION 패턴이 아닌 문구인데(신원 노출 없이 실제 모델처럼 후보만 나열)
    weird_text = "- 이러닝시스템 운영 용역\n- 사회보험료 지원 정보시스템 보완"
    resp = ModelResponse(id="X", answer=weird_text, abstained=True, route="애매_되묻기",
                         structured_answer={"clarification_needed": True,
                                            "candidates": ["RFP-000021", "RFP-000022"]})
    from grader.config import load_config
    C = load_config("config/grader.yaml")
    cfg = {"residual_limit": 20, "clarify_routes": C.retrieval.clarify_routes}
    out = score_item(it, resp, cfg)
    assert out["task_score"].score == 1.0
    assert out["abstention"].abstention_kind == "clarification_ok"


def test_route_signal_overrides_when_text_pattern_absent_but_not_when_route_is_search():
    """route가 검색 라우트('chunks검색_LLM답변')면 되묻기로 오판하면 안 된다."""
    it = _it(task_type="extraction", answer_type="value", answer_raw="5억원",
             unspecified_type="ambiguous_match")
    resp = ModelResponse(id="X", answer="엉뚱한 답", abstained=False, route="chunks검색_LLM답변")
    r = grade_short_answer(it, resp, {"residual_limit": 20})
    assert r.score == 0.0  # 검색 라우트인데 되묻기 취급되면 안 됨


# ── N. 실제 데이터 실행에서 발견 (2026-09-04, 실제 모델 응답으로 채점 돌려봄) ──

def test_list_item_enum_prefix_dropped_by_model_still_matches():
    """실제 모델 응답 재현 — PRAC-EXT-002: 정답 목록 항목엔 전부 ①~⑮ 번호가 있는데,
    모델이 그 중 1개만 번호 없이 냈다. 순번 표시는 값의 일부가 아니므로(이태민
    answer_pipeline._ENUM_PREFIX_RE 와 동일 판단) 정답 처리돼야 한다."""
    assert _item_match("① 입찰참가신청서(서금원 소정양식) 1부",
                       "입찰참가신청서(서금원 소정양식) 1부") is True
    it = _it(task_type="extraction", answer_type="list",
             answer_raw=["① 서류A 1부", "② 서류B 1부"])
    r = grade_list(it, _r(structured_answer=["서류A 1부", "② 서류B 1부"]))
    assert r.detail["missing"] == []  # 번호 빠졌다고 누락 처리되면 안 됨


def test_enum_prefix_strip_does_not_break_wrong_content():
    # 번호를 떼도 내용 자체가 다르면 여전히 오답
    assert _item_match("① 서류A 1부", "서류C 1부") is False
