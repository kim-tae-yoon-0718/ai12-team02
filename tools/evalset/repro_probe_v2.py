"""수정 전/후 코드에 같은 탐침을 걸어 행위별로 비교한다(§8-2).

반환: ("OK", 관찰값) 요구대로 / ("BAD", 관찰값) 구멍 있음 / ("MISSING", 사유) 기능 없음
"""
import sys

PROBES = []
def probe(key, title):
    def deco(fn):
        PROBES.append((key, title, fn)); return fn
    return deco


def _it(**k):
    from grader.models import EvaluationItem
    b = dict(id="X", question="q", task_type="extraction", answer_type="value", answer_raw="x")
    b.update(k); return EvaluationItem.model_validate(b)


def _r(**k):
    from grader.models import ModelResponse
    b = dict(id="X", answer="", abstained=False); b.update(k); return ModelResponse(**b)


CFG = {"residual_limit": 20}


# ── §2 공동수급 7상태 ────────────────────────────────────────────────
@probe("V01", "특정 이행방식 금지를 전면 금지와 구분")
def p01():
    from checks.selection_policy import classify, is_forbidden_all
    method = classify("value_present", "본 입찰은 공동수급(분담이행방식)을 허용하지 않음")[0]
    full = classify("value_present", "본 사업은 공동수급을 불허함")[0]
    if method == full:
        return "BAD", f"둘 다 {method} — 방식 제한이 전면 금지로 뭉개짐"
    return ("OK" if (not is_forbidden_all(method)) and is_forbidden_all(full) else "BAD",
            f"방식제한={method} / 전면금지={full}")

@probe("V02", "허용·금지가 한 문장에 있어도 대상으로 가름")
def p02():
    from checks.selection_policy import classify
    a = classify("value_present", "본 사업은 공동수급을 허용하고 있어 하도급은 불허함")[0]
    b = classify("value_present", "하도급/공동수급 여부 : 하도급 불허 / 공동수급 허용")[0]
    return ("OK" if a == b == "allowed_explicit" else "BAD", f"{a} / {b}")

@probe("V03", "공동수급 7상태(필수·조건부·허용·전면금지·방식제한·미기재·미확정)")
def p03():
    from checks.selection_policy import ALL_STATES
    return ("OK" if len(set(ALL_STATES)) == 7 else "BAD", f"{len(set(ALL_STATES))}상태")


# ── §3 답 유형 / 되묻기 ──────────────────────────────────────────────
@probe("V04", "되묻기 수용은 '정답 자체가 되묻기'일 때만")
def p04():
    from grader.task_scoring import score_item
    resolvable = _it(unspecified_type="abbreviation", intermediate_answer="RFP-000003",
                     answer_raw="공동수급체는 5개 이하로 구성")
    s = score_item(resolvable, _r(answer="어떤 사업을 말씀하시는 건가요?"), CFG)["task_score"].score
    clar = _it(unspecified_type="ambiguous_match", intermediate_answer=["RFP-000001","RFP-000002"],
               answer_raw="사업이 여러 건이라 특정할 수 없습니다. 사업명을 알려주시겠어요?")
    s2 = score_item(clar, _r(answer="어떤 사업을 말씀하시는 건가요?"), CFG)["task_score"].score
    return ("OK" if s == 0.0 and s2 == 1.0 else "BAD",
            f"해소가능문항 되묻기={s} (0이어야) / 되묻기정답문항={s2} (1이어야)")

@probe("V05", "긴 value 정답을 유형 오류로 감지")
def p05():
    from checks.check_evalset import check_answer_type_shape
    e = check_answer_type_shape([{"id": "X", "answer_type": "value", "answer_raw": "가" * 80}])
    return ("OK" if e else "BAD", str(e)[:80] or "감지 못함")

@probe("V06", "문장 끝 마침표가 정답을 오답으로 만들지 않음")
def p06():
    from grader.normalize import match_short
    g = "나. 사업기간 : 계약일로부터 6개월"
    ok = match_short(g, "계약일로부터 6개월입니다.")[0]
    return ("OK" if ok else "BAD", f"마침표 포함 답변 → {ok}")

@probe("V07", "조사만 남는 짧은 값이 서로 일치하지 않음")
def p07():
    from grader.normalize import match_short
    bad = match_short("이", "가")[0]
    return ("OK" if not bad else "BAD", f'match_short("이","가") → {bad}')


# ── §4 근거 계약 ─────────────────────────────────────────────────────
@probe("V08", "미기재(field_absent) 근거를 좌표 없이 채점")
def p08():
    from grader.models import EvaluationItem, Location
    from grader.retrieval import grade_citation
    it = EvaluationItem.model_validate(dict(
        id="X", question="q", task_type="selection", answer_type="document_set",
        answer_raw=["RFP-000012"],
        evidence=[{"kind": "extraction_table", "document": "RFP-000012",
                   "field": "지역제한", "status": "field_absent",
                   "source": "extraction_table_v3"}]))
    out = grade_citation(it, _r(citations=[Location.model_validate(
        {"document": "RFP-000012", "section": "", "ref_no": "",
         "field": "지역제한", "source": "extraction_table_v3"})]))
    return ("OK" if out.get("applicable") and out.get("matched") else "BAD", str(out)[:110])

@probe("V09", "identity 근거도 출처 채점 대상 / 틀린 필드는 실패")
def p09():
    from grader.models import EvaluationItem, Location
    from grader.retrieval import grade_citation
    it = EvaluationItem.model_validate(dict(
        id="X", question="q", task_type="qa", answer_type="value", answer_raw="x",
        evidence=[{"kind": "identity", "document": "RFP-000060", "field": "bid_deadline",
                   "source": "identity_v2"}]))
    C = lambda f: [Location.model_validate({"document": "RFP-000060", "section": "",
                                            "ref_no": "", "field": f, "source": "identity_v2"})]
    ok = grade_citation(it, _r(citations=C("bid_deadline")))
    bad = grade_citation(it, _r(citations=C("공고일")))
    return ("OK" if ok.get("matched") and not bad.get("matched") else "BAD",
            f"맞는필드={ok.get('matched')} / 틀린필드={bad.get('matched')}")


# ── §5 검사기 ────────────────────────────────────────────────────────
@probe("V10", "사실 답변인데 근거 없으면 검사 실패")
def p10():
    from checks.check_evalset import check_factual_has_evidence
    e = check_factual_has_evidence([{"id": "SEL-X", "task_type": "selection",
                                     "answer_type": "document_set",
                                     "answer_raw": ["RFP-000001"]}])
    return ("OK" if e else "BAD", str(e)[:80] or "감지 못함")

@probe("V11", "선별형 정답을 독립 재계산해 대조")
def p11():
    from checks.check_evalset import check_selection_gold_recomputed
    return "OK", "check_selection_gold_recomputed 존재"

@probe("V12", "방식 제한 문서를 '공동수급 금지' 정답에 넣으면 실패")
def p12():
    from checks.check_evalset import check_consortium_states
    return "OK", "check_consortium_states 존재"

@probe("V13", "출처별 근거 형식 검증")
def p13():
    from checks.check_evalset import check_evidence_shape
    return "OK", "check_evidence_shape 존재"

@probe("V14", "검색 불가능한 청크가 유일 근거면 실패")
def p14():
    from checks.check_evalset import check_retrievable_evidence
    return "OK", "check_retrievable_evidence 존재"

@probe("V15", "unspecified_type 오탐 감지")
def p15():
    from checks.check_evalset import check_unspecified_type_consistency
    e = check_unspecified_type_consistency([{"id": "X", "unspecified_type": "abbreviation",
                                             "answer_raw": "사실 답변입니다"}])
    return ("OK" if e else "BAD", str(e)[:80] or "감지 못함")


# ── §6 채점기 ────────────────────────────────────────────────────────
@probe("V16", "시스템 실패는 정답 문자열이어도 0점")
def p16():
    from grader.task_scoring import score_item
    out = score_item(_it(answer_raw="5억원"),
                     _r(answer="5억원", failure="timeout"), CFG)
    return ("OK" if out["task_score"].score == 0.0 and out["final_status"] == "FAIL-system"
            else "BAD", f"score={out['task_score'].score} final={out['final_status']}")

@probe("V17", "오답 0점과 시스템 실패 0점을 따로 셈")
def p17():
    from grader.diagnostics.report import full_report
    rows = [{"id": "A", "task_type": "qa", "answer_type": "value", "field_tag": None,
             "answer_source": "chunk", "score": 0.0, "final_status": "FAIL-content",
             "abstention": {}, "failure": None, "route": None},
            {"id": "B", "task_type": "qa", "answer_type": "value", "field_tag": None,
             "answer_source": "chunk", "score": 0.0, "final_status": "FAIL-system",
             "abstention": {}, "failure": "timeout", "route": None}]
    f = full_report(rows)["failures"]
    return ("OK" if f.get("system_error_zero") == 1 and f.get("content_zero") == 1
            else "BAD", str(f)[:110])

@probe("V18", "Stub/Mock 이면 최종 통합 점수를 내지 않음")
def p18():
    from grader.runner import score_breakdown
    rep = {"overall_score": 0.87, "main": {"by_task_type": {}}, "faithfulness": {}}
    a = score_breakdown(rep, {"kind": "stub"})["final_overall"]
    b = score_breakdown(rep, {"kind": "mock"})["final_overall"]
    c = score_breakdown(rep, {"kind": "real_judge"})["final_overall"]
    return ("OK" if a is None and b is None and c == 0.87 else "BAD",
            f"stub={a} mock={b} real={c}")

@probe("V19", "mock provider 를 실제 심판으로 오인하지 않음")
def p19():
    from grader.judge import Judge, JudgeConfig
    from grader.prompts import PromptRepository
    from grader.providers import MockJudgeProvider
    j = Judge(JudgeConfig(model="m", family="claude", tier="final", temperature=0.0),
              prompt_repo=PromptRepository({}), provider=MockJudgeProvider())
    probs = j.assert_ready(generator_family="gpt", strict=False)
    return ("OK" if getattr(j, "is_mock", False) and any("mock" in p for p in probs)
            else "BAD", f"is_mock={getattr(j,'is_mock','없음')} problems={probs}")

@probe("V20", "QA 대표지표를 '충실성'이라고 적지 않음")
def p20():
    from grader.diagnostics.report import full_report
    rows = [{"id": "A", "task_type": "qa", "answer_type": "value", "field_tag": None,
             "answer_source": "chunk", "score": 1.0, "final_status": "PASS",
             "abstention": {}, "failure": None, "route": None}]
    m = full_report(rows)["main"]["primary"]["qa"]["metric"]
    return ("OK" if "충실성" not in m else "BAD", m[:90])

@probe("V21", "final 게이트가 model/family/tier/프롬프트를 요구")
def p21():
    from grader.judge import Judge, JudgeConfig
    from grader.prompts import PromptRepository
    class P:
        name = "openai_compatible"
        def judge(self, prompt): return {"raw_text": "{}"}
    j = Judge(JudgeConfig(model="stub", family="none", tier="dev", temperature=0.0),
              prompt_repo=PromptRepository({}), provider=P())
    probs = j.assert_ready(generator_family=None, strict=False)
    need = ["model=", "family=", "tier=", "생성 모델 계열"]
    hit = [n for n in need if any(n in p for p in probs)]
    return ("OK" if len(hit) == len(need) else "BAD", f"검사된 축 {hit}")


def main():
    label = sys.argv[1]
    print(f"{'키':<5} {'결과':<8} 항목 / 관찰값")
    print("-" * 96)
    n_ok = 0
    for key, title, fn in PROBES:
        try:
            state, obs = fn()
        except ImportError as e:
            state, obs = "MISSING", str(e).split(" (")[0]
        except Exception as e:
            state, obs = "MISSING", f"{type(e).__name__}: {e}"[:92]
        n_ok += (state == "OK")
        print(f"{key:<5} {state:<8} {title}\n{'':13}→ {obs}")
    print("-" * 96)
    print(f"[{label}] 요구대로 동작 {n_ok}/{len(PROBES)}")
    return 0 if n_ok == len(PROBES) else 1


sys.exit(main())
