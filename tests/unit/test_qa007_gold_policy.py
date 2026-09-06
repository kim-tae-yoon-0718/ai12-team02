"""QA-007 정답 정책 — 원문에 없는 계산값을 정답으로 요구하지 않는다 (2026-09-04).

★배경: 2차 후보본은 체크포인트에 '합계 843백만원'을 넣었다. 359+484=843 은 맞지만
  '843' 은 corpus_v1·v2 와 해당 문서의 청크 323개 전수에서 0건인 **파생값**이다.
  평가셋이 원문에 없는 합계를 요구하면, 원문을 정확히 옮긴 답이 감점된다.

  이 파일은 문항 텍스트를 담지 않는다 — 후보 평가셋 파일을 읽어 규칙만 검사한다.
"""
import json
import os
import re
from pathlib import Path

import pytest

from grader.models import EvaluationItem, ModelResponse
from grader.task_scoring import score_item

CFG = {"residual_limit": 20}


def _candidate_path() -> Path | None:
    """후보 평가셋 위치. 파일 경로만 찾고 **내용은 테스트에 박지 않는다.**

    ① 환경변수 RFP_CANDIDATE_EVALSET
    ② outputs/ 아래 가장 최신 후보본 (평가셋은 저장소가 아니라 산출물 폴더에 있다)
    """
    env = os.environ.get("RFP_CANDIDATE_EVALSET")
    if env and Path(env).exists():
        return Path(env)
    outputs = Path(__file__).resolve().parents[2].parent / "ai12-team02" / "outputs"
    found = list(outputs.glob("*/evalset/candidate/items.jsonl")) if outputs.exists() else []
    # ★사전순이 아니라 **가장 최근에 만든 후보본**을 고른다. 폴더 이름 순으로 고르면
    #   'final_fix' 가 'integration_v2' 보다 앞서서 옛 후보본을 검사하게 된다.
    return max(found, key=lambda q: q.stat().st_mtime) if found else None


CANDIDATE = _candidate_path()


def _load():
    if CANDIDATE is None:
        pytest.skip("후보 평가셋을 찾지 못함 — RFP_CANDIDATE_EVALSET 로 경로를 주면 검사한다")
    return {json.loads(l)["id"]: json.loads(l)
            for l in CANDIDATE.read_text(encoding="utf-8").splitlines() if l.strip()}


# ── 규칙 자체(합성 입력) — 환경과 무관하게 항상 돈다 ──────────────────

def test_summary_partial_credit_does_not_require_every_checkpoint():
    """summary 는 부분점수라, 체크포인트 하나가 빠져도 나머지는 점수를 받는다.
    바꿔 말하면 **원문에 없는 체크포인트를 넣으면 그만큼 상한이 깎인다**."""
    it = EvaluationItem.model_validate(dict(
        id="X", question="q", task_type="qa", answer_type="summary",
        answer_raw=["가나다", "라마바"]))
    r = ModelResponse(id="X", answer="가나다 만 언급", abstained=False)
    assert score_item(it, r, CFG)["task_score"].score == 0.5


def test_derived_total_as_a_checkpoint_caps_a_faithful_answer():
    """원문 두 값을 정확히 옮긴 답이, 원문에 없는 합계 때문에 만점을 못 받는다."""
    faithful = "개발비 359백만원, H/W 484백만원(VAT 포함)"
    with_total = EvaluationItem.model_validate(dict(
        id="X", question="q", task_type="qa", answer_type="summary",
        answer_raw=["개발비 359백만원", "H/W 484백만원(VAT 포함)", "합계 843백만원"]))
    without = EvaluationItem.model_validate(dict(
        id="X", question="q", task_type="qa", answer_type="summary",
        answer_raw=["개발비 359백만원", "H/W 484백만원(VAT 포함)"]))
    r = ModelResponse(id="X", answer=faithful, abstained=False)
    assert score_item(with_total, r, CFG)["task_score"].score < 1.0
    assert score_item(without, r, CFG)["task_score"].score == 1.0


def test_mentioning_the_total_is_not_penalized_either():
    """합계를 말해도 감점되지 않는다 — 요구하지도, 금지하지도 않는다."""
    it = EvaluationItem.model_validate(dict(
        id="X", question="q", task_type="qa", answer_type="summary",
        answer_raw=["개발비 359백만원", "H/W 484백만원(VAT 포함)"]))
    r = ModelResponse(id="X", abstained=False,
                      answer="개발비 359백만원, H/W 484백만원(VAT 포함)으로 합계 843백만원입니다.")
    assert score_item(it, r, CFG)["task_score"].score == 1.0


# ── 후보 평가셋 파일 대조 ─────────────────────────────────────────────

_DERIVED_TOTAL = re.compile(r"합계|총\s*사업예산|843")


def test_candidate_qa007_has_no_derived_total():
    items = _load()
    it = items["QA-007"]
    assert it["answer_type"] == "summary", "answer_type=summary 는 유지한다"
    cps = it["answer_raw"]
    assert isinstance(cps, list) and cps
    for c in cps:
        assert not _DERIVED_TOTAL.search(str(c)), f"원문에 없는 합계가 남아 있다: {c!r}"
    assert len(it.get("evidence") or []) == 2, "기존의 올바른 근거 두 개는 유지한다"


def test_candidate_qa007_checkpoints_are_source_backed():
    """체크포인트의 숫자는 근거 문서의 추출표 값에서 확인되는 것만 쓴다."""
    items = _load()
    nums = {n for c in items["QA-007"]["answer_raw"] for n in re.findall(r"\d+", str(c))}
    assert "843" not in nums
    assert {"359", "484"} <= nums


def test_candidate_version_is_bumped():
    ver = CANDIDATE.parent / "VERSION.txt" if CANDIDATE else None
    if not ver or not ver.exists():
        pytest.skip("VERSION.txt 없음")
    first = ver.read_text(encoding="utf-8").splitlines()[0]
    assert first.startswith("evalset: v1-candidate."), first
    assert first != "evalset: v1-candidate.2", "후보 버전을 올려야 한다"
