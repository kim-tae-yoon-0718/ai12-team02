"""
grader.extraction — 3-2-1 구조화 추출 테이블 평가 / 3-2-2 문서 특정 분리 채점

────────────────────────────────────────────────────────────────
3-2-1 이 왜 3-2(검색 평가)와 별개로 존재하는가
  선별형의 답은 검색이 아니라 추출 테이블(1-12-2, 박예진)에서 나온다.
  테이블이 틀리면 조건 검색이 통째로 틀리는데, ★검색 지표는 아무 이상 신호를 주지 않는다.

★ 순환 문제 (2-8-1과 짝)
  answer_source=table 인 문항의 정답을 그 테이블로 만들었다면 테이블이 틀려도
  시스템은 만점을 받는다 — 답안지와 시스템이 같은 재료에서 나왔기 때문이다.
  ⇒ 테이블 정확도는 **평가셋과 독립적으로** 잰다. 기준은 원문 — 사람이 문서를 열어
     확인한 것과 대조한다(박예진 4-6-2 / 1-12-2 결과물).

[확정 2026-08-31] 12필드 목록·상태 어휘·field_tag 등급은 rfp_extraction_table_v2
(박예진, /srv/rfp) 와 configs/default.yaml gate.column_severity 에서 확정.
[대기 ← 태윤 원문대조 JSONL] pred↔gold 표본 감사 데이터. GATE_BY_SEVERITY 임계값은
지금 임시값이며 baseline 실측 후 조정한다.
────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from collections import defaultdict

from .models import (
    EvaluationItem,
    ExtractState,
    ModelResponse,
    as_id_list,
    norm_extract_state,
)
from .normalize import match_short, parse_amount

# rfp_extraction_table_v2 확정 어휘(박예진). 구 3상태는 models.norm_extract_state 가 흡수.
EXTRACT_STATES: tuple[ExtractState, ...] = (
    "value_present", "field_absent", "not_disclosed",
    "external_reference", "conflict", "extraction_failed", "review_required",
)
_VALUE_STATE = "value_present"       # 값 대조가 필요한 유일한 상태
_FAILURE_STATE = "extraction_failed"  # 기계 실패 = 결측
# "정보성 부재" 상태 — 기계 실패와 뒤바뀌면 부정조건 질의가 조용히 틀린다.
_INFO_ABSENCE_STATES = ("field_absent", "not_disclosed", "external_reference")

# field_tag(critical/major/minor) 별 요구 정확도. [대기] 실측 전 임시값 — baseline 이후 조정.
GATE_BY_SEVERITY = {"critical": 0.95, "major": 0.90, "minor": 0.80}


def grade_extraction_audit(audit_rows: list[dict],
                           column_severity: dict[str, str] | None = None) -> dict:
    """표본 원문 대조 결과(박예진 4-6-2 산출물)를 지표로 바꾼다.

    audit_rows 한 줄의 계약 (태윤 원문대조 산출물):
      {"document_id": "...", "column"|"field_name": "예산",
       "pred_state": "<상태>", "pred_value": "5억원",
       "gold_state": "<상태>", "gold_value": "500,000,000원"}
    상태 어휘 = models.ExtractState (구 value/absent/failed 별칭 자동 흡수).

    반환: 컬럼별 정확도 / 결측률 / ★"정보성 부재 vs 추출 실패" 혼동률 / 오차 방향 / 게이트 판정.
    gold_state=conflict 행은 v1 채점 대상이 아니므로(C 결정) 분모에서 빼고 따로 센다.
    """
    column_severity = column_severity or {}
    by_col: dict[str, list[dict]] = defaultdict(list)
    for r in audit_rows:
        by_col[r.get("column") or r.get("field_name")].append(r)

    report: dict[str, dict] = {}
    for col, all_rows in sorted(by_col.items()):
        rows = [r for r in all_rows if norm_extract_state(r.get("gold_state")) != "conflict"]
        n_conflict = len(all_rows) - len(rows)
        n = len(rows)
        correct = 0
        missing = 0
        confusion = {f"{a}->{b}": 0 for a in EXTRACT_STATES for b in EXTRACT_STATES}
        over = under = 0

        for r in rows:
            gs = norm_extract_state(r.get("gold_state"))
            ps = norm_extract_state(r.get("pred_state"))
            confusion[f"{gs}->{ps}"] += 1
            if ps == _FAILURE_STATE:
                missing += 1
            if gs != ps:
                continue
            if gs != _VALUE_STATE:
                correct += 1
                continue
            ok, _ = match_short(r.get("gold_value"), r.get("pred_value"))
            if ok:
                correct += 1
            else:
                g, p = parse_amount(r.get("gold_value")), parse_amount(r.get("pred_value"))
                if g is not None and p is not None:
                    over += int(p > g)
                    under += int(p < g)

        acc = correct / n if n else 0.0
        absence_as_failure = sum(confusion[f"{a}->{_FAILURE_STATE}"] for a in _INFO_ABSENCE_STATES)
        failure_as_absence = sum(confusion[f"{_FAILURE_STATE}->{a}"] for a in _INFO_ABSENCE_STATES)
        n_absence_gold = sum(
            1 for r in rows if norm_extract_state(r.get("gold_state")) in _INFO_ABSENCE_STATES)
        n_failure_gold = sum(
            1 for r in rows if norm_extract_state(r.get("gold_state")) == _FAILURE_STATE)

        sev = column_severity.get(col, "minor")
        gate = GATE_BY_SEVERITY.get(sev, 0.8)
        entry = {
            "n": n,
            "field_tag": sev,
            "accuracy": round(acc, 4),
            "missing_rate": round(missing / n, 4) if n else 0.0,
            "state_confusion": {k: v for k, v in confusion.items() if v},
            "absent_vs_failed_confusion_rate": round(
                (absence_as_failure + failure_as_absence)
                / max(1, n_absence_gold + n_failure_gold), 4),
            "error_direction": {"over": over, "under": under},
            "gate_threshold": gate,
            "gate_pass": acc >= gate,
            "note": ("★정보성 부재↔추출 실패 혼동은 부정 조건 질의를 조용히 틀리게 만든다"
                     if (absence_as_failure or failure_as_absence) else ""),
        }
        if n_conflict:
            entry["n_conflict_excluded"] = n_conflict  # C 결정: v1 채점 제외
        report[col] = entry

    n_all = sum(v["n"] for v in report.values())
    return {
        "columns": report,
        "n_samples": n_all,
        "n_conflict_excluded": sum(v.get("n_conflict_excluded", 0) for v in report.values()),
        "gate_failed_columns": [c for c, v in report.items() if not v["gate_pass"]],
        "overall_accuracy": round(
            sum(v["accuracy"] * v["n"] for v in report.values()) / n_all, 4) if n_all else 0.0,
        "reading_rule": "★선별형 점수는 이 숫자와 함께 읽는다. 단독 해석 금지(3-4-1).",
    }


def circularity_flag(selection_score: float, extraction_accuracy: float,
                     answer_sources: list[str], gap: float = 0.15) -> dict:
    """D7 순환 경보. 추출 정확도 ↓ + 선별형 점수 ↑ ⇒ 정답도 같은 테이블로 만들었을 가능성.

    answer_sources 는 EvaluationItem.answer_source 값들의 목록 — "table" 인 비율을 본다.
    """
    table_sourced = sum(1 for s in answer_sources if s == "table")
    ratio = table_sourced / len(answer_sources) if answer_sources else 0.0
    suspicious = (selection_score - extraction_accuracy) > gap and ratio > 0.0
    return {
        "suspicious": suspicious,
        "selection_score": selection_score,
        "extraction_accuracy": extraction_accuracy,
        "table_sourced_answer_ratio": round(ratio, 3),
        "message": (
            "★순환 경보: 추출 정확도보다 선별형 점수가 높다. 정답이 시스템과 같은 재료에서 "
            "나왔는지 확인하고, 표본 원문 대조로 정답 신뢰도를 재라 (2-8-1·3-2-1)."
            if suspicious else "순환 신호 없음"),
    }


# ------------------------------------------------------------------ 3-2-2 문서 특정

def grade_doc_selection(item: EvaluationItem, response: ModelResponse,
                        doc_org: dict[str, str] | None = None) -> dict:
    """문서 미특정(v0.2: unspecified_type 존재) 문항의 문서 특정 단계를 따로 잰다.

    intermediate_answer 가 중간 정답(후보 문서 id — v0.2에서 문자열 또는 배열)이다.
    2-2-2 문항의 실패는 두 갈래다 — ★문서를 못 찾았거나, 문서는 맞는데 값을 틀렸거나.
    합쳐서 오답 하나로 세면 어느 쪽을 고쳐야 할지 알 수 없다.
    """
    if not item.document_unspecified:
        return {"applicable": False}
    gold = set(as_id_list(item.intermediate_answer))
    if not gold:
        return {"applicable": False, "reason": "intermediate_answer 미기록 — 3-2-2 분리 채점 불가"}

    got = set(response.selected_document_ids)
    correct = bool(gold & got)

    if correct and got <= gold:
        kind = "correct"
    elif not got:
        kind = "no_selection"  # 되물었거나 아무것도 못 고름
    elif correct:
        kind = "over_selection"
    else:
        doc_org = doc_org or {}
        gold_orgs = {doc_org.get(d) for d in gold}
        got_orgs = {doc_org.get(d) for d in got}
        kind = "same_org_confusion" if gold_orgs & got_orgs else "unrelated_doc"

    return {"applicable": True, "correct": correct, "error_kind": kind,
            "gold": sorted(gold), "selected": sorted(got)}


def aggregate_doc_selection(rows: list[dict]) -> dict:
    rows = [r for r in rows if r.get("applicable")]
    if not rows:
        return {"n": 0}
    kinds: dict[str, int] = defaultdict(int)
    for r in rows:
        kinds[r["error_kind"]] += 1
    n = len(rows)
    return {"n": n,
            "doc_selection_accuracy": round(sum(1 for r in rows if r["correct"]) / n, 4),
            "error_kinds": dict(kinds),
            "note": "1단계에서 놓친 문서는 복구가 안 된다 — 이 값이 추출형 성능의 상한(4-9-6)"}
