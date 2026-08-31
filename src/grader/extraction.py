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

[대기 ← 박예진 1-12-1/1-12-2] 컬럼 목록·3상태·값 형식 확정 시 GATE_BY_SEVERITY(=field_tag
기준) 임계값을 실측으로 채운다. 지금은 임시값이며 baseline 실측 후 조정한다.
────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from collections import defaultdict

from .models import EvaluationItem, ExtractState, ModelResponse, as_id_list
from .normalize import match_short, parse_amount

EXTRACT_STATES: tuple[ExtractState, ...] = ("value", "absent", "failed")

# field_tag(critical/major/minor) 별 요구 정확도. [대기] 실측 전 임시값 — baseline 이후 조정.
GATE_BY_SEVERITY = {"critical": 0.95, "major": 0.90, "minor": 0.80}


def grade_extraction_audit(audit_rows: list[dict],
                           column_severity: dict[str, str] | None = None) -> dict:
    """표본 원문 대조 결과(박예진 4-6-2 산출물)를 지표로 바꾼다.

    audit_rows 한 줄의 계약:
      {"document_id": "...", "column": "budget",
       "pred_state": "value|absent|failed", "pred_value": "5억원",
       "gold_state": "value|absent|failed", "gold_value": "500,000,000원"}

    반환: 컬럼별 정확도 / 결측률 / ★"항목 없음 vs 추출 실패" 혼동률 / 오차 방향 / 게이트 판정
    """
    column_severity = column_severity or {}
    by_col: dict[str, list[dict]] = defaultdict(list)
    for r in audit_rows:
        by_col[r["column"]].append(r)

    report: dict[str, dict] = {}
    for col, rows in sorted(by_col.items()):
        n = len(rows)
        correct = 0
        missing = 0
        confusion = {f"{a}->{b}": 0 for a in EXTRACT_STATES for b in EXTRACT_STATES}
        over = under = 0

        for r in rows:
            gs, ps = r.get("gold_state", "value"), r.get("pred_state", "value")
            confusion[f"{gs}->{ps}"] += 1
            if ps == "failed":
                missing += 1
            if gs != ps:
                continue
            if gs != "value":
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
        absent_as_failed = confusion["absent->failed"]
        failed_as_absent = confusion["failed->absent"]
        n_absent_gold = sum(1 for r in rows if r.get("gold_state") == "absent")
        n_failed_gold = sum(1 for r in rows if r.get("gold_state") == "failed")

        sev = column_severity.get(col, "minor")
        gate = GATE_BY_SEVERITY.get(sev, 0.8)
        report[col] = {
            "n": n,
            "field_tag": sev,
            "accuracy": round(acc, 4),
            "missing_rate": round(missing / n, 4) if n else 0.0,
            "state_confusion": {k: v for k, v in confusion.items() if v},
            "absent_vs_failed_confusion_rate": round(
                (absent_as_failed + failed_as_absent) / max(1, n_absent_gold + n_failed_gold), 4),
            "error_direction": {"over": over, "under": under},
            "gate_threshold": gate,
            "gate_pass": acc >= gate,
            "note": ("★absent↔failed 혼동은 부정 조건 질의를 조용히 틀리게 만든다"
                     if (absent_as_failed or failed_as_absent) else ""),
        }

    n_all = sum(v["n"] for v in report.values())
    return {
        "columns": report,
        "n_samples": n_all,
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
