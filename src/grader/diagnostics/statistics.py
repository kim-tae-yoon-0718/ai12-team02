"""grader.diagnostics.statistics — 집계 프리미티브 (3-12 / 3-3-2 / 3-3 / 2-16).

★ 3-12: 집계 축이 늘수록 칸당 문항이 급감한다 ⇒ 칸 크기를 자동으로 경고한다.
★ 3-3-2: 등급 반영은 ②분리 집계(severity_report) + ③게이트 + ①가중 평균
  (severity_weighted_score) 를 항상 나란히. ①이 ②③를 대체하지 않는다 —
  가중 평균 하나만 보면 "총점 유지 + critical만 악화"가 다시 묻힌다.
"""

from __future__ import annotations

from collections import defaultdict
from statistics import mean

MIN_CELL = 5  # 이보다 적은 칸은 "숫자로 말하지 않는다"


def _mean(xs) -> float:
    xs = list(xs)
    return round(mean(xs), 4) if xs else 0.0


def by_axis(results: list[dict], axis: str) -> dict[str, list[dict]]:
    buckets: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        buckets[str(r.get(axis))].append(r)
    return buckets


def cell_report(results: list[dict], axis: str, min_cell: int = MIN_CELL) -> dict:
    out = {}
    for k, rows in sorted(by_axis(results, axis).items()):
        n = len(rows)
        out[k] = {
            "n": n,
            "score": _mean(r["score"] for r in rows),
            "pp_per_item": round(100.0 / n, 2) if n else None,
            "reliable": n >= min_cell,
            "warning": None if n >= min_cell else f"칸당 문항 {n}개 — 숫자로 말하지 말 것(3-12)",
        }
    return out


def severity_report(results: list[dict], gate: dict[str, float] | None = None) -> dict:
    """3-3-2 ②등급별 분리 집계 + ③문지기(field_tag 축)."""
    gate = gate or {"critical": 0.85}
    rep = cell_report(results, "field_tag")
    gate_result = {}
    for sev, threshold in gate.items():
        cell = rep.get(sev)
        if not cell:
            gate_result[sev] = {"status": "no_items", "threshold": threshold}
            continue
        gate_result[sev] = {
            "status": "pass" if cell["score"] >= threshold else "FAIL",
            "score": cell["score"], "threshold": threshold, "n": cell["n"],
            "reliable": cell["reliable"],
        }
    return {"by_field_tag": rep, "gate": gate_result,
            "note": "critical 필드의 기권은 감점이 아니라 정상 동작일 수 있다(1-5) — abstention 리포트와 함께 볼 것"}


def severity_weighted_score(results: list[dict],
                            field_tag_weight: dict[str, float] | None) -> tuple[float | None, str]:
    """3-3-2 ①가중 평균 — 문항별 점수를 field_tag 가중치(critical:major:minor=5:3:2)로
    가중 평균. gate.field_tag_weight 가 비어 있으면 계산하지 않는다(감으로 안 채움).
    field_tag 없는 문항·가중치 없는 태그의 문항은 제외한다.
    """
    if not field_tag_weight:
        return None, ("field_tag 가중 채점 방식 자체는 확정(2026-08-28 재확인, 3-3-2 ①재도입) "
                      "— 다만 gate.field_tag_weight 값이 아직 없어 계산하지 않는다. 감으로 채운 "
                      "가중치는 근거 없는 우선순위 선언과 같다.")
    total_w = acc = 0.0
    used: list[str] = []
    excluded_no_tag = 0
    for r in results:
        tag = r.get("field_tag")
        if tag is None:
            excluded_no_tag += 1
            continue
        w = field_tag_weight.get(tag)
        if not w:
            continue
        acc += r["score"] * w
        total_w += w
        used.append(tag)
    if total_w == 0:
        return None, "gate.field_tag_weight에 이 실행의 field_tag와 겹치는 항목이 없다."
    note = (f"가중 평균(가중치 출처: gate.field_tag_weight) — 사용된 field_tag: "
            f"{sorted(set(used))}, n={len(used)}")
    if excluded_no_tag:
        note += f", field_tag 없어 제외된 문항 {excluded_no_tag}건"
    return round(acc / total_w, 4), note


def abstention_report(results: list[dict]) -> dict:
    """3-3 기권율 분리. 과잉 거절 / 환각 / ★critical_abstain(정상 동작 후보) 을 따로 센다."""
    kinds: dict[str, int] = defaultdict(int)
    for r in results:
        kinds[(r.get("abstention") or {}).get("abstention_kind", "ok")] += 1
    n = len(results) or 1
    return {
        "counts": dict(kinds),
        "hallucination_rate": round(kinds["hallucination"] / n, 4),
        "over_refusal_rate": round(kinds["over_refusal"] / n, 4),
        "critical_abstain_count": kinds["critical_abstain"],
        "clarification_count": kinds["clarification_ok"],
        "note": "critical_abstain 은 1-5 기준 정상 동작일 수 있음 — 과잉 거절과 합산 금지. "
                "clarification_ok(모호한 문항에 대한 정당한 되묻기)도 과잉 거절과 별도 집계",
    }


def format_report(results: list[dict]) -> dict:
    """3-3/3-6: 형식 계약 위반 비율. 내용이 맞아도 여기서 걸리면 최종 FAIL이다."""
    n = len(results) or 1
    by_status: dict[str, int] = defaultdict(int)
    for r in results:
        by_status[r.get("final_status", "UNKNOWN")] += 1
    fmt_failed = sum(v for k, v in by_status.items() if k.startswith("FAIL-format"))
    return {"by_final_status": dict(by_status),
            "format_failure_rate": round(fmt_failed / n, 4),
            "note": "FAIL-format* 은 내용과 무관하게 output contract 위반으로 떨어진 문항"}


def integrated_score(by_task: dict, task_weight: dict[str, float] | None) -> tuple[float | None, str]:
    """2-16: 통합 점수 산출 방식은 확정 — 가중치는 1-2(업무 빈도·위험). 값이 없으면
    감으로 채우지 않고 None + 이유."""
    if not task_weight:
        return None, ("통합 점수 산출 방식은 확정(2-16) — 가중치는 1-2 업무빈도·위험 기준. "
                      "다만 gate.task_weight 값이 아직 없어 계산하지 않는다. 감으로 채운 "
                      "가중치는 근거 없는 우선순위 선언과 같다.")
    total_w = acc = 0.0
    used = []
    for t, w in task_weight.items():
        cell = by_task.get(t)
        if cell is None or not w:
            continue
        acc += cell["score"] * w
        total_w += w
        used.append(t)
    if total_w == 0:
        return None, "gate.task_weight에 이 실행의 task_type과 겹치는 항목이 없다."
    return round(acc / total_w, 4), f"가중 평균(가중치 출처: gate.task_weight, 1-2 확정) — 사용된 태스크: {used}"
