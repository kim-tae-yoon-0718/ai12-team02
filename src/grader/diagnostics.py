"""
grader.diagnostics — 3-12 집계 / 3-3-2 등급 채점 / 3-5 주 지표 / 3-1-0 진단표(팀 공용) / 3-16

★ 3-12: 집계 축이 늘수록 칸당 문항이 급감한다. task_type × field_tag까지 가면
  칸당 1~2문항이 되어 숫자가 무의미해진다 ⇒ 칸 크기를 자동으로 경고한다.
★ 3-3-2: 등급 반영 방식은 ②분리 집계 + ③문지기(CI 게이트)를 기본으로 유지하고,
  ①가중 평균도 함께 낸다 — 2026-08-28 재확인: field_tag의 의미가 "부분점수 인정
  여부"가 아니라 "오답 심각도(critical/major/minor) 표시 + 가중 채점 입력"으로
  바뀌면서, 한때 "가중치를 감으로 정하면 근거가 없다"고 기각했던 ①방식을 팀이
  다시 채택하기로 했다(severity_weighted_score 참고). ②③은 "총점이 유지돼도
  critical만 나빠진 상태"를 놓치지 않으려고 그대로 남겨 둔다 — ①이 그 역할을
  대신하지 않는다: 가중 평균 하나만 보면 critical 다수 문항의 악화가 다른 문항들
  개선에 묻혀 여전히 안 보일 수 있다. 그래서 이 셋을 항상 나란히 낸다.
★ 3-5: 주 지표는 태스크별로 각각. 선별=완결성 / 추출=정확성 / QA=충실성(+검색 재현율과 짝).
★ 3-1-0: 이 모듈은 개인 도구가 아니라 팀 공용 도구다. 점수 하나는 뭘 고칠지 알려주지 않는다.
  부품별로 나누면 조합이 범인을 특정한다.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from statistics import mean

MIN_CELL = 5  # 이보다 적은 칸은 "숫자로 말하지 않는다"


def _mean(xs) -> float:
    xs = list(xs)
    return round(mean(xs), 4) if xs else 0.0


def by_axis(results: list[dict], axis: str) -> dict[str, list[dict]]:
    buckets: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        v = r.get(axis)
        buckets[str(v)].append(r)
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
            "warning": None if n >= min_cell else
                       f"칸당 문항 {n}개 — 숫자로 말하지 말 것(3-12)",
        }
    return out


def severity_report(results: list[dict], gate: dict[str, float] | None = None) -> dict:
    """3-3-2 ②등급별 분리 집계 + ③문지기(field_tag 축).

    ★같은 무게로 세면 '치명 필드는 나빠졌는데 경 필드가 좋아져서 총점 유지'가
      개선처럼 보인다. 그래서 총점과 별개로 field_tag 점수를 항상 나란히 낸다.
    ★①가중 평균(severity_weighted_score)이 따로 있다고 이 함수를 대체하지 않는다 —
      가중 평균 하나로 합치면 여기서 잡는 "총점 유지 + critical만 악화" 패턴이
      다시 묻힐 수 있다. 셋을 항상 함께 본다(3-3-2, 2026-08-28 재확인).
    """
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
    """3-3-2 ①가중 평균 — 2026-08-28 재확인으로 재도입.

    field_tag(critical/major/minor)를 "critical 오류 1건 발생"처럼 건수·게이트
    통과 여부로만 보고하지 않고, 문항별 점수를 field_tag 가중치로 가중 평균해
    실제 점수 계산에 반영한다. 예: 사업금액(critical)을 틀린 문항은 사업기간
    (minor)을 틀린 문항보다 이 점수를 더 크게 끌어내려야 한다.

    ★ 여전히 가중치 숫자를 감으로 채우지 않는다(3-5 원래 원칙과 동일) —
    gate.field_tag_weight가 비어 있으면 계산하지 않고 이유를 note로 남긴다.
    field_tag가 없는(None) 문항, 또는 gate.field_tag_weight에 없는 태그의 문항은
    가중 평균에서 제외한다 — 중요도가 아예 안 매겨진 문항까지 임의의 가중치를
    주지 않는다.
    """
    if not field_tag_weight:
        return None, ("field_tag 가중 채점 방식 자체는 확정(2026-08-28 재확인, 3-3-2 ①재도입) "
                      "— 다만 gate.field_tag_weight 값이 아직 없어 계산하지 않는다. 감으로 채운 "
                      "가중치는 근거 없는 우선순위 선언과 같다.")
    total_w = 0.0
    acc = 0.0
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
        "note": "critical_abstain 은 1-5 기준 정상 동작일 수 있음 — 과잉 거절과 합산 금지",
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
    """2026-08-27 임현진 2-16 확정: 통합 점수 산출 방식 자체는 확정됐다 — 가중치는
    1-2(업무 빈도·위험)에서 끌어오고, 태스크별 3개 점수와 항상 나란히 표시한다.

    ★ 가중치 숫자(1-2)가 아직 안 왔으면 감으로 채우지 않는다 — 근거 없는 가중치는
    사실상 근거 없는 우선순위 선언이다(3-5 원래 원칙 유지)."""
    if not task_weight:
        return None, ("통합 점수 산출 방식은 확정(2-16) — 가중치는 1-2 업무빈도·위험 기준. "
                      "다만 gate.task_weight 값이 아직 없어 계산하지 않는다. 감으로 채운 "
                      "가중치는 근거 없는 우선순위 선언과 같다.")
    total_w = 0.0
    acc = 0.0
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


def main_metrics(results: list[dict], retrieval_agg: dict | None = None,
                 extraction_agg: dict | None = None, task_weight: dict[str, float] | None = None) -> dict:
    """3-5 태스크별 주 지표 + 2-16 통합 점수(태스크별 3개 점수와 항상 나란히 표시)."""
    by_task = cell_report(results, "task_type")
    out = {"by_task_type": by_task, "primary": {}}

    sel = by_task.get("selection")
    if sel:
        out["primary"]["selection"] = {
            "metric": "완결성(누락 없이 다 찾았는가) — recall 가중 점수",
            "value": sel["score"], "n": sel["n"],
            "must_read_with": "extraction.overall_accuracy (순환 경보)",
            "extraction_accuracy": (extraction_agg or {}).get("overall_accuracy"),
        }
    ext = by_task.get("extraction")
    if ext:
        crit = cell_report([r for r in results if r["task_type"] == "extraction"], "field_tag").get("critical")
        out["primary"]["extraction"] = {
            "metric": "값 정확도 + critical 필드 정답률",
            "value": ext["score"], "n": ext["n"],
            "critical_field_score": (crit or {}).get("score"),
        }
    qa = by_task.get("qa")
    if qa:
        ctx = (retrieval_agg or {}).get("context_k", {})
        out["primary"]["qa"] = {
            "metric": "충실성(judge_faithfulness) — 검색 재현율(context_k)과 짝으로 읽을 것(D7)",
            "value": qa["score"], "n": qa["n"],
            "context_recall": ctx.get("recall@k"),
        }
    score, note = integrated_score(by_task, task_weight)
    out["integrated_score"] = score
    out["integrated_note"] = note
    return out


def full_report(results: list[dict], retrieval_agg=None, extraction_agg=None,
                doc_select_agg=None, gate: dict | None = None,
                task_weight: dict[str, float] | None = None,
                citation_agg: dict | None = None,
                field_tag_weight: dict[str, float] | None = None) -> dict:
    """3-12 최종 집계."""
    sev_score, sev_note = severity_weighted_score(results, field_tag_weight)
    return {
        "n_items": len(results),
        "overall_score": _mean(r["score"] for r in results),
        "main": main_metrics(results, retrieval_agg, extraction_agg, task_weight),
        "severity": severity_report(results, gate),
        # 3-3-2 ①가중 평균 — 위 "severity"(②③)와 항상 나란히 낸다. 이 값이 있어도
        # severity/by_field_tag/gate를 대체하지 않는다(모듈 상단 note 참고).
        "severity_weighted_score": sev_score,
        "severity_weighted_note": sev_note,
        "by_field_tag": cell_report(results, "field_tag"),
        # v0.2(2026-08-27, 임현진): difficulty 필드 자체가 폐기됨(2-4 재확정) — by_difficulty 제거.
        "abstention": abstention_report(results),
        "format": format_report(results),
        "retrieval": retrieval_agg or {},
        "citation": citation_agg or {},
        "extraction": extraction_agg or {},
        "doc_selection": doc_select_agg or {},
        "failures": {
            "system_error_zero": sum(1 for r in results if r.get("failure")),
            "note": "오류로 빈 답이 나온 0점과 모델이 틀린 0점을 구분(4-5)",
        },
    }


# ------------------------------------------------------------------ 3-1-0 팀 공용 진단표

DEFAULT_THRESHOLDS = {
    "retrieval_recall": 0.70,
    "faithfulness": 0.80,
    "extraction_accuracy": 0.90,
    "selection_score": 0.70,
    "doc_selection_accuracy": 0.80,
    "task_score": 0.60,
}

STAGE_MAP = {
    "selection": ["①파싱", "②추출"],
    "extraction": ["①파싱", "③문서특정", "④검색", "⑤생성", "(또는 ②추출 캐시 조회)"],
    "qa": ["①파싱", "④검색", "⑤생성"],
}

RULES = [
    (lambda s: s.get("retrieval_recall_low") and s.get("faithfulness_high"),
     "검색이 범인 (근거를 못 가져옴)",
     "임베딩 교체 / 하이브리드 / 파싱 개선. failure_kind(recall/rank)를 먼저 가를 것"),
    (lambda s: s.get("retrieval_recall_high") and s.get("faithfulness_low"),
     "생성이 범인 (근거는 왔는데 지어냄)",
     "프롬프트 억제 강도 / 기권 관문(4-12-1) / 생성 모델 교체"),
    (lambda s: s.get("retrieval_recall_high") and s.get("faithfulness_high") and s.get("answer_wrong"),
     "★D7 함정 — 충실성은 진실을 재지 않는다",
     "낡거나 틀린 근거를 충실히 옮긴 경우. 검색 재현율과 짝으로 볼 것"),
    (lambda s: s.get("extraction_accuracy_low") and s.get("selection_score_low"),
     "추출 테이블이 범인",
     "검색 튜닝은 무의미. 추출 방법 / 표본 대조 장치부터"),
    (lambda s: s.get("extraction_accuracy_high") and s.get("selection_score_low"),
     "조건 질의 로직이 범인 (4-9-8)",
     "자연어 조건 번역·부정 조건의 '추출 실패' 행 처리·0건 동작 점검"),
    (lambda s: s.get("extraction_accuracy_low") and s.get("selection_score_high"),
     "★순환 경보 (2-8-1·3-2-1)",
     "정답도 같은 테이블로 만들었을 가능성. 표본 원문 대조로 정답 신뢰도를 재라"),
    (lambda s: s.get("doc_selection_accuracy_low") and s.get("extract_score_low"),
     "문서를 못 찾은 것 (3-2-2)",
     "값 추출은 멀쩡할 수 있다. 문서 특정 강제 실행으로 상한을 먼저 확인"),
    (lambda s: s.get("all_metrics_normal") and s.get("score_dropped"),
     "★코퍼스·평가셋이 움직였을 가능성 (regression.attribute_change)",
     "코퍼스 버전이 직전 실행과 같은지 확인. 시스템 변경으로 오독하면 멀쩡한 변경을 되돌리게 된다"),
]


def classify(values: dict, thresholds: dict | None = None) -> dict:
    th = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    s: dict = {}
    for key, t in th.items():
        v = values.get(key)
        if v is None:
            continue
        s[f"{key}_low"] = v < t
        s[f"{key}_high"] = v >= t
    s["answer_wrong"] = values.get("task_score", 1.0) < th["task_score"]
    s["extract_score_low"] = values.get("extract_score", 1.0) < th["task_score"]
    s["score_dropped"] = bool(values.get("score_dropped"))
    # ★"전 지표 정상"은 실제로 넘어온 지표들만 놓고 판단한다. 지표가 아예 없는데
    #   "정상"으로 치면 '전 지표 정상 + 점수 하락 → 코퍼스가 움직임' 규칙이 오발한다.
    present = [k for k in th if k != "task_score" and values.get(k) is not None]
    s["all_metrics_normal"] = bool(present) and all(not s[f"{k}_low"] for k in present)
    return s


def diagnose(values: dict, thresholds: dict | None = None) -> list[dict]:
    signals = classify(values, thresholds)
    out = []
    for cond, suspect, prescription in RULES:
        try:
            if cond(signals):
                out.append({"suspect": suspect, "prescription": prescription})
        except Exception:
            continue
    if not out:
        out.append({"suspect": "특정 안 됨",
                    "prescription": "지표 조합이 규칙에 안 걸린다. 샘플 확인으로 넘어갈 것"})
    return out


def render_table() -> str:
    lines = [
        "부품별 지표 진단표 (3-1-0)",
        "=" * 72,
        " 검색 재현율 ↓ + 충실성 ↑            => 검색이 범인 (근거를 못 가져옴)",
        " 검색 재현율 ↑ + 충실성 ↓            => 생성이 범인 (근거는 왔는데 지어냄)",
        " 검색 재현율 ↑ + 충실성 ↑ + 답 틀림  => ★D7 함정. 낡은 근거를 충실히 옮김",
        "-" * 72,
        " 추출 정확도 ↓ + 선별형 점수 ↓       => 테이블이 범인. 검색 튜닝은 무의미",
        " 추출 정확도 ↑ + 선별형 점수 ↓       => 조건 질의 로직이 범인 (4-9-8)",
        " 추출 정확도 ↓ + 선별형 점수 ↑       => ★순환 경보 (2-8-1·3-2-1)",
        " 문서 특정 ↓ + 추출형 점수 ↓         => 문서를 못 찾은 것. 값 추출은 멀쩡할 수 있다",
        " 전 지표 정상 + 점수만 하락          => ★코퍼스·평가셋이 움직였을 가능성",
        "=" * 72,
        "태스크별로 지나는 단계가 다르다 — 통합 점수 하나로는 진단 불가:",
    ]
    for t, stages in STAGE_MAP.items():
        lines.append(f"  {t:<10} = " + " → ".join(stages))
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="3-1-0 진단표")
    ap.add_argument("--report", help="grader run 이 낸 report.json")
    ap.add_argument("--table", action="store_true", help="진단표만 출력")
    a = ap.parse_args(argv)
    if a.table or not a.report:
        print(render_table())
        return 0
    with open(a.report, encoding="utf-8") as f:
        rep = json.load(f)
    summary = rep.get("summary", rep)
    values = {
        "retrieval_recall": ((summary.get("retrieval") or {}).get("context_k") or {}).get("recall@k"),
        "extraction_accuracy": (summary.get("extraction") or {}).get("overall_accuracy"),
        "selection_score": ((summary.get("main") or {}).get("by_task_type", {}).get("selection") or {}).get("score"),
        "extract_score": ((summary.get("main") or {}).get("by_task_type", {}).get("extraction") or {}).get("score"),
        "doc_selection_accuracy": (summary.get("doc_selection") or {}).get("doc_selection_accuracy"),
        "task_score": summary.get("overall_score"),
    }
    values = {k: v for k, v in values.items() if v is not None}
    print(render_table())
    print("\n[이번 실행 판정]")
    print(json.dumps({"values": values, "diagnosis": diagnose(values)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
