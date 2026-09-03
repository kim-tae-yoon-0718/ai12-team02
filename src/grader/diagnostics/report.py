"""grader.diagnostics.report — 3-12 최종 집계 리포트 + 3-1-0 팀 공용 진단표.

★ 3-1-0: 점수 하나는 뭘 고칠지 알려주지 않는다. 부품별로 나누면 조합이 범인을 특정한다.
"""

from __future__ import annotations

import argparse
import json

from .statistics import (
    _mean,
    abstention_report,
    cell_report,
    format_report,
    integrated_score,
    severity_report,
    severity_weighted_score,
)


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
        # 3-3-2 ①가중 평균 — 위 "severity"(②③)와 항상 나란히. 이 값이 있어도 대체하지 않는다.
        "severity_weighted_score": sev_score,
        "severity_weighted_note": sev_note,
        "by_field_tag": cell_report(results, "field_tag"),
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
     "★코퍼스·평가셋이 움직였을 가능성 (diagnostics.regression.attribute_change)",
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
    # ★"전 지표 정상"은 실제로 넘어온 지표들만 놓고 판단한다.
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
