"""
grader.regression — 3-13 변동 폭 / 3-13-1 회귀 판정 / 3-17 원인 분리

★ 순서가 절대적이다:  변동 폭 측정 → 판정선 → CI ON.
  사람이 볼 때는 "이 정도 차이는 노이즈겠지"로 넘길 수 있지만, CI는 스스로 판단하지 못한다.
  숫자로 된 선을 줘야 통과·실패를 가른다. 선 없이 CI를 켜면 노이즈마다 경보가 울리고,
  곧 아무도 안 보게 된다.

측정은 태스크별·집계 축별로 따로 한다 — 축마다 흔들림의 크기가 다르다.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field
from statistics import mean, pstdev

from .models import PROVENANCE_ASSETS, PROVENANCE_FIELDS


# ------------------------------------------------------------------ 3-13

def measure_variance(runs: list[dict], keys: list[str] | None = None) -> dict:
    """같은 설정으로 여러 번 돌린 결과에서 변동 폭을 잰다.

    runs: [{"overall_score":0.61,"selection":0.55,...}, ...] (같은 설정·같은 평가셋)
    """
    if len(runs) < 2:
        return {"n_runs": len(runs), "usable": False,
                "note": "★재실행 2회 미만 — 변동 폭 없음. 이 상태로 CI를 켜지 말 것"}
    keys = keys or sorted({k for r in runs for k in r})
    out = {}
    for k in keys:
        vals = [r[k] for r in runs if isinstance(r.get(k), (int, float))]
        if len(vals) < 2:
            continue
        sd = pstdev(vals)
        out[k] = {
            "n_runs": len(vals),
            "mean": round(mean(vals), 4),
            "sd": round(sd, 4),
            "range": round(max(vals) - min(vals), 4),
            "noise_band_pp": round(sd * 100, 2),
        }
    return {"n_runs": len(runs), "usable": True, "metrics": out,
            "rule": "이 변동 폭보다 작은 실험 간 차이는 '차이'라고 부르지 않는다"}


def pp_per_item(n_items: int) -> float:
    return round(100.0 / n_items, 2) if n_items else float("inf")


# ------------------------------------------------------------------ 3-13-1

@dataclass
class RegressionPolicy:
    """게이트와 경고를 구분한다.
       ★전부 게이트로 만들면 개발이 멈추고, 전부 경고로 만들면 아무도 안 본다.
    n_sigma·게이트 지표 목록은 [대기] baseline 실측 이후 확정 — 지금은 운영하며 조정하는
    임시값이며, 바꿀 때는 반드시 이력을 남긴다."""
    n_sigma: float = 2.0
    gate_metrics: list[str] = field(default_factory=lambda: [
        "field_tag.critical", "extraction.overall_accuracy"])
    warn_metrics: list[str] = field(default_factory=lambda: [
        "overall_score", "retrieval.context_k.recall@k", "selection", "extraction", "qa"])
    min_cell: int = 5
    absolute_gates: dict = field(default_factory=lambda: {
        "field_tag.critical": 0.85})


def judge_regression(baseline: dict, current: dict, variance: dict,
                     policy: RegressionPolicy | None = None,
                     cell_sizes: dict | None = None,
                     excluded_items: list[str] | None = None) -> dict:
    """회귀 판정. baseline/current 는 {지표경로: 값} 평탄 dict.

    깨진 문항을 제외하고 쟀다면 ★제외 문항 수를 반드시 함께 남긴다.
    """
    policy = policy or RegressionPolicy()
    cell_sizes = cell_sizes or {}
    vmetrics = (variance or {}).get("metrics", {})
    gates, warns, skipped = [], [], []

    def band(metric: str) -> float | None:
        v = vmetrics.get(metric)
        return v["sd"] if v else None

    seen: set[str] = set()
    for metric in policy.gate_metrics + policy.warn_metrics:
        if metric in seen:
            continue
        seen.add(metric)
        cur, base = current.get(metric), baseline.get(metric)
        if cur is None or base is None:
            continue
        n = cell_sizes.get(metric)
        if n is not None and n < policy.min_cell:
            skipped.append({"metric": metric, "n": n,
                            "reason": f"칸당 문항 {n} < {policy.min_cell} — 흔들림이 커서 경보 남발"})
            continue
        sd = band(metric)
        delta = cur - base
        threshold = -(policy.n_sigma * sd) if sd is not None else None
        abs_gate = policy.absolute_gates.get(metric)
        record = {"metric": metric, "baseline": base, "current": cur,
                  "delta": round(delta, 4), "sd": sd,
                  "threshold": round(threshold, 4) if threshold is not None else None,
                  "abs_gate": abs_gate, "n": n}

        reasons: list[str] = []
        if threshold is not None and delta <= threshold:
            reasons.append(f"변동 폭의 {policy.n_sigma}배 이상 하락")
        # ★절대 기준선은 변동 폭(sd) 측정과 무관하게 항상 적용한다 — 이 게이트의 존재
        #   이유가 "분산을 아직 안 쟀어도 이 값 밑으로는 절대 안 된다"이기 때문.
        #   예전에는 sd 미측정 시 절대 기준선 위반도 조용히 skipped 로 새어나갔다.
        if abs_gate is not None and cur < abs_gate:
            reasons.append(f"절대 기준선 {abs_gate} 미달")

        if reasons:
            record["reason"] = "; ".join(reasons)
            (gates if metric in policy.gate_metrics else warns).append(record)
        elif sd is None:
            # 절대 기준선은 통과(또는 없음)했고 상대 판정만 불가능한 경우.
            record["reason"] = "★변동 폭 미측정 — 상대 판정 불가(3-13 선행). 절대 기준선만 적용됨"
            skipped.append(record)

    status = "FAIL" if gates else ("WARN" if warns else "PASS")
    return {
        "status": status,
        "gate_failures": gates,
        "warnings": warns,
        "skipped": skipped,
        "excluded_items": excluded_items or [],
        "excluded_count": len(excluded_items or []),
        "policy": {"n_sigma": policy.n_sigma, "gate_metrics": policy.gate_metrics,
                   "min_cell": policy.min_cell},
        "note": ("★부분 회귀 확인: 총점은 유지되는데 critical 필드만 떨어지는 것이 "
                 "이 도메인의 최악 시나리오다. 판정 근거를 실험 기록에 남길 것."),
    }


# ------------------------------------------------------------------ 3-17

# 팀 확정 6-자산 provenance 축(키, 사람이 읽는 라벨). 키·순서·필드명은
# models.PROVENANCE_ASSETS 한 곳에서만 정의하고 여기서는 라벨만 붙인다 —
# manifest.provenance(runner.provenance())가 내는 6개 필드와 정확히 1:1이어야
# 회귀 원인 분리가 축을 놓치지 않는다.
_AXIS_LABELS = dict(PROVENANCE_ASSETS)
CHANGE_AXES = [(key, _AXIS_LABELS[key]) for key in PROVENANCE_FIELDS]


def attribute_change(prev_manifest: dict, cur_manifest: dict) -> dict:
    """점수가 움직인 원인을 팀 확정 6-자산 provenance 축으로 가른다
    (corpus/preprocess/table/index/evalset/scorer).

    ★못 가르면 코퍼스나 평가셋 때문에 떨어진 점수를 보고 멀쩡한 시스템 변경을 되돌리게 된다.
    가장 값싼 방법은 **한 번에 하나만 바꾸는 것**이다.

    6개 축을 전부 diff 한다 — manifest.provenance 에 키가 없으면(구버전 report 등) None
    으로 읽혀 '안 바뀜'으로 잡히므로, runner 는 6축을 항상 "UNKNOWN"으로라도 채워 낸다.
    """
    changed = []
    for key, label in CHANGE_AXES:
        if prev_manifest.get(key) != cur_manifest.get(key):
            changed.append({"key": key, "axis": label,
                            "from": prev_manifest.get(key), "to": cur_manifest.get(key)})
    axes = sorted({c["axis"] for c in changed})
    return {
        "axes_checked": list(PROVENANCE_FIELDS),  # 이번 비교가 확인한 6축(항상 6개)
        "changed": changed,
        "changed_keys": [c["key"] for c in changed],
        "axes": axes,
        "clean_comparison": len(axes) <= 1,
        "corpus_changed_flag": any(c["key"] == "corpus" for c in changed),
        "preprocess_changed_flag": any(c["key"] == "preprocess" for c in changed),
        "table_changed_flag": any(c["key"] == "table" for c in changed),
        "index_changed_flag": any(c["key"] == "index" for c in changed),
        "evalset_changed_flag": any(c["key"] == "evalset" for c in changed),
        "scorer_changed_flag": any(c["key"] == "scorer" for c in changed),
        # 하위호환 별칭
        "grader_changed_flag": any(c["key"] == "scorer" for c in changed),
        "message": (
            "★한 번에 둘 이상이 바뀌었다(" + ", ".join(axes) + "). 이 비교로는 원인을 가를 수 없다. "
            "이전 버전으로 한 번 더 돌려 차이를 분리하거나, 최소한 결과에 두 버전을 남겨라."
            if len(axes) > 1 else
            ("변경 축 1개 — 비교 가능" if axes else "변경 없음 — 재실행 변동 폭 측정 조건")),
        "rise_check": "갱신 후 점수가 ★올랐을 때도 확인한다. 어려운 문서가 빠져서 오른 경우를 놓친다.",
    }


# ------------------------------------------------------------------ CLI 보조

def flatten_report(report: dict) -> dict:
    """report.json → 회귀 판정용 평탄 지표 dict."""
    s = report.get("summary") or report
    out = {"overall_score": s.get("overall_score")}
    for t, v in (s.get("main", {}).get("by_task_type") or {}).items():
        out[t] = v.get("score")
    for sev, v in (s.get("severity", {}).get("by_field_tag") or {}).items():
        out[f"field_tag.{sev}"] = v.get("score")
    r = (s.get("retrieval") or {}).get("context_k") or {}
    if "recall@k" in r:
        out["retrieval.context_k.recall@k"] = r["recall@k"]
    e = s.get("extraction") or {}
    if "overall_accuracy" in e:
        out["extraction.overall_accuracy"] = e["overall_accuracy"]
    return {k: v for k, v in out.items() if isinstance(v, (int, float))}


def cell_sizes_of(report: dict) -> dict:
    s = report.get("summary") or report
    out = {}
    for t, v in (s.get("main", {}).get("by_task_type") or {}).items():
        out[t] = v.get("n")
    for sev, v in (s.get("severity", {}).get("by_field_tag") or {}).items():
        out[f"field_tag.{sev}"] = v.get("n")
    out["overall_score"] = s.get("n_items")
    return out


def _load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="3-13 변동 폭 측정 / 3-13-1 회귀 판정 / 3-17 원인 분리")
    ap.add_argument("--runs", nargs="*", default=[],
                    help="같은 설정으로 반복 실행한 report.json 들 (변동 폭 측정)")
    ap.add_argument("--baseline", help="비교 기준 report.json")
    ap.add_argument("--current", help="이번 실행 report.json")
    ap.add_argument("--variance", help="이미 측정해 둔 variance.json")
    ap.add_argument("--n-sigma", type=float, default=2.0)
    ap.add_argument("--out", default="out/variance.json")
    a = ap.parse_args(argv)

    var = None
    if a.runs:
        var = measure_variance([flatten_report(_load(p)) for p in a.runs])
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(var, f, ensure_ascii=False, indent=2)
        print(json.dumps(var, ensure_ascii=False, indent=2))
    elif a.variance:
        var = _load(a.variance)

    if a.baseline and a.current:
        base_rep, cur_rep = _load(a.baseline), _load(a.current)
        verdict = judge_regression(
            flatten_report(base_rep), flatten_report(cur_rep), var or {"metrics": {}},
            RegressionPolicy(n_sigma=a.n_sigma), cell_sizes=cell_sizes_of(cur_rep),
            excluded_items=(cur_rep.get("manifest") or {}).get("excluded_items"))
        cause = attribute_change(
            (base_rep.get("manifest") or {}).get("provenance", {}),
            (cur_rep.get("manifest") or {}).get("provenance", {}))
        print(json.dumps({"regression": verdict, "cause_attribution": cause},
                         ensure_ascii=False, indent=2))
        return 1 if verdict["status"] == "FAIL" else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
