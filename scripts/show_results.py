#!/usr/bin/env python3
"""채점 결과를 사람이 읽는 형태로 — 문항별 질문/정답/시스템답/판정/이유.

  python scripts/show_results.py --evaluation-set <evalset.jsonl> --per-item artifacts/scores/per_item.jsonl
  python scripts/show_results.py ... --only FAIL          # 틀린 것만
  python scripts/show_results.py ... --id PRAC-QA-003     # 특정 문항

grader run 이 낸 per_item.jsonl 을 그대로 먹는다. "이 채점기가 정답을 맞다고 했는지
틀리다고 했는지, 왜" 를 눈으로 확인하는 용도(3-14).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from checks.check_evalset import load_jsonl  # noqa: E402


def _s(v, n=90):
    t = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
    return t if len(t) <= n else t[:n] + "…"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--evaluation-set", required=True)
    ap.add_argument("--per-item", default="artifacts/scores/per_item.jsonl")
    ap.add_argument("--only", choices=["PASS", "FAIL", "PENDING"], help="이 판정만")
    ap.add_argument("--id", help="이 문항만")
    a = ap.parse_args(argv)

    items = {it["id"]: it for it in load_jsonl(a.evaluation_set)}
    results = {r["id"]: r for r in load_jsonl(a.per_item)}

    n_pass = n_fail = n_other = 0
    for rid, r in results.items():
        it = items.get(rid, {})
        final = r.get("final_status", "?")
        if a.id and rid != a.id:
            continue
        if a.only and not final.upper().startswith(a.only):
            continue
        if final == "PASS":
            n_pass += 1
        elif final.startswith("FAIL"):
            n_fail += 1
        else:
            n_other += 1

        ts = r.get("task_score", {})
        fs = r.get("format_status", {})
        ab = r.get("abstention", {})
        mark = "○" if final == "PASS" else "✗" if final.startswith("FAIL") else "…"
        print(f"\n{mark} [{rid}]  {it.get('task_type')}/{it.get('answer_type')}"
              f"  field_tag={it.get('field_tag')}  → {final}")
        print(f"   Q      : {_s(it.get('question'))}")
        print(f"   정답   : {_s(it.get('answer_raw'))}")
        print(f"   시스템 : {_s(ts.get('detail', {}).get('pred') or '(구조화 답변)')}")
        why = ts.get("detail", {}).get("why") or ts.get("detail", {}).get("fail_reason") \
            or ts.get("detail", {}).get("note")
        print(f"   판정   : task_score={ts.get('score')}  {why or ''}")
        if not fs.get("passed", True):
            print(f"   형식위반: {fs.get('violations')}")
        if ab.get("should_abstain") or ab.get("abstention_kind") not in (None, "ok"):
            print(f"   기권   : {ab.get('abstention_kind')}")
        miss = ts.get("detail", {}).get("missing")
        extra = ts.get("detail", {}).get("extra")
        if miss:
            print(f"   빠뜨림 : {miss}")
        if extra:
            print(f"   덧붙임 : {extra}")
        wrong = ts.get("detail", {}).get("wrong_cells")
        if wrong:
            print(f"   틀린칸 : {json.dumps(wrong, ensure_ascii=False)}")

    print(f"\n{'─'*60}\n합계: PASS {n_pass} / FAIL {n_fail} / 기타 {n_other}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
