"""수정 전·후 문항별 점수 변화와 변화 이유(§8-9).

같은 실제 응답 50건을 (v1 평가셋 + v1 채점기) 와 (v2 평가셋 + v2 채점기) 로
각각 채점한 결과를 비교한다.
"""
import json, sys
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent


def load(per_item: Path) -> dict:
    rows = {}
    for line in per_item.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            rows[r["id"]] = r
    return rows


def score_of(r):
    return (r.get("task_score") or {}).get("score")


def cite_of(r):
    c = r.get("citation") or {}
    return ("N/A" if not c.get("applicable") else round(c.get("score", 0.0), 3))


def main(before_dir: Path, after_dir: Path, changes_path: Path, out_path: Path) -> int:
    b, a = load(before_dir / "per_item.jsonl"), load(after_dir / "per_item.jsonl")
    ch = json.loads(changes_path.read_text(encoding="utf-8"))
    sel, oth = ch["selection"], ch["others"]

    def why(qid):
        reasons = []
        if qid in sel:
            s = sel[qid]
            if s["question_changed"]:
                reasons.append("질문 재작성(미기재/공동수급 의미 정정)")
            if s["gold_changed"]:
                reasons.append(f"정답 집합 변경(+{len(s['added'])}/-{len(s['removed'])}) "
                               f"— 공동수급 재분류")
            if s.get("n_evidence"):
                reasons.append(f"근거 {s['n_evidence']}건 신규 부착")
        if qid in oth:
            reasons += oth[qid]["changes"]
        return reasons or ["평가셋 변경 없음 — 채점기 수정 효과"]

    rows = []
    for qid in sorted(set(b) | set(a)):
        rows.append({
            "id": qid,
            "score_before": score_of(b.get(qid, {})),
            "score_after": score_of(a.get(qid, {})),
            "status_before": b.get(qid, {}).get("final_status"),
            "status_after": a.get(qid, {}).get("final_status"),
            "citation_before": cite_of(b.get(qid, {})),
            "citation_after": cite_of(a.get(qid, {})),
            "reasons": why(qid),
        })
    changed = [r for r in rows if r["score_before"] != r["score_after"]]
    cite_new = [r for r in rows if r["citation_before"] == "N/A" and r["citation_after"] != "N/A"]
    out = {
        "n_items": len(rows),
        "n_score_changed": len(changed),
        "n_citation_newly_graded": len(cite_new),
        "citation_graded_before": sum(1 for r in rows if r["citation_before"] != "N/A"),
        "citation_graded_after": sum(1 for r in rows if r["citation_after"] != "N/A"),
        "caveat": ("저장된 응답은 v1 평가셋의 **옛 질문**에 대해 생성된 것이다. 질문이 "
                   "재작성된 선별형 14문항과 답 유형이 바뀐 추출형·QA 문항은 점수 하락이 "
                   "모델 성능 저하가 아니라 '옛 답변을 새 계약으로 채점'한 결과다. "
                   "새 질문에 대한 점수는 새 API 실행이 필요하다."),
        "rows": rows,
    }
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"{'문항':<9}{'점수 전':>7}{'점수 후':>7}  {'근거 전':>7}{'근거 후':>7}  변화 이유")
    print("-" * 108)
    for r in rows:
        if r["score_before"] == r["score_after"] and r["citation_before"] == r["citation_after"]:
            continue
        sb = "—" if r["score_before"] is None else f"{r['score_before']:.2f}"
        sa = "—" if r["score_after"] is None else f"{r['score_after']:.2f}"
        print(f"{r['id']:<9}{sb:>7}{sa:>7}  {str(r['citation_before']):>7}{str(r['citation_after']):>7}"
              f"  {'; '.join(r['reasons'])[:60]}")
    print("-" * 108)
    print(f"점수가 바뀐 문항 {len(changed)} / 근거 채점이 새로 가능해진 문항 {len(cite_new)}")
    print(f"근거 채점 대상: {out['citation_graded_before']} → {out['citation_graded_after']}")
    return 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])))
