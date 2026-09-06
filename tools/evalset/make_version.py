"""후보 평가셋 VERSION.txt + provenance.json 생성(§7)."""
import json, sys
from collections import Counter
from pathlib import Path

import argparse, hashlib
sys.path.insert(0, str(Path(__file__).resolve().parent))
import provenance as P                                   # noqa: E402
from paths import add_path_args, repo_root, require      # noqa: E402

_ap = argparse.ArgumentParser(description="후보 평가셋 VERSION.txt·provenance.json 생성")
_ap.add_argument("--out", required=True, help="산출 폴더(evalset/candidate 가 들어 있는 폴더)")
_ap.add_argument("--patch", default=None,
                 help="동결된 최종 patch/worktree.patch — 이 파일의 SHA-256 을 기록")
add_path_args(_ap)
_args = _ap.parse_args()
OUT = Path(_args.out).expanduser().resolve()
require("산출 폴더(--out)", OUT)
FINAL_PATCH_SHA = (hashlib.sha256(require("최종 패치(--patch)", _args.patch).read_bytes()).hexdigest()
                   if _args.patch else None)
WT = repo_root(_args.repo_root)
TOOLS = Path(__file__).resolve().parent
cand = OUT / "evalset/candidate"
items = [json.loads(l) for l in (cand / "items.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
changes = json.loads((OUT / "evalset/changes_v2.json").read_text(encoding="utf-8"))

prov = P.build(
    worktree=WT, candidate_items=cand / "items.jsonl",
    policy_files=[WT / "src/checks/selection_policy.py", WT / "src/checks/check_evalset.py",
                  TOOLS / "build_evalset_v2.py", TOOLS / "items_v2_spec.py"],
    scorer_files=[WT / "src/grader/task_scoring.py", WT / "src/grader/retrieval.py",
                  WT / "src/grader/runner.py", WT / "src/grader/models.py",
                  WT / "src/grader/normalize.py", WT / "src/grader/judge.py",
                  WT / "src/grader/providers.py", WT / "src/grader/diagnostics/report.py",
                  WT / "config/grader.yaml"],
    judge_prompts=[WT / "data/prompts/judge_faithfulness.v1.md",
                   WT / "data/prompts/judge_checkpoint.v1.md",
                   WT / "data/prompts/judge_list_item.v1.md"])

sel, oth = changes["selection"], changes["others"]
qa_changed = sorted([k for k, v in sel.items() if v["question_changed"] or v["gold_changed"]]
                    + [k for k, v in oth.items()
                       if any(("answer_raw" in c) or ("answer_type" in c) or ("비교 셀" in c)
                              for c in v["changes"])])
meta_only = sorted([k for k, v in sel.items() if not (v["question_changed"] or v["gold_changed"])]
                   + [k for k, v in oth.items() if k not in qa_changed and v["changes"]])
prov["change_log"] = {
    "question_or_answer_changed": qa_changed,
    "metadata_only_changed": meta_only,
    "all_changed": sorted(set(qa_changed) | set(meta_only)),
    "note": "선별형 25문항 전부에 reference_time·evidence 가 붙었으므로 '메타데이터만 바뀐' "
            "문항과 '질문·정답이 바뀐' 문항을 나눠 적는다.",
}
prov["quota"] = dict(Counter(i["task_type"] for i in items))
# ★패치 지문은 '실제 최종 patch/worktree.patch 파일'의 해시만 기록한다. 예전의
#   working_patch_sha256(git diff+미추적 연결본)은 실제 패치 파일과 정의가 달라 값이
#   어긋났다 — 혼동을 없애기 위해 제거한다.
prov.pop("working_patch_sha256", None); prov.pop("working_patch_bytes", None)
prov["final_patch_sha256"] = FINAL_PATCH_SHA
prov["final_patch_path"] = "patch/worktree.patch"
CT = OUT / "extraction/rfp_extraction_table_v3_1_candidate"
prov["candidate_extraction_table"] = {
    "version": "v3.1-candidate.1",
    "json_sha256": P.sha256_file(CT / "extraction_table_v3_1_candidate.json"),
    "csv_sha256": P.sha256_file(CT / "extraction_table_v3_1_candidate.csv"),
    "decisions_sha256": P.sha256_file(CT / "semantic_decisions_v3_1_candidate.csv"),
    "metadata": json.loads((CT / "extraction_metadata.json").read_text(encoding="utf-8")),
    "changed_rows": [["RFP-000067", "컨소시엄 요건"], ["RFP-000081", "컨소시엄 요건"]],
}
prov["n_items"] = len(items)
(OUT / "evalset/provenance.json").write_text(
    json.dumps(prov, ensure_ascii=False, indent=2), encoding="utf-8")

q = prov["quota"]
ver = [
 "evalset: v1-candidate.5",
 "note: 후보본 5차 — 공식 evalset v1 에 아직 반영하지 않음. 정답은 후보 추출표 v3.1-candidate.1 로 재계산",
 "extraction_table: v3.1-candidate.1",
 "note_extraction_table: 후보 추출표(공식 v3 에서 RFP-000067·RFP-000081 컨소시엄 요건 두 행만 변경). 공식 v3 는 미변경",
 "corpus: v2", "preprocess: v2", "chunking: v3",
 "registry: v2", "identity: v2", "index: v2",
 "reference_time: 2024-06-01",
 f"items: {len(items)} (selection {q.get('selection',0)} / extraction {q.get('extraction',0)} / qa {q.get('qa',0)})",
 f"evalset_sha256: {prov['candidate_evalset_sha256']}",
 f"base_dev_commit: {prov['base_dev_commit']}",
 f"final_patch_sha256: {prov['final_patch_sha256'] or '(동결 전 — verify_fingerprints 실행 전)'}",
 f"generated: {prov['generated']}",
 "",
 "정답 산출 순서:",
 "  1) 등록부 active=true AND retrieval_eligible=true 인 98문서만 남김",
 "  2) 문항 조건 적용 (추출표 status / 공동수급 7상태 분류)",
 "  3) 기준일 2024-06-01 마감 필터 적용 (identity_v2.bid_deadline)",
 "  4) 마감일 미상은 '미상 표시 후 통과'",
 "  5) 남은 문서 전체를 정답으로 기록 (개수 제한 없음)",
 "",
 "공동수급 7상태: required_all / required_conditional / allowed_explicit /",
 "  forbidden_all / method_restricted / field_absent / undetermined",
 "  ★'명시적 금지'는 forbidden_all 뿐이다 — 특정 이행방식만 금지한 문서는 제외한다.",
 "",
 "입력 자료 SHA-256:",
]
ver += [f"  {k}: {v}" for k, v in prov["official_inputs"].items()]
ver += ["", "정답 생성·검사 코드 SHA-256:"]
ver += [f"  {k}: {v}" for k, v in prov["policy_files"].items()]
ver += ["", "채점기 코드 SHA-256:"]
ver += [f"  {k}: {v}" for k, v in prov["scorer_files"].items()]
ver += ["", "심판 프롬프트 SHA-256:"]
ver += [f"  {k}: {v}" for k, v in prov["judge_prompts"].items()]
(cand / "VERSION.txt").write_text("\n".join(ver) + "\n", encoding="utf-8")
print(f"문항 {prov['n_items']} {q} | 질문·정답 변경 {len(qa_changed)} / 메타만 {len(meta_only)}")
print(f"evalset sha256 {prov['candidate_evalset_sha256'][:16]} | final_patch {(prov['final_patch_sha256'] or '(미기록)')[:16]}")
