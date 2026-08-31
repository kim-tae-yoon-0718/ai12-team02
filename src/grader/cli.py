from __future__ import annotations

import argparse
import os
import sys

from . import diagnostics as diag
from .diagnostics import regression as reg
from .config import load_config
from .prompts import PromptRepository
from .providers import build_provider
from .runner import GraderRunner, execute
from .validation import validate_evaluation_set


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="grader",
        description="RAG evaluation / LLM Judge pipeline",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate", help="평가셋 스키마만 검사(2-17)")
    validate.add_argument("--evaluation-set", required=True)
    validate.add_argument("--strict", action="store_true",
                          help="주석 키(_source_note 등) 를 FAIL 로 잡는다 (최종셋 검사용)")

    run = sub.add_parser("run", help="층 실행기(3-6-1): checks/ci/development/final")
    run.add_argument("--evaluation-set", required=True)
    run.add_argument("--responses", help="4/5층 채점에 필요. checks 모드는 생략 가능")
    run.add_argument("--mode", choices=["checks", "ci", "development", "final"], default="development")
    run.add_argument("--config", default="configs/default.yaml")
    run.add_argument("--overlay", default=None,
                     help="실험별 오버레이(configs/ci.yaml 등) — default.yaml 위에 바뀐 줄만 덮음")
    run.add_argument("--out-dir", default=None, help="기본값은 설정 파일의 out_dir")
    # 팀 확정 6-자산 provenance. scorer 는 채점기 코드 + 심판 프롬프트에서 자동 도출.
    run.add_argument("--corpus", default=None, help="6-자산 ① 원문 코퍼스 버전")
    run.add_argument("--preprocess", default=None, help="6-자산 ② 전처리(파싱·정제·청킹) 버전")
    run.add_argument("--table", default=None, help="6-자산 ③ 구조화 추출 테이블 버전")
    run.add_argument("--index", default=None, help="6-자산 ④ 검색 인덱스(RAG 검색계, 4번) 버전")
    run.add_argument("--evalset", default=None, help="6-자산 ⑤ 평가셋 버전(미지정 시 VERSION.txt)")
    run.add_argument("--scorer", default=None, help="6-자산 ⑥ 채점기 버전")
    run.add_argument("--corpus-doc-ids", default=None, help="2-17 참조 무결성 검사용(JSON 배열)")
    run.add_argument("--excluded-doc-ids", default=None,
                     help="검색 대상 아닌 문서 ID(JSON 배열) — 수집 중복 등. 정답 근거로 쓰면 실패(1-9-1)")
    run.add_argument("--strict", action="store_true",
                     help="주석 키(_source_note 등)를 FAIL 로 잡는다. final 모드는 자동 적용")
    run.add_argument("--leak-check", action="store_true",
                     help="문항 텍스트가 추적 파일에 유출됐는지 검사(【25】). final 모드는 자동 적용")
    run.add_argument("--data-manifest", default=None, help="1층 데이터 정상성 검사용")
    run.add_argument("--extraction-audit", default=None, help="3층 추출 표본 대조 결과")
    run.add_argument("--practice-set", default=None,
                     help="mode=ci 전용 practice 세트(검수 탈락분, practice_items.jsonl). "
                          "없으면 mode=ci 는 하드 실패한다(exit 1) — 최종셋 폴백 없음(임현진 2-17)")
    run.add_argument("--allow-final", action="store_true",
                     help="5층(최종셋) 실행 허용 — 누수 방지 규칙에 따라 명시적으로만")
    run.add_argument("--runner", default=os.environ.get("USER", "unknown"))

    diagnose = sub.add_parser("diagnose", help="3-1-0 팀 공용 진단표")
    diagnose.add_argument("--report", default=None, help="run 이 낸 report.json")
    diagnose.add_argument("--table", action="store_true", help="진단표만 출력")

    regression = sub.add_parser("regression", help="3-13 변동 폭 / 3-13-1 회귀 판정 / 3-17 원인 분리")
    regression.add_argument("--runs", nargs="*", default=[])
    regression.add_argument("--baseline", default=None)
    regression.add_argument("--current", default=None)
    regression.add_argument("--variance", default=None)
    regression.add_argument("--n-sigma", type=float, default=2.0)
    regression.add_argument("--out", default="artifacts/regression/variance.json")

    return parser


def _build_runner(args) -> GraderRunner:
    config = load_config(args.config, overlay=getattr(args, "overlay", None))
    provider = build_provider(config.runtime)
    prompt_repo = PromptRepository(config.judge.prompt_files)
    return GraderRunner(
        config=config,
        provider=provider,
        prompt_repo=prompt_repo,
        corpus=args.corpus,
        preprocess=args.preprocess,
        table=args.table,
        index=args.index,
        evalset=args.evalset,
        scorer=args.scorer,
    )


def main() -> int:
    args = build_parser().parse_args()

    try:
        if args.command == "validate":
            items = validate_evaluation_set(args.evaluation_set, strict_meta=args.strict)
            print(f"VALID: {len(items)} evaluation items")
            return 0

        if args.command == "run":
            runner = _build_runner(args)
            code, report = execute(
                runner,
                evaluation_set_path=args.evaluation_set,
                responses_path=args.responses,
                mode=args.mode,
                allow_final=args.allow_final,
                runner_name=args.runner,
                corpus_doc_ids_path=args.corpus_doc_ids,
                excluded_doc_ids_path=args.excluded_doc_ids,
                data_manifest_path=args.data_manifest,
                extraction_audit_path=args.extraction_audit,
                practice_set_path=args.practice_set,
                strict_meta=args.strict,
                leak_check=args.leak_check,
                out_dir=args.out_dir,
            )
            for L in report["layers"]:
                line = f"  [{L['layer']}층] {L['status']}"
                if L.get("problems"):
                    line += f" — 문제 {len(L['problems'])}건"
                if L.get("reason"):
                    line += f" — {L['reason']}"
                if L.get("warnings"):
                    line += f" — 경고 {len(L['warnings'])}건(게이트 아님)"
                print(line)
            print(f"\n=> {report['message']}  (exit={code})")
            if "summary" in report:
                s = report["summary"]
                print(f"   overall={s['overall_score']}  n={s['n_items']}")
                for t, v in s["main"]["by_task_type"].items():
                    warn = f"  ⚠{v['warning']}" if v.get("warning") else ""
                    print(f"   task_type={t:<10} score={v['score']:<7} n={v['n']:<4}{warn}")
                cit = s.get("citation") or {}
                if cit.get("n"):
                    print(f"   citation(3-4-3): accuracy={cit['citation_accuracy']}"
                          f"  no_citation_rate={cit['no_citation_rate']}  n={cit['n']}")
                if s.get("severity_weighted_score") is not None:
                    print(f"   severity_weighted_score(3-3-2 ①)={s['severity_weighted_score']}"
                          f"  — {s['severity_weighted_note']}")
                if s.get("circularity", {}).get("suspicious"):
                    print("   " + s["circularity"]["message"])
            return code

        if args.command == "diagnose":
            if args.table or not args.report:
                print(diag.render_table())
                return 0
            return diag.main(["--report", args.report])

        if args.command == "regression":
            argv = []
            if args.runs:
                argv += ["--runs", *args.runs]
            if args.baseline:
                argv += ["--baseline", args.baseline]
            if args.current:
                argv += ["--current", args.current]
            if args.variance:
                argv += ["--variance", args.variance]
            argv += ["--n-sigma", str(args.n_sigma), "--out", args.out]
            return reg.main(argv)

        return 1

    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
