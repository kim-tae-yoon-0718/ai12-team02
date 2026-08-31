"""
grader.runner — 3-6 평가 자동화 / 3-6-1 평가를 층으로 쌓아 CI에 올리기

★ 왜 층으로 쌓는가
   전부 한 덩어리로 돌리면 비싸서 자주 못 돌리고, 자주 안 돌리면 CI의 의미가 없다.
   값싼 검사를 앞에 두고 비싼 평가를 뒤에 둔다. 앞에서 걸리면 뒤는 돌리지 않는다.

   1층 데이터 정상성    (박예진 소관 산출물) — 초 단위, 모델 호출 없음
   2층 평가셋 무결성    (2-17, 임현진 소관) — 초 단위
   3층 추출 테이블 정확도 (3-2-1, 박예진 표본 대조) — 비교적 쌈
   4층 소규모 성능 평가  (CI용 층화 부분집합) — 분 단위
   5층 전체 평가        (최종 비교용, mode="final") — ★사람이 명시적으로 돌림

★ 5층을 CI에서 자동으로 돌리지 않는 이유는 비용이 아니라 **누수**다.
   최종셋 점수를 매 커밋마다 보게 되면 결국 그것을 보며 고치게 된다.
   ⇒ --allow-final 플래그 + 실행자 기록을 요구한다.
"""

from __future__ import annotations

import json
import os
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path
import time

from .cache import FileCache, build_cache_key
from .config import GraderConfig, read_base_yaml_assets
from . import diagnostics as diag
from . import extraction as extr
from . import retrieval as retr
from . import task_scoring as ts
from .judge import Judge, JudgeConfig as JudgeCfg, JudgeNotReady, StubJudge, make_item_matcher
from .models import (
    CitationDiagnostic,
    EvaluationItem,
    EvaluationResult,
    ModelResponse,
    Provenance,
    RetrievalDiagnostic,
)
from .prompts import PromptRepository
from .providers import JudgeProvider
from .versioning import read_schema_version, read_versions
from .validation import (
    check_data_sanity,
    check_data_warnings,
    check_evalset_integrity,
    index_by_id,
    load_jsonl,
    validate_evaluation_set,
    validate_model_responses,
)

EXIT_OK, EXIT_FAIL, EXIT_CONFIG = 0, 1, 2


def _git_state(cwd: str | None = None) -> dict:
    """실행 진입점 자동 기록(팀 규약 §2-4): git_commit(짧은 해시) + git_dirty(bool).
    ★git_dirty=true 면 그 실행은 재현 불가 — 최종 실험은 false 여야 한다.
    git 이 없거나 저장소가 아니면 조용히 UNKNOWN/None."""
    def _run(args: list[str]) -> str | None:
        try:
            out = subprocess.run(["git", *args], cwd=cwd, capture_output=True,
                                 text=True, timeout=5)
            return out.stdout.strip() if out.returncode == 0 else None
        except Exception:
            return None

    commit = _run(["log", "-1", "--format=%h"])
    status = _run(["status", "--short"])
    return {
        "git_commit": commit or "UNKNOWN",
        "git_dirty": bool(status) if status is not None else None,
    }


def build_judge(config: GraderConfig, prompt_repo: PromptRepository,
                provider: JudgeProvider | None) -> Judge:
    """use_stub_judge=true 인 동안은 항상 StubJudge — 이것은 심판이 아니라
    배관 점검용 대체물이다(3-8). tier=final 에서는 assert_ready 가 이를 잡아낸다."""
    if config.use_stub_judge:
        return StubJudge()
    cfg = JudgeCfg(
        model=config.judge.model, family=config.judge.family, tier=config.judge.tier,
        temperature=config.runtime.temperature,
        prompt_versions={name: prompt_repo.version_of(name) for name in config.judge.names},
        verification_path=config.judge.verification_path,
        min_agreement=config.judge.min_agreement,
    )
    return Judge(cfg, prompt_repo=prompt_repo, provider=provider)


class GraderRunner:
    def __init__(
        self,
        *,
        config: GraderConfig,
        provider: JudgeProvider,
        prompt_repo: PromptRepository,
        corpus: str | None = None,
        preprocess: str | None = None,
        table: str | None = None,
        index: str | None = None,
        evalset: str | None = None,
        scorer: str | None = None,
    ) -> None:
        self.config = config
        self.provider = provider
        self.prompt_repo = prompt_repo
        # 팀 확정 6-자산 provenance. 우선순위:
        #   CLI 인자(실험 override) > config/base.yaml §①(팀 단일 출처)
        #   > VERSION.txt 자동 조회 > configs/default.yaml 폴백 > "UNKNOWN"
        by = read_base_yaml_assets()
        versions = read_versions()

        def _pick(*cands: str | None) -> str:
            for c in cands:
                if c and c not in ("UNKNOWN", "[대기]", "TODO", "null"):
                    return c
            return "UNKNOWN"

        self.corpus = _pick(corpus, by.get("corpus"), versions.get("corpus"), config.provenance.corpus)
        self.preprocess = _pick(preprocess, by.get("preprocess"), versions.get("preprocess"), config.provenance.preprocess)
        self.table = _pick(table, by.get("table"), versions.get("table"), config.provenance.table)
        self.index = _pick(index, by.get("index"), config.provenance.index)
        self.evalset = _pick(evalset, by.get("evalset"), versions.get("evalset"), config.provenance.evalset)
        self.scorer = _pick(scorer, by.get("scorer"), config.provenance.scorer)
        # 심판 프롬프트 세부 버전 — 6칸이 아니라 manifest 부가 정보(오염 방지 4-9 재료).
        self.judge_prompt_versions = {name: prompt_repo.version_of(name) for name in config.judge.names}
        self.cache = FileCache(config.cache.directory) if config.cache.enabled else None
        self.judge = build_judge(config, prompt_repo, provider)
        # v0.2: schema_version은 더 이상 문항 필드가 아니라 VERSION.txt(evalset:) 출처.
        self.schema_version = read_schema_version()

    def provenance(self) -> Provenance:
        """이 실행의 6-자산 provenance. run_one(문항별)·execute(manifest)가 같은
        출처를 쓰도록 한 곳에서만 만든다 — report와 per_item의 값이 어긋나지 않게."""
        return Provenance(
            corpus=self.corpus,
            preprocess=self.preprocess,
            table=self.table,
            index=self.index,
            evalset=self.evalset,
            scorer=self.scorer,
        )

    def _matcher(self, prompt_name: str, var_name: str):
        return make_item_matcher(self.judge, prompt_name, var_name)

    def _prompt_variables(self, evaluation: EvaluationItem, response: ModelResponse) -> dict:
        return {
            "question": evaluation.question,
            "answer": response.answer,
            "context": "\n".join(c.text for c in response.contexts),
            "ground_truth": evaluation.model_dump_json(exclude_none=True, ensure_ascii=False),
        }

    def run_one(self, evaluation: EvaluationItem, response: ModelResponse, mode: str) -> EvaluationResult:
        """문항 하나를 채점한다.

        ① task_scoring(규칙 기반, list/summary 는 judge_list_item/judge_checkpoint 매처 사용)
        ② retrieval 진단(retrieval_k/reranker_k/context_k 각각)
        ③ 문항 전체 단위 LLM judge(judge_faithfulness 등 whole_item_metrics)
        를 합쳐 하나의 EvaluationResult로 낸다.
        """
        started = time.perf_counter()
        judge_results: list[dict] = []
        any_cache_hit = False

        matcher = None
        if evaluation.answer_type == "list":
            matcher = self._matcher("judge_list_item", "item")
        elif evaluation.answer_type == "summary":
            matcher = self._matcher("judge_checkpoint", "checkpoint")

        scored = ts.score_item(evaluation, response, {
            "allow_partial": self.config.grading.allow_partial,
            "docset_partial_credit": self.config.grading.docset_partial_credit,
            "miss_weight": self.config.grading.miss_weight,
            "require_table_format": self.config.grading.require_table_format,
            "grade_citations": self.config.grading.grade_citations,
        }, matcher)

        stages = retr.grade_retrieval(
            evaluation, response,
            self.config.retrieval.retrieval_k, self.config.retrieval.reranker_k,
            self.config.retrieval.context_k, self.config.retrieval.precision,
            self.config.retrieval.eval_k,
        )
        retrieval_diag = [RetrievalDiagnostic(**s) for s in stages]

        citation_diag = CitationDiagnostic(**retr.grade_citation(
            evaluation, response, self.config.retrieval.precision,
            enabled=self.config.grading.grade_citations,
        ))

        for judge_name in self.config.judge.whole_item_metrics:
            variables = self._prompt_variables(evaluation, response)
            cache_key = build_cache_key(
                evaluation_id=evaluation.id, judge_name=judge_name,
                question=evaluation.question, answer=response.answer,
                contexts=[c.model_dump(exclude_none=True) for c in response.contexts],
                corpus_version=self.corpus,
                extraction_table_version=self.table,
            )
            cached = self.cache.get(cache_key) if self.cache else None
            if cached is not None:
                any_cache_hit = True
                out = cached
            else:
                out = self.judge.score(judge_name, **variables)
                if self.cache:
                    self.cache.set(cache_key, out)
            judge_results.append({
                "metric": judge_name, "raw": out.get("reason", ""), "parsed": out,
                "verdict": out.get("verdict"), "score": out.get("score"), "reason": out.get("reason"),
            })

        runtime_ms = (time.perf_counter() - started) * 1000.0
        provenance = self.provenance()

        return EvaluationResult(
            id=evaluation.id,
            schema_version=self.schema_version,
            mode=mode,
            provider=self.judge.cfg.model,
            judge_results=judge_results,
            task_score=scored["task_score"],
            format_status=scored["format_status"],
            retrieval=retrieval_diag,
            citation=citation_diag,
            abstention=scored["abstention"],
            final_status=scored["final_status"],
            provenance=provenance,
            runtime_ms=runtime_ms,
            cache_hit=any_cache_hit,
        )


def write_results(path: str | Path, results: list[EvaluationResult]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result.model_dump(exclude_none=True), ensure_ascii=False) + "\n")


# ------------------------------------------------------------------ CI 4/5층: 층화 부분집합

def stratified_subset(items: list[EvaluationItem], size: int, seed: int = 20260825) -> list[EvaluationItem]:
    """CI용 부분집합은 ★층화해서 뽑는다(task_type × field_tag).
    무작위로 뽑으면 selection 문항이 0개인 부분집합이 나올 수 있고, 그러면 CI가
    selection 회귀를 영영 못 잡는다."""
    rnd = random.Random(seed)
    strata: dict[tuple, list] = {}
    for it in items:
        strata.setdefault((it.task_type, it.field_tag), []).append(it)
    keys = sorted(strata, key=lambda k: (k[0], str(k[1])))
    out: list[EvaluationItem] = []
    i = 0
    while len(out) < min(size, len(items)):
        k = keys[i % len(keys)]
        pool = strata[k]
        if pool:
            out.append(pool.pop(rnd.randrange(len(pool))))
        elif all(not strata[kk] for kk in keys):
            break
        i += 1
    return out


def _result_row(evaluation: EvaluationItem, result: EvaluationResult) -> dict:
    return {
        "id": evaluation.id,
        "task_type": evaluation.task_type,
        "answer_type": evaluation.answer_type,
        "field_tag": evaluation.field_tag,
        "answer_source": evaluation.answer_source,
        "score": (result.task_score.score if result.task_score else 0.0),
        "final_status": result.final_status,
        "abstention": (result.abstention.model_dump() if result.abstention else {}),
        "failure": None,
        "route": None,
    }


def grade_all(runner: GraderRunner, items: list[EvaluationItem],
             responses: dict[str, ModelResponse], mode: str) -> dict:
    results_rows: list[dict] = []
    eval_results: list[EvaluationResult] = []
    retrieval_stage_rows: list[list[dict]] = []
    citation_rows: list[dict] = []
    doc_select_rows: list[dict] = []
    answer_sources: list[str] = []
    missing: list[str] = []

    for it in items:
        resp = responses.get(it.id)
        if resp is None:
            missing.append(it.id)
            continue
        result = runner.run_one(it, resp, mode)
        eval_results.append(result)
        row = _result_row(it, result)
        row["failure"] = resp.failure
        row["route"] = resp.route
        results_rows.append(row)

        # run_one 이 이미 계산한 검색 진단을 재사용한다 — 같은 채점을 문항마다 두 번 하지 않는다
        # (citation 과 동일한 방침).
        retrieval_stage_rows.append([d.model_dump() for d in result.retrieval])

        if result.citation is not None:
            # run_one이 이미 계산한 걸 재사용 — 같은 채점을 두 번 안 한다.
            citation_rows.append(result.citation.model_dump())

        ds = extr.grade_doc_selection(it, resp, runner.config.gate.doc_org)
        if ds.get("applicable"):
            doc_select_rows.append(ds)
        if it.task_type == "selection":
            answer_sources.append(it.answer_source)

    return {
        "results_rows": results_rows,
        "eval_results": eval_results,
        "retrieval_agg": retr.aggregate_retrieval(retrieval_stage_rows),
        "citation_agg": retr.aggregate_citation(citation_rows),
        "doc_select_agg": extr.aggregate_doc_selection(doc_select_rows),
        "answer_sources": answer_sources,
        "missing_predictions": missing,
    }


def layer_eval(runner: GraderRunner, items: list[EvaluationItem],
              responses: dict[str, ModelResponse], mode: str, layer: int,
              extraction_report: dict | None) -> dict:
    graded = grade_all(runner, items, responses, mode)
    report = diag.full_report(
        graded["results_rows"],
        retrieval_agg=graded["retrieval_agg"],
        citation_agg=graded["citation_agg"],
        extraction_agg=(extraction_report or {}).get("report"),
        doc_select_agg=graded["doc_select_agg"],
        gate=runner.config.gate.severity_gate,
        task_weight=runner.config.gate.task_weight,
        field_tag_weight=runner.config.gate.field_tag_weight,
    )
    sel = (report["main"]["by_task_type"].get("selection") or {}).get("score")
    ext_acc = ((extraction_report or {}).get("report") or {}).get("overall_accuracy")
    if sel is not None and ext_acc is not None:
        report["circularity"] = extr.circularity_flag(sel, ext_acc, graded["answer_sources"])
    report["missing_predictions"] = graded["missing_predictions"]

    gate = report["severity"]["gate"]
    failed = [k for k, v in gate.items() if v.get("status") == "FAIL"]
    status = "FAIL" if (failed and layer >= 4 and runner.config.enforce_gates) else "PASS"
    return {"layer": layer, "status": status, "gate_failed": failed, "report": report,
            "per_item": [r.model_dump(exclude_none=True) for r in graded["eval_results"]]}


def execute(
    runner: GraderRunner,
    evaluation_set_path: str,
    responses_path: str | None,
    mode: str,
    allow_final: bool,
    runner_name: str,
    corpus_doc_ids_path: str | None = None,
    excluded_doc_ids_path: str | None = None,
    data_manifest_path: str | None = None,
    extraction_audit_path: str | None = None,
    practice_set_path: str | None = None,
    strict_meta: bool = False,
    leak_check: bool = False,
    out_dir: str | None = None,
) -> tuple[int, dict]:
    """3-6-1 층 실행기. mode: checks(1~3층) | ci(4층, 층화) | development(4층, 층화) |
    final(5층 전체 — --allow-final 필수 + 심판 assert_ready 필수).

    ★ 임현진 2-17: CI(mode="ci")에 상시 노출되는 것은 practice 세트(검수 탈락분)뿐이어야
    하고, 최종 50문항은 상시 노출하지 않는다. mode="ci"는 --evaluation-set(최종셋)을
    층화 샘플링하지 않고 practice_set_path 파일에서만 채점 대상을 가져온다.
    ★그 파일이 없으면 조용히 최종셋으로 폴백하지 않고 **즉시 하드 실패(exit 1)** 시킨다 —
    유출은 report 경고로 뭉갤 문제가 아니라 바로 드러나야 한다(임현진 확정 방향)."""
    out_dir = out_dir or runner.config.out_dir
    os.makedirs(out_dir, exist_ok=True)
    layers: list[dict] = []

    manifest = {
        "run_id": f"run-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}",
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        # ★6-자산 provenance 전체(6축, base.yaml §① 6칸). run_one 의 per_item.provenance 와
        # 같은 출처(runner.provenance())를 써서 report ↔ per_item 값이 어긋나지 않는다.
        # 회귀 3-17(regression.attribute_change)이 이 6축을 그대로 diff 한다.
        "provenance": runner.provenance().model_dump(),
        # 팀 규약 §2-4 — 실행 진입점에서 git 상태 자동 기록. git_dirty=true 면 재현 불가.
        **_git_state(),
        # 심판 프롬프트 세부 버전(6칸 밖 — 오염 방지 4-9 재료). scorer 축이 바뀌었을 때
        # "코드가 바뀐 건가 프롬프트가 바뀐 건가"를 여기서 가른다.
        "judge_prompt_versions": runner.judge_prompt_versions,
        "schema_version": runner.schema_version,
        "subset": mode,
        "note": f"runner={runner_name}",
    }

    # 1층 — 데이터 정상성
    if data_manifest_path and os.path.exists(data_manifest_path):
        with open(data_manifest_path, encoding="utf-8") as f:
            dm = json.load(f)
        problems = check_data_sanity(dm, runner.config.gate.data_thresholds)
        L1 = {"layer": 1, "status": "FAIL" if problems else "PASS", "problems": problems}
        # ★게이트가 아니라 경고 — 있어도 1층을 FAIL로 만들지 않는다(2026-08-27 박예진).
        warns = check_data_warnings(dm, runner.config.gate.data_warn_thresholds)
        if warns:
            L1["warnings"] = warns
    else:
        L1 = {"layer": 1, "status": "SKIP", "reason": "data_manifest 없음 — 박예진 산출물 대기"}
    layers.append(L1)
    if L1["status"] == "FAIL":
        return _finish(EXIT_FAIL, layers, manifest, out_dir,
                       "1층 실패 — 데이터가 깨진 상태에서 성능을 재면 그 숫자는 아무 뜻이 없다")

    # 2층 — 평가셋 무결성
    # ★최종 모드는 항상 주석 키(_source_note 등) 금지. practice/dev/ci 는 옵션.
    items = validate_evaluation_set(evaluation_set_path, strict_meta=(strict_meta or mode == "final"))
    corpus_ids = None
    if corpus_doc_ids_path and os.path.exists(corpus_doc_ids_path):
        with open(corpus_doc_ids_path, encoding="utf-8") as f:
            corpus_ids = set(json.load(f))
    excluded_ids = None
    if excluded_doc_ids_path and os.path.exists(excluded_doc_ids_path):
        with open(excluded_doc_ids_path, encoding="utf-8") as f:
            excluded_ids = set(json.load(f))
    # 유출 검사: --leak-check 또는 최종 모드일 때. 저장소 루트(pyproject.toml 위치)에서 git ls-files.
    leak_root = str(Path(__file__).resolve().parent.parent.parent) if (leak_check or mode == "final") else None
    leak_exclude = [p for p in (evaluation_set_path, practice_set_path) if p]
    problems = check_evalset_integrity(items, corpus_doc_ids=corpus_ids, quota=runner.config.gate.task_quota,
                                       retrieval_excluded_ids=excluded_ids,
                                       leak_check_root=leak_root, leak_exclude=leak_exclude)
    L2 = {"layer": 2, "status": "FAIL" if problems else "PASS", "problems": problems, "n_items": len(items)}
    layers.append(L2)
    if L2["status"] == "FAIL":
        return _finish(EXIT_FAIL, layers, manifest, out_dir, "2층 실패 — 평가셋 계약 위반")

    # 3층 — 추출 테이블 정확도(표본 대조)
    extraction_report = None
    if extraction_audit_path and os.path.exists(extraction_audit_path):
        rows = (load_jsonl(extraction_audit_path) if extraction_audit_path.endswith(".jsonl")
                else json.load(open(extraction_audit_path, encoding="utf-8")))
        rep = extr.grade_extraction_audit(rows, runner.config.gate.column_severity)
        L3 = {"layer": 3, "status": "FAIL" if rep["gate_failed_columns"] else "PASS", "report": rep}
        extraction_report = L3
    else:
        L3 = {"layer": 3, "status": "SKIP", "reason": "표본 대조 파일 없음 — 박예진 4-6-2 대기"}
    layers.append(L3)
    if L3["status"] == "FAIL" and runner.config.enforce_gates:
        return _finish(EXIT_FAIL, layers, manifest, out_dir,
                       f"3층 실패 — 추출 게이트 미달 컬럼 {L3['report']['gate_failed_columns']}")

    if mode == "checks":
        return _finish(EXIT_OK, layers, manifest, out_dir, "값싼 검사만 실행(1~3층)")

    if mode == "final":
        if not allow_final:
            return _finish(EXIT_CONFIG, layers, manifest, out_dir,
                           "★최종셋(5층)은 자동 실행 금지 — 누수 방지. --allow-final 과 "
                           "--runner 를 명시하라.")
        # ★순환 방지(5, 2-8-1): 최종 평가는 독립 원문 표본 대조 없이 신뢰할 수 없다.
        # 추출 테이블을 그대로 정답으로 쓰면(answer_source=table) 테이블이 틀려도 선별형
        # 만점이 나온다 — 그걸 잡는 건 평가셋과 독립된 표본 대조(박예진 4-6-2)뿐이다.
        # dev/ci/checks 는 종전대로 SKIP 허용(스모크·디버그를 막지 않는다).
        if extraction_report is None:
            return _finish(EXIT_CONFIG, layers, manifest, out_dir,
                           "★최종 평가(5층)는 독립 원문 표본 대조(--extraction-audit) 필수 — "
                           "추출 테이블을 그대로 정답으로 쓰면 테이블 오류가 선별형 점수에 "
                           "숨는다(순환, 2-8-1). 박예진 4-6-2 표본 대조 결과를 연결하라.")
        if runner.config.use_stub_judge or isinstance(runner.judge, StubJudge):
            # ★StubJudge는 tier를 항상 "dev"로 자기 보고하므로 assert_ready()만으로는
            # 걸러지지 않는다 — final에서는 여기서 명시적으로 막는다(3-8).
            return _finish(EXIT_CONFIG, layers, manifest, out_dir,
                           "★use_stub_judge=true 로는 tier=final 실행 불가 — StubJudge는 "
                           "심판이 아니라 배관 점검용이다. configs/default.yaml 에서 끄고 "
                           "실제 심판을 연결하라.")
        try:
            runner.judge.assert_ready(generator_family=None, strict=True)
        except JudgeNotReady as exc:
            return _finish(EXIT_CONFIG, layers, manifest, out_dir,
                           f"★tier=final 심판 준비 안 됨 — {exc}")

    if not responses_path:
        return _finish(EXIT_CONFIG, layers, manifest, out_dir, "--responses 없음 — 4/5층 채점 불가")

    responses = index_by_id(validate_model_responses(responses_path))

    if mode == "ci":
        # ★practice 세트가 없으면 최종셋으로 조용히 새지 않고 즉시 하드 실패시킨다(임현진 확정).
        if not (practice_set_path and os.path.exists(practice_set_path)):
            return _finish(EXIT_FAIL, layers, manifest, out_dir,
                           "★CI(mode=ci) 실행 거부 — practice 세트(--practice-set)가 없다. "
                           "최종 50문항을 CI에 노출하지 않기 위함이다(임현진 2-17). "
                           "practice_items.jsonl 을 연결한 뒤 다시 실행하라.")
        items = validate_evaluation_set(practice_set_path)
        if len(items) > runner.config.ci_subset_size:
            items = stratified_subset(items, runner.config.ci_subset_size)
    elif mode == "development":
        items = stratified_subset(items, runner.config.dev_subset_size)

    layer_no = 5 if mode == "final" else 4
    LX = layer_eval(runner, items, responses, mode, layer_no, extraction_report)
    layers.append(LX)
    code = EXIT_FAIL if LX["status"] == "FAIL" else EXIT_OK
    return _finish(code, layers, manifest, out_dir,
                   f"{layer_no}층 완료 ({len(items)}문항, mode={mode})", LX.get("per_item"))


def _finish(code: int, layers: list[dict], manifest: dict, out_dir: str, message: str,
           per_item: list[dict] | None = None) -> tuple[int, dict]:
    report = {"manifest": manifest, "layers": layers, "message": message, "exit_code": code}
    for L in layers:
        if L.get("layer") in (4, 5) and "report" in L:
            report["summary"] = L["report"]
    with open(os.path.join(out_dir, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    if per_item:
        with open(os.path.join(out_dir, "per_item.jsonl"), "w", encoding="utf-8") as f:
            for r in per_item:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return code, report
