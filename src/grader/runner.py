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

import hashlib
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
from .readiness import evaluate as evaluate_readiness, normalize_family as _norm_family
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
    index_by_id,
    load_jsonl,
    validate_evaluation_set,
    validate_model_responses,
    check_response_contract,
)
from checks.check_evalset import run_all as run_evalset_checks

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
        #   > VERSION.txt 자동 조회 > config/grader.yaml 폴백 > "UNKNOWN"
        by = read_base_yaml_assets()
        versions = read_versions()

        def _pick(*cands: str | None) -> str:
            for c in cands:
                if c and c not in ("UNKNOWN", "[대기]", "TODO", "null"):
                    return c
            return "UNKNOWN"

        self.corpus = _pick(corpus, by.get("corpus"), versions.get("corpus"), config.provenance.corpus)
        self.preprocess = _pick(preprocess, by.get("preprocess"), versions.get("preprocess"), config.provenance.preprocess)
        # table 은 실제 존재하는 rfp_extraction_table_v* 디렉토리 버전을 우선 — base.yaml 선언이
        # 뒤처질 수 있다(팀장: 결과에 실사용값 기록). versioning._latest_table_version 이 감지.
        self.table = _pick(table, versions.get("table"), by.get("table"), config.provenance.table)
        self.index = _pick(index, by.get("index"), config.provenance.index)
        self.evalset = _pick(evalset, by.get("evalset"), versions.get("evalset"), config.provenance.evalset)
        # 명시 인자로 받았는지 기억한다 — 안 받았으면 실제 채점 대상 파일 옆의
        # VERSION.txt 가 더 정확한 출처다(후보본을 공식 v1 로 적지 않기 위해).
        self.evalset_explicit = bool(evalset)
        # scorer 는 채점기가 자기 자신을 기술하는 값 — grader.yaml 이 우선(팀 base.yaml 보다).
        self.scorer = _pick(scorer, config.provenance.scorer, by.get("scorer"))
        # 심판 프롬프트 세부 버전 — 6칸이 아니라 manifest 부가 정보(오염 방지 4-9 재료).
        self.judge_prompt_versions = {name: prompt_repo.version_of(name) for name in config.judge.names}
        # 심판 프롬프트의 실제 파일 경로와 해시 — final 게이트가 존재를 확인하고,
        # manifest 에 해시를 남겨 "어떤 프롬프트로 낸 점수인지"를 재현 가능하게 한다.
        self.judge_prompt_files = dict(getattr(config.judge, "prompt_files", {}) or {})
        self.judge_prompt_hashes = {
            name: (hashlib.sha256(Path(path).read_bytes()).hexdigest()
                   if Path(path).exists() else None)
            for name, path in self.judge_prompt_files.items()
        }
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
            "answer": response.answer_text,
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
            "residual_limit": self.config.grading.residual_limit,
            "accept_natural_absence_phrasing": self.config.grading.accept_natural_absence_phrasing,
            "clarify_routes": self.config.retrieval.clarify_routes,
        }, matcher)

        nsr = self.config.retrieval.non_search_routes
        stages = retr.grade_retrieval(
            evaluation, response,
            self.config.retrieval.retrieval_k, self.config.retrieval.reranker_k,
            self.config.retrieval.context_k, self.config.retrieval.precision,
            self.config.retrieval.eval_k, non_search_routes=nsr,
        )
        retrieval_diag = [RetrievalDiagnostic(**s) for s in stages]

        citation_diag = CitationDiagnostic(**retr.grade_citation(
            evaluation, response, self.config.retrieval.precision,
            enabled=self.config.grading.grade_citations, non_search_routes=nsr,
        ))

        for judge_name in self.config.judge.whole_item_metrics:
            variables = self._prompt_variables(evaluation, response)
            cache_key = build_cache_key(
                evaluation_id=evaluation.id, judge_name=judge_name,
                question=evaluation.question, answer=response.answer_text,
                contexts=[c.model_dump(exclude_none=True) for c in response.contexts],
                corpus_version=self.corpus,
                extraction_table_version=self.table,
                # ★심판 설정이 바뀌면 다른 캐시를 쓴다 — 예전 판정 재사용 금지
                judge_model=self.judge.cfg.model,
                judge_family=self.judge.cfg.family,
                judge_provider=getattr(self.judge, "provider_name", "none"),
                judge_temperature=self.judge.cfg.temperature,
                judge_tier=self.judge.cfg.tier,
                prompt_version=(self.judge_prompt_versions.get(judge_name) or ""),
                prompt_sha256=(self.judge_prompt_hashes.get(judge_name) or ""),
                scorer_version=self.scorer,
                evalset_version=self.evalset,
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


def _gold_evidence_count(evaluation: EvaluationItem) -> int:
    """이 문항에 채점 가능한 정답 근거가 몇 개인가(응답 누락 시 분모 유지용)."""
    ev = list(getattr(evaluation, "evidence", None) or [])
    if ev:
        return len(ev)
    loc = evaluation.location
    if loc is None:
        return 0
    return len(loc) if isinstance(loc, list) else 1


def _missing_row(evaluation: EvaluationItem) -> dict:
    """응답이 아예 없는 문항의 집계 행 — 0점 시스템 실패로 분모에 남긴다."""
    return {
        "id": evaluation.id,
        "task_type": evaluation.task_type,
        "answer_type": evaluation.answer_type,
        "field_tag": evaluation.field_tag,
        "answer_source": evaluation.answer_source,
        "score": 0.0,
        "final_status": "FAIL-system",
        "abstention": {},
        "failure": "MISSING_RESPONSE",
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

    # 팀장: 공식 평가셋이 팀 결정을 아직 반영 못한 문항은 모델 오류로 세지 않고 따로 보고.
    ktm = set(getattr(runner.config.grading, "known_ground_truth_mismatch", ()) or ())
    scored_rows: list[dict] = []

    for it in items:
        resp = responses.get(it.id)
        if resp is None:
            # ★분모에서 빼지 않는다 — 응답이 없는 문항은 0점 시스템 실패다.
            #   건너뛰면 "응답을 안 낼수록 점수가 오르는" 구멍이 된다(4-5).
            missing.append(it.id)
            row = _missing_row(it)
            results_rows.append(row)
            if it.id not in ktm:
                scored_rows.append(row)
            # ★검색·출처·문서선별 지표의 분모에서도 빠지면 안 된다. 주 점수만 0 으로
            #   남기고 진단 축에서 빼면, 응답을 안 낸 문항이 검색 재현율·출처 정확도를
            #   **올려 주는** 결과가 된다(계산 대상이 줄어드니까).
            retrieval_stage_rows.append([
                {"applicable": False, "stage": st, "reason": "응답 누락 — 채점 불가",
                 "missing_response": True}
                for st in ("retrieval_k", "reranker_k", "context_k")])
            if _gold_evidence_count(it):
                citation_rows.append({"applicable": True, "matched": False, "score": 0.0,
                                      "n_citations": 0, "n_gold_locations": _gold_evidence_count(it),
                                      "reason": "응답 누락 — 출처 없음", "missing_response": True})
            continue
        result = runner.run_one(it, resp, mode)
        if it.id in ktm:
            result.final_status = "KNOWN_GROUND_TRUTH_MISMATCH"
        elif (resp.failure or "").strip():
            # ★4-5: 시스템 실패로 인한 0점과 오답 0점을 갈라 놓는다. 분모에는 남는다.
            result.final_status = "FAIL-system"
        eval_results.append(result)
        row = _result_row(it, result)
        row["failure"] = resp.failure
        row["route"] = resp.route
        if it.id in ktm:
            row["excluded_from_aggregate"] = True
        results_rows.append(row)
        if it.id not in ktm:
            scored_rows.append(row)

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
        "results_rows": results_rows,          # 전체 (per_item 출력용)
        "scored_rows": scored_rows,             # 집계 대상 (KNOWN_GROUND_TRUTH_MISMATCH 제외)
        "known_mismatch_ids": sorted(ktm & {r["id"] for r in results_rows}),
        "eval_results": eval_results,
        "retrieval_agg": retr.aggregate_retrieval(retrieval_stage_rows),
        "citation_agg": retr.aggregate_citation(citation_rows),
        "doc_select_agg": extr.aggregate_doc_selection(doc_select_rows),
        "answer_sources": answer_sources,
        "missing_predictions": missing,
    }


def judge_identity(runner: GraderRunner, readiness=None) -> dict:
    """이 실행의 심판이 무엇인지 결과에 명시한다 — stub / mock / real 을 구분한다.

    ★use_stub_judge=false 만으로 '실제 심판'이라고 판단하면 안 된다. provider 기본값이
      mock 이라, stub 을 껐는데 mock 이 붙어 있으면 {"status":"MOCK"} 이 파싱 실패로
      0점 처리되어 **아무도 채점하지 않은 0점**이 심판 점수로 보고된다.
    """
    judge = runner.judge
    stub = bool(runner.config.use_stub_judge) or isinstance(judge, StubJudge)
    mock = (not stub) and bool(getattr(judge, "is_mock", False))
    kind = "stub" if stub else ("mock" if mock else "real_judge")
    notes = {
        "stub": "★StubJudge — 경계 포함 문자열 매처. 의미 비교가 아니며 최종 점수 근거로 "
                "쓸 수 없다(tier=final 가드가 별도로 막는다).",
        "mock": "★더미(mock) provider — 실제 모델을 호출하지 않는다. 이 실행의 심판 점수는 "
                "채점 결과가 아니다.",
        "real_judge": "실심판 호출(프롬프트 버전은 manifest.judge_prompt_versions 참고).",
    }
    # ★usable_for_final 은 '실제 LLM인가'가 아니라 **최종 준비 판정**이 정한다.
    #   검증되지 않은 실제 LLM 을 최종 사용 가능이라고 적으면, 실행은 게이트가
    #   막았는데 보고서만 최종 점수를 내는 상태가 된다.
    usable = bool(getattr(readiness, "ready", False))
    out = {
        "kind": kind,
        "score_source": kind,
        "provider": getattr(judge, "provider_name", "none"),
        "model": judge.cfg.model,
        "family": judge.cfg.family,
        "family_normalized": _norm_family(judge.cfg.family),
        "tier": judge.cfg.tier,
        "usable_for_final": usable,
        "note": notes[kind],
    }
    if readiness is not None:
        out["final_readiness"] = readiness.model_dump()
        if kind == "real_judge" and not usable:
            out["note"] = ("★실제 LLM 이지만 최종 준비 조건을 통과하지 못했다 — "
                           "이 실행의 심판 점수는 잠정값(provisional)이며 최종 점수가 아니다. "
                           "미충족: " + "; ".join(readiness.problems[:3]))
    return out


def score_breakdown(report: dict, judge: dict, readiness=None) -> dict:
    """점수를 출처별로 갈라 낸다 — 무엇이 규칙이고 무엇이 심판인지 섞지 않는다.

    ★심판이 stub/mock 이면 통합 점수(final_overall)를 내지 않는다(null). 문자열
      매처가 낸 목록형·요약형 점수까지 섞인 숫자를 '최종 점수'로 적으면, 읽는 사람이
      그 숫자를 모델 성능으로 받아들인다.
    """
    kind = judge.get("kind")
    # ★judge_identity 와 **같은 판정**을 쓴다. 두 곳이 따로 판단하면 보고서 안에서
    #   usable_for_final=false 인데 final_overall 에 숫자가 있는 모순이 생긴다.
    if readiness is not None:
        real = bool(getattr(readiness, "ready", False))
    else:
        real = bool(judge.get("usable_for_final"))
    by_task = (report.get("main") or {}).get("by_task_type") or {}
    # 규칙 기반으로만 채점되는 유형(문자열/좌표/집합 비교)
    deterministic = {k: v.get("score") for k, v in by_task.items() if k in ("selection", "extraction")}
    return {
        "deterministic": {
            "note": "규칙 기반 채점(문자열·금액·좌표·집합 비교) — 심판 없이 재현 가능",
            "by_task_type": deterministic,
        },
        "provisional_stub": {
            "applicable": kind == "stub",
            "note": ("StubJudge 문자열 비교가 섞인 임시 점수 — 목록형(judge_list_item)·"
                     "요약형(judge_checkpoint) 매칭이 의미 비교가 아니다. 잠정값이다."
                     if kind == "stub" else "해당 없음"),
            "affected_answer_types": ["list", "summary"] if kind == "stub" else [],
        },
        "real_judge_faithfulness": report.get("faithfulness"),
        "final_overall": (report.get("overall_score") if real else None),
        "provisional_real_judge": {
            # 검증 안 된 실제 LLM 점수도 버리지 않는다 — 다만 '최종'이라 부르지 않는다.
            "applicable": kind == "real_judge" and not real,
            "value": (report.get("overall_score")
                      if (kind == "real_judge" and not real) else None),
            "note": ("실제 LLM 이지만 최종 준비 조건 미충족 — 잠정값이며 최종 점수가 아니다"
                     if (kind == "real_judge" and not real) else "해당 없음"),
        },
        "final_overall_note": (
            "최종 준비 조건을 모두 통과한 실심판 점수 — 통합 점수로 사용 가능"
            if real else
            f"★통합 점수를 내지 않는다(null). 심판 종류={kind}, 최종 준비="
            f"{getattr(readiness, 'ready', False)}. "
            + ("미충족: " + "; ".join(getattr(readiness, 'problems', [])[:3])
               if readiness is not None and getattr(readiness, 'problems', None)
               else "실제 심판과 사람 대조 기록이 준비된 뒤에만 최종 점수를 낸다.")),
    }


def faithfulness_axis(results: list[EvaluationResult], judge: dict) -> dict:
    """충실성(judge_faithfulness)을 다른 지표와 섞지 않고 별도 축으로 낸다.

    심판을 아예 안 돌렸거나 StubJudge 였으면 숫자를 만들지 않고 N/A 로 남긴다 —
    0.0 으로 적으면 "충실성이 나쁘다"로 읽히고, 1.0 으로 적으면 근거 없는 만점이 된다.
    """
    # judge_results 원소는 JudgeMetricResult(pydantic) 또는 dict 둘 다 올 수 있다.
    def _f(j, key):
        return j.get(key) if isinstance(j, dict) else getattr(j, key, None)

    rows = [j for r in results for j in (r.judge_results or [])
            if _f(j, "metric") == "judge_faithfulness"]
    if judge.get("kind") != "real_judge":
        return {"applicable": False, "n": len(rows),
                "reason": f"심판이 {judge.get('kind')} — 충실성은 의미 판정이라 "
                          f"문자열 매처나 더미 응답으로 대체할 수 없다"}
    if not rows:
        return {"applicable": False, "n": 0,
                "reason": "judge.whole_item_metrics 에 judge_faithfulness 없음 — 미실행"}
    scores = [float(_f(j, "score")) for j in rows if _f(j, "score") is not None]
    return {
        "applicable": True, "n": len(rows),
        "mean_score": (sum(scores) / len(scores)) if scores else None,
        "n_scored": len(scores),
        "note": "충실성은 정확도와 별개 축이다 — 합산하지 않는다.",
    }


def layer_eval(runner: GraderRunner, items: list[EvaluationItem],
              responses: dict[str, ModelResponse], mode: str, layer: int,
              extraction_report: dict | None) -> dict:
    graded = grade_all(runner, items, responses, mode)
    report = diag.full_report(
        graded["scored_rows"],   # KNOWN_GROUND_TRUTH_MISMATCH 문항은 집계에서 제외(팀장)
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
    readiness = getattr(runner, "final_readiness", None)
    report["judge"] = judge_identity(runner, readiness)
    report["faithfulness"] = faithfulness_axis(graded["eval_results"], report["judge"])
    report["score_breakdown"] = score_breakdown(report, report["judge"], readiness)
    if graded["known_mismatch_ids"]:
        report["known_ground_truth_mismatch"] = graded["known_mismatch_ids"]

    gate = report["severity"]["gate"]
    failed = [k for k, v in gate.items() if v.get("status") == "FAIL"]
    status = "FAIL" if (failed and layer >= 4 and runner.config.enforce_gates) else "PASS"
    return {"layer": layer, "status": status, "gate_failed": failed, "report": report,
            "per_item": [r.model_dump(exclude_none=True) for r in graded["eval_results"]]}


def execute(
    runner: GraderRunner,
    evaluation_set_path: str,
    *,
    generator_family: str | None = None,
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
    force_context: bool = False,
    force_doc: bool = False,
    chunks_path: str | None = None,
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
        "judge_prompt_hashes": runner.judge_prompt_hashes,
        # ★실제로 로딩한 채점·검색 설정을 그대로 남긴다 — 보고서와 실행값이 어긋나면
        #   "어느 정책으로 낸 점수인지"를 나중에 알 수 없다.
        "grading_config": {
            "accept_natural_absence_phrasing": runner.config.grading.accept_natural_absence_phrasing,
            "allow_partial": runner.config.grading.allow_partial,
            "grade_citations": runner.config.grading.grade_citations,
            "residual_limit": runner.config.grading.residual_limit,
        },
        "retrieval_config": {
            "non_search_routes": sorted(runner.config.retrieval.non_search_routes),
            "clarify_routes": sorted(runner.config.retrieval.clarify_routes),
            "retrieval_k": runner.config.retrieval.retrieval_k,
            "reranker_k": runner.config.retrieval.reranker_k,
            "context_k": runner.config.retrieval.context_k,
        },
        "schema_version": runner.schema_version,
        "subset": mode,
        # ★생성 모델 계열 — 최종 준비 검사(자기 편향 확인)가 이 값을 쓴다. 예전에는
        #   manifest 에 넣는 경로가 없어 항상 None 이었고, 그래서 모든 준비를 끝내도
        #   final 실행이 계속 차단됐다. CLI --generator-family 로 들어온다.
        "generator_family": generator_family,
        "generator_family_normalized": _norm_family(generator_family),
        "note": f"runner={runner_name}",
    }

    # ★최종 준비 판정은 여기서 **한 번만** 계산하고, 게이트·judge_identity·
    #   score_breakdown 이 모두 이 결과를 쓴다.
    readiness = evaluate_readiness(
        runner.judge, mode=mode, generator_family=generator_family,
        config=runner.config, prompt_hashes=runner.judge_prompt_hashes)
    runner.final_readiness = readiness
    manifest["judge"] = judge_identity(runner, readiness)
    manifest["final_readiness"] = readiness.model_dump()

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
    # ⚠️ [2026-09-04] 최종 모드 **설정 가드는 데이터 검사보다 먼저** 본다.
    #    최종 실행에서 평가셋 할당량(strict)을 강제하게 되면서, 설정이 틀린 실행이
    #    설정 오류(EXIT_CONFIG) 대신 데이터 실패(EXIT_FAIL)로 끝나 원인이 가려졌다.
    #    설정 문제는 데이터를 읽기 전에 설정 오류로 끝내는 것이 맞다.
    if mode == "final":
        if not allow_final:
            return _finish(EXIT_CONFIG, layers, manifest, out_dir,
                           "★최종셋(5층)은 자동 실행 금지 — 누수 방지. --allow-final 과 "
                           "--runner 를 명시하라.")
        if not (extraction_audit_path and os.path.exists(extraction_audit_path)):
            return _finish(EXIT_CONFIG, layers, manifest, out_dir,
                           "★최종 평가(5층)는 독립 원문 표본 대조(--extraction-audit) 필수 — "
                           "추출 테이블을 그대로 정답으로 쓰면 테이블 오류가 선별형 점수에 "
                           "숨는다(순환, 2-8-1). 박예진 4-6-2 표본 대조 결과를 연결하라.")
        # ★단일 판정. 게이트와 보고서가 서로 다른 기준을 쓰지 않는다.
        if not readiness.ready:
            return _finish(EXIT_CONFIG, layers, manifest, out_dir,
                           "★최종 심판 준비 안 됨 — " + "; ".join(readiness.problems))

    layers.append(L1)
    if L1["status"] == "FAIL":
        return _finish(EXIT_FAIL, layers, manifest, out_dir,
                       "1층 실패 — 데이터가 깨진 상태에서 성능을 재면 그 숫자는 아무 뜻이 없다")

    # 2층 — 평가셋 계약 검사 (2-17). 규칙 단일 출처: checks.check_evalset (임현진 소관).
    # ★raw JSONL 위에서 돈다 — 파일을 믿기 전에. pydantic 파싱(items)은 채점용 별개 단계.
    # ★최종 모드는 항상 주석 키(_source_note 등) 금지. practice/dev/ci 는 옵션.
    items = validate_evaluation_set(evaluation_set_path, strict_meta=(strict_meta or mode == "final"))
    raw_records = load_jsonl(evaluation_set_path)
    excluded_ids = None
    if excluded_doc_ids_path and os.path.exists(excluded_doc_ids_path):
        with open(excluded_doc_ids_path, encoding="utf-8") as f:
            excluded_ids = set(json.load(f))
    # C5 코퍼스 버전 대조 — RAG_ROOT 있으면 VERSION.txt 경로 조립
    rag_root = os.environ.get("RAG_ROOT")
    # ★실제로 채점하는 평가셋 파일 옆의 VERSION.txt 를 먼저 본다. 공식 경로를 무조건
    #   읽으면 후보본을 채점하면서 결과에는 공식 버전(v1)이 적혀, 어떤 정답으로 낸
    #   점수인지 나중에 구분할 수 없게 된다.
    _local_ver = Path(evaluation_set_path).resolve().parent / "VERSION.txt"
    evalset_ver = (str(_local_ver) if _local_ver.exists()
                   else (os.path.join(rag_root, "evalset/v1/VERSION.txt") if rag_root else None))
    corpus_ver = os.path.join(rag_root, "shared_data/processed/corpus_v2/VERSION.txt") if rag_root else None
    chunk_ver = os.path.join(rag_root, "shared_data/processed/chunks_v3/VERSION.txt") if rag_root else None
    # 유출 검사: --leak-check 또는 최종 모드일 때. 저장소 루트(pyproject.toml 위치)에서 git ls-files.
    leak_root = str(Path(__file__).resolve().parent.parent.parent) if (leak_check or mode == "final") else None
    leak_exclude = [p for p in (evaluation_set_path, practice_set_path) if p]
    problems = run_evalset_checks(
        raw_records,
        doc_ids_path=corpus_doc_ids_path,
        excluded_ids=excluded_ids,
        practice_path=practice_set_path,
        evalset_version_path=evalset_ver,
        corpus_version_path=corpus_ver,
        chunking_version_path=chunk_ver,
        # ⚠️ [2026-09-04 정정] 예전에는 strict 를 넘기지 않아 최종 실행에서도 총량·할당량
        #    검사가 꺼져 49문항·51문항·비율 오류가 그대로 통과했다.
        strict=(mode == "final"),
        final_set=(mode in ("ci", "final")),  # evaluation_set 이 최종셋인 모드
        leak_repo_root=leak_root,
        leak_exclude=leak_exclude,
    )
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

    if mode == "final" and extraction_report is None:
        # 안전망 — 파일은 있었는데 표본 대조 결과가 비어 있는 경우(위 설정 가드 통과분).
        return _finish(EXIT_CONFIG, layers, manifest, out_dir,
                       "★최종 평가(5층) 독립 원문 표본 대조 결과가 비어 있음 — "
                       "--extraction-audit 내용을 확인하라.")

    if not responses_path:
        return _finish(EXIT_CONFIG, layers, manifest, out_dir, "--responses 없음 — 4/5층 채점 불가")

    raw_responses = validate_model_responses(responses_path)
    # 중복 id 는 index_by_id 가 예외로 죽기 전에 여기서 잡아 종료코드를 확정한다.
    _dup = sorted({r.id for r in raw_responses if [x.id for x in raw_responses].count(r.id) > 1})
    if _dup:
        return _finish(EXIT_FAIL, layers, manifest, out_dir,
                       f"★모델 응답 id 중복 {len(_dup)}건 — 채점 중단: {_dup[:10]}")
    responses = index_by_id(raw_responses)

    # 3-1 / 3-2-2 강제 주입 — 검색·문서특정 단계를 완벽하다고 가정. 상한 측정용.
    if force_context or force_doc:
        from . import force_inject as fi
        items_by_id = {it.id: it for it in validate_evaluation_set(evaluation_set_path)}
        forced: list[str] = []
        if force_doc:
            forced.append("doc")
            fi.apply_forced_doc(items_by_id, responses)
        if force_context:
            if not (chunks_path and os.path.exists(chunks_path)):
                return _finish(EXIT_CONFIG, layers, manifest, out_dir,
                               "★--force-context 는 --chunks(청크 jsonl) 가 필요하다 — "
                               "정답 근거 청크를 주입해야 하므로.")
            forced.append("context")
            fi.apply_forced_context(items_by_id, responses, fi.load_chunk_index(chunks_path))
        manifest["forced"] = forced
        manifest["forced_note"] = ("★강제 주입 실행 — 일반 실행과 비교 금지. "
                                   "이 점수는 해당 단계가 완벽할 때의 상한이다.")

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

    # ★채점 직전, 실제로 채점할 문항 집합에 대해 응답 1:1 대응을 확인한다.
    #   (부분집합 모드는 응답이 더 많은 게 정상이므로 '추가 id'는 요구하지 않는다.)
    contract = check_response_contract(items, raw_responses,
                                       require_exact=(mode in ("final", "full")))
    manifest["response_contract"] = contract
    if not contract["ok"]:
        return _finish(EXIT_FAIL, layers, manifest, out_dir,
                       "★모델 응답 완전성 계약 위반 — 채점 중단: " + " / ".join(contract["problems"]))

    # provenance ⑤ 평가셋 축도 같은 출처를 따른다(명시 인자가 없을 때만).
    if (not getattr(runner, "evalset_explicit", False)
            and evalset_ver and os.path.exists(evalset_ver)):
        from .versioning import read_schema_version
        resolved = read_schema_version(evalset_ver)
        if resolved and resolved != "UNKNOWN":
            runner.evalset = resolved
            manifest["provenance"] = runner.provenance().model_dump()

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
