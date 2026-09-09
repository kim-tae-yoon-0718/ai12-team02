#!/usr/bin/env python3
"""회사 참여 후보 추천(기본 ON) 뒤 현재 3단계 질의응답을 실행하는 진입점."""
from __future__ import annotations

import argparse
import sys
import time
from datetime import timedelta
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import company_match as C  # noqa: E402

sys.path.insert(0, str(REPO_ROOT / "src" / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "src" / "rag"))
import answer_pipeline as AP  # noqa: E402
from config import load_config  # noqa: E402
from identity_metadata import reference_datetime_from_config  # noqa: E402


TOP_N_DEFAULT = 5
YES_ANSWERS = {"예", "네", "응", "y", "yes", "ㅇ"}
NO_ANSWERS = {"아니오", "아니요", "아니", "n", "no", "ㄴ"}
UNKNOWN_ANSWERS = {"모름", "모르겠음", "unknown", "u", ""}
QUIT_WORDS = {"q", "quit", "exit", "취소", "그만", "종료"}


class CancelledByUser(Exception):
    pass


def recommendation_enabled(cfg: dict[str, Any]) -> bool:
    value = cfg.get("company_recommendation_enabled", True)
    if not isinstance(value, bool):
        raise ValueError("company_recommendation_enabled는 true 또는 false여야 합니다.")
    return value


def input_or_cancel(prompt: str) -> str:
    value = input(prompt).strip()
    if value.lower() in QUIT_WORDS:
        raise CancelledByUser()
    return value


def input_confirmed(prompt: str) -> str:
    while True:
        value = input_or_cancel(prompt)
        confirmed = input_or_cancel(
            f"입력하신 값 '{value or '(빈 값)'}'이(가) 맞나요? (예/아니오): "
        ).lower()
        if confirmed in YES_ANSWERS:
            return value
        if confirmed not in NO_ANSWERS:
            print("예 또는 아니오로 답해주세요.")
        else:
            print("다시 입력해주세요.")


def input_consortium_needed() -> bool | None:
    while True:
        value = input_or_cancel(
            "공동수급이 꼭 필요한 회사인가요? (예/아니오/모름): "
        ).lower()
        if value in YES_ANSWERS:
            return True
        if value in NO_ANSWERS:
            return False
        if value in UNKNOWN_ANSWERS:
            return None
        print("예, 아니오 또는 모름으로 답해주세요.")


def try_load_profile(store_dir: Path, company_name: str) -> C.CompanyProfile | None:
    try:
        return C.load_profile(store_dir, company_name)
    except LookupError:
        return None


def register_profile_interactive(store_dir: Path, company_name: str) -> C.CompanyProfile:
    region = input_confirmed("소재 지역(예: 서울): ")
    business_fields = [
        item.strip()
        for item in input_confirmed("사업분야(쉼표로 구분, 없으면 엔터): ").split(",")
        if item.strip()
    ]
    certifications = [
        item.strip()
        for item in input_confirmed("보유 자격·신고증(쉼표로 구분, 없으면 엔터): ").split(",")
        if item.strip()
    ]
    profile = C.CompanyProfile(
        company_name=company_name,
        region=region,
        business_fields=business_fields,
        certifications=certifications,
        consortium_needed=input_consortium_needed(),
    )
    C.save_profile(store_dir, profile)
    print(f"[{company_name}] 회사 정보를 저장했습니다.")
    return profile


def get_or_register_profile(
    store_dir: Path,
    initial_company_name: str | None,
) -> tuple[str, C.CompanyProfile]:
    company_name = initial_company_name
    while True:
        if not company_name:
            company_name = input_or_cancel("회사 이름을 입력하세요(종료: 취소): ")
        profile = try_load_profile(store_dir, company_name)
        if profile is not None:
            print(f"[{company_name}] 저장된 회사 정보를 불러왔습니다.")
            return company_name, profile
        answer = input_or_cancel(
            f"[{company_name}] 저장된 정보가 없습니다. 등록할까요? (예/아니오): "
        ).lower()
        if answer in YES_ANSWERS:
            return company_name, register_profile_interactive(store_dir, company_name)
        company_name = None


def _document_label(document_id: str, identity: Any) -> str:
    record = identity.get(document_id) if identity is not None else None
    if record is None or not record.project_name:
        return document_id
    return f"{document_id} · {record.project_name}"


def _location_text(location: dict[str, Any] | str | None) -> str:
    if isinstance(location, str):
        return location
    if not isinstance(location, dict):
        return ""
    heading = str(location.get("heading") or "").strip()
    line = location.get("line_start") or location.get("line")
    if heading and line:
        return f"{heading} (line {line})"
    return heading or (f"line {line}" if line else "")


def _print_candidate(item: dict[str, Any], identity: Any) -> None:
    print(f"- {_document_label(item['document_id'], identity)}")
    for field_name in C.MATCHED_FIELDS:
        detail = item["reasons"][field_name]
        if detail["status"] not in {C.FIELD_CONFIRMED, C.FIELD_REVIEW, C.FIELD_CONTRADICTION}:
            continue
        print(f"  · {field_name}: {detail['status']} — {detail['reason']}")
        if detail["status"] == C.FIELD_REVIEW and detail["evidence"]:
            print(f"    원문 근거: {detail['evidence'][:220]}")
            location = _location_text(detail["location"])
            if location:
                print(f"    위치: {location}")


def print_recommendations(
    company_name: str,
    results: list[dict[str, Any]],
    identity: Any,
    top_n: int,
) -> None:
    priority = [item for item in results if item["verdict"] == C.VERDICT_PRIORITY]
    review = [item for item in results if item["verdict"] == C.VERDICT_REVIEW]
    excluded = [item for item in results if item["verdict"] == C.VERDICT_EXCLUDED]

    print(f"\n=== {company_name} · 우선 검토 후보 {min(len(priority), top_n)}건 ===")
    if not priority:
        print("(닫힌 구조 값에서 명시적으로 일치한 후보가 없습니다.)")
    for item in priority[:top_n]:
        _print_candidate(item, identity)

    print(f"\n=== 사람 확인이 필요한 후보 {min(len(review), top_n)}건 ===")
    if not review:
        print("(사람 확인이 필요한 후보가 없습니다.)")
    for item in review[:top_n]:
        _print_candidate(item, identity)

    print(f"\n명시적 조건 불일치로 제외된 공고: {len(excluded)}건")
    print("※ 우선 검토 후보도 참가 가능 확정이 아닙니다. 사람 확인 필요 표시를 확인하세요.")


def print_urgent_candidates(
    results: list[dict[str, Any]],
    identity: Any,
    cfg: dict[str, Any],
    top_n: int,
    urgent_days: int,
) -> None:
    if identity is None:
        return
    candidate_ids = {
        item["document_id"] for item in results if item["verdict"] != C.VERDICT_EXCLUDED
    }
    reference = reference_datetime_from_config(cfg)
    end = reference + timedelta(days=urgent_days)
    urgent = sorted(
        (
            (document_id, record.bid_deadline)
            for document_id, record in identity.records.items()
            if document_id in candidate_ids
            and record.bid_deadline is not None
            and reference <= record.bid_deadline <= end
        ),
        key=lambda item: (item[1], item[0]),
    )[:top_n]
    print(f"\n=== 후보 중 {urgent_days}일 안에 마감하는 공고 {len(urgent)}건 ===")
    if not urgent:
        print("(해당 공고가 없습니다.)")
    for document_id, deadline in urgent:
        print(f"- {_document_label(document_id, identity)} · {deadline:%Y-%m-%d}")


def run_search_loop(rt: dict[str, Any], cfg: dict[str, Any]) -> None:
    session = AP.SessionState()
    cache: dict[str, Any] = {}
    eligible = rt["registry_scope"].eligible_ids if rt["registry_scope"] is not None else None

    def get_embed_client():
        if "embed" not in cache:
            cache["embed"] = AP.EmbeddingClient(cfg)
        cache["embed"].reset_usage()
        return cache["embed"]

    def get_gen_client():
        if "gen" not in cache:
            cache["gen"] = AP.GenerationClient(cfg)
        cache["gen"].reset_usage()
        return cache["gen"]

    def get_stage1_planner():
        if "planner" not in cache:
            cache["planner"] = AP.Stage1Planner(cfg, rt["identity"], eligible)
        cache["planner"].reset_usage()
        return cache["planner"]

    def get_stage2_agent():
        if "agent2" not in cache:
            cache["agent2"] = AP.Stage2Agent(cfg, rt["identity"], eligible)
        cache["agent2"].reset_usage()
        return cache["agent2"]

    print("\n=== 현재 3단계 질의응답을 시작합니다 (종료: q 또는 빈 입력) ===")
    while True:
        question = input("\n질문: ").strip()
        if not question or question.lower() in QUIT_WORDS:
            print("질의응답을 종료합니다.")
            return
        started = time.perf_counter()
        try:
            result = AP.answer(
                question,
                rt["store"],
                get_embed_client,
                get_gen_client,
                rt["table"],
                cfg,
                identity=rt["identity"],
                session=session,
                locator=rt["locator"],
                registry_scope=rt["registry_scope"],
                get_stage1_planner=get_stage1_planner,
                get_stage2_agent=get_stage2_agent,
            )
        except Exception as error:  # 대화형 화면은 질문 하나가 실패해도 다음 질문을 받는다.
            print(f"[오류] {error}")
            continue
        latency_ms = round((time.perf_counter() - started) * 1000)
        print(f"(처리 시간 {latency_ms}ms · 경로 {result.route_matched_rule})")
        if result.session_banner:
            print(f"[안내] {result.session_banner}")
        print(result.text)


def run_session(
    args: argparse.Namespace,
    cfg: dict[str, Any],
    rt: dict[str, Any],
) -> int:
    if recommendation_enabled(cfg):
        try:
            company_name, profile = get_or_register_profile(Path(args.store_dir), args.company)
        except CancelledByUser:
            print("\n회사 정보 입력을 취소했습니다.")
            return 0
        eligible_ids = (
            set(rt["registry_scope"].eligible_ids)
            if rt["registry_scope"] is not None else None
        )
        results = C.match_company(profile, rt["table"], eligible_ids)
        print_recommendations(company_name, results, rt["identity"], args.top_n)
        print_urgent_candidates(
            results, rt["identity"], cfg, args.top_n, args.urgent_days,
        )
    else:
        print("회사 추천 기능이 꺼져 있어 기존 3단계 질의응답으로 바로 이동합니다.")

    if not args.skip_search:
        run_search_loop(rt, cfg)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="회사 참여 후보 추천 + 현재 3단계 질의응답")
    parser.add_argument("--company", default=None, help="생략하면 화면에서 입력")
    parser.add_argument("--store-dir", default=str(C.DEFAULT_STORE_DIR))
    parser.add_argument("--top-n", type=int, default=TOP_N_DEFAULT)
    parser.add_argument("--urgent-days", type=int, default=14)
    parser.add_argument("--skip-search", action="store_true", help="추천 화면까지만 확인")
    AP.add_common_args(parser)
    args = parser.parse_args()
    if args.top_n <= 0 or args.urgent_days < 0:
        parser.error("top-n은 양수, urgent-days는 0 이상이어야 합니다.")

    cfg = load_config(args.experiment_config)
    # 실행 자료 검증을 먼저 끝낸 뒤에만 회사 정보를 입력·저장한다.
    rt = AP.build_runtime(args, cfg)
    return run_session(args, cfg, rt)


if __name__ == "__main__":
    raise SystemExit(main())
