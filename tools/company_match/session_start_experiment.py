#!/usr/bin/env python3
"""실사용 세션 시작 흐름 실험 — 회사 조회/등록 → 추천 5 + 마감임박 5 → 검색 시작.

⚠️ 원본은 하나도 안 건드린다. company_match.py / loose_match_experiment.py /
   identity_metadata.py / reference_clock_experiment.py / answer_pipeline.py를
   전부 import만 해서 이어붙인 새 진입점이다.

흐름:
    1. 회사 이름을 입력받는다. 프로필이 있으면 불러오고, 없으면 그 자리에서
       등록(지역·사업분야·자격증·컨소시엄 필요 여부)한다.
    2. loose_match_experiment로 전체 문서를 매칭해서 "적합" 상위 N건을 추천하고,
       identity_v2 기준 "마감 임박" 상위 N건을 따로 보여준다.
    3. 그다음부터는 질문을 받아 공식 answer_pipeline.answer()로 실제 검색·답변을
       띄운다(select/extract는 API 키 없이도 동작, QA 라우트만 API 키 필요).

⚠️ 마감임박 표시도 검색과 똑같이 공식 reference_datetime_from_config(고정값,
   base.yaml)을 쓴다 — reference_clock_experiment(진짜 현재 시각)는 이제 여기서
   안 쓴다(껐다). 평가용 identity_v2 데이터가 전부 2024년 공고라 "진짜 지금"
   기준으로 계산하면 마감임박이 항상 0건으로 나와서 화면 자체가 의미가 없었다
   — 실사용 시점이 오면 그때 다시 reference_clock_experiment를 붙이면 된다
   (파일은 그대로 남아 있다, tools/company_match와 무관하게 동작).

사용:
    RAG_ROOT=/srv/rfp python3 session_start_experiment.py [--company "OO소프트"]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import company_match as C                              # noqa: E402
import loose_match_experiment as L                      # noqa: E402

sys.path.insert(0, str(REPO_ROOT / "src" / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "src" / "rag"))
import answer_pipeline as AP                             # noqa: E402
from config import load_config, identity_path            # noqa: E402
from identity_metadata import (                          # noqa: E402
    load_identity, reference_datetime_from_config, urgent_deadline_documents,
)

TOP_N_DEFAULT = 5


YES_ANSWERS = {"예", "y", "yes", "ㅇ", "네", "응"}
QUIT_WORDS = {"q", "quit", "exit", "취소", "그만", "종료"}


class CancelledByUser(Exception):
    """등록/입력 도중 사용자가 그만두기로 한 경우 — main()에서 잡아 조용히 끝낸다."""


def is_yes(text: str) -> bool:
    return text.strip() in YES_ANSWERS


def input_or_cancel(prompt: str) -> str:
    """입력값이 QUIT_WORDS(q/취소/그만 등)면 전체를 취소한다 — 등록 중간에
    치기 싫어졌을 때 끝까지 다 채우지 않고 빠져나올 수 있게 하기 위함."""
    value = input(prompt).strip()
    if value.lower() in QUIT_WORDS:
        raise CancelledByUser()
    return value


def prompt_yes_no(msg: str) -> bool:
    return is_yes(input_or_cancel(f"{msg} (예/아니오, 취소하려면 '취소'): "))


def input_confirmed(prompt: str) -> str:
    """값을 입력받고 맞는지 확인한다. '아니오'면 같은 항목을 다시 물어본다
    (오타로 잘못된 값이 그대로 저장되는 걸 막기 위함). 값 대신 '취소'를 치면
    등록 전체를 그만둔다."""
    while True:
        value = input_or_cancel(prompt)
        shown = value if value else "(빈 값)"
        if prompt_yes_no(f"입력하신 값 '{shown}'이(가) 맞나요?"):
            return value
        print("다시 입력해주세요.")


def try_load_profile(store_dir: Path,
                     company_name: str) -> tuple[C.CompanyProfile, bool | None] | None:
    """있으면 (프로필, 컨소시엄필요여부)를, 없으면 None을 돌려준다(에러로 안 죽음)."""
    try:
        profile = C.load_profile(store_dir, company_name)
    except SystemExit:
        return None
    consortium_needed = L.load_consortium_needed(store_dir, company_name)
    return profile, consortium_needed


def register_profile_interactive(store_dir: Path,
                                 company_name: str) -> tuple[C.CompanyProfile, bool | None]:
    region = input_confirmed("소재 지역(예: 서울): ")
    business_fields = [s.strip() for s in
                       input_confirmed("사업분야(쉼표로 구분, 없으면 엔터): ").split(",")
                       if s.strip()]
    certifications = [s.strip() for s in
                      input_confirmed("보유 자격증/신고증(쉼표로 구분, 없으면 엔터): ").split(",")
                      if s.strip()]
    consortium_needed = prompt_yes_no("컨소시엄(공동수급) 참여가 필요한 회사인가요?")
    profile = C.CompanyProfile(company_name=company_name, region=region,
                               business_fields=business_fields,
                               certifications=certifications)
    C.save_profile(store_dir, profile)
    L.save_consortium_needed(store_dir, company_name, consortium_needed)
    print(f"[{company_name}] 프로필 저장 완료.")
    return profile, consortium_needed


def get_or_register_profile(
    store_dir: Path, initial_company_name: str | None,
) -> tuple[str, C.CompanyProfile, bool | None]:
    """회사 이름을 물어보고, 있으면 불러오고, 없으면 등록 여부를 먼저 확인한다.

    '아니오'를 고르면(오타 등으로 다른 회사를 찾던 경우 대비) 등록하지 않고
    이름을 다시 입력받는다 — 원치 않는 신규 등록이 실수로 생기지 않게 한다."""
    company_name = initial_company_name
    while True:
        if not company_name:
            company_name = input_or_cancel("회사 이름을 입력하세요(그만두려면 '취소'): ")
        found = try_load_profile(store_dir, company_name)
        if found is not None:
            profile, consortium_needed = found
            print(f"[{company_name}] 기존 프로필을 불러왔습니다 "
                 f"(지역={profile.region!r}, 사업분야={profile.business_fields}, "
                 f"자격증={profile.certifications}, 컨소시엄 필요={consortium_needed})")
            return company_name, profile, consortium_needed
        answer = input_or_cancel(
            f"[{company_name}] 등록된 프로필이 없습니다. 등록할까요? "
            f"등록하시려면 '예', 재입력하시려면 '아니오', 그만두려면 '취소'를 쳐주세요: ")
        if is_yes(answer):
            profile, consortium_needed = register_profile_interactive(store_dir, company_name)
            return company_name, profile, consortium_needed
        company_name = None


def print_recommendations(company_name: str, profile: C.CompanyProfile,
                          consortium_needed: bool | None, extraction_table: Path,
                          top_n: int) -> None:
    rows_by_doc = C.load_extraction_rows(extraction_table)
    results = L.match_company_loose(profile, consortium_needed, rows_by_doc)
    # '적합'은 최소 한 필드가 회사 값과 실제로 대조돼서 나온 결론만 포함한다
    # (company_match.overall_verdict 참고) — 조건 자체가 안 적혀 있어서 그냥
    # 통과된 '적합(근거 약함)'은 따로 세기만 하고 추천 목록엔 안 넣는다.
    fits = [r for r in results if r["verdict"] == "적합"][:top_n]
    weak_count = sum(1 for r in results if r["verdict"] == "적합(근거 약함)")
    print(f"\n=== {company_name}님께 추천하는 공고 (적합 상위 {len(fits)}건) ===")
    if not fits:
        print("(실제 근거로 확인된 '적합' 공고가 없습니다 — '확인 필요' 목록을 직접 보세요)")
    for r in fits:
        cert = r["reasons"][C.CERT_FIELD]["reason"]
        print(f"- {r['document_id']}: {cert}")
    if weak_count:
        print(f"(참고: 조건 자체가 명시 안 돼 있어 판단 근거가 약한 공고 {weak_count}건은 "
             f"추천에서 제외했습니다)")


def print_urgent_deadlines(identity, cfg: dict[str, Any], top_n: int, urgent_days: int) -> None:
    # 마감임박도 검색과 같은 공식 고정 기준일을 쓴다(현재 시각 기준 사용 안 함).
    ref = reference_datetime_from_config(cfg)
    urgent = urgent_deadline_documents(identity, ref, urgent_days)[:top_n]
    print(f"\n=== 마감 임박 공고 (지금부터 {urgent_days}일 이내, 상위 {len(urgent)}건) ===")
    if not urgent:
        print(f"(지금부터 {urgent_days}일 이내 마감인 공고가 없습니다)")
    for doc_id, deadline in urgent:
        rec = identity.get(doc_id)
        name = rec.project_name if rec and rec.project_name else ""
        print(f"- {doc_id} (마감: {deadline:%Y-%m-%d}) {name}")


def build_answer_runtime(experiment_config: str | None) -> tuple[dict[str, Any], dict[str, Any]]:
    cfg = load_config(experiment_config)
    args = argparse.Namespace(
        index=None, extraction_table=None, extraction_metadata=None, identity=None,
        registry=None, chunks=None, allow_no_chunks=False, allow_no_deadline_filter=False,
        allow_no_identity=False, allow_unofficial_table=False,
    )
    rt = AP.build_runtime(args, cfg)
    return rt, cfg


def run_search_loop(rt: dict[str, Any], cfg: dict[str, Any]) -> None:
    session = AP.SessionState()
    cache: dict[str, Any] = {}

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

    print("\n=== 검색을 시작합니다 (종료: q/취소, 빈 입력) ===")
    while True:
        question = input("\n질문: ").strip()
        if not question or question.lower() in QUIT_WORDS:
            print("검색을 종료합니다.")
            return
        t0 = time.perf_counter()
        try:
            result = AP.answer(question, rt["store"], get_embed_client, get_gen_client,
                               rt["table"], cfg, identity=rt["identity"], session=session,
                               locator=rt["locator"], registry_scope=rt["registry_scope"])
        except Exception as e:  # noqa: BLE001 — 대화형 데모 루프, 한 질문 실패로 전체를 안 죽인다
            print(f"[오류] {e}")
            continue
        print(f"(latency {round((time.perf_counter() - t0) * 1000)}ms, "
             f"route={result.route_matched_rule})")
        if result.session_banner:
            print(f"[안내] {result.session_banner}")
        print(result.text)


def main() -> int:
    ap = argparse.ArgumentParser(description="회사 조회/등록 → 추천 → 검색, 실험 진입점")
    ap.add_argument("--company", default=None, help="생략하면 입력창에서 물어봄")
    ap.add_argument("--store-dir", default=str(C.DEFAULT_STORE_DIR))
    ap.add_argument("--extraction-table", default=str(C.DEFAULT_EXTRACTION_TABLE))
    ap.add_argument("--top-n", type=int, default=TOP_N_DEFAULT)
    ap.add_argument("--urgent-days", type=int, default=14)
    ap.add_argument("--experiment-config", default=None)
    ap.add_argument("--skip-search", action="store_true",
                    help="추천만 보고 검색 루프는 안 켠다(자동화 테스트용)")
    args = ap.parse_args()

    store_dir = Path(args.store_dir)
    try:
        company_name, profile, consortium_needed = get_or_register_profile(
            store_dir, args.company)
    except CancelledByUser:
        print("\n취소했습니다. 프로그램을 종료합니다.")
        return 0
    print_recommendations(company_name, profile, consortium_needed,
                         Path(args.extraction_table), args.top_n)

    rt, cfg = build_answer_runtime(args.experiment_config)
    print_urgent_deadlines(rt["identity"], cfg, args.top_n, args.urgent_days)

    if args.skip_search:
        return 0
    run_search_loop(rt, cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
