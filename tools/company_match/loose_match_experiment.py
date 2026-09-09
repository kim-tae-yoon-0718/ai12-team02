#!/usr/bin/env python3
"""company_match.py 실험 변형 — (1) 느슨한 키워드 대조 (2) '컨소시엄 요건' 필드 추가.

⚠️ 본문(company_match.py)은 전혀 수정하지 않는다 — CompanyProfile/load_profile/
   load_extraction_rows/evaluate_region/evaluate_business_field/overall_verdict/
   MATCHED_FIELDS를 그대로 가져다 쓰고, 여기서 두 가지만 새로 시도한다.

(1) 자격증 대조를 "정확히 같은 문구가 통째로 들어있는가"에서 "회사가 입력한 문구를
    낱말로 쪼갠 것 중 하나라도 들어있는가"로 느슨하게 바꾼다. 동의어 표를 만드는
    대신 택한 방법이라 트레이드오프가 있다 — "확인 필요"로 남던 것 중 일부가
    "충족"으로 바뀌는데, 그중 일부는 실제로는 다른 요건과 우연히 겹친 오탐일 수
    있다. 근거 문자열에 어떤 낱말이 왜 매칭됐는지 항상 남겨서 사람이 바로 알아볼
    수 있게 한다.

(2) 컨소시엄 요건은 회사 프로필에 없던 새 속성(consortium_needed)이 필요하다.
    본문 CompanyProfile 스키마를 건드리지 않으려고, 같은 명부 CSV(company_profiles.csv)의
    같은 회사 줄에 이 실험 모듈만 읽고 쓰는 추가 열을 얹는다 — 본문 쪽
    read_all_rows/write_all_rows는 알려진 열 목록 밖의 값도 그대로 보존해주고,
    CompanyProfile.from_row는 모르는 열을 그냥 무시한다(하위 호환).
    판정 자체는 tools/evalset/build_extraction_table.py의 consortium_policy_kind()를
    그대로 재사용한다(그 파일도 수정하지 않는다) — 100건 표에서 이미 검증된
    "허용/불허/조건부/제약" 분류를 새로 안 만들고 재사용하는 것.

사용:
    # 기존 프로필에 컨소시엄 필요 여부 추가/수정
    python3 loose_match_experiment.py set-consortium --company "부산소프트" \
        --consortium-needed true

    # 느슨한 자격증 대조 + 컨소시엄 요건까지 포함해서 매칭
    python3 loose_match_experiment.py match --company "부산소프트"
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
import unicodedata
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import company_match as C                                            # noqa: E402

sys.path.insert(0, str(HERE.parent / "evalset"))
from build_extraction_table import consortium_policy_kind             # noqa: E402

CONSORTIUM_FIELD = "컨소시엄 요건"
EXPERIMENTAL_FIELDS = C.MATCHED_FIELDS + (CONSORTIUM_FIELD,)

# 대조에서 제외할, 너무 흔해서 의미가 없는 낱말(이것 하나만 매칭돼도 실제로는
# 아무 근거가 안 됨). 동의어 표는 안 만들지만 이 정도 오탐 방지는 최소한으로 둔다.
GENERIC_TOKENS = frozenset({
    "업체", "사업자", "등록", "등록증", "신고", "신고증", "자격", "요건", "관련",
    "법률", "시행령", "시행규칙", "규정", "및", "또는", "제출", "증명", "보유",
    "면허", "실적",
})


def _tokenize(phrase: str) -> list[str]:
    """회사가 입력한 문구 하나를 대조용 낱말로 쪼갠다. 동의어 사전 없이 하는
    가장 단순한 방법 — 공백·괄호·쉼표로 나누고 너무 짧거나 너무 흔한 토큰은 뺀다."""
    parts = re.split(r"[\s,()·/]+", unicodedata.normalize("NFC", phrase or ""))
    return [p for p in parts if len(p) >= 2 and p not in GENERIC_TOKENS]


def loose_keyword_match(text: str, candidates: list[str]) -> tuple[str, str] | None:
    """candidates 각각을 낱말로 쪼개서, 그중 하나라도 text에 들어있으면
    (원래 후보 문구, 실제 매칭된 낱말)을 돌려준다. 못 찾으면 None."""
    text = unicodedata.normalize("NFC", text or "")
    for candidate in candidates:
        for token in _tokenize(candidate):
            if token in text:
                return candidate, token
    return None


def evaluate_certification_loose(profile: C.CompanyProfile, row: dict | None) -> tuple[str, str]:
    if row is None or row["status"] == "field_absent":
        return "정보없음", "참가 자격 명시 없음"
    text = row.get("answer_normalized") or row.get("answer_raw") or ""
    if not profile.certifications:
        return "확인 필요", "참가 자격 요건이 있으나 회사 보유 자격증 정보 없음 — 직접 대조 필요"
    hit = loose_keyword_match(text, profile.certifications)
    if hit:
        candidate, token = hit
        return "충족", (f"회사가 등록한 '{candidate}'의 낱말 '{token}' — 참가 자격 요건 문구에서 "
                       f"발견됨(느슨한 대조 — 전체 문구가 아니라 낱말 단위 일치라 오탐 가능성 있음)")
    return "확인 필요", f"낱말 단위로도 안 보임 — 직접 대조 필요: {text[:150]}"


def evaluate_consortium(consortium_needed: bool | None, row: dict | None) -> tuple[str, str]:
    if row is None or row["status"] == "field_absent":
        return "정보없음", "컨소시엄 요건 명시 없음"
    text = row.get("answer_normalized") or row.get("answer_raw") or ""
    policy = consortium_policy_kind(text)
    if policy == "allow":
        return "충족", f"컨소시엄 허용됨({policy}): {text[:150]}"
    if policy == "conditional_allow":
        # 조건부 허용은 그 조건(구성원 수·지역·업종 제한 등)을 회사 값과
        # 대조해서 확인한 게 아니라 "조건이 있다"는 것만 안다 — 그대로
        # '충족'으로 단정하면 조건을 못 채우는 회사도 적합으로 보일 수 있다
        # (회사가 처음 만든 이후로 코드리뷰에서 지적된 문제).
        return "확인 필요", f"컨소시엄 조건부 허용({policy}) — 조건 충족 여부 직접 확인 필요: {text[:150]}"
    if policy == "deny":
        if consortium_needed is True:
            return "부적합", f"이 사업은 컨소시엄을 허용하지 않음 — 회사는 컨소시엄 참여가 필요: {text[:150]}"
        if consortium_needed is False:
            return "충족", f"단독 입찰만 가능한 사업이고 회사도 단독 참여 가능(등록값 기준): {text[:150]}"
        # consortium_needed 미등록(None)은 "회사가 단독 참여 가능한지 모른다"는
        # 뜻이지 "가능하다"가 아니다 — 모르는 걸 충족으로 단정하지 않는다.
        return "확인 필요", f"단독 입찰만 가능한 사업 — 회사가 단독 참여 가능한지 등록된 정보 없음: {text[:150]}"
    if policy == "constraint":
        return "확인 필요", f"컨소시엄 구성에 조건이 있음 — 직접 확인 필요: {text[:150]}"
    return "확인 필요", f"컨소시엄 정책을 자동 분류 못함 — 직접 확인 필요: {text[:150]}"


_CONSORTIUM_CSV_COLUMN = "consortium_needed"
_BOOL_TEXT = {True: "true", False: "false"}


def load_consortium_needed(store_dir: Path, company_name: str) -> bool | None:
    """명부 CSV의 같은 회사 줄에서 consortium_needed 열을 읽는다(본문 company_match.py의
    CompanyProfile 스키마엔 없는 열이지만, write_all_rows가 알려진 열 밖의 값도
    그대로 보존해줘서 같은 파일에 얹을 수 있다)."""
    rows = C.read_all_rows(store_dir)
    idx = C._find_row_index(rows, company_name)
    if idx is None:
        raise SystemExit(f"❌ 프로필을 찾을 수 없습니다: {company_name!r} "
                         f"(명부: {C._profiles_csv_path(store_dir)}) — company_match.py save로 먼저 만드세요.")
    raw = (rows[idx].get(_CONSORTIUM_CSV_COLUMN) or "").strip().lower()
    return {"true": True, "false": False}.get(raw)


def save_consortium_needed(store_dir: Path, company_name: str, consortium_needed: bool) -> Path:
    rows = C.read_all_rows(store_dir)
    idx = C._find_row_index(rows, company_name)
    if idx is None:
        raise SystemExit(f"❌ 프로필을 찾을 수 없습니다: {company_name!r} "
                         f"(명부: {C._profiles_csv_path(store_dir)}) — company_match.py save로 먼저 만드세요.")
    rows[idx][_CONSORTIUM_CSV_COLUMN] = _BOOL_TEXT[consortium_needed]
    return C.write_all_rows(store_dir, rows)


def match_company_loose(profile: C.CompanyProfile, consortium_needed: bool | None,
                        rows_by_doc: dict[str, dict[str, dict]]) -> list[dict]:
    results = []
    for document_id in sorted(rows_by_doc):
        fields = rows_by_doc[document_id]
        field_results = {
            C.REGION_FIELD: C.evaluate_region(profile, fields.get(C.REGION_FIELD)),
            C.BUSINESS_FIELD: C.evaluate_business_field(profile, fields.get(C.BUSINESS_FIELD)),
            C.CERT_FIELD: evaluate_certification_loose(profile, fields.get(C.CERT_FIELD)),
            CONSORTIUM_FIELD: evaluate_consortium(consortium_needed, fields.get(CONSORTIUM_FIELD)),
        }
        results.append({"document_id": document_id, "verdict": C.overall_verdict(field_results),
                        "reasons": {name: {"status": s, "reason": r}
                                   for name, (s, r) in field_results.items()}})
    # 추천 도구가 목적이라 "왜 되는지"(적합)를 먼저 보여준다 — "왜 안 되는지"
    # (부적합)는 그 다음. 근거는 검증용 company_match.py와 똑같이 필드마다
    # 항상 다 붙는다(적합이어도 충족 이유가, 부적합이어도 어긋난 이유가 나옴).
    order = {"적합": 0, "확인 필요": 1, "적합(근거 약함)": 2, "부적합": 3}
    return sorted(results, key=lambda r: (order[r["verdict"]], r["document_id"]))


def print_report(company: str, results: list[dict]) -> None:
    from collections import Counter
    print(f"=== [실험: 느슨한 대조 + 컨소시엄 요건] {company} 매칭 결과 ({len(results)}건) ===")
    print(dict(Counter(r["verdict"] for r in results).most_common()))
    print()
    for r in results:
        print(f"[{r['verdict']}] {r['document_id']}")
        for name in EXPERIMENTAL_FIELDS:
            info = r["reasons"][name]
            print(f"    - {name} ({info['status']}): {info['reason']}")


def main() -> int:
    ap = argparse.ArgumentParser(description="company_match 실험: 느슨한 자격증 대조 + 컨소시엄 요건")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sp_c = sub.add_parser("set-consortium", help="프로필에 컨소시엄 필요 여부 추가/수정")
    sp_c.add_argument("--company", required=True)
    sp_c.add_argument("--consortium-needed", required=True, choices=["true", "false"])
    sp_c.add_argument("--store-dir", default=str(C.DEFAULT_STORE_DIR))

    sp_m = sub.add_parser("match", help="느슨한 대조 + 컨소시엄 요건 포함 매칭")
    sp_m.add_argument("--company", required=True)
    sp_m.add_argument("--store-dir", default=str(C.DEFAULT_STORE_DIR))
    sp_m.add_argument("--extraction-table", default=str(C.DEFAULT_EXTRACTION_TABLE))
    sp_m.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.cmd == "set-consortium":
        path = save_consortium_needed(Path(args.store_dir), args.company,
                                      args.consortium_needed == "true")
        print(f"저장됨: {path}")
        return 0

    store_dir = Path(args.store_dir)
    profile = C.load_profile(store_dir, args.company)
    consortium_needed = load_consortium_needed(store_dir, args.company)
    table_path = Path(args.extraction_table)
    if not table_path.exists():
        raise SystemExit(f"❌ 추출표가 없습니다: {table_path}")
    rows_by_doc = C.load_extraction_rows(table_path)
    results = match_company_loose(profile, consortium_needed, rows_by_doc)
    print_report(args.company, results)

    if args.out:
        out_path = Path(args.out)
        with out_path.open("w", encoding="utf-8-sig", newline="") as fh:
            w = csv.writer(fh, quoting=csv.QUOTE_ALL, lineterminator="\n")
            w.writerow(["document_id", "verdict"]
                      + [f"{n}_status" for n in EXPERIMENTAL_FIELDS]
                      + [f"{n}_reason" for n in EXPERIMENTAL_FIELDS])
            for r in results:
                w.writerow([r["document_id"], r["verdict"]]
                          + [r["reasons"][n]["status"] for n in EXPERIMENTAL_FIELDS]
                          + [r["reasons"][n]["reason"] for n in EXPERIMENTAL_FIELDS])
        print(f"\nCSV 저장됨: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
