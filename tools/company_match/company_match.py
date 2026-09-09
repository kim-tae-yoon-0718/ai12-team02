#!/usr/bin/env python3
"""회사 프로필과 공식 추출표를 안전하게 대조하는 참여 후보 추천 도구.

이 모듈은 자연어의 뜻을 코드가 추측하지 않는다. 공식 추출표의 필드·상태와
폐쇄된 값 목록으로 완전히 비교할 수 있는 조건만 자동 확인한다. 자유 서술인
자격·업종·공동수급 조건은 근거를 그대로 보여주고 사람 확인 대상으로 남긴다.

최종 판정은 다음 세 가지뿐이다.

* 우선 검토 후보: 명시적으로 맞는 조건이 하나 이상이고 명시적 충돌은 없음
* 확인 필요: 명시적 충돌은 없지만 사람이 읽어야 하는 조건이 남음
* 제외: 폐쇄된 구조 값에서 명시적 충돌이 확인됨

어떤 경우에도 이 결과를 법적 의미의 입찰 참가 자격 확정으로 표현하지 않는다.
"""
from __future__ import annotations

import argparse
import csv
import json
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STORE_DIR = REPO_ROOT / "data" / "company_profiles"

REGION_FIELD = "지역제한"
BUSINESS_FIELD = "사업분야"
CERT_FIELD = "참가 자격(면허·실적)"
CONSORTIUM_FIELD = "컨소시엄 요건"
MATCHED_FIELDS = (REGION_FIELD, BUSINESS_FIELD, CERT_FIELD, CONSORTIUM_FIELD)

FIELD_CONFIRMED = "확인됨"
FIELD_NOT_STATED = "조건 미기재"
FIELD_REVIEW = "사람 확인 필요"
FIELD_CONTRADICTION = "명시적 불일치"

VERDICT_PRIORITY = "우선 검토 후보"
VERDICT_REVIEW = "확인 필요"
VERDICT_EXCLUDED = "제외"

# 광역 시·도는 의미가 정해진 닫힌 목록이므로 코드로 안전하게 비교할 수 있다.
# 자유로운 동의어를 새로 추측하지 않고, 행정구역의 정식 명칭과 통상 약칭만 쓴다.
REGION_GROUPS: tuple[tuple[str, ...], ...] = (
    ("서울특별시", "서울"), ("부산광역시", "부산"),
    ("대구광역시", "대구"), ("인천광역시", "인천"),
    ("광주광역시", "광주"), ("대전광역시", "대전"),
    ("울산광역시", "울산"), ("세종특별자치시", "세종"),
    ("경기도", "경기"), ("강원특별자치도", "강원도", "강원"),
    ("충청북도", "충북"), ("충청남도", "충남"),
    ("전북특별자치도", "전라북도", "전북"),
    ("전라남도", "전남"), ("경상북도", "경북"),
    ("경상남도", "경남"), ("제주특별자치도", "제주도", "제주"),
)


def _norm(value: Any) -> str:
    return unicodedata.normalize("NFC", str(value or "")).strip()


def _compact(value: Any) -> str:
    return "".join(_norm(value).split()).casefold()


def _row_is_active(row: dict[str, Any]) -> bool:
    return str(row.get("active", "true")).lower() != "false"


def _row_is_eligible(row: dict[str, Any]) -> bool:
    return str(row.get("retrieval_eligible", "true")).lower() != "false"


def _row_text(row: dict[str, Any] | None) -> str:
    if row is None:
        return ""
    return _norm(row.get("answer_normalized") or row.get("answer_raw"))


def _row_location(row: dict[str, Any] | None) -> dict[str, Any] | str | None:
    if row is None:
        return None
    return row.get("representative_location") or None


def _canonical_region(value: str) -> str | None:
    target = _norm(value)
    for group in REGION_GROUPS:
        if target in group:
            return group[0]
    return None


def _regions_in_curated_field(value: str) -> list[str]:
    """검수된 지역제한 필드에서 행정구역 이름만 뽑는다.

    같은 그룹의 정식명과 약칭이 함께 나타나도 한 지역으로 합치며, 둘 이상의
    지역이 나오면 호출부가 자동 판정을 포기한다.
    """
    text = _norm(value)
    found: list[str] = []
    for group in REGION_GROUPS:
        if any(alias in text for alias in sorted(group, key=len, reverse=True)):
            found.append(group[0])
    return found


@dataclass
class CompanyProfile:
    company_name: str
    region: str = ""
    business_fields: list[str] = field(default_factory=list)
    certifications: list[str] = field(default_factory=list)
    consortium_needed: bool | None = None

    def to_row(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        consortium = "" if self.consortium_needed is None else str(self.consortium_needed).lower()
        row = {
            "company_name": self.company_name,
            "region": self.region,
            "business_fields": ";".join(self.business_fields),
            "certifications": ";".join(self.certifications),
            "consortium_needed": consortium,
        }
        row.update(extra or {})
        return row

    @classmethod
    def from_row(cls, row: dict[str, str]) -> "CompanyProfile":
        raw_consortium = (row.get("consortium_needed") or "").strip().lower()
        consortium = {"true": True, "false": False}.get(raw_consortium)
        return cls(
            company_name=row["company_name"],
            region=row.get("region", ""),
            business_fields=[v for v in (row.get("business_fields") or "").split(";") if v],
            certifications=[v for v in (row.get("certifications") or "").split(";") if v],
            consortium_needed=consortium,
        )


PROFILES_CSV_NAME = "company_profiles.csv"
BASE_CSV_FIELDS = [
    "company_name", "region", "business_fields", "certifications", "consortium_needed",
]


def _profiles_csv_path(store_dir: Path) -> Path:
    return store_dir / PROFILES_CSV_NAME


def read_all_rows(store_dir: Path) -> list[dict[str, str]]:
    path = _profiles_csv_path(store_dir)
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_all_rows(store_dir: Path, rows: list[dict[str, str]]) -> Path:
    store_dir.mkdir(parents=True, exist_ok=True)
    path = _profiles_csv_path(store_dir)
    fieldnames = list(BASE_CSV_FIELDS)
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fieldnames, quoting=csv.QUOTE_ALL, lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in fieldnames} for row in rows)
    return path


def _find_row_index(rows: list[dict[str, str]], company_name: str) -> int | None:
    target = _compact(company_name)
    for index, row in enumerate(rows):
        if _compact(row.get("company_name")) == target:
            return index
    return None


def save_profile(store_dir: Path, profile: CompanyProfile) -> Path:
    rows = read_all_rows(store_dir)
    index = _find_row_index(rows, profile.company_name)
    if index is None:
        rows.append(profile.to_row())
    else:
        rows[index].update(profile.to_row())
    return write_all_rows(store_dir, rows)


def load_profile(store_dir: Path, company_name: str) -> CompanyProfile:
    rows = read_all_rows(store_dir)
    index = _find_row_index(rows, company_name)
    if index is None:
        raise LookupError(f"등록된 회사 프로필이 없습니다: {company_name!r}")
    return CompanyProfile.from_row(rows[index])


def group_rows_by_document(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        if not _row_is_active(row) or not _row_is_eligible(row):
            continue
        document_id = _norm(row.get("document_id"))
        field_name = _norm(row.get("field_name"))
        if not document_id or field_name not in MATCHED_FIELDS:
            continue
        fields = grouped.setdefault(document_id, {})
        if field_name in fields:
            raise ValueError(f"활성 추출표 행이 중복되었습니다: {document_id}/{field_name}")
        fields[field_name] = row
    return grouped


def load_extraction_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".json":
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        rows = payload.get("rows") if isinstance(payload, dict) else None
        if not isinstance(rows, list):
            raise ValueError(f"추출표 JSON에 rows 목록이 없습니다: {path}")
        return rows
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _decision(status: str, reason: str, row: dict[str, Any] | None) -> dict[str, Any]:
    return {
        "status": status,
        "reason": reason,
        "evidence": _row_text(row),
        "location": _row_location(row),
    }


def evaluate_region(profile: CompanyProfile, row: dict[str, Any] | None) -> dict[str, Any]:
    if row is None or row.get("status") == "field_absent":
        return _decision(FIELD_NOT_STATED, "추출표에 지역 제한이 명시되지 않았습니다.", row)
    if row.get("status") != "value_present":
        return _decision(FIELD_REVIEW, "지역 제한 자료의 상태가 확정값이 아니어서 사람이 확인해야 합니다.", row)
    company_region = _canonical_region(profile.region)
    required_regions = _regions_in_curated_field(_row_text(row))
    if company_region is None:
        return _decision(FIELD_REVIEW, "회사 지역을 정해진 행정구역 값으로 확인할 수 없습니다.", row)
    if len(required_regions) != 1:
        return _decision(FIELD_REVIEW, "지역 제한에 지역이 없거나 여러 개여서 자동 판정을 하지 않습니다.", row)
    required = required_regions[0]
    if company_region == required:
        return _decision(FIELD_CONFIRMED, f"회사 지역과 공고의 지역 제한이 모두 {required}입니다.", row)
    return _decision(
        FIELD_CONTRADICTION,
        f"공고는 {required} 소재 업체로 제한되지만 회사 지역은 {company_region}입니다.",
        row,
    )


def evaluate_business_field(profile: CompanyProfile, row: dict[str, Any] | None) -> dict[str, Any]:
    if row is None or row.get("status") == "field_absent":
        return _decision(FIELD_NOT_STATED, "추출표에 사업분야 조건이 명시되지 않았습니다.", row)
    if row.get("status") != "value_present":
        return _decision(FIELD_REVIEW, "사업분야 자료의 상태가 확정값이 아니어서 사람이 확인해야 합니다.", row)
    value = _compact(_row_text(row))
    registered = {_compact(item) for item in profile.business_fields if _compact(item)}
    if value and value in registered:
        return _decision(FIELD_CONFIRMED, "공고의 사업분야 값이 회사가 입력한 값과 정확히 같습니다.", row)
    return _decision(
        FIELD_REVIEW,
        "사업분야는 비슷한 단어만으로 일치 처리하지 않습니다. 사람이 원문 조건을 확인해야 합니다.",
        row,
    )


def evaluate_certification(profile: CompanyProfile, row: dict[str, Any] | None) -> dict[str, Any]:
    if row is None or row.get("status") == "field_absent":
        return _decision(FIELD_NOT_STATED, "추출표에 참가 자격 조건이 명시되지 않았습니다.", row)
    if row.get("status") != "value_present":
        return _decision(FIELD_REVIEW, "참가 자격 자료의 상태가 확정값이 아니어서 사람이 확인해야 합니다.", row)
    value = _compact(_row_text(row))
    registered = {_compact(item) for item in profile.certifications if _compact(item)}
    if value and value in registered:
        return _decision(FIELD_CONFIRMED, "참가 자격 값 전체가 회사가 입력한 자격 값과 정확히 같습니다.", row)
    return _decision(
        FIELD_REVIEW,
        "참가 자격은 법령·실적·유효기간 등 여러 조건이 섞일 수 있어 단어가 겹쳐도 자동 충족 처리하지 않습니다.",
        row,
    )


def evaluate_consortium(profile: CompanyProfile, row: dict[str, Any] | None) -> dict[str, Any]:
    if row is None or row.get("status") == "field_absent":
        return _decision(FIELD_NOT_STATED, "추출표에 공동수급 조건이 명시되지 않았습니다.", row)
    # 허용·금지·조건부 문장의 뜻을 코드 패턴으로 추측하지 않는다. 회사가 공동수급을
    # 원하지 않아도 공고가 공동수급을 필수로 할 수 있으므로 값이 있으면 항상 확인한다.
    return _decision(
        FIELD_REVIEW,
        "공동수급 조건은 허용 방식·구성원 수·지분율 등을 사람이 원문과 함께 확인해야 합니다.",
        row,
    )


EVALUATORS = {
    REGION_FIELD: evaluate_region,
    BUSINESS_FIELD: evaluate_business_field,
    CERT_FIELD: evaluate_certification,
    CONSORTIUM_FIELD: evaluate_consortium,
}


def overall_verdict(field_results: dict[str, dict[str, Any]]) -> str:
    statuses = {item["status"] for item in field_results.values()}
    if FIELD_CONTRADICTION in statuses:
        return VERDICT_EXCLUDED
    # '우선 검토'는 참가 가능 확정이 아니다. 닫힌 값에서 확인된 조건이 하나라도
    # 있으면 먼저 보이게 하되, 남은 자유 서술은 각 필드의 '사람 확인 필요'
    # 표시를 그대로 유지한다.
    if FIELD_CONFIRMED in statuses:
        return VERDICT_PRIORITY
    return VERDICT_REVIEW


def match_company(
    profile: CompanyProfile,
    rows: Iterable[dict[str, Any]],
    eligible_document_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    grouped = group_rows_by_document(rows)
    results: list[dict[str, Any]] = []
    for document_id in sorted(grouped):
        if eligible_document_ids is not None and document_id not in eligible_document_ids:
            continue
        fields = grouped[document_id]
        decisions = {
            field_name: EVALUATORS[field_name](profile, fields.get(field_name))
            for field_name in MATCHED_FIELDS
        }
        results.append({
            "document_id": document_id,
            "verdict": overall_verdict(decisions),
            "reasons": decisions,
            "confirmed_count": sum(
                item["status"] == FIELD_CONFIRMED for item in decisions.values()
            ),
            "review_count": sum(item["status"] == FIELD_REVIEW for item in decisions.values()),
        })
    order = {VERDICT_PRIORITY: 0, VERDICT_REVIEW: 1, VERDICT_EXCLUDED: 2}
    return sorted(
        results,
        key=lambda item: (
            order[item["verdict"]], -item["confirmed_count"], item["review_count"],
            item["document_id"],
        ),
    )


def print_report(company_name: str, results: list[dict[str, Any]]) -> None:
    from collections import Counter

    print(f"=== {company_name} 참여 후보 대조 결과 ({len(results)}건) ===")
    print(dict(Counter(item["verdict"] for item in results)))
    for item in results:
        print(f"\n[{item['verdict']}] {item['document_id']}")
        for field_name in MATCHED_FIELDS:
            detail = item["reasons"][field_name]
            print(f"  - {field_name} ({detail['status']}): {detail['reason']}")
            if detail["status"] == FIELD_REVIEW and detail["evidence"]:
                print(f"    근거: {detail['evidence'][:240]}")


def main() -> int:
    parser = argparse.ArgumentParser(description="회사 프로필 저장 및 안전한 참여 후보 대조")
    subparsers = parser.add_subparsers(dest="command", required=True)

    save_parser = subparsers.add_parser("save", help="회사 프로필 저장")
    save_parser.add_argument("--company", required=True)
    save_parser.add_argument("--region", default="")
    save_parser.add_argument("--business-fields", default="")
    save_parser.add_argument("--certifications", default="")
    save_parser.add_argument("--consortium-needed", choices=("true", "false", "unknown"),
                             default="unknown")
    save_parser.add_argument("--store-dir", default=str(DEFAULT_STORE_DIR))

    match_parser = subparsers.add_parser("match", help="공식 추출표와 안전하게 대조")
    match_parser.add_argument("--company", required=True)
    match_parser.add_argument("--extraction-table", required=True)
    match_parser.add_argument("--store-dir", default=str(DEFAULT_STORE_DIR))
    args = parser.parse_args()

    store_dir = Path(args.store_dir)
    if args.command == "save":
        consortium = {"true": True, "false": False, "unknown": None}[args.consortium_needed]
        profile = CompanyProfile(
            company_name=args.company,
            region=args.region,
            business_fields=[v.strip() for v in args.business_fields.split(",") if v.strip()],
            certifications=[v.strip() for v in args.certifications.split(",") if v.strip()],
            consortium_needed=consortium,
        )
        print(f"저장됨: {save_profile(store_dir, profile)}")
        return 0

    profile = load_profile(store_dir, args.company)
    rows = load_extraction_rows(Path(args.extraction_table))
    print_report(args.company, match_company(profile, rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
