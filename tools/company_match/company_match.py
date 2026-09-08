#!/usr/bin/env python3
"""기업 프로필 저장 + 자동 매칭 — "원래 기획" 되살리기, 1차 수직 슬라이스.

회사 이름 하나로 프로필(지역·사업분야·자격증)을 저장해두고, 이미 만들어진
`extraction_table_v4.csv`(12필드 구조화 추출표, 100건, 공식·읽기 전용)의
지역제한·사업분야·참가 자격(면허·실적) 3필드와 코드로(= LLM 없이) 대조해서
공고별 적합/부적합 + 근거를 보여준다. select/extract 라우트와 같은 "코드로
찾는다" 원칙을 그대로 따른다.

⚠️ 실제 공식 자료(data/preprocessed/rfp_extraction_table_v4/*)는 읽기만 한다.
⚠️ 이 도구가 만드는 회사 프로필 저장소(기본 data/company_profiles/)는 완전히
   새 자료다 — 다른 어떤 공식 산출물과도 안 겹친다.
⚠️ answer_pipeline.py 등 공식 실행 경로에는 연결돼 있지 않다(우리만 손으로 쓰는 도구).

실제 추출표를 까보면 지역제한은 100건 중 95건, 사업분야는 99건이 "명시 없음"
(field_absent)이다 — 그래서 이 두 필드는 대부분 "정보 없음(통과)"로 나온다.
실질적인 신호는 참가 자격(면허·실적) 쪽에 많다(91건 값 있음). 값이 있는데
회사 자격증 목록과 겹치는 표현을 못 찾으면 "확인 필요"로만 남기고 절대
"부적합"으로 단정하지 않는다 — 법령 인용문이 많아 표현이 다르면 실제로는
충족하는데 못 찾은 것일 수 있다(추출표 자체의 "값을 지어내지 않는다" 원칙과
같은 이유). 반대로 지역제한·사업분야는 텍스트가 짧고 구체적(예: "부산광역시
소재", "농림수산")이라 회사 정보와 다르면 "부적합"으로 단정할 수 있다.

사용:
    # 회사 프로필 저장
    python3 company_match.py save --company "OO소프트" --region 서울 \
        --business-fields "소프트웨어개발,시스템통합" \
        --certifications "소프트웨어사업자 신고필증,정보통신공사업 등록증"

    # 회사 이름으로 불러와서 매칭
    python3 company_match.py match --company "OO소프트"
"""
from __future__ import annotations

import argparse
import csv
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_STORE_DIR = Path(__file__).resolve().parents[2] / "data" / "company_profiles"
DEFAULT_EXTRACTION_TABLE = (Path(__file__).resolve().parents[2] / "data" / "preprocessed"
                           / "rfp_extraction_table_v4" / "extraction_table_v4.csv")

REGION_FIELD = "지역제한"
BUSINESS_FIELD = "사업분야"
CERT_FIELD = "참가 자격(면허·실적)"
MATCHED_FIELDS = (REGION_FIELD, BUSINESS_FIELD, CERT_FIELD)

# 17개 광역 시·도 — 정식 명칭과 흔한 줄임말을 같은 그룹으로 묶는다(별칭이
# 늘어도 이 표만 고치면 된다. 코드에 지역명을 흩어 넣지 않는다).
REGION_GROUPS: list[list[str]] = [
    ["서울특별시", "서울"], ["부산광역시", "부산"], ["대구광역시", "대구"],
    ["인천광역시", "인천"], ["광주광역시", "광주"], ["대전광역시", "대전"],
    ["울산광역시", "울산"], ["세종특별자치시", "세종"],
    ["경기도", "경기"],
    ["강원특별자치도", "강원도", "강원"],
    ["충청북도", "충북"], ["충청남도", "충남"],
    ["전북특별자치도", "전라북도", "전북"], ["전라남도", "전남"],
    ["경상북도", "경북"], ["경상남도", "경남"],
    ["제주특별자치도", "제주도", "제주"],
]


def _norm(s: str) -> str:
    return unicodedata.normalize("NFC", (s or "").strip())


def _region_group(name: str) -> list[str] | None:
    name = _norm(name)
    for group in REGION_GROUPS:
        if any(name == alias or name in alias or alias in name for alias in group):
            return group
    return None


def find_region_in_text(text: str) -> str | None:
    """본문에서 광역 시·도 이름을 하나 찾는다(가장 먼저 등장하는 것). 못 찾으면 None."""
    text = text or ""
    for group in REGION_GROUPS:
        for alias in sorted(group, key=len, reverse=True):
            if alias in text:
                return group[0]
    return None


def region_matches(company_region: str, required_region: str) -> bool:
    a, b = _region_group(company_region), _region_group(required_region)
    return bool(a and b and a == b)


def find_substring_match(text: str, candidates: list[str]) -> str | None:
    """candidates 중 text 안에 그대로 들어있는(부분 문자열) 첫 표현을 돌려준다."""
    text = _norm(text)
    for c in candidates:
        c = _norm(c)
        if c and c in text:
            return c
    return None


@dataclass
class CompanyProfile:
    company_name: str
    region: str = ""
    business_fields: list[str] = field(default_factory=list)
    certifications: list[str] = field(default_factory=list)

    def to_row(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        row = {"company_name": self.company_name, "region": self.region,
              "business_fields": ";".join(self.business_fields),
              "certifications": ";".join(self.certifications)}
        row.update(extra or {})
        return row

    @classmethod
    def from_row(cls, row: dict[str, str]) -> "CompanyProfile":
        return cls(company_name=row["company_name"], region=row.get("region", ""),
                   business_fields=[s for s in (row.get("business_fields") or "").split(";") if s],
                   certifications=[s for s in (row.get("certifications") or "").split(";") if s])


# 회사가 늘어나도 회사당 파일 하나씩 만들지 않는다 — identity_v2/document_registry_v2와
# 같은 결로, 명부 파일 하나에 회사 한 줄씩 쌓이는 CSV 하나만 쓴다. 실험 모듈
# (loose_match_experiment.py)이 여기에 없는 열(consortium_needed)을 얹어 쓸 수 있게
# CSV_FIELDS 밖의 열이 있어도 읽기/쓰기 둘 다 그 값을 보존한다.
PROFILES_CSV_NAME = "company_profiles.csv"
BASE_CSV_FIELDS = ["company_name", "region", "business_fields", "certifications"]


def _profiles_csv_path(store_dir: Path) -> Path:
    return store_dir / PROFILES_CSV_NAME


def read_all_rows(store_dir: Path) -> list[dict[str, str]]:
    path = _profiles_csv_path(store_dir)
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return [dict(r) for r in csv.DictReader(fh)]


def write_all_rows(store_dir: Path, rows: list[dict[str, str]]) -> Path:
    store_dir.mkdir(parents=True, exist_ok=True)
    path = _profiles_csv_path(store_dir)
    fieldnames = list(BASE_CSV_FIELDS)
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames, quoting=csv.QUOTE_ALL, lineterminator="\n")
        w.writeheader()
        w.writerows({k: r.get(k, "") for k in fieldnames} for r in rows)
    return path


def _find_row_index(rows: list[dict[str, str]], company_name: str) -> int | None:
    target = _norm(company_name)
    for i, row in enumerate(rows):
        if _norm(row.get("company_name", "")) == target:
            return i
    return None


def save_profile(store_dir: Path, profile: CompanyProfile) -> Path:
    """명부 CSV에 회사를 한 줄 추가하거나(신규), 같은 이름 줄을 갱신한다(기존).
    실험 모듈이 얹어둔 열(예: consortium_needed)은 갱신 시에도 그대로 보존한다."""
    rows = read_all_rows(store_dir)
    idx = _find_row_index(rows, profile.company_name)
    if idx is None:
        rows.append(profile.to_row())
    else:
        rows[idx].update(profile.to_row())
    return write_all_rows(store_dir, rows)


def load_profile(store_dir: Path, company_name: str) -> CompanyProfile:
    """회사 이름으로 명부에서 한 줄을 찾는다(공백·정규화 차이는 봐주되, 다른
    회사로 착각하지 않도록 정확히 같은 이름만 허용한다)."""
    rows = read_all_rows(store_dir)
    idx = _find_row_index(rows, company_name)
    if idx is None:
        raise SystemExit(f"❌ 프로필을 찾을 수 없습니다: {company_name!r} "
                         f"(명부: {_profiles_csv_path(store_dir)})")
    return CompanyProfile.from_row(rows[idx])


def load_extraction_rows(path: Path) -> dict[str, dict[str, dict]]:
    """document_id -> {field_name: row} — active·retrieval_eligible 아닌 문서는 뺀다
    (document_registry.py가 검색·선별 대상을 거르는 것과 같은 기준)."""
    with path.open(encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    by_doc: dict[str, dict[str, dict]] = {}
    for r in rows:
        if r.get("active") != "true" or r.get("retrieval_eligible") != "true":
            continue
        by_doc.setdefault(r["document_id"], {})[r["field_name"]] = r
    return by_doc


def evaluate_region(profile: CompanyProfile, row: dict | None) -> tuple[str, str]:
    # 다른 필드들과 같은 규칙: '명시 없음'은 실제로 대조해서 나온 결론이
    # 아니므로 "정보없음"으로 남긴다("충족"은 회사 값과 직접 비교해 확인된
    # 경우에만 쓴다) — 그래야 "실제 근거가 있는 적합"과 "확인할 게 없어서
    # 그냥 통과된 적합"을 뒤에서 구분할 수 있다.
    if row is None or row["status"] == "field_absent":
        return "정보없음", "지역 제한 명시 없음"
    text = row.get("answer_normalized") or row.get("answer_raw") or ""
    required = find_region_in_text(text)
    if required is None:
        return "확인 필요", f"지역 제한 문구가 있으나 지역명을 자동 인식 못함 — 직접 확인: {text[:150]}"
    if not profile.region:
        return "확인 필요", f"이 사업은 {required} 소재 업체 대상 — 회사 지역 정보 없음"
    if region_matches(profile.region, required):
        return "충족", f"이 사업의 소재지 요건({required})이 회사 지역({profile.region})과 일치"
    return "부적합", f"이 사업은 {required} 소재 업체만 가능 — 회사 지역({profile.region})과 다름"


def evaluate_business_field(profile: CompanyProfile, row: dict | None) -> tuple[str, str]:
    if row is None or row["status"] == "field_absent":
        return "정보없음", "사업분야 명시 없음 — 참고만 하고 통과 처리"
    text = row.get("answer_normalized") or row.get("answer_raw") or ""
    if not profile.business_fields:
        return "확인 필요", f"이 사업의 사업분야는 '{text}' — 회사 사업분야 정보 없음"
    matched = find_substring_match(text, profile.business_fields)
    if matched:
        return "충족", f"이 사업의 사업분야 '{text}'가 회사 사업분야 '{matched}'와 일치"
    return "부적합", f"이 사업의 사업분야는 '{text}' — 회사가 등록한 사업분야와 다름"


def evaluate_certification(profile: CompanyProfile, row: dict | None) -> tuple[str, str]:
    if row is None or row["status"] == "field_absent":
        return "정보없음", "참가 자격 명시 없음"
    text = row.get("answer_normalized") or row.get("answer_raw") or ""
    if not profile.certifications:
        return "확인 필요", "참가 자격 요건이 있으나 회사 보유 자격증 정보 없음 — 직접 대조 필요"
    matched = find_substring_match(text, profile.certifications)
    if matched:
        return "충족", f"회사 보유 자격 '{matched}'가 참가 자격 요건 문구에서 확인됨"
    # 법령 인용이 섞인 자유 서술이라 못 찾았다고 곧바로 '부적합'으로 단정하지 않는다.
    return "확인 필요", f"회사가 등록한 자격증 표현이 요건 문구에서 안 보임 — 직접 대조 필요: {text[:150]}"


EVALUATORS = {REGION_FIELD: evaluate_region, BUSINESS_FIELD: evaluate_business_field,
              CERT_FIELD: evaluate_certification}


def overall_verdict(field_results: dict[str, tuple[str, str]]) -> str:
    """부적합 > 확인 필요 > 적합 > 적합(근거 약함) 순으로 판정한다.

    '적합'은 최소 한 필드가 회사 값과 실제로 대조해서 확인된("충족") 경우에만
    쓴다. 전부 '정보없음'(=이 공고엔 그 조건 자체가 없어서 아무것도 못 걸러낸
    상태)이면 '적합(근거 약함)'으로 따로 표시한다 — 안 그러면 "이 회사한테
    맞다고 확인된 공고"와 "그냥 조건이 하나도 안 적힌 공고"가 똑같이 보인다."""
    statuses = {s for s, _ in field_results.values()}
    if "부적합" in statuses:
        return "부적합"
    if "확인 필요" in statuses:
        return "확인 필요"
    if "충족" in statuses:
        return "적합"
    return "적합(근거 약함)"


def match_company(profile: CompanyProfile,
                  rows_by_doc: dict[str, dict[str, dict]]) -> list[dict]:
    results = []
    for document_id in sorted(rows_by_doc):
        fields = rows_by_doc[document_id]
        field_results = {name: EVALUATORS[name](profile, fields.get(name))
                         for name in MATCHED_FIELDS}
        results.append({
            "document_id": document_id, "verdict": overall_verdict(field_results),
            "reasons": {name: {"status": status, "reason": reason}
                       for name, (status, reason) in field_results.items()},
        })
    order = {"부적합": 0, "확인 필요": 1, "적합": 2, "적합(근거 약함)": 3}
    return sorted(results, key=lambda r: (order[r["verdict"]], r["document_id"]))


def print_report(company: str, results: list[dict]) -> None:
    from collections import Counter
    print(f"=== {company} 매칭 결과 ({len(results)}건) ===")
    print(dict(Counter(r["verdict"] for r in results).most_common()))
    print()
    for r in results:
        print(f"[{r['verdict']}] {r['document_id']}")
        for name in MATCHED_FIELDS:
            info = r["reasons"][name]
            print(f"    - {name} ({info['status']}): {info['reason']}")


def main() -> int:
    ap = argparse.ArgumentParser(description="기업 프로필 저장 + RFP 자동 매칭")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sp_save = sub.add_parser("save", help="회사 프로필 저장")
    sp_save.add_argument("--company", required=True)
    sp_save.add_argument("--region", default="")
    sp_save.add_argument("--business-fields", default="",
                         help="쉼표로 구분(예: 소프트웨어개발,시스템통합)")
    sp_save.add_argument("--certifications", default="",
                         help="쉼표로 구분(예: 소프트웨어사업자 신고필증,정보통신공사업 등록증)")
    sp_save.add_argument("--store-dir", default=str(DEFAULT_STORE_DIR))

    sp_match = sub.add_parser("match", help="회사 이름으로 불러와서 매칭")
    sp_match.add_argument("--company", required=True)
    sp_match.add_argument("--store-dir", default=str(DEFAULT_STORE_DIR))
    sp_match.add_argument("--extraction-table", default=str(DEFAULT_EXTRACTION_TABLE))
    sp_match.add_argument("--out", default=None, help="결과 CSV 저장 경로(선택)")

    args = ap.parse_args()

    if args.cmd == "save":
        profile = CompanyProfile(
            company_name=args.company, region=args.region,
            business_fields=[s.strip() for s in args.business_fields.split(",") if s.strip()],
            certifications=[s.strip() for s in args.certifications.split(",") if s.strip()])
        path = save_profile(Path(args.store_dir), profile)
        print(f"저장됨: {path}")
        return 0

    profile = load_profile(Path(args.store_dir), args.company)
    table_path = Path(args.extraction_table)
    if not table_path.exists():
        raise SystemExit(f"❌ 추출표가 없습니다: {table_path}")
    rows_by_doc = load_extraction_rows(table_path)
    results = match_company(profile, rows_by_doc)
    print_report(args.company, results)

    if args.out:
        out_path = Path(args.out)
        with out_path.open("w", encoding="utf-8-sig", newline="") as fh:
            w = csv.writer(fh, quoting=csv.QUOTE_ALL, lineterminator="\n")
            w.writerow(["document_id", "verdict"] + [f"{n}_status" for n in MATCHED_FIELDS]
                      + [f"{n}_reason" for n in MATCHED_FIELDS])
            for r in results:
                w.writerow([r["document_id"], r["verdict"]]
                          + [r["reasons"][n]["status"] for n in MATCHED_FIELDS]
                          + [r["reasons"][n]["reason"] for n in MATCHED_FIELDS])
        print(f"\nCSV 저장됨: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
