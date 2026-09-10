"""데이터 자산 검증 — CI 1층 데이터 정상성(1-19-1, 박예진 산출물 data_manifest).

★ 코퍼스 문서 참조 무결성·할당량(2-17)은 checks.check_evalset 로 이동했다(단일 출처).
"""

from __future__ import annotations


# ------------------------------------------------------------------ CI 1층: 데이터 정상성 (1-19-1)

def check_data_sanity(manifest: dict, thresholds: dict | None = None) -> list[str]:
    """데이터 계약 검사(1-19-1, 박예진 소관 산출물 기준).

    ★ 모델 성능 평가가 아니라 데이터 계약 검사다. 훨씬 싸고 훨씬 빨리 잡는다.
    thresholds 가 비어 있으면(팀 실측 전) 임의의 숫자를 기준으로 쓰지 않고 통과시킨다.
    """
    th = thresholds or {}
    problems: list[str] = []

    exp = th.get("expected_docs")
    if exp is not None and manifest.get("doc_count") != exp:
        problems.append(f"문서 수 {manifest.get('doc_count')} ≠ 예상 {exp}")
    lr = manifest.get("load_rate")
    if lr is not None and "min_load_rate" in th and lr < th["min_load_rate"]:
        problems.append(f"정상 로드율 {lr} < 기준 {th['min_load_rate']}")
    if manifest.get("duplicate_doc_ids"):
        problems.append(f"문서 ID 중복 {manifest['duplicate_doc_ids']}")
    for k in ("corpus_version", "extraction_table_version"):
        if not manifest.get(k):
            problems.append(f"버전 꼬리표 없음: {k} — 섞임 검출 불가")
    miss = manifest.get("critical_column_missing_rate")
    if miss is not None and "max_critical_missing_rate" in th and miss > th["max_critical_missing_rate"]:
        problems.append(f"critical 컬럼 결측률 {miss} > 기준 {th['max_critical_missing_rate']}")

    raw_count = manifest.get("raw_doc_count")
    md_count = manifest.get("md_doc_count")
    if raw_count is not None and md_count is not None and "max_raw_md_count_diff" in th:
        diff = abs(raw_count - md_count)
        if diff > th["max_raw_md_count_diff"]:
            problems.append(
                f"raw/md 문서 수 불일치: raw={raw_count} md={md_count} (허용 diff {th['max_raw_md_count_diff']})")
    fn_mismatches = manifest.get("filename_normalization_mismatches")
    if fn_mismatches is not None and "max_filename_mismatches" in th and fn_mismatches > th["max_filename_mismatches"]:
        problems.append(f"파일명 정규화 불일치 {fn_mismatches}건 > 기준 {th['max_filename_mismatches']}")
    csv_fail = manifest.get("csv_decode_failure_rows")
    if csv_fail is not None and "max_csv_decode_failures" in th and csv_fail > th["max_csv_decode_failures"]:
        problems.append(f"CSV 디코딩 실패 행 {csv_fail}건 > 기준 {th['max_csv_decode_failures']}")

    table_rate = manifest.get("table_preservation_rate")
    if table_rate is not None and "min_table_preservation_rate" in th and table_rate < th["min_table_preservation_rate"]:
        problems.append(f"표 보존율 {table_rate} < 기준 {th['min_table_preservation_rate']}")
    enc_fail = manifest.get("encoding_corruption_count")
    if enc_fail is not None and "max_encoding_corruption" in th and enc_fail > th["max_encoding_corruption"]:
        problems.append(f"인코딩 깨짐 {enc_fail}건 > 기준 {th['max_encoding_corruption']}")

    return problems


def check_data_warnings(manifest: dict, warn_thresholds: dict | None = None) -> list[str]:
    """게이트가 아니라 경고(박예진). 실행을 막지 않고 사람이 한 번 보고 넘긴다.

    ★ 표 0개 문서: 스캔된 한글 파일(OCR 미실행) 신호. 1-11 게이트 4종을 다 통과하므로
    이 값이 유일한 값싼 신호다. 게이트로 두면 정상 RFP 하나에 전체가 멈추므로 경고로만 둠.
    한계(1-19-1): 문서 일부만 스캔인 경우는 못 잡는다.
    """
    th = warn_thresholds or {}
    warnings: list[str] = []
    zero_table = manifest.get("zero_table_doc_count")
    if zero_table is not None and "max_zero_table_docs" in th and zero_table > th["max_zero_table_docs"]:
        warnings.append(
            f"[경고] 표 0개인 문서 {zero_table}건 — 스캔된 한글 파일(OCR 미실행) 가능성. "
            "게이트 아님, 실행을 막지 않는다. 사람이 직접 열어 확인할 것 "
            "(한계: 문서 일부만 스캔인 경우는 못 잡음 — 1-19-1)")
    return warnings
