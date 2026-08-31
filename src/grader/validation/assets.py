"""데이터 자산 검증 — 코퍼스 문서 참조 무결성(2-17) + CI 1층 데이터 정상성(1-19-1)."""

from __future__ import annotations

from typing import Iterable

from ..models import EvaluationItem, as_id_list


# ------------------------------------------------------------------ 참조 무결성 (2-17)

def check_references(
    items: Iterable[EvaluationItem],
    corpus_doc_ids: set[str] | None = None,
    retrieval_excluded_ids: set[str] | None = None,
) -> list[str]:
    """정답 근거가 가리키는 문서가 (1) 코퍼스에 실재하고 (2) 검색 대상인지.

    retrieval_excluded_ids: 등록부에서 검색 대상이 아닌 문서(수집 중복 등, 1-9-1
      확정: RFP-000006/RFP-000017 은 각각 RFP-000075/RFP-000098 의 중복 → 검색대상 98건).
    """
    problems: list[str] = []
    excluded = retrieval_excluded_ids or set()

    if corpus_doc_ids is not None:
        for it in items:
            if it.location is not None and it.location.document not in corpus_doc_ids:
                problems.append(f"[참조] {it.id}: 코퍼스에 없는 document {it.location.document}")
            for d in as_id_list(it.document_id):
                if d not in corpus_doc_ids:
                    problems.append(f"[참조] {it.id}: document_id 미존재 {d}")
            for d in as_id_list(it.intermediate_answer):
                if d not in corpus_doc_ids:
                    problems.append(f"[참조] {it.id}: intermediate_answer 미존재 문서 {d}")
            if it.answer_type == "document_set":
                for d in as_id_list(it.answer_raw):
                    if d not in corpus_doc_ids:
                        problems.append(f"[참조] {it.id}: answer_raw 문서 미존재 {d}")

    if excluded:
        for it in items:
            gold_docs = set(as_id_list(it.document_id)) | set(as_id_list(it.intermediate_answer))
            if it.answer_type == "document_set":
                gold_docs |= set(as_id_list(it.answer_raw))
            if it.location is not None:
                gold_docs.add(it.location.document)
            hit = sorted(gold_docs & excluded)
            if hit:
                problems.append(
                    f"[중복제외] {it.id}: 정답 근거가 검색 대상 아닌 문서 {hit} "
                    f"(수집 중복 → 검색대상 98건에서 빠짐, 1-9-1)")
    return problems


def check_quota(items: Iterable[EvaluationItem], quota: dict[str, int] | None,
                quota_tolerance: float = 0.2) -> list[str]:
    problems: list[str] = []
    if not quota:
        return problems
    counts = {t: 0 for t in quota}
    for it in items:
        if it.task_type in counts:
            counts[it.task_type] += 1
    for t, target in quota.items():
        got = counts.get(t, 0)
        if target and abs(got - target) > max(1, target * quota_tolerance):
            problems.append(f"[할당량] task_type={t} 설계 {target} vs 실제 {got}")
    return problems


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
