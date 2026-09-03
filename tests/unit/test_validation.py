from grader.validation import check_data_sanity, check_data_warnings

# 2026-08-27 박예진 1-11 확정 — CI 1층 데이터 정상성 게이트 4종("논쟁 없음").
THRESHOLDS = {
    "min_load_rate": 1.0,
    "max_raw_md_count_diff": 0,
    "max_filename_mismatches": 0,
    "max_csv_decode_failures": 0,
    "min_table_preservation_rate": 1.0,
    "max_encoding_corruption": 0,
}


def _clean_manifest(**over):
    base = dict(
        doc_count=10,
        load_rate=1.0,
        duplicate_doc_ids=[],
        corpus_version="corpus-1",
        extraction_table_version="extract-1",
        raw_doc_count=10,
        md_doc_count=10,
        filename_normalization_mismatches=0,
        csv_decode_failure_rows=0,
        table_preservation_rate=1.0,
        encoding_corruption_count=0,
    )
    base.update(over)
    return base


def test_clean_manifest_passes_all_four_gates():
    assert check_data_sanity(_clean_manifest(), THRESHOLDS) == []


def test_load_rate_below_threshold_is_flagged():
    problems = check_data_sanity(_clean_manifest(load_rate=0.98), THRESHOLDS)
    assert any("정상 로드율" in p for p in problems)


def test_raw_md_count_mismatch_is_flagged():
    problems = check_data_sanity(_clean_manifest(raw_doc_count=12, md_doc_count=10), THRESHOLDS)
    assert any("raw/md 문서 수 불일치" in p for p in problems)


def test_filename_normalization_mismatch_is_flagged():
    problems = check_data_sanity(_clean_manifest(filename_normalization_mismatches=2), THRESHOLDS)
    assert any("파일명 정규화 불일치" in p for p in problems)


def test_csv_decode_failure_is_flagged():
    problems = check_data_sanity(_clean_manifest(csv_decode_failure_rows=3), THRESHOLDS)
    assert any("CSV 디코딩 실패" in p for p in problems)


def test_missing_manifest_keys_are_skipped_not_flagged():
    """아직 그 산출물을 안 만든 단계(키 자체가 없음)에서는 임의의 기준으로 잡지 않는다."""
    manifest = dict(doc_count=10, load_rate=1.0, duplicate_doc_ids=[],
                    corpus_version="c1", extraction_table_version="e1")
    assert check_data_sanity(manifest, THRESHOLDS) == []


def test_empty_thresholds_never_flags_the_new_gates():
    """팀 실측/합의 전(기준 없음)에는 임의의 숫자를 기준으로 쓰지 않는다."""
    manifest = _clean_manifest(raw_doc_count=99, md_doc_count=1,
                               filename_normalization_mismatches=5, csv_decode_failure_rows=5)
    assert check_data_sanity(manifest, {}) == []


def test_table_preservation_rate_below_threshold_is_flagged():
    problems = check_data_sanity(_clean_manifest(table_preservation_rate=0.97), THRESHOLDS)
    assert any("표 보존율" in p for p in problems)


def test_encoding_corruption_is_flagged():
    problems = check_data_sanity(_clean_manifest(encoding_corruption_count=2), THRESHOLDS)
    assert any("인코딩 깨짐" in p for p in problems)


# ------------------------------------------------------------------ 표 0개 문서 (경고, 게이트 아님)

WARN_THRESHOLDS = {"max_zero_table_docs": 0}


def test_zero_table_docs_is_a_warning_not_a_gate():
    """★게이트가 아니라 경고 — check_data_sanity(문제=게이트)에는 절대 안 섞인다."""
    manifest = _clean_manifest(zero_table_doc_count=2)
    assert check_data_sanity(manifest, THRESHOLDS) == []  # 게이트는 안 걸림
    warnings = check_data_warnings(manifest, WARN_THRESHOLDS)
    assert any("표 0개" in w for w in warnings)


def test_zero_table_docs_absent_is_silent():
    manifest = _clean_manifest()  # zero_table_doc_count 키 자체가 없음
    assert check_data_warnings(manifest, WARN_THRESHOLDS) == []


def test_zero_table_docs_without_warn_threshold_is_silent():
    """팀 실측/합의 전(기준 없음)에는 임의의 숫자를 기준으로 쓰지 않는다."""
    manifest = _clean_manifest(zero_table_doc_count=5)
    assert check_data_warnings(manifest, {}) == []
