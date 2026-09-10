"""§4 검사기 S16 — 정답 출처 라벨과 근거 종류의 일치.

수정 전 후보본에서는 EXT-07 이 answer_source=metadata 인데 근거가 chunk/chunks_v3 이고
좌표가 "CSV: bid_deadline" 이었다(생성기가 명세 근거를 덮어썼다). 그 상태를 재현해
S16 이 실패하는지, 고친 상태에서 통과하는지 함께 본다.
"""
import json

import pytest

from checks.check_evalset import AssetMissing, check_answer_source_evidence_kind

TABLE = {"rows": [
    {"document_id": "RFP-000011", "field_name": "지역제한", "status": "field_absent",
     "answer_raw": "", "active": "true"},
    {"document_id": "RFP-000096", "field_name": "예산", "status": "value_present",
     "answer_raw": "3억원", "active": "true"},
]}
IDENTITY = ('"document_id","source_filename_nfc","buyer_org","project_name",'
            '"notice_date","bid_deadline","metadata_found"\n'
            '"RFP-000014","a.md","기관","사업","2024-01-01","2024-09-19 17:00:00","true"\n')
CHUNKS = [{"document_id": "RFP-000014", "section_path": ["2. 절차"],
           "location_label": "2. 절차 · 문단 2-86", "md_line_start": 130,
           "md_line_end": 150, "search_text": "입찰 마감"}]


@pytest.fixture
def assets(tmp_path):
    t = tmp_path / "table.json"; t.write_text(json.dumps(TABLE), encoding="utf-8")
    i = tmp_path / "identity.csv"; i.write_text("﻿" + IDENTITY, encoding="utf-8")
    c = tmp_path / "chunks.jsonl"
    c.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in CHUNKS), encoding="utf-8")
    return str(t), str(i), str(c)


def _run(items, assets, spec=None):
    t, i, c = assets
    return check_answer_source_evidence_kind(items, t, i, c, spec)


CHUNK_LOC = {"document": "RFP-000014", "section": "2. 절차",
             "ref_no": "2. 절차 · 문단 2-86", "line": 141}


def test_ext07_before_fix_is_rejected(assets, tmp_path):
    """수정 전 모양 — metadata 라벨인데 근거는 CSV 참조를 단 청크 하나."""
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"evidence": {"EXT-07": [
        {"kind": "identity", "document": "RFP-000014", "field": "bid_deadline"},
        {"kind": "chunk", "document": "RFP-000014"}]}}), encoding="utf-8")
    bad = [{"id": "EXT-07", "answer_type": "value", "answer_source": "metadata",
            "answer_raw": "2024-09-19 17:00:00",
            "evidence": [{"kind": "chunk", "document": "RFP-000014", "source": "chunks_v3",
                          "location": {"document": "RFP-000014", "section": "CSV",
                                       "ref_no": "CSV: bid_deadline"}}]}]
    errs = _run(bad, assets, str(spec))
    assert any("identity 근거가 없음" in e for e in errs)
    assert any("CSV 참조" in e for e in errs)
    assert any("명세가 요구한 근거가 생성 결과에 없음" in e for e in errs)


def test_ext07_after_fix_passes(assets, tmp_path):
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"evidence": {"EXT-07": [
        {"kind": "identity", "document": "RFP-000014", "field": "bid_deadline"},
        {"kind": "chunk", "document": "RFP-000014"}]}}), encoding="utf-8")
    good = [{"id": "EXT-07", "answer_type": "value", "answer_source": "metadata",
             "answer_raw": "2024-09-19 17:00:00",
             "evidence": [{"kind": "identity", "document": "RFP-000014",
                           "field": "bid_deadline", "source": "identity_v2"},
                          {"kind": "chunk", "document": "RFP-000014",
                           "source": "chunks_v3", "location": CHUNK_LOC}]}]
    assert _run(good, assets, str(spec)) == []


def test_table_answer_needs_extraction_table_evidence(assets):
    bad = [{"id": "X", "answer_type": "value", "answer_source": "table", "answer_raw": "v",
            "evidence": [{"kind": "chunk", "document": "RFP-000014", "location": CHUNK_LOC}]}]
    assert any("extraction_table 근거가 없음" in e for e in _run(bad, assets))


def test_chunk_evidence_must_point_at_a_real_chunk(assets):
    bad = [{"id": "X", "answer_type": "value", "answer_source": "verified", "answer_raw": "v",
            "evidence": [{"kind": "chunk", "document": "RFP-000014",
                          "location": {"document": "RFP-000014", "section": "없음",
                                       "ref_no": "없는 절 · 문단 99", "line": 1}}]}]
    assert any("chunks_v3 에 없음" in e for e in _run(bad, assets))


def test_identity_evidence_needs_a_real_column(assets):
    bad = [{"id": "X", "answer_type": "value", "answer_source": "metadata", "answer_raw": "v",
            "evidence": [{"kind": "identity", "document": "RFP-000014",
                          "field": "없는컬럼"}]}]
    assert any("컬럼" in e for e in _run(bad, assets))


def test_extraction_table_status_must_match(assets):
    bad = [{"id": "X", "answer_type": "value", "answer_source": "table", "answer_raw": "v",
            "evidence": [{"kind": "extraction_table", "document": "RFP-000011",
                          "field": "지역제한", "status": "value_present"}]}]
    assert any("근거 상태" in e for e in _run(bad, assets))


def test_empty_selection_and_unanswerable_are_exempt_but_not_blanket(assets):
    ok = [{"id": "SEL-020", "answer_type": "document_set", "answer_source": "table",
           "answer_raw": [], "evidence": []},
          {"id": "QA-009", "answer_type": "unanswerable", "answer_source": "metadata",
           "answer_raw": "그런 사업을 찾을 수 없습니다"}]
    assert _run(ok, assets) == []
    # 면제 문항이라도 **가지고 있는 근거**는 검사한다
    bad = [{"id": "SEL-020", "answer_type": "document_set", "answer_source": "table",
            "answer_raw": [], "evidence": [{"kind": "extraction_table",
                                            "document": "RFP-000011", "field": "없는필드"}]}]
    assert any("행 없음" in e for e in _run(bad, assets))


def test_missing_asset_is_reported(tmp_path):
    with pytest.raises(AssetMissing):
        check_answer_source_evidence_kind([], str(tmp_path / "no.json"),
                                          str(tmp_path / "no.csv"), str(tmp_path / "no.jsonl"))
