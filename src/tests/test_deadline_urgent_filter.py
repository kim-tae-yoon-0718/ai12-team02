"""마감 임박 필터(deadline_urgent_days) 회귀 테스트.

⚠️ 문서 ID는 평가셋과 겹치지 않는 대역(RFP-0009xx)을 쓴다.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from answer_pipeline import apply_deadline_filter, urgent_deadline_banner, SessionState
from identity_metadata import IdentityIndex, IdentityRecord

REF = datetime(2024, 6, 1)

BASE_CFG = {
    "reference_datetime_source": "external",
    "reference_datetime": "2024-06-01",
    "deadline_missing_policy": "show_as_unknown",
    "deadline_filter_disclosure": False,
    "deadline_filter_default": {"select": True, "extract": False,
                                "qa": False, "compare": False},
}


def cfg(**over) -> dict:
    out = dict(BASE_CFG)
    out.update(over)
    return out


def _record(doc_id: str, deadline: datetime | None) -> IdentityRecord:
    return IdentityRecord(
        document_id=doc_id,
        source_filename_nfc=f"{doc_id}.md",
        notice_number=doc_id,
        buyer_org="테스트기관",
        project_name="테스트사업",
        notice_date="2024-05-01",
        bid_deadline_raw="" if deadline is None else deadline.strftime("%Y-%m-%d"),
        bid_deadline=deadline,
        metadata_found=True,
    )


def _identity(*records: IdentityRecord) -> IdentityIndex:
    index = IdentityIndex(path=Path("unused"))
    for rec in records:
        index.records[rec.document_id] = rec
    return index


def test_urgent_days_none_keeps_all_non_expired_docs():
    identity = _identity(
        _record("RFP-000901", datetime(2024, 6, 2)),
        _record("RFP-000902", datetime(2024, 6, 30)),
    )
    kept, _, _ = apply_deadline_filter(
        ["RFP-000901", "RFP-000902"], identity, cfg(deadline_urgent_days=None))
    assert kept == ["RFP-000901", "RFP-000902"]


def test_urgent_days_drops_docs_outside_window():
    identity = _identity(
        _record("RFP-000901", datetime(2024, 6, 2)),   # 창 안(1일 후)
        _record("RFP-000902", datetime(2024, 6, 30)),  # 창 밖(29일 후)
    )
    kept, _, _ = apply_deadline_filter(
        ["RFP-000901", "RFP-000902"], identity, cfg(deadline_urgent_days=7))
    assert kept == ["RFP-000901"]


def test_urgent_days_boundary_is_inclusive():
    boundary = REF + (datetime(2024, 6, 8) - datetime(2024, 6, 1))
    identity = _identity(_record("RFP-000901", boundary))
    kept, _, _ = apply_deadline_filter(
        ["RFP-000901"], identity, cfg(deadline_urgent_days=7))
    assert kept == ["RFP-000901"]


def test_urgent_days_does_not_exclude_unknown_deadline():
    identity = _identity(_record("RFP-000901", None))
    kept, notes, _ = apply_deadline_filter(
        ["RFP-000901"], identity, cfg(deadline_urgent_days=7))
    assert kept == ["RFP-000901"]
    assert notes


def test_urgent_days_still_drops_already_expired_docs():
    identity = _identity(_record("RFP-000901", datetime(2024, 5, 1)))
    kept, _, _ = apply_deadline_filter(
        ["RFP-000901"], identity, cfg(deadline_urgent_days=7))
    assert kept == []


# ---------------------------------------------------------------------------
# 대화 시작 배너 (urgent_deadline_banner / SessionState.urgent_banner_shown)
# ---------------------------------------------------------------------------

def test_banner_none_when_urgent_days_unset():
    identity = _identity(_record("RFP-000901", datetime(2024, 6, 2)))
    assert urgent_deadline_banner(identity, cfg(deadline_urgent_days=None)) is None


def test_banner_none_when_nothing_urgent():
    identity = _identity(_record("RFP-000901", datetime(2024, 6, 30)))  # 창 밖
    assert urgent_deadline_banner(identity, cfg(deadline_urgent_days=7)) is None


def test_banner_lists_urgent_docs_sorted_by_deadline():
    identity = _identity(
        _record("RFP-000902", datetime(2024, 6, 6)),
        _record("RFP-000901", datetime(2024, 6, 2)),
    )
    text = urgent_deadline_banner(identity, cfg(deadline_urgent_days=7))
    assert text is not None
    assert text.index("RFP-000901") < text.index("RFP-000902")


def test_banner_shows_once_per_session(tmp_path):
    from test_selection_fix import (
        write_table, write_registry, write_identity, run_answer, present,
        cfg as select_cfg,
    )

    table = write_table(tmp_path, {"RFP-000901": {"예산": present("3억원")}})
    scope = write_registry(tmp_path, [
        {"document_id": "RFP-000901", "active": True, "retrieval_eligible": True}])
    identity = write_identity(tmp_path, [
        ("RFP-000901", "테스트기관", "테스트사업", "2024-06-03 00:00:00"),
    ])
    session = SessionState(active_document_id="RFP-000901")
    c = select_cfg(deadline_urgent_days=7)

    first = run_answer("그 사업 예산이 얼마야?", table, identity=identity,
                       registry_scope=scope, session=session, config=c)
    assert first.session_banner is not None
    assert "RFP-000901" in first.session_banner

    second = run_answer("그 사업 예산이 얼마야?", table, identity=identity,
                        registry_scope=scope, session=session, config=c)
    assert second.session_banner is None


def test_banner_absent_without_session(tmp_path):
    from test_selection_fix import (
        write_table, write_registry, write_identity, run_answer, present,
        cfg as select_cfg,
    )

    table = write_table(tmp_path, {"RFP-000901": {"예산": present("3억원")}})
    scope = write_registry(tmp_path, [
        {"document_id": "RFP-000901", "active": True, "retrieval_eligible": True}])
    identity = write_identity(tmp_path, [
        ("RFP-000901", "테스트기관", "테스트사업", "2024-06-03 00:00:00"),
    ])
    c = select_cfg(deadline_urgent_days=7)

    r = run_answer("RFP-000901의 예산이 얼마야?", table, identity=identity,
                   registry_scope=scope, config=c)
    assert r.session_banner is None
