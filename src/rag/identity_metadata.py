"""
identity_v2 로더 — 마감일·발주기관·사업명의 **유일한 공식 출처**.

공식 파일: shared_data/processed/document_registry_v2/document_identity_v2.csv
(UTF-8 BOM, 100행, 12컬럼)

  document_id, source_filename_nfc, collection_system, source_record_id,
  source_url, notice_number, notice_round, buyer_org, project_name,
  notice_date, bid_deadline, metadata_found

⚠️ 2026-09-02 변경 (4-4 확정): 마감일을 `shared_data/raw/` 아래 원본 수집 CSV에서
   직접 읽던 경로를 폐기하고 identity_v2로 통일했다. 발주기관·사업명도 같은
   파일에서 읽는다 — 문서 특정과 마감일이 서로 다른 원본을 보면 값이 어긋난다.
   identity_v2에 값이 없으면 다른 자료로 조용히 fallback하지 않고 '없음'을 알린다.

⚠️ 청킹·임베딩 대상이 아니다 — document_id로 바로 조회하는 구조화 자료다.
"""
from __future__ import annotations

import csv
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from text_normalize import (
    normalize_display,
    normalize_org_key,
    normalize_project_key,
    strip_leading_org_from_project,
)

# 공식 identity_v2가 반드시 가져야 하는 컬럼(없으면 즉시 중단 — 조용한 fallback 금지)
IDENTITY_REQUIRED_COLUMNS = (
    "document_id",
    "source_filename_nfc",
    "notice_number",
    "buyer_org",
    "project_name",
    "notice_date",
    "bid_deadline",
    "metadata_found",
)

# 마감일 값이 들어 있는 실제 컬럼명(지어내지 않고 파일의 이름 그대로)
DEADLINE_COLUMN = "bid_deadline"
# 근거 위치 표기 — 임현진 평가셋 location 형식과 동일하게 맞춘다
DEADLINE_SECTION_LABEL = "CSV"
DEADLINE_REF_NO = f"CSV: {DEADLINE_COLUMN}"

_DEADLINE_FORMATS = ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d")


class IdentityFormatError(ValueError):
    pass


@dataclass
class IdentityRecord:
    document_id: str
    source_filename_nfc: str
    notice_number: str
    buyer_org: str                 # 원본 표기 그대로(표시용)
    project_name: str              # 원본 표기 그대로(표시용)
    notice_date: str
    bid_deadline_raw: str
    bid_deadline: datetime | None  # 파싱 실패·빈 값이면 None(=미상)
    metadata_found: bool

    @property
    def org_key(self) -> str:
        return normalize_org_key(self.buyer_org)

    @property
    def project_key(self) -> str:
        return normalize_project_key(self.project_name)

    @property
    def project_core_key(self) -> str:
        """사업명에서 발주기관 접두부만 뗀 비교 보조 키(없으면 빈 문자열)."""
        return strip_leading_org_from_project(self.project_name, self.buyer_org)


@dataclass
class IdentityIndex:
    """identity_v2 전체를 담은 조회 인덱스. 원본 표시는 record에 그대로 남는다."""

    path: Path
    records: dict[str, IdentityRecord] = field(default_factory=dict)
    by_org_key: dict[str, list[str]] = field(default_factory=dict)
    by_project_key: dict[str, list[str]] = field(default_factory=dict)
    by_project_core_key: dict[str, list[str]] = field(default_factory=dict)
    org_display_by_key: dict[str, str] = field(default_factory=dict)

    # ---- 조회 ----
    def get(self, document_id: str) -> IdentityRecord | None:
        return self.records.get(document_id)

    def document_ids(self) -> list[str]:
        return list(self.records)

    def deadline(self, document_id: str) -> datetime | None:
        rec = self.records.get(document_id)
        return rec.bid_deadline if rec else None

    def has_document(self, document_id: str) -> bool:
        return document_id in self.records

    def deadline_map(self) -> dict[str, datetime | None]:
        """document_id -> 마감일. 값이 비어 있으면 None(미상)."""
        return {doc_id: rec.bid_deadline for doc_id, rec in self.records.items()}

    def known_org_keys(self) -> set[str]:
        return set(self.by_org_key)


def _parse_deadline(raw: str) -> datetime | None:
    raw = (raw or "").strip()
    if not raw:
        return None
    for fmt in _DEADLINE_FORMATS:
        try:
            return datetime.strptime(raw, fmt)
        except ValueError:
            continue
    return None


def load_identity(path: Path | str) -> IdentityIndex:
    """공식 identity_v2 CSV를 읽어 조회 인덱스를 만든다.

    필수 컬럼이 하나라도 없으면 IdentityFormatError로 즉시 중단한다 —
    다른 원본 수집 파일로 조용히 갈아타지 않는다(4-4 확정)."""
    path = Path(path)
    if not path.exists():
        raise IdentityFormatError(
            f"identity_v2 파일을 찾지 못했습니다: {path}. 마감일·발주기관·사업명은 "
            f"이 파일이 유일한 공식 출처입니다(다른 자료로 대체하지 않습니다)."
        )
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = [(n or "").strip().lstrip("﻿") for n in (reader.fieldnames or [])]
        missing = [c for c in IDENTITY_REQUIRED_COLUMNS if c not in fieldnames]
        if missing:
            raise IdentityFormatError(
                f"{path}: identity_v2에 필수 컬럼이 없습니다: {missing}. "
                f"실제 컬럼: {fieldnames}"
            )
        rows = list(reader)

    index = IdentityIndex(path=path)
    for row in rows:
        doc_id = (row.get("document_id") or "").strip()
        if not doc_id:
            raise IdentityFormatError(f"{path}: document_id가 빈 행이 있습니다.")
        if doc_id in index.records:
            raise IdentityFormatError(f"{path}: document_id 중복: {doc_id}")
        raw_deadline = (row.get(DEADLINE_COLUMN) or "").strip()
        rec = IdentityRecord(
            document_id=doc_id,
            source_filename_nfc=unicodedata.normalize(
                "NFC", (row.get("source_filename_nfc") or "").strip()
            ),
            notice_number=(row.get("notice_number") or "").strip(),
            buyer_org=normalize_display(row.get("buyer_org")),
            project_name=normalize_display(row.get("project_name")),
            notice_date=(row.get("notice_date") or "").strip(),
            bid_deadline_raw=raw_deadline,
            bid_deadline=_parse_deadline(raw_deadline),
            metadata_found=str(row.get("metadata_found", "")).strip().lower() == "true",
        )
        index.records[doc_id] = rec

        if rec.org_key:
            index.by_org_key.setdefault(rec.org_key, []).append(doc_id)
            index.org_display_by_key.setdefault(rec.org_key, rec.buyer_org)
        if rec.project_key:
            index.by_project_key.setdefault(rec.project_key, []).append(doc_id)
        core = rec.project_core_key
        if core:
            index.by_project_core_key.setdefault(core, []).append(doc_id)

    return index


# ---------------------------------------------------------------------------
# 마감일 — 값·근거 위치·기준 시각
# ---------------------------------------------------------------------------

def deadline_citation(document_id: str) -> dict:
    """마감일 근거 위치. 평가셋 location 형식과 동일한 구조로 만든다."""
    return {
        "document": document_id,
        "section": DEADLINE_SECTION_LABEL,
        "ref_no": DEADLINE_REF_NO,
    }


def deadline_evidence(index: IdentityIndex, document_id: str) -> dict:
    """마감일 답변에 함께 넣을 구조화 근거 — 값·출처파일·컬럼·버전."""
    rec = index.get(document_id)
    return {
        "document_id": document_id,
        "source": "identity_v2",
        "source_file": index.path.name,
        "column": DEADLINE_COLUMN,
        "value_raw": rec.bid_deadline_raw if rec else None,
        "value_normalized": (
            rec.bid_deadline.strftime("%Y-%m-%d") if rec and rec.bid_deadline else None
        ),
        "value_datetime": (
            rec.bid_deadline.strftime("%Y-%m-%d %H:%M") if rec and rec.bid_deadline else None
        ),
    }


def format_deadline_answer(
    document_id: str, index: IdentityIndex,
) -> tuple[str, bool, dict]:
    """(답변 문구, 기권 여부, 구조화 값) — 마감일 단건 조회용."""
    rec = index.get(document_id)
    if rec is None:
        return (
            f"{document_id}는 identity_v2(document_identity_v2.csv)에 없습니다 — "
            f"마감일을 확인할 수 없습니다.",
            True,
            deadline_evidence(index, document_id),
        )
    if rec.bid_deadline is None:
        return (
            f"{document_id}의 입찰 참여 마감일은 identity_v2에 값이 없습니다(미상). "
            f"다른 자료로 대체하지 않습니다.",
            True,
            deadline_evidence(index, document_id),
        )
    return (
        f"{document_id}의 입찰 참여 마감일: "
        f"{rec.bid_deadline.strftime('%Y-%m-%d %H:%M')}",
        False,
        deadline_evidence(index, document_id),
    )


def is_before_deadline(
    document_id: str,
    index: IdentityIndex,
    reference_datetime: datetime,
) -> tuple[bool | None, str]:
    """(마감 전인지, 상태 문구). 미상·미등록이면 None + 안내(제외하지 않음)."""
    rec = index.get(document_id)
    if rec is None:
        return None, "마감일 정보 없음(identity_v2에 문서가 없음)"
    if rec.bid_deadline is None:
        return None, "마감일 미상(identity_v2 bid_deadline 값 없음)"
    return (rec.bid_deadline >= reference_datetime), ""


def reference_datetime_from_config(cfg: dict[str, Any]) -> datetime:
    """평가 기준 시각. 코드가 now()를 읽지 않는다(4-10-2 확정)."""
    if cfg.get("reference_datetime_source") not in (None, "external"):
        raise RuntimeError(
            "reference_datetime_source가 'external'이 아닙니다 — 현재 시각(now)을 "
            "쓰면 평가 결과가 날짜에 따라 달라집니다."
        )
    raw = cfg.get("reference_datetime")
    if not raw:
        raise RuntimeError(
            "base.yaml에 reference_datetime이 없습니다. now()로 대체하지 않습니다."
        )
    for fmt in _DEADLINE_FORMATS:
        try:
            return datetime.strptime(str(raw), fmt)
        except ValueError:
            continue
    raise RuntimeError(f"reference_datetime 형식을 읽지 못했습니다: {raw!r}")
