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
    filename_project_alias,
    fold_possessive_particles,
    is_generic_project_key,
    project_rounds,
    strip_bracketed,
    normalize_display,
    normalize_org_key,
    normalize_org_key_unbracketed,
    normalize_project_key,
    normalize_project_key_particle_folded,
    normalize_project_key_unbracketed,
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
    def org_key_unbracketed(self) -> str:
        """괄호 주석을 뗀 기관명 키. "한국철도공사 (용역)" → "한국철도공사"."""
        return normalize_org_key_unbracketed(self.buyer_org)

    @property
    def project_key_unbracketed(self) -> str:
        """괄호 주석을 뗀 사업명 키.
        "봉화군 재난통합관리시스템 고도화 사업(협상)(긴급)" → "봉화군재난통합관리시스템고도화사업"."""
        return normalize_project_key_unbracketed(self.project_name)

    @property
    def project_core_key(self) -> str:
        """사업명에서 발주기관 접두부만 뗀 비교 보조 키(없으면 빈 문자열)."""
        return strip_leading_org_from_project(self.project_name, self.buyer_org)

    @property
    def filename_project_alias(self) -> str:
        """공식 파일명 `발주기관명_짧은 사업명.md`의 짧은 사업명(표시 표기 그대로).

        파일명 앞 기관명이 같은 행 `buyer_org`와 정확히 일치할 때만 값이 있다.
        일치하지 않으면 빈 문자열 — 추측으로 별칭을 만들지 않는다.
        """
        return filename_project_alias(self.source_filename_nfc, self.buyer_org,
                                     self.project_name)

    @property
    def project_alias_keys(self) -> list[str]:
        """공식 파일명에서 얻은 사업명 별칭의 비교 키(원래 키와 다를 때만)."""
        alias = self.filename_project_alias
        if not alias:
            return []
        keys = [normalize_project_key(alias),
                normalize_project_key_unbracketed(alias)]
        known = {self.project_key, self.project_key_unbracketed, self.project_core_key}
        return [k for k in dict.fromkeys(keys) if k and k not in known]

    def project_names_for_keys(self) -> list[str]:
        """이 행이 가진 **공식** 사업명 표기 전부(정식 사업명 + 파일명 별칭)."""
        return [n for n in (self.project_name, self.filename_project_alias) if n]


@dataclass
class IdentityIndex:
    """identity_v2 전체를 담은 조회 인덱스. 원본 표시는 record에 그대로 남는다."""

    path: Path
    records: dict[str, IdentityRecord] = field(default_factory=dict)
    by_org_key: dict[str, list[str]] = field(default_factory=dict)
    by_project_key: dict[str, list[str]] = field(default_factory=dict)
    by_project_core_key: dict[str, list[str]] = field(default_factory=dict)
    # 공식 파일명 `발주기관명_짧은 사업명.md`에서 얻은 짧은 사업명 별칭
    by_project_alias_key: dict[str, list[str]] = field(default_factory=dict)
    # 낱말 사이의 조사 "의" 하나만 접은 보조 키(원래 키는 그대로 남는다)
    by_project_folded_key: dict[str, list[str]] = field(default_factory=dict)
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
    # 회차를 잃은 보조 키가 **다른 문서의 정식 사업명**과 같아지는지 보려면 전체
    # 정식 키를 먼저 알아야 한다(한 번 훑고 색인은 그다음에 만든다).
    primary_keys_by_doc = {
        (r.get("document_id") or "").strip():
            normalize_project_key(normalize_display(r.get("project_name")))
        for r in rows
    }
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
        # 괄호 주석을 뗀 보조 키도 같은 색인에 넣는다. 사람은 "한국철도공사 (용역)"을
        # "한국철도공사"라고 부르고, "…고도화 사업(협상)(긴급)"을 "…고도화 사업"이라고
        # 부른다. 임의 유사도가 아니라 **괄호 제거**라는 결정적 변환만 쓴다.
        org_unbr = rec.org_key_unbracketed
        if org_unbr and org_unbr != rec.org_key:
            index.by_org_key.setdefault(org_unbr, []).append(doc_id)
            index.org_display_by_key.setdefault(org_unbr, rec.buyer_org)
        # ⚠️ 같은 문서의 여러 공식 별칭이 한 키로 합쳐질 수 있다(정식 사업명과
        #    파일명 별칭이 같은 글자로 접히는 경우). 문서 ID를 두 번 넣으면
        #    "후보가 2건"으로 잘못 세어 되묻기가 발생하므로 항상 중복을 막는다.
        def _add(table: dict[str, list[str]], key: str) -> None:
            if not key:
                return
            docs = table.setdefault(key, [])
            if doc_id not in docs:
                docs.append(doc_id)

        rec_rounds = project_rounds(rec.project_name)

        def _add_auxiliary(table: dict[str, list[str]], key: str,
                           source: str | None = None) -> None:
            """보조 키 전용 — 문서를 특정하지 못하게 된 키는 색인하지 않는다.

            ⚠️ ① 주석을 떼다 보면 "위탁용역"처럼 여러 사업에 나올 수 있는 표현만
               남을 수 있다. 그런 키는 부분문자열 비교에서 아무 질문에나 걸린다.
            ⚠️ ② 사업 회차는 사업을 **구분하는** 정보다. 괄호 주석을 떼면서 회차까지
               사라진 키("…기능개선 사업(1차)" → "…기능개선사업")를 색인하면, 회차가
               없는 다른 사업과 키가 같아져 후보가 번지고 되묻기가 늘어난다.
               회차를 잃은 보조 키는 넣지 않는다(원래 키는 그대로 남는다).
            """
            if is_generic_project_key(key):
                return
            if source is not None and project_rounds(source) != rec_rounds:
                # 회차를 잃은 키는, 그 키가 **다른 문서의 정식 사업명**과 똑같아질
                # 때만 뺀다. 그때만 회차가 다른 두 사업이 한 이름으로 뭉개진다.
                # 충돌이 없으면 넣어 둔다 — 괄호 주석을 뗀 이름으로 부르는 질문
                # ("모바일오피스 시스템 고도화 용역")을 계속 찾을 수 있어야 한다.
                if any(other != doc_id and pk == key
                       for other, pk in primary_keys_by_doc.items()):
                    return
            _add(table, key)

        _add(index.by_project_key, rec.project_key)          # 공식 사업명 전체 — 그대로
        proj_unbr = rec.project_key_unbracketed
        if proj_unbr and proj_unbr != rec.project_key:
            _add_auxiliary(index.by_project_key, proj_unbr,
                           source=strip_bracketed(rec.project_name))
        _add_auxiliary(index.by_project_core_key, rec.project_core_key,
                       source=rec.project_core_key)
        # 공식 파일명의 짧은 사업명 — 기관 접두가 정확히 일치할 때만 채워진다
        for alias_key in rec.project_alias_keys:
            _add_auxiliary(index.by_project_alias_key, alias_key,
                           source=rec.filename_project_alias)
        # 조사 "의"만 접은 보조 키 — 정식 사업명·괄호 제거형·기관접두 제거형·
        # 파일명 별칭 전부에 같은 규칙을 적용한다(원래 키는 위에 그대로 남는다).
        for name in rec.project_names_for_keys():
            for variant in (name, strip_bracketed(name)):
                _add_auxiliary(index.by_project_folded_key,
                               normalize_project_key_particle_folded(variant),
                               source=variant)
            # 기관 접두를 뗀 형태도 같은 규칙으로 한 번 더 본다. 접두 제거는
            # 공백을 없앤 뒤에 일어나므로 **접기를 먼저** 적용해야 한다.
            folded_core = strip_leading_org_from_project(
                fold_possessive_particles(name), rec.buyer_org)
            if folded_core and folded_core != strip_leading_org_from_project(
                    name, rec.buyer_org):
                _add_auxiliary(index.by_project_folded_key, folded_core)

    _link_aliases_to_overlapping_official_names(index)
    return index


def _link_aliases_to_overlapping_official_names(index: "IdentityIndex") -> None:
    """보조 키가 **다른 문서의 공식 사업명 안에도 글자 그대로 있으면** 그 문서도 후보다.

    ⚠️ 2026-09-03 적대적 점검에서 실제 퇴행이 나왔다. 파일명 별칭은 자기 행만
    보고 만들어지므로, 그 이름이 다른 문서의 정식 사업명 안에도 그대로 들어 있는
    경우를 놓쳤다. 그러면 두 문서에 다 해당하는 이름인데도 별칭 소유 문서 하나로
    조용히 확정되고(임의 선택), 반대로 기관명까지 붙여 부른 질문은 기관과 사업명이
    서로 다른 문서를 가리킨다며 되묻기로 퇴행했다.

    여기서 쓰는 건 결정적 부분문자열 비교뿐이다 — 유사도가 아니다. 겹치는 문서를
    후보로 **남기기만** 하므로, 실행부의 기존 규칙(기관+사업명 교집합 우선, 후보가
    여럿이면 되묻기)이 그대로 판단한다.
    """
    for table in (index.by_project_alias_key, index.by_project_folded_key):
        for key, docs in table.items():
            for doc_id, rec in index.records.items():
                if doc_id in docs:
                    continue
                official = (rec.project_key, rec.project_key_unbracketed,
                            rec.project_core_key)
                if any(o and key in o for o in official):
                    docs.append(doc_id)


# ---------------------------------------------------------------------------
# 마감일 — 값·근거 위치·기준 시각
# ---------------------------------------------------------------------------

def deadline_citation(document_id: str, value: str | None = None) -> dict:
    """마감일 근거 — identity_v2 의 컬럼 자체가 근거다(원문 줄이 아니다).

    ★[2026-09-04 §8] 예전에는 section="CSV" / ref_no="CSV: bid_deadline" 처럼 **좌표
      모양**으로 냈다. 그건 원문 위치가 아니라 CSV 컬럼 이름이라, 실제로 존재하지 않는
      좌표를 가리키는 인용이 된다(위장). 이제 Evidence 형으로 무엇을 봤는지 그대로
      적는다 — kind=identity / document / field / value / source. 좌표 키는 넣지 않는다.
    """
    cite = {
        "kind": "identity",
        "document": document_id,
        "field": DEADLINE_COLUMN,
        "source": "identity_v2",
    }
    if value:
        cite["value"] = value
    return cite


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
