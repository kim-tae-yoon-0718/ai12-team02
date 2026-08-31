"""
4-10-2 — 선별형 마감 필터용 메타데이터 로더.
공식 data_list.csv(UTF-8 BOM, 12컬럼, 100행)를 document_registry_v2.json과
연결해 document_id -> 마감일 매핑을 만든다.

⚠️ 이 데이터는 청킹·임베딩 대상이 아니다(4-9-8 확정: "숫자·날짜·부정 조건은
벡터 검색이 아니라 검증된 구조화 질의"). document_id로 바로 조회하는
구조화 조회 대상이라 G-2와 같은 성격 — 그냥 로더 하나면 된다.

연결 키: CSV의 '파일명'(확장자 .hwp 등)과 registry의 'output_filename'
(확장자 .md)이 확장자만 빼면 동일하다(100/100 매핑 확인 완료, NFC 정규화 필요
— 체크리스트의 "파일명은 NFD 저장" 경고와 일치).

deadline_filter_field(base.yaml, "입찰 참여 마감일") 컬럼명은 코드에 다시
하드코딩하지 않고 cfg에서 그대로 가져온다 — 컬럼명이 바뀌면 base.yaml만
고치면 된다.
"""
from __future__ import annotations
import csv
import os
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any


def _stem_nfc(filename: str) -> str:
    """확장자를 떼고 NFC로 정규화 — CSV·registry 파일명을 같은 기준으로 비교."""
    return unicodedata.normalize("NFC", os.path.splitext(filename)[0])


def load_deadline_by_document_id(
    csv_path: Path, registry_path: Path, cfg: dict[str, Any],
) -> dict[str, datetime | None]:
    """document_id -> 마감일(datetime) 매핑. 마감일이 빈 값(미상)이면 None.
    CSV·registry 어느 한쪽에도 없는 문서는 매핑에서 아예 빠진다(호출측이
    '미상'과 '매핑 자체가 없음'을 구분할 수 있게)."""
    deadline_field = cfg.get("deadline_filter_field")
    if not deadline_field:
        raise RuntimeError(
            "base.yaml에 deadline_filter_field가 비어 있습니다. "
            "CSV의 실제 컬럼명을 채워야 합니다."
        )

    import json
    with open(registry_path, "r", encoding="utf-8") as f:
        registry_doc = json.load(f)
    if "documents" not in registry_doc:
        raise ValueError(f"{registry_path}: 'documents' 키가 없습니다.")
    stem_to_doc_id = {
        _stem_nfc(row["output_filename"]): row["document_id"]
        for row in registry_doc["documents"]
    }

    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if deadline_field not in (reader.fieldnames or []):
            raise KeyError(
                f"{csv_path}: 컬럼 '{deadline_field}'가 없습니다. "
                f"실제 컬럼: {reader.fieldnames}"
            )
        rows = list(reader)

    mapping: dict[str, datetime | None] = {}
    unmatched_csv_rows = 0
    for row in rows:
        doc_id = stem_to_doc_id.get(_stem_nfc(row["파일명"]))
        if doc_id is None:
            unmatched_csv_rows += 1
            continue
        raw = (row.get(deadline_field) or "").strip()
        if not raw:
            mapping[doc_id] = None  # 마감일 미상 — deadline_missing_policy로 처리
            continue
        try:
            mapping[doc_id] = datetime.strptime(raw, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            # 형식이 다른 값이 섞여 있으면 조용히 무시하지 않고 미상으로 남기되 알린다
            print(f"⚠️  {doc_id}: 마감일 형식을 못 읽었습니다({raw!r}) — 미상으로 처리")
            mapping[doc_id] = None

    if unmatched_csv_rows:
        print(f"⚠️  CSV {unmatched_csv_rows}행이 registry 문서와 매핑되지 않았습니다.")

    return mapping


def is_before_deadline(
    document_id: str, deadline_map: dict[str, datetime | None],
    reference_datetime: datetime, missing_policy: str = "show_as_unknown",
) -> tuple[bool | None, str]:
    """(마감 전인지, 상태 문구) 반환. 마감 전이면 True, 지났으면 False,
    미상이거나 매핑 자체가 없으면 None + 안내 문구(4-10-2 확정: 미상은
    제외하지 않고 '미상'으로 표시하고 통과시킨다 — 호출측이 그대로 노출)."""
    if document_id not in deadline_map:
        return None, "마감일 정보 없음(문서-CSV 매핑 실패)"
    deadline = deadline_map[document_id]
    if deadline is None:
        return None, "마감일 미상"
    return (deadline >= reference_datetime), ""
