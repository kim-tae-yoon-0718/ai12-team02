"""공식 문서 등록부(document_registry_v2.json) 조회 — 검색·선별 대상 범위의 단일 출처.

공식 파일: shared_data/processed/document_registry_v2/document_registry_v2.json
최상위 'documents' 키 아래 100건. 각 행에서 이 모듈이 보는 값은 두 개다.

  active               — soft delete 플래그(false면 아예 대상 아님)
  retrieval_eligible   — 콘텐츠 중복 대표 선정 결과. false면 검색·선별 대상에서 제외하고
                         duplicate_of_document_id 가 대표 문서를 가리킨다.

⚠️ 문서 ID를 코드에 하드코딩해 제외하지 않는다. 어떤 문서가 몇 건 빠지는지는
   전적으로 이 파일의 값으로 결정된다(현재 공식본 기준 100건 중 98건이 대상).

⚠️ 추출표의 행 단위 `active` 와는 다른 층이다. 추출표 쪽은
   "그 행이 살아 있는가", 등록부 쪽은 "그 문서가 검색·선별 대상인가"를 뜻한다.
   선별 경로는 **둘 다** 봐야 한다(예전엔 추출표 active 만 봐서 중복 문서가 결과에 섞였다).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class DocumentRegistryError(ValueError):
    pass


@dataclass
class RegistryScope:
    """등록부가 정한 검색·선별 대상 범위."""

    path: Path | None = None
    document_count: int = 0
    eligible_ids: set[str] = field(default_factory=set)
    excluded: list[dict] = field(default_factory=list)   # 제외 사유·대표 문서

    def is_eligible(self, document_id: str) -> bool:
        return document_id in self.eligible_ids

    def representative_of(self, document_id: str) -> str | None:
        for row in self.excluded:
            if row["document_id"] == document_id:
                return row.get("duplicate_of_document_id")
        return None

    def summary(self) -> dict[str, Any]:
        return {
            "source": "document_registry",
            "path": str(self.path) if self.path else None,
            "document_count": self.document_count,
            "eligible_count": len(self.eligible_ids),
            "excluded_count": len(self.excluded),
            "excluded": self.excluded,
        }


def _row_flag(row: dict, key: str, default: bool = True) -> bool:
    """등록부 값은 JSON 불리언이지만 CSV 유래 문자열("true"/"false")도 들어올 수 있다."""
    v = row.get(key, default)
    if isinstance(v, bool):
        return v
    if v is None:
        return default
    return str(v).strip().lower() not in ("false", "0", "no", "n")


def load_registry_scope(path: Path | str | None) -> RegistryScope | None:
    """등록부를 읽어 검색·선별 대상 범위를 만든다. path가 없으면 None(범위 제한 없음).

    호출측은 None을 "등록부를 못 읽었다"로 취급하고, 그 사실을 사용자에게 알려야 한다 —
    조용히 전체 문서를 대상으로 삼아 중복 문서를 섞지 않는다."""
    if path is None:
        return None
    path = Path(path)
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        doc = json.load(f)
    if "documents" not in doc:
        raise DocumentRegistryError(
            f"{path}: 최상위 객체에 'documents' 키가 없습니다. "
            f"존재하지 않는 registry.jsonl 형식을 가정하지 마세요."
        )
    rows = doc["documents"]
    eligible: set[str] = set()
    excluded: list[dict] = []
    for row in rows:
        doc_id = row["document_id"]
        active = _row_flag(row, "active", True)
        retrievable = _row_flag(row, "retrieval_eligible", True)
        if active and retrievable:
            eligible.add(doc_id)
            continue
        excluded.append({
            "document_id": doc_id,
            "active": active,
            "retrieval_eligible": retrievable,
            "duplicate_of_document_id": row.get("duplicate_of_document_id") or None,
            "relation_status": row.get("relation_status"),
        })
    return RegistryScope(path=path, document_count=len(rows),
                         eligible_ids=eligible, excluded=excluded)


def load_retrieval_eligible_ids(path: Path | str | None) -> set[str] | None:
    """검색 대상 document_id 집합만 필요할 때 쓰는 얇은 래퍼."""
    scope = load_registry_scope(path)
    return scope.eligible_ids if scope is not None else None
