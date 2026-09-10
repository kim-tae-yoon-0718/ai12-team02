"""
청크 좌표 조회 — `document_id` + 원문 줄 번호 → 청크 좌표.

팀 확정 좌표 규약 (박예진·임현진 2026-09-02, 하루님 채점기 `Location.from_chunk`):

    document = document_id
    section  = section_path 의 **리프**(마지막) 요소
    ref_no   = location_label 을 **그대로** (예: "4. 제안 요청내용 · 문단 1-57")
    line     = md_line_start,  line_end = md_line_end

평가셋 정답 좌표도 같은 규약이라 문자열/line 범위로 그대로 대조된다.
추출표의 `representative_location`·`additional_locations`는
`heading`/`block_index`/`line`만 갖고 있어 이 규약과 형식이 다르다 — 그래서
줄 번호로 **그 줄을 포함하는 청크**를 찾아 청크 좌표로 변환한다.

⚠️ 검색 대상(retrieval_eligible) 청크만으로는 커버리지가 57%뿐이다(별첨·서식 구간이
   검색에서 빠지기 때문). 좌표 변환은 **전체 청크**(chunks.jsonl 19,381개)를 쓴다.
   검색 여부와 좌표 존재 여부는 다른 문제다.
"""
from __future__ import annotations

import json
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class ChunkLocation:
    document_id: str
    section: str
    ref_no: str          # = location_label
    line: int | None     # = md_line_start
    line_end: int | None  # = md_line_end
    chunk_id: str = ""

    def as_citation(self, source: str = "chunks_v3", field: str | None = None) -> dict:
        out = {
            "document": self.document_id,
            "section": self.section,
            "ref_no": self.ref_no,
            "line": self.line,
            "line_end": self.line_end,
            "source": source,
        }
        if field:
            out["field"] = field
        return out


def _section_leaf(section_path: Any, location_label: str = "") -> str:
    """section_path 의 리프. dict 원소(추출표 형식)와 문자열 원소(청크 형식) 둘 다 받는다."""
    items: list[str] = []
    for s in (section_path or []):
        if isinstance(s, dict):
            s = s.get("title") or s.get("heading") or ""
        if s:
            items.append(str(s))
    if items:
        return items[-1]
    # location_label 이 "<절> · 문단 N-M" 형태면 앞부분이 절 이름이다
    if " · " in location_label:
        return location_label.rsplit(" · ", 1)[0].strip()
    return ""


class ChunkLocator:
    """문서별 청크 줄 범위 색인. `locate(document_id, line)` 로 좌표를 찾는다."""

    def __init__(self) -> None:
        # document_id -> (starts[], entries[])  — starts 는 정렬된 md_line_start
        self._by_doc: dict[str, tuple[list[int], list[ChunkLocation]]] = {}
        self._by_chunk_id: dict[str, ChunkLocation] = {}

    # ---- 적재 ----
    @classmethod
    def from_records(cls, records: Iterable[dict]) -> "ChunkLocator":
        """records: chunks.jsonl 한 줄과 같은 dict들."""
        loc = cls()
        raw: dict[str, list[ChunkLocation]] = {}
        for c in records:
            start, end = c.get("md_line_start"), c.get("md_line_end")
            if start is None:
                continue
            entry = ChunkLocation(
                document_id=c["document_id"],
                section=_section_leaf(c.get("section_path"), c.get("location_label", "") or ""),
                ref_no=str(c.get("location_label") or ""),
                line=int(start),
                line_end=int(end) if end is not None else int(start),
                chunk_id=str(c.get("chunk_id") or ""),
            )
            raw.setdefault(entry.document_id, []).append(entry)
            if entry.chunk_id:
                loc._by_chunk_id[entry.chunk_id] = entry
        for doc_id, entries in raw.items():
            entries.sort(key=lambda e: (e.line, e.line_end))
            loc._by_doc[doc_id] = ([e.line for e in entries], entries)
        return loc

    @classmethod
    def from_chunks_jsonl(cls, path: Path | str) -> "ChunkLocator":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"청크 파일이 없습니다: {path}")

        def _iter():
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        yield json.loads(line)

        return cls.from_records(_iter())

    @classmethod
    def from_vector_store(cls, store) -> "ChunkLocator":
        """인덱스 메타데이터로 만든다(검색 제외 청크는 빠져 커버리지가 낮다 — 보조용)."""
        return cls.from_records({
            "document_id": m.document_id, "section_path": m.section_path,
            "location_label": m.location_label, "md_line_start": m.md_line_start,
            "md_line_end": m.md_line_end, "chunk_id": m.chunk_id,
        } for m in store.metadata)

    # ---- 조회 ----
    def __len__(self) -> int:
        return sum(len(v[1]) for v in self._by_doc.values())

    @property
    def document_count(self) -> int:
        return len(self._by_doc)

    def by_chunk_id(self, chunk_id: str) -> ChunkLocation | None:
        return self._by_chunk_id.get(chunk_id)

    def locate(self, document_id: str, line: int | None) -> ChunkLocation | None:
        """그 줄을 포함하는 청크 중 **가장 좁은** 것. 없으면 None(추측하지 않는다)."""
        if line is None:
            return None
        found = self._by_doc.get(document_id)
        if not found:
            return None
        starts, entries = found
        # md_line_start <= line 인 마지막 위치까지 훑으며 포함하는 것 중 최소 범위 선택
        idx = bisect_right(starts, line)
        best: ChunkLocation | None = None
        for e in entries[:idx][::-1]:
            if e.line_end is not None and e.line_end >= line:
                if best is None or (e.line_end - e.line) < (best.line_end - best.line):
                    best = e
        return best

    def citation_for_location(self, document_id: str, loc: dict,
                              field: str | None = None,
                              source: str = "extraction_table") -> dict | None:
        """추출표 위치 객체 → 청크 규약 좌표. 줄 번호가 없거나 매핑 실패면 None.

        ⚠️ 매핑에 실패하면 근거를 **지어내지 않는다** — 호출측이 "근거 위치 없음"으로
        처리해야 한다(무관한 청크를 억지로 붙이지 않는다)."""
        if not isinstance(loc, dict):
            return None
        line = loc.get("line", loc.get("line_start"))
        hit = self.locate(document_id, line)
        if hit is None:
            return None
        out = hit.as_citation(source=source, field=field)
        out["matched_from_line"] = line
        return out
