"""grader.force_inject — 3-1 / 3-2-2 강제 주입 실행.

검색·문서특정 단계를 "완벽했다고 가정"하고 채점해서, 그 단계만 고치면 성능이
얼마나 오르는지의 **상한**을 잰다.

  --force-context : 정답 근거 청크를 그대로 context 로 주입 (+ citation 도 정답 좌표로).
                    → "검색이 완벽하면 생성이 얼마나 잘하나" (3-1)
  --force-doc     : 정답 문서 ID 를 selected_document_ids 로 주입.
                    → "문서 특정이 완벽하면 값 추출이 얼마나 잘하나" (3-2-2)

★ 강제 주입 실행의 점수는 일반 실행과 **비교 대상이 아니다** — report.manifest.forced 에
  표시하고, 사람이 "상한"으로만 읽는다.

★ 좌표 매칭 정밀도: 박예진 청크가 아직 extraction_v3 block_index 를 안 실어 보내므로
  ref_no 단위 매칭은 성립하지 않는다. 여기서는 section 단위로 gold 청크를 고른다
  (박예진 확정 후 precision 인자로 올린다).
"""

from __future__ import annotations

import json
from pathlib import Path

from .models import ContextChunk, EvaluationItem, Location, ModelResponse
from .normalize import match_location


def load_chunk_index(path: str | Path) -> dict[str, list[dict]]:
    """chunks.jsonl → {document_id: [chunk dict, ...]}."""
    by_doc: dict[str, list[dict]] = {}
    p = Path(path)
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        by_doc.setdefault(rec.get("document_id"), []).append(rec)
    return by_doc


def _gold_chunks(item: EvaluationItem, by_doc: dict[str, list[dict]],
                 precision: str = "section") -> list[ContextChunk]:
    """문항의 정답 좌표에 해당하는 청크들. 좌표가 문서만 있으면 그 문서의 청크 전부."""
    out: list[ContextChunk] = []
    for gl in item.gold_locations():
        docs = by_doc.get(gl.document, [])
        for rec in docs:
            cc = ContextChunk.model_validate(rec)
            if cc.location is None:
                continue
            if gl.section and match_location(gl, cc.location, precision):
                out.append(cc)
        if not gl.section:  # 좌표에 section 이 없으면 문서 전체
            out.extend(ContextChunk.model_validate(r) for r in docs)
    return out


def apply_forced_context(items: dict[str, EvaluationItem],
                         responses: dict[str, ModelResponse],
                         by_doc: dict[str, list[dict]],
                         precision: str = "section") -> list[str]:
    """각 응답의 contexts/retrieved/reranked/citations 를 정답 근거로 교체. 바뀐 id 목록 반환."""
    changed = []
    for rid, resp in responses.items():
        item = items.get(rid)
        if item is None or not item.gold_locations():
            continue
        gold = _gold_chunks(item, by_doc, precision)
        if not gold:
            continue
        resp.contexts = gold
        resp.retrieved = []
        resp.reranked = []
        resp.citations = [Location(**gl.model_dump()) for gl in item.gold_locations()]
        changed.append(rid)
    return changed


def apply_forced_doc(items: dict[str, EvaluationItem],
                     responses: dict[str, ModelResponse]) -> list[str]:
    """각 응답의 selected_document_ids 를 정답 문서로 교체. 바뀐 id 목록 반환."""
    changed = []
    for rid, resp in responses.items():
        item = items.get(rid)
        if item is None:
            continue
        gold_docs = sorted({gl.document for gl in item.gold_locations()})
        if not gold_docs and isinstance(item.document_id, str):
            gold_docs = [item.document_id]
        elif not gold_docs and isinstance(item.document_id, list):
            gold_docs = list(item.document_id)
        if not gold_docs:
            continue
        resp.selected_document_ids = gold_docs
        if len(gold_docs) == 1:
            resp.active_document_id = gold_docs[0]
        changed.append(rid)
    return changed
