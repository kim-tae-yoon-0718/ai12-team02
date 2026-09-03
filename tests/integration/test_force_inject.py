"""3-1 / 3-2-2 강제 주입 — 검색·문서특정을 완벽하다고 가정한 상한 측정."""

import json

from grader.force_inject import (
    apply_forced_context,
    apply_forced_doc,
    load_chunk_index,
)
from grader.models import EvaluationItem, ModelResponse


def _item(**over):
    base = dict(id="Q1", question="예산?", task_type="extraction", answer_type="value",
               answer_raw="5억", document_id="RFP-1",
               location={"document": "RFP-1", "section": "3. 사업개요", "ref_no": "paragraph 3"})
    base.update(over)
    return EvaluationItem.model_validate(base)


def _chunks_file(tmp_path):
    p = tmp_path / "chunks.jsonl"
    rows = [
        {"document_id": "RFP-1", "section_path": [{"title": "3. 사업개요"}],
         "block_type": "paragraph", "block_index": 3, "search_text": "사업 예산은 5억원"},
        {"document_id": "RFP-1", "section_path": [{"title": "9. 딴 절"}],
         "block_type": "table", "block_index": 1, "search_text": "무관한 표"},
    ]
    p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    return p


def test_forced_context_injects_gold_chunk(tmp_path):
    items = {"Q1": _item()}
    responses = {"Q1": ModelResponse(id="Q1", answer="5억원", abstained=False, contexts=[], retrieved=[])}
    by_doc = load_chunk_index(_chunks_file(tmp_path))

    changed = apply_forced_context(items, responses, by_doc, precision="section")
    assert changed == ["Q1"]
    r = responses["Q1"]
    # gold 절("3. 사업개요")의 청크만, 딴 절 표는 제외
    assert len(r.contexts) == 1
    assert "5억원" in r.contexts[0].text
    # citation 도 정답 좌표로
    assert r.citations and r.citations[0].document == "RFP-1"


def test_forced_doc_sets_selected_ids():
    items = {"Q1": _item()}
    responses = {"Q1": ModelResponse(id="Q1", answer="x", abstained=False, selected_document_ids=["WRONG"])}
    apply_forced_doc(items, responses)
    assert responses["Q1"].selected_document_ids == ["RFP-1"]
    assert responses["Q1"].active_document_id == "RFP-1"


def test_forced_context_comparison_multi_doc(tmp_path):
    p = tmp_path / "c.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in [
        {"document_id": "RFP-A", "section_path": [{"title": "예산절"}],
         "block_type": "paragraph", "block_index": 1, "search_text": "A 예산"},
        {"document_id": "RFP-B", "section_path": [{"title": "예산절"}],
         "block_type": "paragraph", "block_index": 1, "search_text": "B 예산"},
    ]), encoding="utf-8")
    it = EvaluationItem.model_validate({
        "id": "C1", "question": "비교", "task_type": "qa", "answer_type": "comparison",
        "answer_raw": [{"항목": "예산", "RFP-A": "1억", "RFP-B": "2억"}],
        "location": [
            {"document": "RFP-A", "section": "예산절", "ref_no": "paragraph 1"},
            {"document": "RFP-B", "section": "예산절", "ref_no": "paragraph 1"},
        ],
    })
    responses = {"C1": ModelResponse(id="C1", answer="표", abstained=False)}
    apply_forced_context({"C1": it}, responses, load_chunk_index(p), precision="section")
    docs = {c.document_id for c in responses["C1"].contexts}
    assert docs == {"RFP-A", "RFP-B"}
    assert len(responses["C1"].citations) == 2
