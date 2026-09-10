"""Reflection reinterpretation: approved-only memory without code-side NLP."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def _write_assets(tmp_path: Path, records: list[dict], vectors: list[list[float]]):
    records_path = tmp_path / "approved.jsonl"
    vectors_path = tmp_path / "vectors.npy"
    manifest_path = tmp_path / "manifest.json"
    records_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records),
        encoding="utf-8",
    )
    np.save(vectors_path, np.asarray(vectors, dtype=np.float32), allow_pickle=False)
    manifest = {
        "records_sha256": hashlib.sha256(records_path.read_bytes()).hexdigest(),
        "vectors_sha256": hashlib.sha256(vectors_path.read_bytes()).hexdigest(),
        "embedding_model": "text-embedding-3-small",
        "record_count": len(records),
        "vector_dimension": len(vectors[0]),
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return records_path, vectors_path, manifest_path


def _record(record_id: str, question: str, status: str = "approved") -> dict:
    return {
        "record_id": record_id,
        "question": question,
        "promotion_status": status,
        "approved_plan": {"action": "vector_search"},
        "verified_strategy": ["승인된 방법"],
        "review_reason": "개발자 확인",
        "reviewed_by": "태윤님 위임 LLM 검토",
        "reviewed_at": "2026-09-07T00:00:00+09:00",
        "removable_tag": "test-tag",
    }


def _cfg(paths, threshold=0.5, top_k=2):
    records, vectors, manifest = paths
    return {
        "reflection_memory_records": str(records),
        "reflection_memory_vectors": str(vectors),
        "reflection_memory_manifest": str(manifest),
        "reflection_memory_embedding_model": "text-embedding-3-small",
        "reflection_memory_min_similarity": threshold,
        "reflection_memory_top_k": top_k,
    }


class _Usage:
    def __init__(self):
        self.tokens = 0

    def add_embedding(self, tokens):
        self.tokens += tokens


class _Client:
    def __init__(self, vector):
        self.vector = vector
        self.embeddings = self

    def create(self, **kwargs):
        assert kwargs["model"] == "text-embedding-3-small"
        assert len(kwargs["input"]) == 1
        return SimpleNamespace(
            data=[SimpleNamespace(embedding=self.vector)],
            usage=SimpleNamespace(total_tokens=7),
        )


def test_memory_rejects_unapproved_candidate(tmp_path):
    from reflection_memory import ReflectionMemory, ReflectionMemoryError

    paths = _write_assets(tmp_path, [_record("C1", "질문", "candidate")], [[1, 0]])
    with pytest.raises(ReflectionMemoryError, match="unapproved"):
        ReflectionMemory(_cfg(paths), _Client([1, 0]))


def test_memory_rejects_manifest_mismatch(tmp_path):
    from reflection_memory import ReflectionMemory, ReflectionMemoryError

    paths = _write_assets(tmp_path, [_record("A1", "질문")], [[1, 0]])
    manifest = json.loads(paths[2].read_text(encoding="utf-8"))
    manifest["records_sha256"] = "0" * 64
    paths[2].write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ReflectionMemoryError, match="manifest mismatch"):
        ReflectionMemory(_cfg(paths), _Client([1, 0]))


def test_memory_retrieves_by_numeric_similarity_and_threshold(tmp_path):
    from reflection_memory import ReflectionMemory

    paths = _write_assets(
        tmp_path,
        [_record("A1", "첫 질문"), _record("A2", "둘째 질문")],
        [[1, 0], [0, 1]],
    )
    usage = _Usage()
    selected = ReflectionMemory(
        _cfg(paths, threshold=0.6, top_k=2), _Client([0.9, 0.1])
    ).retrieve("처음 보는 현재 질문", usage)
    assert [row["record_id"] for row in selected] == ["A1"]
    assert selected[0]["reviewed_by"] == "태윤님 위임 LLM 검토"
    assert "promotion_status" not in selected[0]
    assert usage.tokens == 7


def test_primary_and_adjudicator_see_memory_but_independent_does_not():
    from stage2_agent import Stage2Agent

    agent = Stage2Agent.__new__(Stage2Agent)
    agent.catalog_lines = ["RFP-000001 | 기관 | 사업"]
    agent.system_prompt = "primary"
    agent.independent_system_prompt = "independent"
    agent.adjudicator_system_prompt = "adjudicator"
    agent.current_memory_A = [{"record_id": "A1"}]
    candidate = SimpleNamespace(as_dict=lambda: {"action": "vector_search"})

    primary = json.loads(agent._messages("질문", [], None)[1]["content"])
    independent = json.loads(agent._independent_messages("질문", None)[1]["content"])
    adjudicator = json.loads(agent._adjudicator_messages(
        "질문", candidate, candidate, {}, {}, None)[1]["content"])

    assert primary["verified_memory_A"] == [{"record_id": "A1"}]
    assert "verified_memory_A" not in independent
    assert adjudicator["verified_memory_A"] == [{"record_id": "A1"}]


def test_memory_source_has_no_question_language_rules():
    source = (Path(__file__).parents[1] / "rag" / "reflection_memory.py").read_text(
        encoding="utf-8")
    forbidden = ["정규식", "추진배경", "공동수급", "사업명", "선별형", "추출형"]
    assert all(token not in source for token in forbidden)


def test_reflection_prompts_call_memory_precedent_not_gold_answer():
    prompts = Path(__file__).parents[1] / "prompts"
    primary = (prompts / "stage2_agent_reflection_v1.txt").read_text(encoding="utf-8")
    adjudicator = (prompts / "stage2_execution_adjudicator_reflection_v1.txt").read_text(
        encoding="utf-8")
    assert "정답이나 강제 명령이 아니라" in primary
    assert "현재 질문" in primary
    assert "보조 근거로만" in adjudicator


def test_off_config_keeps_memory_assets_optional():
    from stage2_agent import Stage2Agent

    agent = Stage2Agent.__new__(Stage2Agent)
    agent.reflection_memory = None
    agent.current_memory_A = [{"record_id": "old"}]
    assert agent.prepare_question("새 질문") == []
