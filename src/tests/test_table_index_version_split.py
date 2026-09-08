"""추출표 버전(현재 조회용)과 인덱스 추출표 계보(생성 당시)의 분리 — 회귀 고정.

배경
----
예전에는 `extraction_version` 하나가 두 뜻으로 쓰였다.

  ① 지금 구조화 조회에 쓰는 추출표 버전            (공식 v4)
  ② 이 인덱스를 만든 청크의 추출표 계보            (index_v2 = chunks_v3 = v3)

`validate_index_tag` 가 둘을 같은 값으로 검사해서, 추출표만 v4 로 승격한 순간
멀쩡한 index_v2 가 "낡았다"고 차단되고 베이스라인 평가가 통째로 막혔다.

여기서 고정하는 것
------------------
  - ①②가 달라도 통과하되, ②가 실제 인덱스와 다르면 여전히 차단
  - 코퍼스·청킹·임베딩·차원 등 기존 엄격 검사는 그대로 유지
  - 인덱스 생성이 cfg 의 ①을 꼬리표에 복사하지 않고 실제 청크 계보를 적음
  - 추출표 경로 탐색 우선순위(명시 > 저장소 > 서버) 와 충돌 시 중단
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
OFFICIAL_PROCESSED = Path(os.environ.get("RAG_ROOT_OFFICIAL", "/srv/rfp")) / \
    "shared_data" / "processed"
OFFICIAL_INDEX_V2 = OFFICIAL_PROCESSED / "index_v2"


def _sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# ①② 분리 — 인덱스 꼬리표 검증
# ---------------------------------------------------------------------------

class TestIndexTagLineageSeparation:
    """cfg.extraction_version(조회용) 과 cfg.index_source_extraction_version(계보)."""

    def _idx(self, tmp_path, small_store, **over):
        from conftest import write_valid_index
        return write_valid_index(tmp_path, small_store, **over)

    def test_1_current_table_v4_with_index_lineage_v3_passes(self, tmp_path,
                                                             small_store, base_cfg):
        """①=v4, ②=v3, 인덱스 꼬리표=v3 → 통과. (이번 수정의 핵심 시나리오)"""
        from vector_store import validate_index_tag
        cfg = dict(base_cfg, table="v4", extraction_version="v4",
                   index_source_extraction_version="v3")
        idx = self._idx(tmp_path, small_store, extraction_version="v3")
        out = validate_index_tag(idx, cfg)
        assert out["tag"]["extraction_version"] == "v3"

    def test_3_index_lineage_v2_against_configured_v3_fails(self, tmp_path,
                                                            small_store, base_cfg):
        """인덱스 계보 v2 ↔ 설정 계보 v3 → 차단(계보 검사는 살아 있어야 한다)."""
        from vector_store import validate_index_tag, IndexTagError
        cfg = dict(base_cfg, table="v4", extraction_version="v4",
                   index_source_extraction_version="v3")
        idx = self._idx(tmp_path, small_store, extraction_version="v2")
        with pytest.raises(IndexTagError) as ei:
            validate_index_tag(idx, cfg)
        assert "extraction_version" in str(ei.value)

    def test_lineage_mismatch_message_explains_two_axes(self, tmp_path,
                                                        small_store, base_cfg):
        """오류 문구가 '조회용'과 '계보'를 구분해 알려준다."""
        from vector_store import validate_index_tag, IndexTagError
        cfg = dict(base_cfg, extraction_version="v4",
                   index_source_extraction_version="v3")
        idx = self._idx(tmp_path, small_store, extraction_version="v2")
        with pytest.raises(IndexTagError) as ei:
            validate_index_tag(idx, cfg)
        assert "계보" in str(ei.value)

    def test_backward_compat_falls_back_to_extraction_version(self, tmp_path,
                                                              small_store, base_cfg):
        """계보 키가 없는 옛 config 는 예전 그대로 동작한다(닫히는 쪽으로)."""
        from vector_store import validate_index_tag, IndexTagError, \
            index_source_extraction_version
        cfg = dict(base_cfg, extraction_version="v3")
        cfg.pop("index_source_extraction_version", None)
        assert index_source_extraction_version(cfg) == "v3"
        validate_index_tag(self._idx(tmp_path, small_store,
                                     extraction_version="v3"), cfg)
        with pytest.raises(IndexTagError):
            validate_index_tag(self._idx(tmp_path, small_store, name="idx2",
                                         extraction_version="v4"), cfg)

    @pytest.mark.parametrize("over,needle", [
        ({"corpus_version": "v1"}, "corpus_version"),
        ({"chunking_version": "v1"}, "chunking_version"),
        ({"preprocess_version": "v1"}, "preprocess_version"),
        ({"registry_version": "v1"}, "registry_version"),
        ({"embedding_model": "text-embedding-3-large"}, "embedding_model"),
        ({"embedding_provider": "local"}, "embedding_provider"),
        ({"chunk_size": 700}, "chunk_size"),
        ({"chunk_overlap": 50}, "chunk_overlap"),
    ])
    def test_4_other_axes_still_blocked(self, tmp_path, small_store, base_cfg,
                                        over, needle):
        """계보 축을 분리했다고 다른 축 검사가 느슨해지지 않았는지."""
        from vector_store import validate_index_tag, IndexTagError
        cfg = dict(base_cfg, extraction_version="v4",
                   index_source_extraction_version="v3")
        idx = self._idx(tmp_path, small_store, extraction_version="v3", **over)
        with pytest.raises(IndexTagError) as ei:
            validate_index_tag(idx, cfg)
        assert needle in str(ei.value)

    def test_4b_vector_dimension_mismatch_still_blocked(self, tmp_path, small_store,
                                                        base_cfg):
        """꼬리표 차원 ≠ 실제 vectors.npy → 계속 차단."""
        from vector_store import validate_index_tag, IndexTagError
        cfg = dict(base_cfg, extraction_version="v4",
                   index_source_extraction_version="v3")
        idx = self._idx(tmp_path, small_store, extraction_version="v3")
        tag = json.loads((idx / "index_tag.json").read_text(encoding="utf-8"))
        tag["vector_dimension"] = 999
        (idx / "index_tag.json").write_text(json.dumps(tag), encoding="utf-8")
        with pytest.raises(IndexTagError, match="vector_dimension"):
            validate_index_tag(idx, cfg)


# ---------------------------------------------------------------------------
# ② 추출표 메타데이터 자체 검사는 '조회용' 축으로 남아 있어야 한다
# ---------------------------------------------------------------------------

class TestCurrentTableVersionStillChecked:
    OFFICIAL_V4 = REPO_ROOT / "data" / "preprocessed" / \
        "rfp_extraction_table_v4" / "extraction_table_v4.json"

    @pytest.mark.skipif(not OFFICIAL_V4.exists(), reason="공식 추출표 v4 없음")
    def test_2_table_metadata_v3_with_config_v4_fails(self, base_cfg):
        """실제 추출표가 v3인데 설정이 v4 → 실패(계보 분리와 무관하게 유지).

        ★이 검사는 '지금 조회에 쓰는 추출표' 축이다. 인덱스 계보 축을 분리했다고
          이쪽까지 느슨해지면, 엉뚱한 버전의 표로 답을 만들어도 통과해 버린다.
        """
        from table_query import validate_extraction_table, ExtractionTableFormatError
        doc = json.loads(self.OFFICIAL_V4.read_text(encoding="utf-8"))
        assert doc["extraction_version"] == "v4"
        doc["extraction_version"] = "v3"          # 파일이 v3라고 선언
        cfg = dict(base_cfg, table="v4", extraction_version="v4",
                   corpus="v2", document_registry_version="v2",
                   schema_version="1-12-2/v3",
                   index_source_extraction_version="v3")
        with pytest.raises(ExtractionTableFormatError, match="extraction_version"):
            validate_extraction_table(doc, cfg=cfg, official=True)

    @pytest.mark.skipif(not OFFICIAL_V4.exists(), reason="공식 추출표 v4 없음")
    def test_2b_official_v4_passes_with_config_v4(self, base_cfg):
        """정상 조합(파일 v4 + 설정 v4)은 계보가 v3이어도 통과한다."""
        from table_query import validate_extraction_table
        doc = json.loads(self.OFFICIAL_V4.read_text(encoding="utf-8"))
        cfg = dict(base_cfg, table="v4", extraction_version="v4",
                   corpus="v2", document_registry_version="v2",
                   schema_version="1-12-2/v3",
                   index_source_extraction_version="v3")
        validate_extraction_table(doc, cfg=cfg, official=True)


# ---------------------------------------------------------------------------
# 추출표 경로 탐색 우선순위
# ---------------------------------------------------------------------------

class TestExtractionTablePathResolution:
    """① 명시 > ② 저장소 > ③ 서버 공용 > ④ 명확한 오류."""

    def _make_table_dir(self, root: Path, version: str, marker: str) -> Path:
        d = root / f"rfp_extraction_table_{version}"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"extraction_table_{version}.json").write_text(
            json.dumps({"marker": marker}), encoding="utf-8")
        (d / "extraction_metadata.json").write_text(
            json.dumps({"extraction_version": version, "marker": marker}),
            encoding="utf-8")
        return d

    def _cfg_with_repo(self, tmp_path, monkeypatch, base_cfg):
        """RAG_CONFIG_PATH 로 저장소 루트를 tmp_path 로 고정한다."""
        cfgdir = tmp_path / "config"
        cfgdir.mkdir(exist_ok=True)
        (cfgdir / "base.yaml").write_text("top_k: 5\n", encoding="utf-8")
        monkeypatch.setenv("RAG_CONFIG_PATH", str(cfgdir / "base.yaml"))
        return dict(base_cfg, table="v4")

    def test_6_repo_path_is_found_automatically(self, tmp_path, monkeypatch, base_cfg):
        """서버 공용 경로에 없어도 저장소 data/preprocessed 를 찾는다."""
        from config import extraction_table_path, extraction_metadata_path
        cfg = self._cfg_with_repo(tmp_path, monkeypatch, base_cfg)
        self._make_table_dir(tmp_path / "data" / "preprocessed", "v4", "repo")
        assert json.loads(extraction_table_path(cfg).read_text())["marker"] == "repo"
        assert extraction_metadata_path(cfg).name == "extraction_metadata.json"

    def test_7_explicit_path_wins(self, tmp_path, monkeypatch, base_cfg):
        """명시 경로가 있으면 저장소·서버보다 우선한다(_try_derive 계약)."""
        from answer_pipeline import _try_derive
        from config import extraction_table_path
        cfg = self._cfg_with_repo(tmp_path, monkeypatch, base_cfg)
        self._make_table_dir(tmp_path / "data" / "preprocessed", "v4", "repo")
        explicit = tmp_path / "explicit.json"
        explicit.write_text(json.dumps({"marker": "explicit"}), encoding="utf-8")
        got = _try_derive(str(explicit), extraction_table_path, cfg, "--extraction-table")
        assert json.loads(Path(got).read_text())["marker"] == "explicit"

    def test_server_path_used_when_repo_has_none(self, tmp_path, monkeypatch, base_cfg):
        """저장소에 없고 서버에만 있으면 서버 경로를 쓴다."""
        from config import extraction_table_path
        cfg = self._cfg_with_repo(tmp_path, monkeypatch, base_cfg)
        server = tmp_path / "srv" / "shared_data" / "processed"
        self._make_table_dir(server, "v4", "server")
        monkeypatch.setenv("RAG_ROOT", str(tmp_path / "srv"))
        assert json.loads(extraction_table_path(cfg).read_text())["marker"] == "server"

    def test_8_same_version_different_content_stops(self, tmp_path, monkeypatch,
                                                    base_cfg):
        """두 위치에 같은 버전이 있는데 지문이 다르면 임의 선택 금지 — 중단."""
        from config import extraction_table_path, ExtractionTablePathError
        cfg = self._cfg_with_repo(tmp_path, monkeypatch, base_cfg)
        self._make_table_dir(tmp_path / "data" / "preprocessed", "v4", "repo")
        server = tmp_path / "srv" / "shared_data" / "processed"
        self._make_table_dir(server, "v4", "server-different")
        monkeypatch.setenv("RAG_ROOT", str(tmp_path / "srv"))
        with pytest.raises(ExtractionTablePathError) as ei:
            extraction_table_path(cfg)
        msg = str(ei.value)
        assert "내용이 다릅니다" in msg and "v4" in msg

    def test_same_version_identical_content_is_ok(self, tmp_path, monkeypatch,
                                                  base_cfg):
        """두 위치가 **같은 내용**이면 충돌이 아니다 — 저장소를 쓴다."""
        from config import extraction_table_path
        cfg = self._cfg_with_repo(tmp_path, monkeypatch, base_cfg)
        self._make_table_dir(tmp_path / "data" / "preprocessed", "v4", "same")
        server = tmp_path / "srv" / "shared_data" / "processed"
        self._make_table_dir(server, "v4", "same")
        monkeypatch.setenv("RAG_ROOT", str(tmp_path / "srv"))
        assert extraction_table_path(cfg).parts[-3] == "preprocessed"

    def test_9_missing_everywhere_raises_clear_error(self, tmp_path, monkeypatch,
                                                     base_cfg):
        """어디에도 없으면 조용한 버전 강등이 아니라 명확한 오류."""
        from config import extraction_table_path, ExtractionTablePathError
        cfg = self._cfg_with_repo(tmp_path, monkeypatch, base_cfg)
        monkeypatch.setenv("RAG_ROOT", str(tmp_path / "srv"))
        with pytest.raises(ExtractionTablePathError) as ei:
            extraction_table_path(cfg)
        msg = str(ei.value)
        assert "찾지 못했습니다" in msg and "다른 버전으로 자동 대체하지 않습니다" in msg

    def test_9b_no_silent_downgrade_to_older_version(self, tmp_path, monkeypatch,
                                                     base_cfg):
        """v3 만 있는데 v4 를 요구하면 v3 로 되돌아가지 않는다."""
        from config import extraction_table_path, ExtractionTablePathError
        cfg = self._cfg_with_repo(tmp_path, monkeypatch, base_cfg)
        self._make_table_dir(tmp_path / "data" / "preprocessed", "v3", "old")
        monkeypatch.setenv("RAG_ROOT", str(tmp_path / "srv"))
        with pytest.raises(ExtractionTablePathError):
            extraction_table_path(cfg)

    def test_path_error_is_not_swallowed_by_try_derive(self, tmp_path, monkeypatch,
                                                       base_cfg):
        """_try_derive 가 경로 충돌 원인을 경고로 덮지 않는다."""
        from answer_pipeline import _try_derive
        from config import extraction_table_path, ExtractionTablePathError
        cfg = self._cfg_with_repo(tmp_path, monkeypatch, base_cfg)
        monkeypatch.setenv("RAG_ROOT", str(tmp_path / "srv"))
        with pytest.raises(ExtractionTablePathError):
            _try_derive(None, extraction_table_path, cfg, "--extraction-table")


# ---------------------------------------------------------------------------
# 인덱스 생성 — 꼬리표에 실제 청크 계보를 적는다
# ---------------------------------------------------------------------------

class TestBuildIndexRecordsChunkLineage:
    def test_10_tag_records_chunk_lineage_not_config(self, tmp_path, base_yaml_path,
                                                     chunks_jsonl_path, registry_path,
                                                     chunks_version_txt, monkeypatch):
        """cfg 는 v4(조회용)인데 청크는 v3 → 꼬리표는 **v3** 여야 한다."""
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from test_pr10_fixes import _run_build_index

        text = base_yaml_path.read_text(encoding="utf-8")
        text = text.replace("table: v3", "table: v4").replace(
            "extraction_version: v3",
            "extraction_version: v4\nindex_source_extraction_version: v3")
        base_yaml_path.write_text(text, encoding="utf-8")

        bi = _run_build_index(tmp_path, base_yaml_path, chunks_jsonl_path, monkeypatch,
                              registry=registry_path, version_file=chunks_version_txt)
        bi.main()
        tag = json.loads((tmp_path / "idx_built" / "index_tag.json").read_text(
            encoding="utf-8"))
        assert tag["extraction_version"] == "v3", \
            "인덱스 꼬리표에 cfg 의 조회용 추출표(v4)가 복사되면 거짓 계보가 된다"

    def test_lineage_resolver_prefers_version_file(self):
        from build_index import resolve_index_extraction_lineage
        cfg = {"extraction_version": "v4", "index_source_extraction_version": "v3"}
        assert resolve_index_extraction_lineage(
            cfg, {"extraction_version": "v3"}, []) == "v3"

    def test_lineage_resolver_rejects_config_conflict(self):
        """청크 계보가 설정 계보와 다르면 중단한다."""
        from build_index import resolve_index_extraction_lineage, ChunkLineageError
        cfg = {"extraction_version": "v4", "index_source_extraction_version": "v4"}
        with pytest.raises(ChunkLineageError, match="계보"):
            resolve_index_extraction_lineage(cfg, {"extraction_version": "v3"}, [])

    def test_lineage_resolver_never_copies_current_table_version(self):
        """계보 근거가 하나도 없으면 cfg.extraction_version 을 몰래 쓰지 않는다."""
        from build_index import resolve_index_extraction_lineage, ChunkLineageError
        with pytest.raises(ChunkLineageError):
            resolve_index_extraction_lineage({"extraction_version": "v4"}, {}, [])

    def test_lineage_resolver_rejects_mixed_chunks(self):
        from build_index import resolve_index_extraction_lineage, ChunkLineageError
        with pytest.raises(ChunkLineageError, match="섞여"):
            resolve_index_extraction_lineage(
                {"index_source_extraction_version": "v3"}, {},
                [{"extraction_version": "v3"}, {"extraction_version": "v4"}])

    def test_build_and_runtime_share_one_lineage_helper(self):
        """빌드 경로와 실행 경로가 같은 함수를 쓴다(기준 이중화 금지)."""
        import build_index
        from vector_store import index_source_extraction_version
        assert build_index.index_source_extraction_version is \
            index_source_extraction_version


# ---------------------------------------------------------------------------
# 공식 자료 — 꼬리표를 고치지 않고 현재 조합이 실제로 뜨는가
# ---------------------------------------------------------------------------

_OFFICIAL_READY = (OFFICIAL_INDEX_V2 / "index_tag.json").exists() and \
    (OFFICIAL_PROCESSED / "chunks_v3" / "chunks.jsonl").exists()


@pytest.mark.skipif(not _OFFICIAL_READY, reason="공식 index_v2/chunks_v3 가 없는 환경")
class TestOfficialCombinationBoots:
    def test_5_official_index_tag_is_untouched_and_runtime_boots(self, monkeypatch):
        """index_v2 꼬리표를 **고치지 않고** 런타임 초기화가 성공해야 한다."""
        import argparse
        monkeypatch.delenv("RAG_CONFIG_PATH", raising=False)
        monkeypatch.setenv("RAG_ROOT", str(Path(
            os.environ.get("RAG_ROOT_OFFICIAL", "/srv/rfp"))))
        monkeypatch.setenv("RAG_CONFIG_PATH", str(REPO_ROOT / "config" / "base.yaml"))

        tag = json.loads((OFFICIAL_INDEX_V2 / "index_tag.json").read_text(
            encoding="utf-8"))
        assert tag["extraction_version"] == "v3", \
            "공식 index_v2 꼬리표는 v3 그대로여야 한다(거짓 v4 로 고치지 않는다)"

        from config import load_config
        from answer_pipeline import build_runtime, add_common_args
        cfg = load_config(None)
        assert cfg["extraction_version"] == "v5"
        assert cfg["index_source_extraction_version"] == "v3"

        parser = argparse.ArgumentParser()
        add_common_args(parser)
        rt = build_runtime(parser.parse_args([]), cfg)
        assert rt["index_check"]["vector_dimension"] == 1536
        assert len(rt["table"]) == 1200

    def test_11_official_input_tests_actually_run(self):
        """공식 입력 테스트가 저장소 v5 를 찾아 skip 되지 않는지."""
        import importlib
        mod = importlib.import_module("test_official_inputs")
        assert mod.EXTRACTION_DIR is not None
        assert mod.EXTRACTION_DIR == REPO_ROOT / "data" / "preprocessed" / \
            "rfp_extraction_table_v5"
        assert mod._MISSING == [], f"skip 사유가 남아 있다: {mod._MISSING}"
        assert mod.pytestmark.args[0] is False


# ---------------------------------------------------------------------------
# 고정 자료 지문 — 이번 수정이 평가셋·채점기·추출표를 건드리지 않았는가
# ---------------------------------------------------------------------------

class TestFixedAssetsUnchanged:
    def test_12_evalset_and_table_fingerprints(self):
        preserved = {
            "data/evalsets/final/v2/items.jsonl":
                "0cf0868aaa1466b91fee79cba9ac7eb0fe935b32254a9e6b8272d3ca4bbeb007",
            "data/preprocessed/rfp_extraction_table_v4/extraction_table_v4.json":
                "44c973d79d7fe4d7b5693fdad070572e846c11a15b8bf8bcaca66c4e16d2a1df",
        }
        released = {
            "data/evalsets/final/v3/items.jsonl":
                "88b28b480ddfa0ad37b25db5b656f2dd94900eb8b477eafda9587d9065223197",
            "data/preprocessed/rfp_extraction_table_v5/extraction_table_v5.json":
                "b6082b9bee4ab71b0260d0912f5ac71ea2ccea5fcf9f61390792579b11a5070e",
        }
        for rel, want in {**preserved, **released}.items():
            p = REPO_ROOT / rel
            assert p.exists(), f"{rel} 가 없습니다"
            assert _sha256(p) == want, f"{rel} 지문이 바뀌었습니다 — 고정 자료 수정 금지"

    def test_12b_evalset_has_exactly_50_items(self):
        p = REPO_ROOT / "data" / "evalsets" / "final" / "v3" / "items.jsonl"
        assert sum(1 for line in p.read_text(encoding="utf-8").splitlines()
                   if line.strip()) == 50

    def test_12c_grader_package_untracked_by_this_change(self):
        """채점기 폴더는 이번 수정 대상이 아니다 — 파일이 그대로 있는지만 확인."""
        grader = REPO_ROOT / "src" / "grader"
        assert (grader / "task_scoring.py").exists()
        assert (grader / "runner.py").exists()
        assert (REPO_ROOT / "config" / "grader.yaml").exists()
