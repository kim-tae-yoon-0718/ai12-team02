"""VERSION.txt 파싱 + RAG_ROOT 경로 규칙 (팀 확정 2026-08-30).

    $RAG_ROOT/evalset/v1/VERSION.txt
        evalset: v1
        corpus: [대기]
        created: 2026-08-30
"""

from grader.versioning import read_schema_version, read_versions


def _write(root, text):
    d = root / "evalset" / "v1"
    d.mkdir(parents=True)
    (d / "VERSION.txt").write_text(text, encoding="utf-8")


def test_reads_key_value_from_rag_root(tmp_path, monkeypatch):
    _write(tmp_path, "evalset: v1\ncorpus: [대기]\ncreated: 2026-08-30\n")
    monkeypatch.setenv("RAG_ROOT", str(tmp_path))
    v = read_versions()
    assert v == {"evalset": "v1", "corpus": "[대기]", "created": "2026-08-30"}
    assert read_schema_version() == "v1"


def test_legacy_single_line_format(tmp_path, monkeypatch):
    _write(tmp_path, "v0.2\n")
    monkeypatch.setenv("RAG_ROOT", str(tmp_path))
    assert read_schema_version() == "v0.2"


def test_missing_file_is_unknown(tmp_path, monkeypatch):
    monkeypatch.setenv("RAG_ROOT", str(tmp_path))  # evalset/v1/VERSION.txt 없음
    assert read_versions() == {}
    assert read_schema_version() == "UNKNOWN"


def test_explicit_path_overrides_env(tmp_path):
    p = tmp_path / "V.txt"
    p.write_text("evalset: v9\n", encoding="utf-8")
    assert read_schema_version(p) == "v9"
