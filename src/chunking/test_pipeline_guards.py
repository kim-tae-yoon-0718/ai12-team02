"""
실행 안전장치 회귀 시험 — PR #7 3차 리뷰 지적 9번

표 처리 시험(test_chunking_tables.py)과 달리 build_chunks.py 를 실제로 실행해
중단 조건과 종료코드, 기존 파일 보존을 확인한다.

⚠️ 서버의 공식 폴더를 건드리지 않는다. 매 시험마다 임시 RAG_ROOT 를 새로 만들고
   끝나면 지운다. 가짜 문서 3건 · 필드 2개 = 6행짜리 작은 데이터를 쓴다.

실행:
    python3 test_pipeline_guards.py
    python3 -m pytest test_pipeline_guards.py -v
"""
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unicodedata
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPT = HERE / "build_chunks.py"
REPO = HERE.parents[1]                       # <repo>/src/chunking/ 기준

DOCS = 3
FIELDS = ["사업명", "사업기간"]
ROWS = DOCS * len(FIELDS)

PIPE = "| 구분 | 역할 |\n| --- | --- |\n| 기관 | 운영 |"
HTML = ('<table><tr><th>항목</th><th>값</th></tr>'
        '<tr><td>사업기간</td><td>12개월</td></tr></table>')


# ─────────────────────────────────────────────────────────────
# 픽스처
# ─────────────────────────────────────────────────────────────

def build_fixture(root: Path, *, rows=None, field_count=None,
                  duplicate_pair=False, split_version=False,
                  extra_doc=False, missing_doc=False,
                  extraction_version="v3", schema_version="1-12-2/v3",
                  corpus_version="v2", registry_version="v2",
                  drop_metadata=False, chunk_size=1500,
                  cfg_schema_version="1-12-2/v3",
                  meta_drop=(), meta_extra=None):
    """가짜 코퍼스·등록부·추출표를 만든다. 인자로 특정 결함을 주입한다."""
    proc = root / "shared_data" / "processed"
    md = proc / "corpus_v2" / "md"
    sc = proc / "corpus_v2" / "sidecar"
    reg_dir = proc / "document_registry_v2"
    tbl_dir = proc / "rfp_extraction_table_v3"
    for d in (md, sc, reg_dir, tbl_dir):
        d.mkdir(parents=True, exist_ok=True)

    registry = []
    for n in range(1, DOCS + 1):
        name = unicodedata.normalize("NFD", f"문서{n}.md")
        body = (f"# 1. 개요\n\n본문 {n}.\n\n## 1.1 목표\n\n{PIPE}\n\n"
                f"# 2. 요구사항\n\n{HTML}\n")
        (md / name).write_text(body, encoding="utf-8")
        side = {"doc_no": n, "output_filename": name,
                "blank_fields": [], "sections": [], "source_filename_is_nfc": False}
        (sc / (name + ".sections.json")).write_text(
            json.dumps(side, ensure_ascii=False), encoding="utf-8")
        h = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
        registry.append({
            "document_id": f"RFP-{n:06d}", "document_version": "1", "active": "true",
            "legacy_doc_no": str(n), "source_filename_nfd_raw": name,
            "source_filename_nfc": name, "output_filename": name,
            "sidecar_filename": name + ".sections.json",
            "source_sha256": h(md / name), "processed_sha256": h(md / name),
            "sidecar_sha256": h(sc / (name + ".sections.json")),
            "corpus_version": "v2", "preprocess_version": "v2", "rules_sha256": "x" * 64,
            "content_duplicate_group": "", "duplicate_of_document_id": "",
            "retrieval_eligible": "true", "relation_status": "independent"})

    with (reg_dir / "document_registry_v2.csv").open(
            "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(registry[0]), quoting=csv.QUOTE_ALL)
        w.writeheader()
        w.writerows(registry)

    (reg_dir / "registry_metadata.json").write_text(json.dumps({
        "corpus_version": "v2", "document_count": DOCS,
        "generated_at": "2026-08-30T10:46:10+00:00"}, ensure_ascii=False),
        encoding="utf-8")

    # ── 추출표
    table = []
    for r in registry:
        if missing_doc and r["document_id"] == "RFP-000003":
            continue
        for fld in FIELDS:
            table.append({"document_id": r["document_id"],
                          "document_version": "1", "field_name": fld})
    if extra_doc:
        for fld in FIELDS:
            table.append({"document_id": "RFP-000099",
                          "document_version": "1", "field_name": fld})
    if duplicate_pair:
        table.append(dict(table[0]))
    if split_version:
        # 필드 수·행 수는 그대로 두고 한 행의 버전만 바꿔 '문서 안 갈림'만 만든다
        for t in table:
            if t["document_id"] == "RFP-000001" and t["field_name"] == FIELDS[1]:
                t["document_version"] = "9"
                break
    if rows == "short":
        table.pop()
    elif rows == "long":
        table.append({"document_id": "RFP-000001",
                      "document_version": "1", "field_name": "여분필드"})

    with (tbl_dir / "extraction_table_v3.csv").open(
            "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["document_id", "document_version", "field_name"])
        w.writeheader()
        w.writerows(table)

    if not drop_metadata:
        # 기본은 v3·v4 형식(생성 근거를 corpus_dir·rules_sha256·decisions_file 로 적는다).
        # meta_drop 으로 키를 빼고 meta_extra 로 덮어써서 v5 형식이나 결측을 만든다.
        meta = {
            "schema_version": schema_version, "extraction_version": extraction_version,
            "corpus_version": corpus_version, "registry_version": registry_version,
            "row_count": len(table),
            "document_count": DOCS if field_count is None else DOCS,
            "field_count": len(FIELDS) if field_count is None else field_count,
            "generated_at": "2026-09-01T04:34:32+00:00",
            "generator": "build_extraction_table.py",
            "corpus_dir": "/x/corpus_v2", "registry_dir": "/x/document_registry_v2",
            "rules_sha256": "be3a" * 16, "decisions_file": "semantic_decisions_v3.csv"}
        for k in meta_drop:
            meta.pop(k, None)
        meta.update(meta_extra or {})
        (tbl_dir / "extraction_metadata.json").write_text(
            json.dumps(meta, ensure_ascii=False), encoding="utf-8")

    cfg_dir = root / "repo" / "config"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    # cfg_schema_version 은 base.yaml 쪽 값이다. 위의 schema_version 인자는
    # extraction_metadata.json 쪽 값이라 서로 다른 값을 넣어 불일치를 만들 수 있다.
    # None 을 주면 키 자체를 쓰지 않는다(선언 누락 상황 재현).
    schema_line = (f'schema_version: "{cfg_schema_version}"\n'
                   if cfg_schema_version is not None else "")
    (cfg_dir / "base.yaml").write_text(
        f"corpus: v2\npreprocess: v2\ntable: v3\n"
        f"chunk_size: {chunk_size}\nchunk_overlap: 150\nchunking_version: v1\n"
        f'chunk_unit: "char"\ntable_chunk_threshold: 1500\n'
        f'table_degraded_threshold: 0.6\ntable_format: "html"\n'
        f'table_empty_cell: "skip_in_search"\n'
        + schema_line +
        f'embedding_model: "text-embedding-3-small"\nembedding_max_length: 8192\n',
        encoding="utf-8")
    return root


def run(root: Path, *args):
    script = root / "repo" / "src" / "chunking" / "build_chunks.py"
    script.parent.mkdir(parents=True, exist_ok=True)
    if not script.exists():
        shutil.copy(SCRIPT, script)
    env = dict(os.environ, RAG_ROOT=str(root))
    return subprocess.run([sys.executable, str(script), *args],
                          capture_output=True, text=True, env=env)


def _check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def with_fixture(**kw):
    root = Path(tempfile.mkdtemp(prefix="chunkguard_"))
    build_fixture(root, **kw)
    return root


def expect_stop(root, *args, code=1, must_contain=None):
    r = run(root, *args)
    _check(r.returncode == code,
           f"종료코드 {code} 를 기대했으나 {r.returncode}\n{r.stderr[:400]}")
    if must_contain:
        _check(must_contain in r.stderr,
               f"중단 사유에 '{must_contain}' 가 없다\n{r.stderr[:400]}")
    return r


# ─────────────────────────────────────────────────────────────
# 정상 경로
# ─────────────────────────────────────────────────────────────

def test_normal_run_succeeds():
    root = with_fixture()
    try:
        r = run(root)
        _check(r.returncode == 0, f"정상 실행이 실패했다\n{r.stderr[:400]}")
        _check("추출표 검증" in r.stdout, "추출표 검증 결과가 출력되지 않았다")
        out = root / "shared_data" / "processed" / "chunks_v1" / "chunks.jsonl"
        _check(out.exists(), "산출물이 생성되지 않았다")
    finally:
        shutil.rmtree(root, ignore_errors=True)


# ─────────────────────────────────────────────────────────────
# 추출표 구조 (리뷰 지적 3)
# ─────────────────────────────────────────────────────────────

def test_stop_on_row_count_short():
    root = with_fixture(rows="short")
    try:
        # 메타데이터 row_count 는 실제와 맞지만 문서당 필드 수가 깨진다
        expect_stop(root, must_contain="필드 수가")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_stop_on_row_count_long():
    root = with_fixture(rows="long")
    try:
        expect_stop(root, must_contain="필드 수가")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_stop_on_metadata_row_count_mismatch():
    root = with_fixture()
    try:
        # CSV 에서 한 행을 빼 메타데이터와 어긋나게 만든다
        p = (root / "shared_data" / "processed" / "rfp_extraction_table_v3"
             / "extraction_table_v3.csv")
        lines = p.read_text(encoding="utf-8-sig").splitlines()
        p.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8-sig")
        expect_stop(root, must_contain="행 수가 메타데이터와 다릅니다")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_stop_on_duplicate_pair():
    root = with_fixture(duplicate_pair=True)
    try:
        expect_stop(root, must_contain="중복")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_stop_on_split_document_version():
    root = with_fixture(split_version=True)
    try:
        expect_stop(root, must_contain="document_version")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_stop_on_extra_document():
    root = with_fixture(extra_doc=True)
    try:
        expect_stop(root)
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_stop_on_missing_document():
    root = with_fixture(missing_doc=True)
    try:
        expect_stop(root)
    finally:
        shutil.rmtree(root, ignore_errors=True)


# ─────────────────────────────────────────────────────────────
# 메타데이터 교차 확인 (리뷰 지적 4)
# ─────────────────────────────────────────────────────────────

def test_stop_when_extraction_version_is_v2():
    root = with_fixture(extraction_version="v2")
    try:
        expect_stop(root, must_contain="extraction_version")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_stop_on_wrong_schema_version():
    root = with_fixture(schema_version="1-12-2/v2")
    try:
        expect_stop(root, must_contain="schema_version")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_stop_when_config_schema_version_missing():
    """base.yaml 에 schema_version 이 없으면 대조를 건너뛰지 말고 중단해야 한다.

    ⚠️ 예전에는 cfg 에 없으면 조용히 건너뛰고, 대신 메타데이터의 schema 문자열이
       '.../{table}' 로 끝나는지 봤다. 추출표 v4 가 v3 스키마를 그대로 쓰면서
       그 검사를 걷어냈으므로, 이 대조가 스키마를 지키는 유일한 자리다.
       선택 항목으로 되돌리면 스키마가 틀려도 아무도 못 막는다.
    """
    root = with_fixture(cfg_schema_version=None)
    try:
        expect_stop(root, must_contain="schema_version")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_stop_on_wrong_corpus_or_registry_version():
    for kw, key in (({"corpus_version": "v1"}, "corpus_version"),
                    ({"registry_version": "v1"}, "registry_version")):
        root = with_fixture(**kw)
        try:
            expect_stop(root, must_contain=key)
        finally:
            shutil.rmtree(root, ignore_errors=True)


# ── 생성 근거(provenance) — 이름이 아니라 '목적'으로 요구한다
#
# 추출표 v5 는 v3·v4 와 같은 목적을 다른 키로 적는다. 이름만 보던 예전 검사는
# 멀쩡한 v5 를 막았다. 아래 시험이 두 형식 모두 통과하는 것과, 목적별로 대안 키가
# 전부 없을 때 중단하는 것을 고정한다.

# v5 형식 — v3 형식의 생성 근거 키를 빼고 v5 키로 채운다.
V5_DROP = ("generated_at", "corpus_dir", "registry_dir",
           "rules_sha256", "decisions_file")
V5_EXTRA = {
    "released_at": "2026-09-08",
    "source_table_sha256": "44c9" * 16,
    "table_sha256": "b608" * 16,
    "decisions": "tools/evalset/extraction_table_v5_decisions.json",
}


def test_metadata_accepts_v5_style_provenance():
    """v5 형식(released_at·table_sha256·decisions)만으로도 통과해야 한다."""
    root = with_fixture(meta_drop=V5_DROP, meta_extra=V5_EXTRA)
    try:
        r = run(root)
        _check(r.returncode == 0,
               f"v5 형식 생성 근거가 거부됐다\n{r.stderr[:400]}")
        out = root / "shared_data" / "processed" / "chunks_v1" / "chunks.jsonl"
        _check(out.exists(), "산출물이 생성되지 않았다")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_stop_when_provenance_time_missing():
    root = with_fixture(meta_drop=V5_DROP, meta_extra={
        k: v for k, v in V5_EXTRA.items() if k != "released_at"})
    try:
        expect_stop(root, must_contain="생성 시각")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_stop_when_provenance_generator_missing():
    root = with_fixture(meta_drop=("generator",))
    try:
        expect_stop(root, must_contain="생성 코드")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_stop_when_provenance_inputs_missing():
    """corpus_dir·registry_dir 과 corpus_version·registry_version 이 모두 없을 때.

    ⚠️ 앞선 want 대조가 corpus_version 을 먼저 보므로 그쪽 문구로 먼저 죽을 수 있다.
       여기서는 '중단한다'만 고정한다 — 어느 검사가 잡든 통과시키지 않는 것이 요점.
    """
    root = with_fixture(meta_drop=("corpus_dir", "registry_dir",
                                   "corpus_version", "registry_version"))
    try:
        expect_stop(root)
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_stop_when_provenance_digest_missing():
    root = with_fixture(meta_drop=V5_DROP, meta_extra={
        k: v for k, v in V5_EXTRA.items()
        if k not in ("source_table_sha256", "table_sha256")})
    try:
        expect_stop(root, must_contain="무결성 지문")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_stop_when_provenance_decisions_missing():
    root = with_fixture(meta_drop=V5_DROP, meta_extra={
        k: v for k, v in V5_EXTRA.items() if k != "decisions"})
    try:
        expect_stop(root, must_contain="결정 근거")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_stop_when_metadata_missing():
    root = with_fixture(drop_metadata=True)
    try:
        expect_stop(root, must_contain="extraction_metadata.json")
    finally:
        shutil.rmtree(root, ignore_errors=True)


# ─────────────────────────────────────────────────────────────
# 부분 갱신 (리뷰 지적 3 / 이전 지적)
# ─────────────────────────────────────────────────────────────

def test_stop_partial_without_existing():
    root = with_fixture()
    try:
        expect_stop(root, "--only", "RFP-000001", must_contain="기존 산출물이 없습니다")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_stop_partial_when_config_changed():
    root = with_fixture()
    try:
        _check(run(root).returncode == 0, "선행 전체 실행이 실패했다")
        cfg = root / "repo" / "config" / "base.yaml"
        cfg.write_text(cfg.read_text(encoding="utf-8")
                       .replace("chunk_size: 1500", "chunk_size: 1200"),
                       encoding="utf-8")
        expect_stop(root, "--only", "RFP-000001", must_contain="chunk_size")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_stop_partial_without_stats():
    root = with_fixture()
    try:
        _check(run(root).returncode == 0, "선행 전체 실행이 실패했다")
        (root / "shared_data" / "processed" / "chunks_v1" / "stats.json").unlink()
        expect_stop(root, "--only", "RFP-000001", must_contain="stats.json")
    finally:
        shutil.rmtree(root, ignore_errors=True)


# ─────────────────────────────────────────────────────────────
# 검사 생략 · 실패 시 보존 (리뷰 지적 4·6)
# ─────────────────────────────────────────────────────────────

def test_skip_check_writes_to_unchecked_only():
    root = with_fixture(drop_metadata=True)
    try:
        r = run(root, "--skip-extraction-check")
        _check(r.returncode == 0, f"우회 실행이 실패했다\n{r.stderr[:300]}")
        proc = root / "shared_data" / "processed"
        _check((proc / "chunks_v1_unchecked" / "chunks.jsonl").exists(),
               "_unchecked 폴더에 저장되지 않았다")
        _check(not (proc / "chunks_v1").exists(),
               "검사를 건너뛴 결과가 공식 폴더에 저장됐다")
        errs = (proc / "chunks_v1_unchecked" / "errors.jsonl").read_text(encoding="utf-8")
        _check("skip-extraction-check" in errs, "우회 사실이 기록되지 않았다")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_tokenize_without_config_stops_and_preserves():
    root = with_fixture()
    try:
        _check(run(root).returncode == 0, "선행 전체 실행이 실패했다")
        out = root / "shared_data" / "processed" / "chunks_v1" / "chunks.jsonl"
        before = hashlib.sha256(out.read_bytes()).hexdigest()

        cfg = root / "repo" / "config" / "base.yaml"
        cfg.write_text(cfg.read_text(encoding="utf-8")
                       .replace("embedding_max_length: 8192",
                                "embedding_max_length: null"), encoding="utf-8")
        r = run(root, "--tokenize")
        _check(r.returncode == 3, f"종료코드 3을 기대했으나 {r.returncode}")
        after = hashlib.sha256(out.read_bytes()).hexdigest()
        _check(before == after, "실패한 실행이 기존 공식 청크를 바꿨다")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_processing_error_preserves_existing():
    root = with_fixture()
    try:
        _check(run(root).returncode == 0, "선행 전체 실행이 실패했다")
        out = root / "shared_data" / "processed" / "chunks_v1" / "chunks.jsonl"
        before = hashlib.sha256(out.read_bytes()).hexdigest()

        md = root / "shared_data" / "processed" / "corpus_v2" / "md"
        target = sorted(md.iterdir())[0]
        hidden = target.with_suffix(".hidden")
        target.rename(hidden)
        r = run(root)
        _check(r.returncode == 1, f"종료코드 1을 기대했으나 {r.returncode}")
        _check(hashlib.sha256(out.read_bytes()).hexdigest() == before,
               "실패한 실행이 기존 공식 청크를 바꿨다")
        hidden.rename(target)
    finally:
        shutil.rmtree(root, ignore_errors=True)


# ─────────────────────────────────────────────────────────────
# 병합 정책 · 파이프 표 (리뷰 지적 8·9)
# ─────────────────────────────────────────────────────────────

def test_no_cross_section_merge_in_output():
    root = with_fixture()
    try:
        _check(run(root).returncode == 0, "정상 실행이 실패했다")
        out = root / "shared_data" / "processed" / "chunks_v1" / "chunks.jsonl"
        rows = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines()]
        multi = [r for r in rows if len(r.get("section_paths", [])) > 1]
        _check(not multi, f"서로 다른 절이 병합됐다: {len(multi)}건")
        _check(all("section_paths" in r for r in rows), "section_paths 가 없다")
    finally:
        shutil.rmtree(root, ignore_errors=True)


# ─────────────────────────────────────────────────────────────
# location 전수 대조 게이트 (--verify-location, 팀 결정 2026-09-03)
# ─────────────────────────────────────────────────────────────

# 한 절 안에 문단 6개. chunk_size 를 줄이면 조각이 늘어 count_changed 가 생긴다.
LONG_DOC = "# 1. 개요\n\n" + "\n\n".join(
    f"문단 {i} " + "가" * 200 for i in range(1, 7)) + "\n"

# 사업명은 사업기간에게 pop 당해 소실 → 뒤따르는 표의 search_text 로 간다.
TABLE_LEAD_DOC = (
    "# 1. 개요\n\n"
    "### □ 사업명 : 통합 정보시스템\n\n"
    "### □ 사업기간 : 12개월\n\n"
    f"{HTML}\n"
)

CHUNKS = ("shared_data", "processed", "chunks_v1")


def _rewrite_doc(root: Path, n: int, body: str):
    """가짜 문서 한 건의 본문만 갈아끼운다.

    등록부의 processed_sha256 은 청킹 코드가 대조하지 않고 그대로 메타데이터에
    싣기만 하므로(build_chunks 는 등록부 조인 값을 신뢰한다) 본문만 바꿔도 된다.
    """
    md = root / "shared_data" / "processed" / "corpus_v2" / "md"
    (md / unicodedata.normalize("NFD", f"문서{n}.md")).write_text(body, encoding="utf-8")


def _set_chunk_size(root: Path, size: int):
    cfg = root / "repo" / "config" / "base.yaml"
    cfg.write_text(cfg.read_text(encoding="utf-8")
                   .replace("chunk_size: 1500", f"chunk_size: {size}"), encoding="utf-8")


def _make_reference(root: Path) -> Path:
    """정상 실행 산출물을 대조용 참조 파일로 떠 둔다."""
    _check(run(root).returncode == 0, "참조용 선행 실행이 실패했다")
    ref = root / "ref.jsonl"
    shutil.copy(root.joinpath(*CHUNKS) / "chunks.jsonl", ref)
    return ref


def _rewrite_jsonl(path: Path, fn):
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    fn(rows)
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
                    encoding="utf-8")


def _location_verification(root: Path) -> dict:
    stats = json.loads((root.joinpath(*CHUNKS) / "stats.json").read_text(encoding="utf-8"))
    lv = stats.get("location_verification")
    _check(lv is not None, "stats.json 에 location_verification 이 없다")
    return lv


def test_verify_location_same_input_passes():
    root = with_fixture()
    try:
        ref = _make_reference(root)
        r = run(root, "--verify-location", str(ref))
        _check(r.returncode == 0, f"같은 입력 대조가 실패했다\n{r.stderr[:400]}")
        lv = _location_verification(root)
        for k in ("document_missing", "document_added", "location_missing",
                  "location_added", "eligible_changed", "count_changed"):
            _check(lv[k] == 0, f"{k} 가 0이 아니다: {lv[k]}")
        _check(lv["result"] == "ok" and lv["mode"] == "set", f"요약이 이상하다: {lv}")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_verify_location_mismatch_stops_and_writes_nothing():
    root = with_fixture()
    try:
        ref = _make_reference(root)
        _rewrite_jsonl(ref, lambda rows: rows[0].__setitem__(
            "location_label", rows[0]["location_label"] + " (변조)"))

        # '미생성' 을 확인하려면 빈 상태에서 돌려야 한다
        out = root.joinpath(*CHUNKS)
        shutil.rmtree(out)

        r = run(root, "--verify-location", str(ref))
        _check(r.returncode == 4, f"종료코드 4를 기대했으나 {r.returncode}\n{r.stderr[:400]}")
        for name in ("chunks.jsonl", "stats.json", "VERSION.txt"):
            _check(not (out / name).exists(), f"중단했는데 {name} 이 생성됐다")
        _check(not list(out.glob("*.tmp")), f".tmp 잔재가 남았다: {list(out.glob('*.tmp'))}")

        errs = [json.loads(l) for l in
                (out / "errors.jsonl").read_text(encoding="utf-8").splitlines()]
        kinds = {e.get("kind") for e in errs}
        _check({"location_missing", "location_added"} <= kinds,
               f"불일치 상세가 기록되지 않았다: {kinds}")
        summary = [e for e in errs if e.get("kind") == "summary"][0]
        _check(summary["level"] == "error" and summary["location_missing"] == 1,
               f"요약이 이상하다: {summary}")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_verify_location_count_change_passes_in_set_mode():
    """chunk_size 를 바꾸면 같은 라벨의 개수만 늘어난다 — 집합 비교는 통과한다."""
    root = with_fixture()
    try:
        _rewrite_doc(root, 1, LONG_DOC)
        ref = _make_reference(root)
        _set_chunk_size(root, 500)

        r = run(root, "--verify-location", str(ref))
        _check(r.returncode == 0, f"집합 비교가 중단됐다\n{r.stderr[:400]}")
        lv = _location_verification(root)
        _check(lv["location_missing"] == 0 and lv["location_added"] == 0,
               f"라벨 값이 바뀌었다: {lv}")
        # 개수 변화가 실제로 안 생기면 이 시험도, 다음 strict 시험도 무의미해진다
        _check(lv["count_changed"] > 0,
               "chunk_size 를 줄였는데 개수 변화가 없다 — 픽스처가 시험을 못 하고 있다")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_verify_location_strict_stops_on_count_change():
    """같은 조건에서 --verify-location-strict 는 중단해야 한다."""
    root = with_fixture()
    try:
        _rewrite_doc(root, 1, LONG_DOC)
        ref = _make_reference(root)
        _set_chunk_size(root, 500)

        expect_stop(root, "--verify-location", str(ref), "--verify-location-strict",
                    code=4, must_contain="개수 변화")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_verify_location_strict_requires_verify_location():
    root = with_fixture()
    try:
        expect_stop(root, "--verify-location-strict", code=1,
                    must_contain="--verify-location")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_verify_location_bad_reference_stops():
    """참조 경로가 없거나 코퍼스가 다르면 처리 전에 멈춘다 (입력 오류이므로 1)."""
    root = with_fixture()
    try:
        expect_stop(root, "--verify-location", str(root / "없는파일.jsonl"),
                    code=1, must_contain="참조 파일이 없습니다")
    finally:
        shutil.rmtree(root, ignore_errors=True)

    root = with_fixture()
    try:
        ref = _make_reference(root)
        _rewrite_jsonl(ref, lambda rows: [r.__setitem__("corpus_version", "v1")
                                          for r in rows])
        expect_stop(root, "--verify-location", str(ref),
                    code=1, must_contain="corpus_version")
    finally:
        shutil.rmtree(root, ignore_errors=True)


# ─────────────────────────────────────────────────────────────
# 소실 헤딩 복원 — 표는 content 를 건드리지 않는다
# ─────────────────────────────────────────────────────────────

def test_recovered_heading_touches_table_search_text_only():
    root = with_fixture()
    try:
        _rewrite_doc(root, 1, TABLE_LEAD_DOC)
        _check(run(root).returncode == 0, "정상 실행이 실패했다")

        rows = [json.loads(l) for l in
                (root.joinpath(*CHUNKS) / "chunks.jsonl").read_text(
                    encoding="utf-8").splitlines()]
        tables = [r for r in rows
                  if r["document_id"] == "RFP-000001" and r["block_type"] == "table"]
        _check(tables, "표 청크가 없다")

        hit = [r for r in tables if "□ 사업명 : 통합 정보시스템" in r["search_text"]]
        _check(hit, "소실 헤딩이 표 청크의 search_text 에 실리지 않았다")

        c = hit[0]
        _check("사업명" not in c["content"],
               f"복원 텍스트가 표 content(HTML) 에 섞였다: {c['content'][:120]!r}")
        _check(c["content"].strip() == HTML,
               "표 content 가 원본 HTML 과 다르다")
        # 살아남은 형제만 경로에 남는다 — 복원은 경로를 바꾸지 않는다
        _check("□ 사업명 : 통합 정보시스템" not in c["section_path"],
               f"복원 텍스트가 section_path 에 들어갔다: {c['section_path']}")

        stats = json.loads((root.joinpath(*CHUNKS) / "stats.json").read_text(
            encoding="utf-8"))
        hr = stats["heading_recovery"]
        _check(hr["heading_to_table"] >= 1,
               f"표로 간 복원 건수가 기록되지 않았다: {hr}")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_edgeless_pipe_line_is_not_a_table():
    """양 끝 파이프가 없는 줄을 표로 오인하면 안 된다.

    실제 코퍼스 100문서 전수 측정 결과 양 끝 파이프가 없는 표는 0건이었고
    (구분선 존재 · 대시 3개 이상 · 열 수 일치 기준), 현재 인식 규칙을 유지한다.
    산문에 파이프가 있어도 표로 잡지 않는지 확인한다.
    """
    sys.path.insert(0, str(HERE))
    os.environ.setdefault("RAG_ROOT", "/tmp")
    from build_chunks import split_blocks
    doc = "# 1. 개요\n\n조건 A | 조건 B 를 함께 본다.\n\n일반 문장입니다.\n"
    kinds = [k for k, *_ in split_blocks(doc)]
    _check("pipe_table" not in kinds, f"산문을 표로 잡았다: {kinds}")


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failed = []
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except AssertionError as e:
            print(f"  FAIL  {name}\n          {e}")
            failed.append(name)
        except Exception as e:                                  # noqa: BLE001
            print(f"  ERROR {name}\n          {type(e).__name__}: {e}")
            failed.append(name)
    print(f"\n{len(tests) - len(failed)} / {len(tests)} 통과")
    sys.exit(1 if failed else 0)
