#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
체크리스트4 C 청킹 — 구축 진입점

확정정리(체크리스트4_파싱~청킹, 예진)의 C-1~C-5를 그대로 구현한다.
설정은 config/base.yaml에서 읽고, 실험별 override를 그 위에 덮어쓴다.

사용:
    python build_chunks.py
    python build_chunks.py --config-override runs/yj001_I_chunk1500/config.yaml
    python build_chunks.py --only RFP-000001,RFP-000002      # 부분 갱신
    python build_chunks.py --tokenize                        # 토큰 길이 집계까지 (tiktoken)
    python build_chunks.py --verify-location .../chunks_v3/chunks.jsonl
                                                             # location 전수 대조 (불일치 시 중단)

산출물:  $RAG_ROOT/shared_data/processed/chunks_<chunking_version>/
    chunks.jsonl        청크 본문 + 메타데이터
    stats.json          문서별·유형별 집계, 길이 분포, 초과 건수
    errors.jsonl        중단 사유·경고
    VERSION.txt         설정·git·입력 버전 기록
"""

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
import unicodedata
from collections import Counter, defaultdict
collections_Counter = Counter
from datetime import datetime, timezone
from pathlib import Path

# ─────────────────────────────────────────────────────────────
# 설정 로드
# ─────────────────────────────────────────────────────────────

# 코드가 읽는 설정 키. base.yaml에 없으면 중단한다(값을 지어내지 않는다).
OPTIONAL_KEYS = ["embedding_model", "embedding_max_length",
                 "extraction_version", "schema_version"]

REQUIRED_KEYS = [
    "corpus", "preprocess", "table",
    "chunk_size", "chunk_overlap", "chunking_version", "chunk_unit",
    "table_chunk_threshold", "table_degraded_threshold",
    "table_format", "table_empty_cell",
]

# 추출표 메타데이터가 '무엇을' 증언해야 하는가 — 키 이름이 아니라 목적으로 요구한다.
#
# ⚠️ 이름만 보는 검사는 추출표 v5 를 막았다. v5 는 같은 목적을 다른 키로 적는다
#    (released_at / corpus_version / source_table_sha256 / decisions). 게다가 v3·v4 의
#    corpus_dir 은 /srv/rfp 절대경로, calibration_csv 는 /home/spai1205 절대경로라
#    규약 §2-3(절대경로 하드코딩 금지)에 어긋난다 — v5 형식이 규약에 더 맞다.
#    데이터가 옳고 검사의 전제가 낡았던 경우다.
#
# 각 목적은 대안 중 **한 묶음**이 다 채워지면 통과한다. 어느 묶음도 못 채우면 중단.
# 키를 지우지 않는다 — 예전 형식(v3·v4)도 그대로 통과해야 chunks_v3 을 재현할 수 있다.
PROVENANCE_REQUIREMENTS = (
    ("생성 시각", (("generated_at",), ("released_at",))),
    ("생성 코드", (("generator",),)),
    # ⚠️ v5 형식(corpus_version+registry_version)에서는 바로 위 want 대조와 겹쳐
    #    실효가 없다 — 거기서 이미 값이 있고 cfg 와 같아야 통과하기 때문이다.
    #    그래도 목적 목록의 완결성을 위해 남긴다. 이 목록이 사실상 "추출표
    #    메타데이터가 무엇을 증언해야 하는가"의 문서 역할을 한다.
    ("입력 자산", (("corpus_dir", "registry_dir"),
                   ("corpus_version", "registry_version"))),
    ("무결성 지문", (("rules_sha256",), ("source_table_sha256",), ("table_sha256",))),
    ("결정 근거", (("decisions_file",), ("decisions",))),
)


def _coerce(value: str):
    """YAML 스칼라를 파이썬 값으로. 최소 구현."""
    v = value.strip()
    if v.startswith("#") or v == "":
        return None
    # 인라인 주석 제거 (따옴표 밖의 #)
    out, quote = [], None
    for ch in v:
        if quote:
            out.append(ch)
            if ch == quote:
                quote = None
        elif ch in "\"'":
            quote = ch
            out.append(ch)
        elif ch == "#":
            break
        else:
            out.append(ch)
    v = "".join(out).strip()
    if v == "":
        return None
    if v.lower() in ("null", "~"):
        return None
    if v.lower() == "true":
        return True
    if v.lower() == "false":
        return False
    if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
        return v[1:-1]
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        pass
    return v


def load_yaml_flat(path: Path) -> dict:
    """base.yaml의 최상위 스칼라 키만 읽는다.

    PyYAML이 있으면 그것을 쓰고, 없으면 최소 파서로 대체한다.
    중첩 블록(deadline_filter_default 등)은 이 스크립트가 쓰지 않으므로 건너뛴다.
    """
    try:
        import yaml  # noqa
        with path.open(encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data
    except ImportError:
        pass

    data = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            if line[0] in " \t-":          # 중첩 블록·리스트는 무시
                continue
            if ":" not in line:
                continue
            key, _, rest = line.partition(":")
            key = key.strip()
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
                continue
            data[key] = _coerce(rest)
    return data


def load_config(repo_root: Path, override: Path = None) -> dict:
    base_path = repo_root / "config" / "base.yaml"
    if not base_path.exists():
        die(f"config/base.yaml을 찾을 수 없습니다: {base_path}")
    cfg = load_yaml_flat(base_path)
    if override:
        if not override.exists():
            die(f"override 설정을 찾을 수 없습니다: {override}")
        cfg.update({k: v for k, v in load_yaml_flat(override).items() if v is not None})

    missing = [k for k in REQUIRED_KEYS if cfg.get(k) is None]
    if missing:
        die("base.yaml에 값이 없습니다 (지어내지 않고 중단합니다): " + ", ".join(missing))

    if cfg["chunk_unit"] != "char":
        die(f"이 스크립트는 chunk_unit='char'만 구현합니다. 현재: {cfg['chunk_unit']}")
    if cfg["table_format"] != "html":
        die(f"이 스크립트는 table_format='html'만 구현합니다. 현재: {cfg['table_format']}")
    return cfg


def die(msg: str, code: int = 1):
    print(f"[중단] {msg}", file=sys.stderr)
    sys.exit(code)


def git_info(repo_root: Path) -> dict:
    """규약 §2-4 — 실행 진입점에서 git 상태를 기록한다."""
    def run(args):
        try:
            return subprocess.run(
                args, cwd=repo_root, capture_output=True, text=True, timeout=10
            ).stdout.strip()
        except Exception:
            return ""
    commit = run(["git", "log", "-1", "--format=%h"])
    dirty = run(["git", "status", "--short"]) != ""
    branch = run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    return {"git_commit": commit or None, "git_dirty": dirty, "git_branch": branch or None}


# ─────────────────────────────────────────────────────────────
# 입력 읽기
# ─────────────────────────────────────────────────────────────

def read_registry(path: Path) -> list:
    """등록부 CSV.

    A 단계 실측: BOM이 있어 utf-8-sig로 읽어야 한다.
    utf-8로 읽으면 첫 컬럼 키가 '\\ufeff"document_id"'가 되어 KeyError.
    """
    with path.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        die(f"등록부가 비어 있습니다: {path}")
    if "document_id" not in rows[0]:
        die(f"등록부에 document_id 컬럼이 없습니다: {list(rows[0].keys())[:5]}")
    # C-5 ① 재현성 — 등록부 document_id 순서로 고정한다.
    rows.sort(key=lambda r: r["document_id"])
    return rows


def load_extraction_metadata(table_dir: Path, cfg: dict):
    """추출표 공식 메타데이터를 읽고 버전을 교차 확인한다.

    폴더 이름이 v3이라고 안의 내용도 v3인 것은 아니다.
    공식 이름표는 extraction_metadata.json 이다(팀 결정 2026-09-01).
    산출물별 이름표: 코퍼스 VERSION.txt / 등록부 registry_metadata.json /
    추출표 extraction_metadata.json. VERSION.txt 를 새로 만들지 않는다.
    """
    meta_path = table_dir / "extraction_metadata.json"
    if not meta_path.exists():
        die(f"추출표 공식 메타데이터가 없습니다: {meta_path}")
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as e:                                     # noqa: BLE001
        die(f"extraction_metadata.json 을 읽지 못했습니다: {e}")

    want = {
        "extraction_version": cfg["table"],
        "corpus_version": cfg["corpus"],
        "registry_version": cfg["corpus"],   # 등록부는 코퍼스와 같은 버전을 쓴다
    }

    # ⚠️ base.yaml 안에서 table 과 extraction_version·schema_version 이 어긋나면
    #    사람이 읽을 때 "추출표는 v3인데 추출 버전은 v2"가 된다.
    #    코드가 읽는 값은 table 하나뿐이라 조용히 통과하므로 여기서 막는다.
    cfg_ext = cfg.get("extraction_version")
    if cfg_ext is not None and str(cfg_ext) != str(cfg["table"]):
        die(f"base.yaml 안에서 충돌합니다 — table: {cfg['table']} / "
            f"extraction_version: {cfg_ext}. 두 값은 항상 같아야 합니다.")
    # ⚠️ schema_version 은 필수다. 없으면 건너뛰지 않고 중단한다.
    #    예전에는 cfg 에 없으면 이 대조를 건너뛰고, 대신 아래에서 schema 문자열이
    #    '.../{table}' 로 끝나는지 봤다. 그 검사를 걷어냈으므로(추출표 v4 가 v3
    #    스키마를 그대로 쓴다) 이 대조가 스키마를 지키는 유일한 자리다.
    #    선택 항목으로 두면 schema_version 을 안 적은 설정에서 추출표 메타데이터의
    #    스키마가 틀려도 아무도 못 막는다 — 폴더 이름만 보고 내용을 믿게 된다.
    cfg_schema = cfg.get("schema_version")
    if cfg_schema is None:
        die("base.yaml 에 schema_version 이 없습니다. 추출표 메타데이터의 스키마를 "
            "대조할 수 없으므로 중단합니다 (지어내지 않습니다).")
    if str(cfg_schema) != str(meta.get("schema_version")):
        die(f"base.yaml 의 schema_version({cfg_schema})이 "
            f"extraction_metadata.json({meta.get('schema_version')})과 다릅니다.")

    bad = {k: (meta.get(k), v) for k, v in want.items() if meta.get(k) != v}
    if bad:
        die("추출표 메타데이터의 버전이 설정과 다릅니다: " +
            " / ".join(f"{k}: {a} (기대 {b})" for k, (a, b) in bad.items()))

    # 스키마 이름표는 있어야 한다. 다만 "스키마 문자열이 추출표 버전을 담는다"고
    # 보면 안 된다.
    # ⚠️ 예전에는 schema_version 이 '.../{table}' 로 끝나는지 봤다. v4 가 v3 스키마를
    #    그대로 쓰면서(추출표 v4 는 RFP-000067·RFP-000081 두 행만 정정한 것이라
    #    schema_version 이 "1-12-2/v3" 이다) 이 검사가 멀쩡한 조합을 막았다.
    #    버전 고정은 이미 위에서 두 번 한다 — base.yaml 의 schema_version 과
    #    메타데이터 대조, 그리고 want["extraction_version"] == cfg["table"] 대조.
    #    (같은 유형의 이중 의미 문제를 팀이 extraction_version /
    #     index_source_extraction_version 분리로 한 번 겪었다 — base.yaml 참조)
    schema = meta.get("schema_version")
    if not schema:
        die("추출표 메타데이터에 schema_version 이 없습니다. "
            "공식 산출물로 쓸 수 없습니다.")

    for k in ("row_count", "document_count", "field_count"):
        if not isinstance(meta.get(k), int) or meta[k] <= 0:
            die(f"추출표 메타데이터의 {k} 가 비었거나 잘못됐습니다: {meta.get(k)}")

    # 생성 근거가 비어 있으면 어느 코드·입력으로 만든 표인지 되짚을 수 없다.
    # 목적별 허용 키는 PROVENANCE_REQUIREMENTS 참조 — 이름이 아니라 목적을 요구한다.
    for purpose, alternatives in PROVENANCE_REQUIREMENTS:
        if not any(all(meta.get(k) for k in group) for group in alternatives):
            names = " 또는 ".join("+".join(g) for g in alternatives)
            die(f"추출표 메타데이터에 '{purpose}' 를 적은 키가 없습니다 "
                f"(허용: {names}). 공식 산출물로 쓸 수 없습니다.")

    return meta


def check_registry_metadata(registry_dir: Path, cfg: dict, ext_meta: dict):
    """등록부 공식 메타데이터(registry_metadata.json)와 교차 확인한다."""
    meta_path = registry_dir / "registry_metadata.json"
    if not meta_path.exists():
        die(f"등록부 공식 메타데이터가 없습니다: {meta_path}")
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as e:                                     # noqa: BLE001
        die(f"registry_metadata.json 을 읽지 못했습니다: {e}")

    if meta.get("corpus_version") != cfg["corpus"]:
        die(f"등록부 메타데이터의 corpus_version 이 설정과 다릅니다: "
            f"{meta.get('corpus_version')} (기대 {cfg['corpus']})")
    if meta.get("document_count") != ext_meta["document_count"]:
        die(f"등록부와 추출표의 문서 수가 다릅니다: "
            f"등록부 {meta.get('document_count')} / 추출표 {ext_meta['document_count']}")
    return meta


def validate_extraction_table(table_dir: Path, cfg: dict, meta: dict, registry: list):
    """추출표 CSV의 기본 구조를 강제로 확인한다.

    ⚠️ 폴더의 첫 번째 *.csv 를 쓰면 보조 CSV(field_alias_candidates 등)를
       잘못 읽는다. 공식 파일명을 명시해서 연다.
    """
    csv_path = table_dir / f"extraction_table_{cfg['table']}.csv"
    if not csv_path.exists():
        die(f"공식 추출표 CSV가 없습니다: {csv_path}")

    with csv_path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        for need in ("document_id", "document_version", "field_name"):
            if need not in cols:
                die(f"추출표에 {need} 컬럼이 없습니다: {cols[:8]}")
        rows = list(reader)

    n_rows = len(rows)
    if n_rows != meta["row_count"]:
        die(f"추출표 행 수가 메타데이터와 다릅니다: 실제 {n_rows} / "
            f"메타데이터 {meta['row_count']}")

    blank = [i + 2 for i, r in enumerate(rows)
             if not (r["document_id"] or "").strip() or not (r["field_name"] or "").strip()]
    if blank:
        die(f"document_id 또는 field_name 이 빈 행이 {len(blank)}건 있습니다: {blank[:5]}")

    by_doc, versions, pairs = {}, {}, collections_Counter()
    for r in rows:
        d, fld = r["document_id"], r["field_name"]
        by_doc.setdefault(d, []).append(fld)
        versions.setdefault(d, set()).add(r["document_version"])
        pairs[(d, fld)] += 1

    dup = [k for k, v in pairs.items() if v > 1]
    if dup:
        die(f"(document_id, field_name) 중복이 {len(dup)}건 있습니다: {dup[:5]}")

    if len(by_doc) != meta["document_count"]:
        die(f"추출표 문서 수가 메타데이터와 다릅니다: 실제 {len(by_doc)} / "
            f"메타데이터 {meta['document_count']}")

    fc = meta["field_count"]
    wrong = {d: len(f) for d, f in by_doc.items() if len(f) != fc}
    if wrong:
        die(f"필드 수가 {fc}가 아닌 문서가 {len(wrong)}건 있습니다: "
            f"{list(wrong.items())[:5]}")

    field_sets = {frozenset(f) for f in by_doc.values()}
    if len(field_sets) != 1:
        die(f"문서마다 필드 구성이 다릅니다 (서로 다른 조합 {len(field_sets)}종)")

    split = [{"document_id": d, "versions": sorted(v)}
             for d, v in versions.items() if len(v) > 1]
    if split:
        die(f"한 문서 안에서 document_version 이 갈립니다 ({len(split)}건): {split[:3]}")

    reg_map = {r["document_id"]: r["document_version"] for r in registry}
    missing = sorted(set(reg_map) - set(by_doc))
    extra = sorted(set(by_doc) - set(reg_map))
    if missing:
        die(f"추출표에 없는 등록부 문서가 {len(missing)}건 있습니다: {missing[:5]}")
    if extra:
        die(f"등록부에 없는 문서가 추출표에 {len(extra)}건 있습니다: {extra[:5]}")

    mismatch = [{"document_id": d, "registry": reg_map[d],
                 "extraction": sorted(versions[d])[0]}
                for d in reg_map if sorted(versions[d])[0] != reg_map[d]]
    if mismatch:
        die(f"등록부와 추출표의 document_version 이 다릅니다 ({len(mismatch)}건): "
            f"{mismatch[:3]}")

    return {"csv": csv_path.name, "rows": n_rows, "documents": len(by_doc),
            "fields_per_document": fc}


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# ─────────────────────────────────────────────────────────────
# location 전수 대조 (--verify-location)
#
# 팀 결정 2026-09-03 — 모든 청킹 실험은 평가셋 location 을 바꾸면 안 된다.
# section_path·location_label 이 이전 버전과 달라지는 수정은 실험에 반영하지 않고
# 한계점으로만 기록한다. 여기서 그 대조를 강제한다.
# ─────────────────────────────────────────────────────────────

MAX_CHUNK_IDS = 5             # 진단용으로 키마다 남길 chunk_id 개수
MAX_MISMATCH_DETAILS = 200    # errors.jsonl 에 남길 상세 레코드 상한

# 이 종류가 하나라도 나오면 산출물을 만들지 않는다.
BLOCKING_KINDS = ("document_missing", "document_added",
                  "location_missing", "location_added", "eligible_changed")


def location_key(chunk: dict):
    """대조 키 — (document_id, section_path, location_label).

    ⚠️ chunk_id 는 못 쓴다. C-3 ① 의 순번은 문서 내 위치라서 청크 수가 하나만
       달라져도 그 뒤가 전부 밀린다. chunk_size 를 바꾸는 실험에서는 항상 밀린다.
       md_line_start/end 도 그룹의 첫·마지막 unit 에서 오므로 병합 양상이
       바뀌면 같이 움직이고, 한 그룹을 쪼갠 조각들은 같은 줄 범위를 공유한다.
    ⚠️ section_path 를 함께 잡는다. para_label 이 path[-1](마지막 절 이름)만
       쓰므로 3.1 아래 '배점기준' 과 4.2 아래 '배점기준' 이 같은 라벨이 된다.
       둘을 묶어야 키가 유일해지고, 라벨은 그대로인데 경로만 바뀐 변화도 잡힌다.
    ⚠️ 문자열을 정규화하지 않는다. 헤딩이 없는 문서는 para_label 이 doc_title 을
       쓰는데 그 값이 NFD 다(파일명 유래). 정규화 차이를 조용히 흡수하지 말고
       location_missing + location_added 한 쌍으로 드러나게 둔다.
    """
    return (chunk["document_id"],
            tuple(chunk.get("section_path") or []),
            chunk.get("location_label"))


def _sort_key(key):
    """키에 None 이 섞여도 정렬이 죽지 않게."""
    return (key[0], tuple(str(p) for p in key[1]), str(key[2]))


def _add_to_index(index: dict, chunk: dict):
    slot = index.setdefault(chunk["document_id"], {}).setdefault(
        location_key(chunk), {"count": 0, "eligible": set(), "chunk_ids": []})
    slot["count"] += 1
    slot["eligible"].add(bool(chunk.get("retrieval_eligible")))
    if len(slot["chunk_ids"]) < MAX_CHUNK_IDS:
        slot["chunk_ids"].append(chunk.get("chunk_id"))


def index_locations(chunks) -> dict:
    """새 산출물을 대조용 투영으로 바꾼다. document_id -> {key: slot}"""
    index = {}
    for c in chunks:
        _add_to_index(index, c)
    return index


def load_prev_locations(path: Path, cfg: dict, errors: list):
    """이전 chunks.jsonl 에서 대조에 필요한 값만 뽑는다.

    ⚠️ 통째로 리스트에 담지 않는다. chunks_v3 는 19,381 레코드에 content 와
       search_text 가 들어 있어 전량 적재는 낭비다. 한 줄씩 읽어 투영만 남긴다.
    """
    if not path.exists():
        die(f"--verify-location 참조 파일이 없습니다: {path}")

    index, versions, corpora, total = {}, set(), set(), 0
    try:
        with path.open(encoding="utf-8") as f:
            for lineno, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                try:
                    c = json.loads(line)
                except json.JSONDecodeError as e:
                    die(f"--verify-location 참조 파일 {lineno}번째 줄을 읽지 "
                        f"못했습니다: {e}")
                for need in ("document_id", "section_path", "location_label"):
                    if need not in c:
                        die(f"--verify-location 참조 파일 {lineno}번째 줄에 {need} 가 "
                            f"없습니다. 청크 산출물이 맞는지 확인하세요.")
                _add_to_index(index, c)
                versions.add(c.get("chunking_version"))
                corpora.add(c.get("corpus_version"))
                total += 1
    except OSError as e:                                       # noqa: BLE001
        die(f"--verify-location 참조 파일을 열지 못했습니다: {e}")

    if total == 0:
        die(f"--verify-location 참조 파일이 비어 있습니다: {path}")

    if len(corpora) > 1:
        die(f"참조 파일 안에서 corpus_version 이 갈립니다: "
            f"{sorted(str(v) for v in corpora)}")
    if len(versions) > 1:
        die(f"참조 파일 안에서 chunking_version 이 갈립니다: "
            f"{sorted(str(v) for v in versions)}")
    prev_corpus = next(iter(corpora))
    prev_version = next(iter(versions))

    # 코퍼스가 다르면 문서 집합도 줄 번호도 달라 대조 자체가 성립하지 않는다.
    if prev_corpus != cfg["corpus"]:
        die(f"--verify-location 참조 파일의 corpus_version 이 다릅니다: "
            f"{prev_corpus} (현재 설정 {cfg['corpus']}). "
            f"코퍼스가 다르면 location 대조가 성립하지 않습니다.")

    # 같은 버전을 참조한다는 것은 지금 덮어쓸 산출물과 대조한다는 뜻이다.
    # 코드만 고친 재실행이면 정상이고, 실험이면 참조 경로를 잘못 준 것이다.
    if prev_version == cfg["chunking_version"]:
        errors.append({"level": "warn", "stage": "verify_location",
                       "msg": f"참조 파일의 chunking_version 이 현재 설정과 같습니다 "
                              f"({prev_version}). 지금 덮어쓸 산출물을 참조하고 "
                              f"있는지 확인하세요.",
                       "path": str(path)})

    meta = {
        "reference_file": str(path),
        "reference_sha256": sha256_of(path),
        "reference_chunking_version": prev_version,
        "reference_corpus_version": prev_corpus,
        "locations_prev": total,
        "documents_prev": len(index),
    }
    return index, meta


def compare_locations(prev: dict, new: dict, only, strict: bool):
    """이전·새 투영을 전수 대조한다. (상세 레코드, 종류별 건수)를 돌려준다.

    기본은 집합 비교다. 같은 라벨의 '개수'만 달라진 경우는 중단하지 않는다 —
    split_by_paragraph 가 한 그룹을 여러 조각으로 나눠도 조각들이 그룹 전체의
    para_start/para_end 를 그대로 받으므로, chunk_size 를 바꾸면 라벨 '값'은
    하나도 안 바뀌고 개수만 늘어난다. 평가셋이 참조하는 좌표는 전부 남아 있다.
    개수까지 강제하려면 --verify-location-strict 를 준다.
    """
    details, counts = [], Counter()

    def scope_of(doc_id):
        if only is None:
            return "full"
        return "regenerated" if doc_id in only else "kept"

    def add(kind, level, doc_id, key, p, n, msg, **extra):
        counts[kind] += 1
        if len(details) >= MAX_MISMATCH_DETAILS:
            return
        rec = {"level": level, "stage": "verify_location", "kind": kind,
               "document_id": doc_id, "scope": scope_of(doc_id)}
        if key is not None:
            rec["section_path"] = list(key[1])
            rec["location_label"] = key[2]
        rec.update({
            "prev_count": p["count"] if p else 0,
            "new_count": n["count"] if n else 0,
            "prev_chunk_ids": p["chunk_ids"] if p else [],
            "new_chunk_ids": n["chunk_ids"] if n else [],
        })
        rec.update(extra)
        rec["msg"] = msg
        details.append(rec)

    prev_docs, new_docs = set(prev), set(new)
    for doc_id in sorted(prev_docs - new_docs):
        add("document_missing", "error", doc_id, None, None, None,
            "이전 파일에 있던 문서가 새 산출물에 없습니다")
    for doc_id in sorted(new_docs - prev_docs):
        add("document_added", "error", doc_id, None, None, None,
            "이전 파일에 없던 문서가 새 산출물에 있습니다")

    for doc_id in sorted(prev_docs & new_docs):
        p, n = prev[doc_id], new[doc_id]
        for key in sorted(set(p) - set(n), key=_sort_key):
            add("location_missing", "error", doc_id, key, p[key], None,
                "이전 파일에 있던 location 이 새 산출물에 없습니다")
        for key in sorted(set(n) - set(p), key=_sort_key):
            add("location_added", "error", doc_id, key, None, n[key],
                "이전 파일에 없던 location 이 새 산출물에 생겼습니다")
        for key in sorted(set(p) & set(n), key=_sort_key):
            if p[key]["eligible"] != n[key]["eligible"]:
                add("eligible_changed", "error", doc_id, key, p[key], n[key],
                    "같은 location 의 retrieval_eligible 이 달라졌습니다",
                    prev_eligible=sorted(p[key]["eligible"]),
                    new_eligible=sorted(n[key]["eligible"]))
            if p[key]["count"] != n[key]["count"]:
                add("count_changed", "error" if strict else "warn",
                    doc_id, key, p[key], n[key],
                    "같은 location 의 청크 개수가 달라졌습니다 "
                    "(라벨 값은 그대로)")

    return details, counts


# ─────────────────────────────────────────────────────────────
# 블록 분해 — 헤딩 / 표 / 문단
# ─────────────────────────────────────────────────────────────

HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(.*\S)\s*$")
TABLE_OPEN_RE = re.compile(r"<table\b", re.I)
TABLE_CLOSE_RE = re.compile(r"</table\s*>", re.I)
TR_OPEN_RE = re.compile(r"<tr\b", re.I)
CELL_RE = re.compile(r"<t[dh]\b[^>]*>(.*?)</t[dh]\s*>", re.S | re.I)
TAG_RE = re.compile(r"<[^>]+>")
PIPE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")
# | --- | :--- | 구분선.
# ⚠️ 대시를 필수로 둔다. 없으면 전부 빈 행(|  |  |  |)이 구분선으로 오인되어
#    빈 셀 비율에서 빠지고 검색 본문에서도 사라진다.
PIPE_SEP_RE = re.compile(r"^\s*\|[\s:|-]*-[\s:|-]*\|\s*$")


def find_tables(text: str):
    """중첩을 고려해 바깥 <table> 범위만 찾는다.

    C-2 ③-b — 안쪽 표는 바깥 표의 한 행 안에 통째로 들어간다.
    단순 non-greedy 정규식은 안쪽 </table>에서 잘리므로 깊이를 센다.
    """
    spans = []
    tokens = []
    for m in TABLE_OPEN_RE.finditer(text):
        tokens.append((m.start(), "open", m.end()))
    for m in TABLE_CLOSE_RE.finditer(text):
        tokens.append((m.start(), "close", m.end()))
    tokens.sort()
    depth, start = 0, None
    for pos, kind, end in tokens:
        if kind == "open":
            if depth == 0:
                start = pos
            depth += 1
        else:
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    spans.append((start, end))
                    start = None
    return spans


def cell_is_blank(cell_html: str) -> bool:
    stripped = TAG_RE.sub("", cell_html)
    stripped = stripped.replace("&nbsp;", "").replace("\u00a0", "")
    return stripped.strip() == ""


CELL_OPEN_RE = re.compile(r"<t([dh])\b[^>]*>", re.I)
CELL_CLOSE_RE = re.compile(r"</t([dh])\s*>", re.I)


def outer_cells(row_html: str):
    """행에 직접 속한 셀의 내용만 뽑는다.

    ⚠️ 단순 non-greedy 정규식(<td>(.*?)</td>)은 셀 안에 중첩 표가 있을 때
       안쪽 </th>를 바깥 셀의 끝으로 봐서 바깥 셀 내용을 통째로 놓친다.
       실제 코퍼스에 <th><table>…</table><br>용역명 : …</th> 형태가 있고,
       그 표들은 검색 본문이 비고 빈 셀 비율이 1.0으로 잘못 계산됐다.
    """
    tokens = []
    for m in CELL_OPEN_RE.finditer(row_html):
        tokens.append((m.start(), m.end(), "open"))
    for m in CELL_CLOSE_RE.finditer(row_html):
        tokens.append((m.start(), m.end(), "close"))
    tokens.sort()

    cells, depth, start = [], 0, None
    for s, e, kind in tokens:
        if kind == "open":
            if depth == 0:
                start = e
            depth += 1
        else:
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    cells.append(row_html[start:s])
                    start = None
    return cells


def cell_text(cell_html: str) -> str:
    """셀의 표시 텍스트. 안쪽 표가 있으면 그 내용도 함께 들어간다."""
    return re.sub(r"\s+", " ", TAG_RE.sub(" ", cell_html)).strip()


def table_kind(text: str) -> str:
    """HTML 표인지 파이프 표(| a | b |)인지."""
    return "html" if TABLE_OPEN_RE.search(text) else "pipe"


def _outer_tr_spans(html: str):
    """바깥 표에 직접 속한 <tr>의 시작 위치와, 바깥 </table>의 시작 위치.

    C-2 ③-b — 중첩 표는 바깥 기준으로만 자른다.
    ⚠️ 단순 <tr> 분리는 안쪽 표의 행까지 바깥 행으로 세어
       중첩 표 한가운데를 가르고 HTML 구조를 깬다. 깊이를 센다.
    """
    tokens = []
    for m in TABLE_OPEN_RE.finditer(html):
        tokens.append((m.start(), "table_open"))
    for m in TABLE_CLOSE_RE.finditer(html):
        tokens.append((m.start(), "table_close"))
    for m in TR_OPEN_RE.finditer(html):
        tokens.append((m.start(), "tr"))
    tokens.sort(key=lambda t: t[0])

    depth, tr_starts, close_start = 0, [], len(html)
    for pos, kind in tokens:
        if kind == "table_open":
            depth += 1
        elif kind == "table_close":
            depth -= 1
            if depth == 0:
                close_start = pos
                break
        elif depth == 1:                        # 바깥 표에 직접 속한 행만
            tr_starts.append(pos)
    return tr_starts, close_start


def table_parts(text: str):
    """표를 (kind, head, header_rows, body_rows, tail)로 나눈다.

    head        HTML의 <table ...> 여는 부분 (파이프 표는 빈 문자열)
    header_rows 분할 시 각 조각에 반복할 머리글 (C-2 ③-a)
    body_rows   본문 행. row_start/row_end는 이 목록의 1-based 순번
    tail        HTML의 </table> (파이프 표는 빈 문자열)
    """
    kind = table_kind(text)

    if kind == "pipe":
        lines = [ln for ln in text.split("\n") if ln.strip()]
        if not lines:
            return kind, "", [], [], ""
        header = [lines[0]]
        rest = lines[1:]
        if rest and PIPE_SEP_RE.match(rest[0]):
            header.append(rest[0])              # 마크다운 구분선도 머리글에 포함
            rest = rest[1:]
        return kind, "", header, rest, ""

    tr_starts, close_start = _outer_tr_spans(text)
    if not tr_starts:
        return kind, text[:close_start], [], [], text[close_start:]
    head = text[:tr_starts[0]]
    rows = []
    for i, s in enumerate(tr_starts):
        e = tr_starts[i + 1] if i + 1 < len(tr_starts) else close_start
        rows.append(text[s:e])
    return kind, head, rows[:1], rows[1:], text[close_start:]


def table_row_count(text: str) -> int:
    """본문 행 수. row_end 기본값 계산용."""
    _, _, _, body, _ = table_parts(text)
    return len(body)


def pipe_cells(line: str):
    """파이프 표 한 행의 셀. 양끝 구분자는 버린다."""
    return [c.strip() for c in line.strip().strip("|").split("|")]


def table_blank_ratio(text: str):
    """빈 셀 비율. C-2 ② degraded 판정의 입력.

    행에 직접 속한 셀만 센다. 중첩 표의 셀은 바깥 셀 내용의 일부로 흡수된다.
    ⚠️ 임계값 0.6은 이전의 잘못된 셀 추출로 잰 분포가 근거였다.
       셀 추출을 고쳤으므로 분포를 다시 재고 임계값을 재확인해야 한다.
    """
    kind = table_kind(text)
    if kind == "html":
        _, _, header, body, _ = table_parts(text)
        cells = []
        for row in header + body:
            cells.extend(outer_cells(row))
        if not cells:
            return None, 0, 0
        blank = sum(1 for c in cells if cell_is_blank(c))
        return blank / len(cells), blank, len(cells)

    # 파이프 표 — 이전에는 <td>가 없어 항상 None이었고 degraded 판정이 아예 안 됐다.
    _, _, header, body, _ = table_parts(text)
    cells = []
    for line in header + body:
        if PIPE_SEP_RE.match(line):
            continue
        cells.extend(pipe_cells(line))
    if not cells:
        return None, 0, 0
    blank = sum(1 for c in cells if c == "")
    return blank / len(cells), blank, len(cells)


def table_search_text(text: str) -> str:
    """검색용 텍스트.

    C-2 ④ — 빈 셀만 건너뛰고 행 구조는 유지한다.
    원본(HTML/마크다운)은 수정하지 않는다. 이 함수는 별도 텍스트를 만든다.

    ⚠️ 파이프 표는 <td>가 없어 이전 구현에서 빈 문자열이 나왔다.
       검색 본문이 장절 접두만 남아 사실상 색인되지 않았다.
    """
    kind, _, header, body, _ = table_parts(text)
    lines = []

    for row in header + body:
        if kind == "pipe":
            if PIPE_SEP_RE.match(row):
                continue                        # 마크다운 구분선은 내용이 아니다
            vals = [c for c in pipe_cells(row) if c != ""]
        else:
            vals = [cell_text(c) for c in outer_cells(row) if not cell_is_blank(c)]
            if not vals:
                # 셀 구조가 깨진 행(닫는 태그 누락 등)이라도 텍스트가 있으면 살린다.
                fallback = cell_text(row)
                if fallback:
                    vals = [fallback]
        vals = [v for v in vals if v]
        if vals:
            lines.append(" | ".join(vals))

    return "\n".join(lines)


def plain_search_text(md: str) -> str:
    txt = TAG_RE.sub(" ", md)
    txt = re.sub(r"[ \t]+", " ", txt)
    txt = re.sub(r"\n{3,}", "\n\n", txt)
    return txt.strip()


def split_blocks(md_text: str):
    """문서를 순서대로 (kind, text, start_line, end_line) 블록으로 나눈다.

    kind: heading / table / text
    표는 먼저 떼어내고, 남은 구간에서 헤딩을 찾는다.
    """
    line_start = [0]
    for line in md_text.split("\n"):
        line_start.append(line_start[-1] + len(line) + 1)

    def line_of(offset: int) -> int:
        lo, hi = 0, len(line_start) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if line_start[mid] <= offset:
                lo = mid
            else:
                hi = mid - 1
        return lo + 1                            # 1-based

    blocks = []
    cursor = 0
    for ts, te in find_tables(md_text):
        if ts > cursor:
            blocks.extend(_split_text_region(md_text[cursor:ts], cursor, line_of))
        blocks.append(("table", md_text[ts:te], line_of(ts), line_of(te - 1)))
        cursor = te
    if cursor < len(md_text):
        blocks.extend(_split_text_region(md_text[cursor:], cursor, line_of))
    return blocks


def _split_text_region(region: str, base_offset: int, line_of):
    """표가 아닌 구간을 헤딩과 문단으로 나눈다.

    파이프 표(| a | b |)는 연속 줄을 한 덩어리로 묶어 표처럼 다룬다.
    """
    out = []
    buf, buf_start = [], None
    pipe, pipe_start = [], None
    offset = base_offset

    def flush_text():
        nonlocal buf, buf_start
        if buf:
            body = "\n".join(buf)
            if body.strip():
                out.append(("text", body, line_of(buf_start),
                            line_of(buf_start + len(body))))
            buf, buf_start = [], None

    def flush_pipe():
        nonlocal pipe, pipe_start
        if pipe:
            body = "\n".join(pipe)
            out.append(("pipe_table", body, line_of(pipe_start),
                        line_of(pipe_start + len(body))))
            pipe, pipe_start = [], None

    for line in region.split("\n"):
        start = offset
        offset += len(line) + 1
        m = HEADING_RE.match(line)
        if m:
            flush_pipe()
            flush_text()
            out.append(("heading", line, line_of(start), line_of(start)))
            continue
        if PIPE_ROW_RE.match(line):
            flush_text()
            if pipe_start is None:
                pipe_start = start
            pipe.append(line)
            continue
        flush_pipe()
        if buf_start is None:
            buf_start = start
        buf.append(line)
    flush_pipe()
    flush_text()
    return out


# ─────────────────────────────────────────────────────────────
# 청크 조립
# ─────────────────────────────────────────────────────────────

def clean_heading(title: str) -> str:
    """헤딩 제목에서 HTML 태그를 걷어낸다.

    ⚠️ chapter_title_recovered(1-17 C-1)가 표 안 한 줄을 제목으로 복구하면서
       <u>일반현황 및 연혁</u> 같은 태그가 딸려온다. 이 값이 사용자에게 보여줄
       원문 위치(C-3)와 김하루 님 Location.section에 그대로 나간다.
    """
    return re.sub(r"\s+", " ", TAG_RE.sub("", title)).strip()


def build_prefix(doc_title: str, path: tuple) -> str:
    """C-3 ③ — 문서명 + 전체 장절 경로. 기계 조립, LLM 생성 없음."""
    return " > ".join([doc_title] + list(path))


# ─────────────────────────────────────────────────────────────
# 소실 헤딩 복원
#
# 실측(2026-09-04) — 헤딩 8,609개 중 4,080개가 어떤 청크의 section_path 에도
# 들어가지 못하고 본문에도 남지 않는다. 같은 레벨의 형제 헤딩이 연달아 나오면
# 뒤 헤딩이 앞 헤딩을 pop 하는데, 그 사이에 본문이 없으면 앞 헤딩의 텍스트는
# 어디에도 남지 않는다.
#     ### □ 사업명 : 통합 정보시스템 구축 사전 컨설팅   ← pop 당하고 소실
#     ### □ 사업기간 : 계약체결일로부터 ~ …            ← pop 당하고 소실
#     ### □ 사업예산 : 50,000,000                     ← 이것만 남아 뒤 표의 경로가 된다
# 사업명·사업기간·사업예산 같은 핵심 메타데이터가 이렇게 사라진다.
#
# ⚠️ 복원은 '새 문단을 만들지 않는' 방식이어야 한다. 문단이 하나라도 늘면
#    para_no 가 밀려 location_label 의 문단 번호가 전부 바뀌고, 평가셋 좌표가
#    깨진다(팀 결정 2026-09-03).
#      - 뒤가 문단이면 : 그 문단 앞에 빈 줄 없이 이어 붙인다 → 문단 수 불변
#      - 뒤가 표면     : 그 표 청크의 search_text 앞에만 넣는다 → content(HTML) 무변경
#    실측상 표가 먼저 오는 1,105건 중 1,102건은 헤딩 텍스트가 그 표 안에
#    없었다. 그 헤딩들은 대개 바로 뒤 표의 제목 역할을 한다.
# ─────────────────────────────────────────────────────────────

LEADING_BLANK_RE = re.compile(r"\A(?:[ \t]*\n)+")


def _produces_unit(kind: str, body: str) -> bool:
    """이 블록이 청크 후보 단위(unit)를 만드는가.

    ⚠️ 헤딩은 unit 을 만들지 않는다. 헤딩을 unit 으로 세면 "형제 헤딩이
       뒤따르니 소실이 아니다"로 잘못 판정해 복원 대상이 대량으로 빠진다.
    """
    if kind == "heading":
        return False
    if kind in ("table", "pipe_table"):
        return True
    return any(p.strip() for p in re.split(r"\n\s*\n", body))


def find_lost_headings(blocks, boiler_of):
    """어떤 청크에도 남지 못하는 헤딩을 찾는다 → {블록 순번: 제목}, 별첨 제외 수

    소실 판정 = 그 헤딩의 서브트리(다음에 나오는 level <= 자기 level 인 헤딩
    직전까지) 안에서 unit 이 하나도 안 만들어짐. 그런 헤딩은 pop 될 때까지
    아무 unit 도 거치지 않으므로 section_path 에 들어갈 기회 자체가 없다.

    ⚠️ clean_heading 이 빈 문자열을 만든 줄은 제외한다. process_document 가
       스택에 올리지도 않고(레벨 비교조차 하지 않는다) 살릴 제목도 없다.
       그래서 서브트리 경계 계산에서도 그런 줄은 헤딩으로 세지 않는다.
    ⚠️ 별첨·서식 구간(sidecar sections) 안의 헤딩도 제외한다. 살려도
       retrieval_eligible=false 라 검색에 들어가지 않는다.
    """
    heads = []
    for i, (kind, body, ln0, _ln1) in enumerate(blocks):
        if kind != "heading":
            continue
        m = HEADING_RE.match(body)
        title = clean_heading(m.group(2))
        if not title:
            continue
        heads.append((i, len(m.group(1)), title, ln0))

    lost, skipped = {}, 0
    for n, (i, level, title, ln0) in enumerate(heads):
        end = len(blocks)
        for j in range(n + 1, len(heads)):
            if heads[j][1] <= level:
                end = heads[j][0]
                break
        if any(_produces_unit(blocks[k][0], blocks[k][1])
               for k in range(i + 1, end)):
            continue                      # 서브트리에 unit 있음 → 소실 아님
        if boiler_of(ln0)[0] is not None:
            skipped += 1
            continue                      # 별첨·서식 구간
        lost[i] = title
    return lost, skipped


def attach_lead(body: str, pending: list) -> str:
    """문단 앞에 소실 헤딩 텍스트를 빈 줄 없이 이어 붙인다.

    ⚠️ 빈 줄(\\n\\n)로 이으면 문단이 하나 늘어 para_no 가 밀린다.
       원문 첫머리의 빈 줄도 걷어내야 한다. "제목" + "\\n" + "\\n\\n본문" 이면
       결국 빈 줄이 생겨 같은 사고가 난다.
    """
    return "\n".join(pending) + "\n" + LEADING_BLANK_RE.sub("", body)


def para_label(path: tuple, doc_title: str, start: int, end: int) -> str:
    """C-3 ④ — 절 안의 문단 순번. 헤딩이 없으면 문서명을 0단 절로 쓴다."""
    where = path[-1] if path else doc_title
    if start == end:
        return f"{where} · 문단 {start}"
    return f"{where} · 문단 {start}-{end}"


def can_merge(path_a: tuple, path_b: tuple) -> bool:
    """C-1 ④ (2026-09-01 개정) — 같은 장절 경로일 때만 병합한다.

    ⚠️ 이전에는 형제 절(3.1과 3.2)도 합쳤다. 그러면 청크의 장절 경로가
       공통 부모(3장)로 깎여 근거 위치가 3.1인지 3.2인지 알 수 없게 된다.
       구성원 경로를 section_paths에 남겨도 다음 단계(인덱스)가 section_path만
       쓰므로 버려진다. 추출표 v3가 "실제 장·절 경로"를 저장하므로
       청크도 실제 절을 가져야 좌표 대조가 성립한다.

    비용: 텍스트 그룹 5,520 → 6,887 (+1,367, 24.8%). 전체 청크 기준 약 +7.5%.
    """
    return path_a == path_b


def common_prefix(paths):
    if not paths:
        return ()
    out = []
    for parts in zip(*paths):
        if len(set(parts)) == 1:
            out.append(parts[0])
        else:
            break
    return tuple(out)


def split_by_paragraph(text: str, budget: int, overlap: int):
    """C-1 ⑥ — 꼬리 구간은 문단 경계로 분할.

    문단 하나가 budget을 넘으면 자르지 않고 그대로 두고 초과를 기록한다.
    """
    paras = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paras:
        return [], 0
    pieces, cur, oversize = [], [], 0
    for p in paras:
        if len(p) > budget:
            if cur:
                pieces.append(("\n\n".join(cur), False))
                cur = []
            pieces.append((p, True))              # 문단 하나가 초과 — 자르지 않는다
            oversize += 1
            continue
        cand = ("\n\n".join(cur + [p])) if cur else p
        if len(cand) <= budget:
            cur = cur + [p]
        else:
            joined = "\n\n".join(cur)
            pieces.append((joined, False))
            tail = joined[-overlap:] if overlap > 0 else ""
            cur = ([tail, p] if tail else [p])
    if cur:
        pieces.append(("\n\n".join(cur), False))
    return pieces, oversize


def split_table(text: str, budget: int):
    """C-2 — 행 경계로 분할하고 머리글을 반복한다. HTML·파이프 표 공통.

    part/of는 글자 수가 아니라 행 경계로 계산한다 (C-2 ③).
    중첩 표는 바깥 행 단위로만 잘린다 (C-2 ③-b, table_parts가 보장).
    한 행이 budget을 넘으면 자르지 않고 초과를 기록한다 (C-2 ③-c).
    """
    kind, head, header, body, tail = table_parts(text)
    if not body:
        return [(text, 0, 0, 1, 1, False)], 0

    joiner = "\n" if kind == "pipe" else ""

    def assemble(rowset):
        segs = ([head] if head else []) + header + list(rowset)
        out = joiner.join(segs) if kind == "pipe" else head + "".join(header + list(rowset))
        return out + (("\n" + tail) if (kind == "pipe" and tail) else tail)

    base = len(assemble([]))

    parts, cur, cur_start, oversize = [], [], 1, 0
    for idx, row in enumerate(body, start=1):
        if base + len(row) > budget:
            # 이 행은 혼자서도 budget을 넘는다 — 자르지 않고 초과로 표시 (C-2 ③-c)
            # ⚠️ 판정을 루프 첫머리에서 해야 한다. cur가 비었을 때만 보면
            #    큰 행 앞에 작은 행이 하나라도 있을 때 표시가 누락된다.
            if cur:
                parts.append((cur, cur_start, cur_start + len(cur) - 1, False))
                cur = []
            parts.append(([row], idx, idx, True))
            oversize += 1
            cur_start = idx + 1
            continue
        if not cur:
            cur, cur_start = [row], idx
            continue
        if base + sum(len(r) + len(joiner) for r in cur) + len(row) <= budget:
            cur.append(row)
        else:
            parts.append((cur, cur_start, cur_start + len(cur) - 1, False))
            cur, cur_start = [row], idx
    if cur:
        parts.append((cur, cur_start, cur_start + len(cur) - 1, False))

    of = len(parts)
    return [(assemble(rowset), r0, r1, i, of, over)
            for i, (rowset, r0, r1, over) in enumerate(parts, start=1)], oversize


# ─────────────────────────────────────────────────────────────
# 문서 하나 처리
# ─────────────────────────────────────────────────────────────

def process_document(reg_row, md_dir, sidecar_dir, cfg, errors):
    doc_id = reg_row["document_id"]
    doc_title = Path(reg_row["output_filename"]).stem
    # A 단계 실측: 파일명이 NFD. 등록부 값을 그대로 쓰고 정규화하지 않는다.
    md_path = md_dir / reg_row["output_filename"]
    sc_path = sidecar_dir / reg_row["sidecar_filename"]
    if not md_path.exists():
        errors.append({"level": "error", "document_id": doc_id,
                       "msg": "본문 파일 없음", "path": str(md_path)})
        return [], {}
    md_text = md_path.read_text(encoding="utf-8")

    boiler_spans = []
    if sc_path.exists():
        try:
            sc = json.loads(sc_path.read_text(encoding="utf-8"))
            for s in sc.get("sections", []):
                boiler_spans.append((s.get("start_line", 0), s.get("end_line", 0),
                                     s.get("type"), s.get("label")))
        except Exception as e:
            errors.append({"level": "warn", "document_id": doc_id,
                           "msg": f"sidecar 파싱 실패: {e}"})
    else:
        errors.append({"level": "warn", "document_id": doc_id,
                       "msg": "sidecar 파일 없음", "path": str(sc_path)})

    def boiler_of(line_no):
        for a, b, t, label in boiler_spans:
            if a <= line_no <= b:
                return t, label
        return None, None

    doc_eligible = str(reg_row.get("retrieval_eligible", "true")).lower() == "true"
    size = int(cfg["chunk_size"])
    overlap = int(cfg["chunk_overlap"])
    tbl_threshold = int(cfg["table_chunk_threshold"])
    degraded_th = float(cfg["table_degraded_threshold"])

    blocks = split_blocks(md_text)

    # 어떤 청크에도 남지 못하는 헤딩. 판정은 매 실행 런타임에 다시 계산한다.
    lost_headings, lost_skipped_boiler = find_lost_headings(blocks, boiler_of)
    pending_lead = []                            # 아직 못 붙인 소실 헤딩 제목
    recov = {"text": 0, "table": 0, "boilerplate": lost_skipped_boiler}

    # 장절 경로 스택과 절 안 문단 순번
    stack = []                                   # [(level, title)]
    para_no = defaultdict(int)
    table_idx = 0
    units = []                                   # 청크 후보 단위

    for bi, (kind, body, ln0, ln1) in enumerate(blocks):
        if kind == "heading":
            if bi in lost_headings:
                # 스택에는 평소대로 올라가지만 어차피 unit 없이 pop 된다.
                # 텍스트만 뒤따르는 첫 문단·표로 실어 보낸다.
                pending_lead.append(lost_headings[bi])
            m = HEADING_RE.match(body)
            level, title = len(m.group(1)), clean_heading(m.group(2))
            if not title:
                continue                        # 태그만 있던 줄은 헤딩으로 안 센다
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            continue

        path = tuple(t for _, t in stack)
        b_type, b_label = boiler_of(ln0)

        if kind in ("table", "pipe_table"):
            table_idx += 1
            # 표 HTML 은 건드리지 않는다. 소실 헤딩 텍스트는 search_text 에만 실린다.
            lead = "\n".join(pending_lead) if pending_lead else None
            if pending_lead:
                recov["table"] += len(pending_lead)
                pending_lead = []
            units.append({
                "kind": "table", "html": body, "path": path,
                "line_start": ln0, "line_end": ln1,
                "table_idx": table_idx, "boiler_type": b_type, "boiler_label": b_label,
                "lead_text": lead,
            })
        else:
            # ⚠️ 문단 수는 '원문 기준으로' 먼저 센다. 복원 텍스트를 붙인 뒤에 세면
            #    para_no 가 밀려 location_label 의 문단 번호가 통째로 바뀐다.
            n_paras = len([p for p in re.split(r"\n\s*\n", body) if p.strip()])
            if n_paras == 0:
                continue                        # 빈 블록 — pending 은 다음으로 넘긴다
            start_no = para_no[path] + 1
            para_no[path] += n_paras
            if pending_lead:
                body = attach_lead(body, pending_lead)
                recov["text"] += len(pending_lead)
                pending_lead = []
            units.append({
                "kind": "text", "text": body, "path": path,
                "line_start": ln0, "line_end": ln1,
                "para_start": start_no, "para_end": para_no[path],
                "boiler_type": b_type, "boiler_label": b_label,
            })

    # ── 병합: 표를 만나면 끊고(C-1 ⑤), 같은 장절 경로일 때만 합친다(C-1 ④)
    chunks_raw = []
    group = []

    def flush_group():
        nonlocal group
        if not group:
            return
        paths = [u["path"] for u in group]
        uniq = list(dict.fromkeys(paths))       # 순서 유지, 중복 제거
        path = common_prefix(paths) if len(uniq) > 1 else paths[0]
        prefix = build_prefix(doc_title, path)
        budget = max(size - len(prefix) - 1, 200)
        text = "\n\n".join(u["text"] for u in group)
        pieces, oversize = split_by_paragraph(text, budget, overlap)
        for i, (piece, over) in enumerate(pieces):
            chunks_raw.append({
                "kind": "text", "path": path, "content": piece,
                "line_start": group[0]["line_start"], "line_end": group[-1]["line_end"],
                "para_start": group[0]["para_start"], "para_end": group[-1]["para_end"],
                "member_paths": uniq,
                "boiler_type": group[0]["boiler_type"], "boiler_label": group[0]["boiler_label"],
                "oversize": 1 if over else 0,
                "prefix": prefix,
            })
        group = []

    for u in units:
        if u["kind"] == "table":
            flush_group()
            path = u["path"]
            prefix = build_prefix(doc_title, path)
            budget = max(tbl_threshold - len(prefix) - 1, 200)
            ratio, blank, total = table_blank_ratio(u["html"])
            if len(u["html"]) <= budget:
                parts = [(u["html"], 1, max(table_row_count(u["html"]), 1), 1, 1, False)]
            else:
                parts, _ = split_table(u["html"], budget)
            for html, r0, r1, part, of, over in parts:
                chunks_raw.append({
                    "kind": "table", "path": path, "content": html,
                    "line_start": u["line_start"], "line_end": u["line_end"],
                    "table_idx": u["table_idx"], "row_start": r0, "row_end": r1,
                    "part": part, "of": of,
                    "blank_ratio": ratio, "blank_cells": blank, "total_cells": total,
                    "boiler_type": u["boiler_type"], "boiler_label": u["boiler_label"],
                    # 표 제목 역할을 하던 소실 헤딩. 조각마다 반복한다 —
                    # 머리글 반복(C-2 ③-a)과 같은 이유로, 조각 하나만 검색에
                    # 걸려도 무슨 표인지 알 수 있어야 한다.
                    "lead_text": u.get("lead_text"),
                    # C-2 ③-c — 자르지 못해 남긴 조각에만 표시한다.
                    # ⚠️ len(html) > budget 으로 재면 안 된다. 머리글 반복분 때문에
                    #    정상 분할된 조각도 넘을 수 있어 오탐이 생긴다(950자 조각이
                    #    초과로 찍히던 사례). split_table이 알려주는 값을 쓴다.
                    "oversize": 1 if over else 0,
                    "prefix": prefix,
                })
        else:
            if group and not can_merge(group[-1]["path"], u["path"]):
                flush_group()
            group.append(u)
    flush_group()

    # ── 메타데이터 부착
    out = []
    for seq, c in enumerate(chunks_raw, start=1):
        chunk_id = f"{doc_id}-{seq:04d}"          # C-3 ①
        is_boiler = c.get("boiler_type") is not None
        eligible = doc_eligible and not is_boiler  # C-3 ⑤ 별첨·서식은 검색 제외

        if c["kind"] == "table":
            body_search = table_search_text(c["content"])
            if c.get("lead_text"):
                # content(HTML)는 그대로 두고 검색 텍스트에만 얹는다.
                body_search = c["lead_text"] + "\n" + body_search
            # C-2 ② degraded 판정은 검색 대상 표에만 적용한다
            degraded = bool(eligible and c["blank_ratio"] is not None
                            and c["blank_ratio"] > degraded_th)
            location = (f"{c['path'][-1] if c['path'] else doc_title} · "
                        f"표 {c['table_idx']}"
                        + (f" ({c['part']}/{c['of']})" if c["of"] > 1 else ""))
        else:
            body_search = plain_search_text(c["content"])
            degraded = False
            location = para_label(c["path"], doc_title, c["para_start"], c["para_end"])

        out.append({
            "chunk_id": chunk_id,
            "document_id": doc_id,
            "document_version": reg_row["document_version"],
            "processed_sha256": reg_row["processed_sha256"],
            "sidecar_sha256": reg_row["sidecar_sha256"],
            "corpus_version": reg_row["corpus_version"],
            "preprocess_version": reg_row.get("preprocess_version"),
            "chunking_version": cfg["chunking_version"],
            "source_document_title": doc_title,
            "section_path": list(c["path"]),
            # 서로 다른 절을 병합한 경우 구성원 경로를 전부 남긴다.
            # section_path만 두면 3.1과 3.2를 합친 뒤 "3장"만 남아 근거 위치를 잃는다.
            "section_paths": [list(p) for p in c.get("member_paths", [c["path"]])],
            "location_label": location,
            "block_type": c["kind"],
            "table_idx": c.get("table_idx"),
            "row_start": c.get("row_start"),
            "row_end": c.get("row_end"),
            "part": c.get("part"),
            "of": c.get("of"),
            "table_blank_ratio": (round(c["blank_ratio"], 4)
                                  if c.get("blank_ratio") is not None else None),
            "table_degraded": degraded,
            "boilerplate_type": c.get("boiler_type"),
            "boilerplate_label": c.get("boiler_label"),
            "retrieval_eligible": eligible,
            "md_line_start": c["line_start"],       # 내부 검증용
            "md_line_end": c["line_end"],
            "content": c["content"],                # 표는 HTML 구조 유지
            "search_text": c["prefix"] + "\n" + body_search,
            "prefix": c["prefix"],
            "char_len": len(c["prefix"]) + 1 + len(body_search),
            "oversize": bool(c.get("oversize")),
        })

    doc_stat = {
        "chunks": len(out),
        "text": sum(1 for c in out if c["block_type"] == "text"),
        "table": sum(1 for c in out if c["block_type"] == "table"),
        "eligible": sum(1 for c in out if c["retrieval_eligible"]),
        "degraded": sum(1 for c in out if c["table_degraded"]),
        "oversize": sum(1 for c in out if c["oversize"]),
        # 소실 헤딩 복원 내역. unrecovered 는 문서 끝까지 붙일 곳이 없던 것.
        "heading_lost": len(lost_headings),
        "heading_to_text": recov["text"],
        "heading_to_table": recov["table"],
        "heading_unrecovered": len(pending_lead),
        "heading_skipped_boilerplate": recov["boilerplate"],
    }
    return out, doc_stat


# ─────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────

def pct(sorted_vals, p):
    if not sorted_vals:
        return None
    return sorted_vals[min(int(len(sorted_vals) * p), len(sorted_vals) - 1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-override", type=Path, default=None)
    ap.add_argument("--only", type=str, default=None,
                    help="쉼표로 구분한 document_id. 해당 문서의 청크만 다시 만든다 (C-5)")
    ap.add_argument("--skip-extraction-check", action="store_true",
                    help="추출표 대조를 건너뛴다. C-5 ③ 안전장치를 끄는 것이므로 "
                         "사유를 기록해야 한다")
    ap.add_argument("--verify-location", type=Path, default=None,
                    help="이전 chunks.jsonl 과 location_label·section_path 를 전수 "
                         "대조한다. 불일치가 있으면 산출물을 만들지 않고 중단 "
                         "(팀 결정 2026-09-03)")
    ap.add_argument("--verify-location-strict", action="store_true",
                    help="--verify-location 대조에 같은 라벨의 '개수'까지 포함한다")
    ap.add_argument("--tokenize", action="store_true",
                    help="토큰 길이 집계까지 수행 (tiktoken 필요)")
    ap.add_argument("--repo-root", type=Path, default=None)
    args = ap.parse_args()

    # argparse 로는 의존관계를 표현할 수 없어 여기서 막는다.
    if args.verify_location_strict and not args.verify_location:
        die("--verify-location-strict 는 --verify-location 과 함께 써야 합니다.")

    rag_root = os.environ.get("RAG_ROOT")
    if not rag_root:
        die("환경변수 RAG_ROOT가 없습니다. (규약 §2-3 — 절대 경로를 코드에 박지 않습니다)")
    rag_root = Path(rag_root)
    repo_root = args.repo_root or Path(__file__).resolve().parents[2]

    cfg = load_config(repo_root, args.config_override)
    gi = git_info(repo_root)

    processed = rag_root / "shared_data" / "processed"
    corpus_dir = processed / f"corpus_{cfg['corpus']}"
    md_dir = corpus_dir / "md"
    sidecar_dir = corpus_dir / "sidecar"
    registry_path = (processed / f"document_registry_{cfg['corpus']}"
                     / f"document_registry_{cfg['corpus']}.csv")
    table_dir = processed / f"rfp_extraction_table_{cfg['table']}"
    out_dir = processed / f"chunks_{cfg['chunking_version']}"

    for p in (md_dir, sidecar_dir, registry_path):
        if not p.exists():
            die(f"입력을 찾을 수 없습니다: {p}")

    registry = read_registry(registry_path)
    errors = []

    registry_dir = registry_path.parent
    if args.skip_extraction_check:
        # ⚠️ 안전장치를 끈 결과가 공식 폴더에 저장되면 안 된다.
        out_dir = processed / f"chunks_{cfg['chunking_version']}_unchecked"
        errors.append({"level": "warn",
                       "msg": "--skip-extraction-check — C-5 ③ 추출표 검증을 건너뜀. "
                              f"공식 폴더가 아닌 {out_dir.name} 에 저장한다"})
        ext_report = None
    else:
        ext_meta = load_extraction_metadata(table_dir, cfg)
        check_registry_metadata(registry_dir, cfg, ext_meta)
        ext_report = validate_extraction_table(table_dir, cfg, ext_meta, registry)
        print(f"추출표 검증: {ext_report['csv']} — {ext_report['rows']}행 / "
              f"문서 {ext_report['documents']}건 / "
              f"문서당 {ext_report['fields_per_document']}필드 · "
              f"schema {ext_meta['schema_version']} — 이상 없음")

    only = set(x.strip() for x in args.only.split(",")) if args.only else None
    targets = [r for r in registry if (only is None or r["document_id"] in only)]
    if only and len(targets) != len(only):
        die(f"--only 에 등록부에 없는 document_id가 있습니다: "
            f"{sorted(only - {r['document_id'] for r in targets})}")

    chunks_path = out_dir / "chunks.jsonl"
    existing = []
    if only:
        # ⚠️ 부분 갱신은 "이미 완전한 전체 파일이 있다"를 전제로 한다.
        #    전제를 확인하지 않으면 100문서짜리 자리에 1문서짜리가 생기거나,
        #    설정을 바꾼 뒤 한 문서만 갱신해 99개는 옛 설정인 혼합 파일이 된다.
        if not chunks_path.exists():
            die(f"부분 갱신할 기존 산출물이 없습니다: {chunks_path}\n"
                f"       --only 없이 전체 청킹을 먼저 실행하세요.")
        with chunks_path.open(encoding="utf-8") as f:
            existing = [json.loads(line) for line in f]

        prev_docs = {c["document_id"] for c in existing}
        reg_docs = {r["document_id"] for r in registry}
        if prev_docs != reg_docs:
            die(f"기존 산출물의 문서 집합이 등록부와 다릅니다 "
                f"(기존 {len(prev_docs)}건 / 등록부 {len(reg_docs)}건). "
                f"전체 청킹을 먼저 실행하세요.")

        # 설정이 바뀐 채 부분 갱신하면 한 파일 안에 두 설정의 청크가 섞인다.
        for key in ("chunking_version", "corpus_version"):
            want = cfg["chunking_version"] if key == "chunking_version" else cfg["corpus"]
            prev = {c.get(key) for c in existing}
            if prev != {want}:
                die(f"기존 산출물의 {key}가 현재 설정과 다릅니다 "
                    f"(기존 {sorted(prev)} / 현재 {want}). "
                    f"설정이 바뀌었다면 전체 청킹을 실행하세요.")

        # ⚠️ 버전 두 칸만 보면 chunk_size·table_chunk_threshold 등이 바뀌었는데
        #    버전 번호를 안 올린 경우를 못 잡는다. stats.json의 설정 전체와 대조한다.
        stats_path = out_dir / "stats.json"
        if not stats_path.exists():
            die(f"기존 stats.json이 없어 설정을 대조할 수 없습니다: {stats_path}\n"
                f"       전체 청킹을 먼저 실행하세요.")
        prev_cfg = json.loads(stats_path.read_text(encoding="utf-8")).get("config", {})
        changed = {k: (prev_cfg.get(k), cfg[k])
                   for k in REQUIRED_KEYS if prev_cfg.get(k) != cfg[k]}
        if changed:
            die("기존 산출물과 설정이 다릅니다. 부분 갱신하면 한 파일에 두 설정의 "
                "청크가 섞입니다.\n       " +
                " / ".join(f"{k}: {a} → {b}" for k, (a, b) in changed.items()) +
                "\n       전체 청킹을 실행하세요.")

    # ── location 대조 참조 파일 (팀 결정 2026-09-03)
    #    처리 루프보다 앞에서 읽는다. 경로나 버전이 틀린 실행이 100문서를 다 돌고
    #    나서 죽으면 몇 분을 버린다.
    prev_loc, prev_meta = None, None
    if args.verify_location:
        prev_loc, prev_meta = load_prev_locations(args.verify_location, cfg, errors)
        print(f"location 대조 참조: {args.verify_location}\n"
              f"  청크 {prev_meta['locations_prev']} / 문서 {prev_meta['documents_prev']}건 "
              f"· chunking {prev_meta['reference_chunking_version']} "
              f"· corpus {prev_meta['reference_corpus_version']}")

    started = datetime.now(timezone.utc)
    all_chunks, per_doc = [], {}
    for r in targets:
        chunks, stat = process_document(r, md_dir, sidecar_dir, cfg, errors)
        all_chunks.extend(chunks)
        per_doc[r["document_id"]] = stat

    out_dir.mkdir(parents=True, exist_ok=True)

    if only:
        # C-5 — 해당 document_id의 청크만 교체한다.
        kept = [c for c in existing if c["document_id"] not in only]
        merged = kept + all_chunks
        merged.sort(key=lambda c: (c["document_id"], c["chunk_id"]))
    else:
        merged = all_chunks

    chunks_this_run = len(all_chunks)

    # ⚠️ 오류가 있으면 기존 공식 파일을 건드리지 않는다.
    #    먼저 덮어쓰면 실패한 실행의 불완전한 결과가 공식 자리를 차지한다.
    hard_errors = [e for e in errors if e["level"] == "error"]
    if hard_errors:
        (out_dir / "errors.jsonl").write_text(
            "\n".join(json.dumps(e, ensure_ascii=False) for e in hard_errors) + "\n",
            encoding="utf-8")
        die(f"처리 오류 {len(hard_errors)}건. 기존 산출물을 보존하고 중단합니다. "
            f"errors.jsonl 확인.")

    # ── location 전수 대조 (팀 결정 2026-09-03)
    #    ⚠️ 반드시 .tmp 쓰기 '전에' 한다. .tmp 와 replace() 사이에서 죽으면 공식
    #       폴더에 chunks.jsonl.tmp 잔재가 남고, 2만 줄을 쓴 것도 헛일이 된다.
    #       --tokenize 앞이기도 해서 버릴 산출물에 tiktoken 을 돌리지 않는다.
    #    ⚠️ hard_errors 뒤여야 한다. 파싱 실패한 문서가 청크를 못 만든 것을
    #       "location 대규모 회귀"로 오해하게 만들면 안 된다.
    loc_report = None
    if prev_loc is not None:
        details, counts = compare_locations(
            prev_loc, index_locations(merged), only, args.verify_location_strict)
        blocking = list(BLOCKING_KINDS)
        if args.verify_location_strict:
            blocking.append("count_changed")
        n_block = sum(counts[k] for k in blocking)

        summary = dict(prev_meta)
        summary.update({
            "level": "error" if n_block else "info",
            "stage": "verify_location",
            "kind": "summary",
            "mode": "strict" if args.verify_location_strict else "set",
            "documents_compared": len(set(prev_loc) | {c["document_id"] for c in merged}),
            "locations_new": len(merged),
            "document_missing": counts["document_missing"],
            "document_added": counts["document_added"],
            "location_missing": counts["location_missing"],
            "location_added": counts["location_added"],
            "eligible_changed": counts["eligible_changed"],
            "count_changed": counts["count_changed"],
            "details_written": len(details),
            "details_truncated": sum(counts.values()) > len(details),
            "msg": (f"location 대조 불일치 {n_block}건 — 산출물을 만들지 않고 중단"
                    if n_block else "location 대조 이상 없음"),
        })
        errors.extend(details)
        errors.append(summary)

        if n_block:
            with (out_dir / "errors.jsonl").open("w", encoding="utf-8") as f:
                for e in errors:
                    f.write(json.dumps(e, ensure_ascii=False) + "\n")
            die(f"location 대조에서 불일치 {n_block}건이 나왔습니다 "
                f"(문서 {counts['document_missing'] + counts['document_added']} / "
                f"없어짐 {counts['location_missing']} / "
                f"새로 생김 {counts['location_added']} / "
                f"검색대상 변경 {counts['eligible_changed']}"
                + (f" / 개수 변화 {counts['count_changed']}"
                   if args.verify_location_strict else "") + "). "
                f"기존 산출물을 보존하고 중단합니다. "
                f"{out_dir / 'errors.jsonl'} 확인.", code=4)

        loc_report = {k: v for k, v in summary.items()
                      if k not in ("level", "stage", "kind", "msg")}
        loc_report["result"] = "ok"
        print(f"location 대조: {prev_meta['reference_chunking_version']} "
              f"{prev_meta['locations_prev']}청크 대조 — 이상 없음"
              + (f" (개수 변화 {counts['count_changed']}건 — 라벨 값은 그대로)"
                 if counts["count_changed"] else ""))

    written = len(merged)

    # ── 집계
    # ⚠️ 부분 갱신에서도 파일 전체를 기준으로 센다.
    #    이번에 처리한 문서만 세면 chunks.jsonl은 chunks.jsonl은 전체인데
    #    stats.json·VERSION.txt에는 232 같은 숫자가 남아 서로 다른 기록이 된다.
    all_chunks = merged
    lens = sorted(c["char_len"] for c in all_chunks)
    stats = {
        "generated_at": started.isoformat(),
        "elapsed_sec": round((datetime.now(timezone.utc) - started).total_seconds(), 1),
        "config": {k: cfg[k] for k in REQUIRED_KEYS},
        "embedding": {k: cfg.get(k) for k in OPTIONAL_KEYS},
        "extraction_check": ext_report,
        "location_verification": loc_report,
        "heading_recovery": {
            k: sum(s.get(k, 0) for s in per_doc.values())
            for k in ("heading_lost", "heading_to_text", "heading_to_table",
                      "heading_unrecovered", "heading_skipped_boilerplate")
        } if per_doc else None,
        "git": gi,
        "mode": "partial" if only else "full",
        "documents_in_file": len({c["document_id"] for c in all_chunks}),
        "documents_this_run": len(targets),
        "chunks_written_total": written,
        "chunks_this_run": chunks_this_run,
        "by_type": dict(Counter(c["block_type"] for c in all_chunks)),
        "retrieval_eligible": sum(1 for c in all_chunks if c["retrieval_eligible"]),
        "retrieval_excluded": sum(1 for c in all_chunks if not c["retrieval_eligible"]),
        "boilerplate_chunks": dict(Counter(
            c["boilerplate_type"] for c in all_chunks if c["boilerplate_type"])),
        "table_degraded": sum(1 for c in all_chunks if c["table_degraded"]),
        "oversize_chunks": sum(1 for c in all_chunks if c["oversize"]),
        "char_len": {
            "mean": round(sum(lens) / len(lens), 1) if lens else None,
            "median": pct(lens, 0.50), "p95": pct(lens, 0.95),
            "max": lens[-1] if lens else None,
        },
        "per_document": per_doc,
    }

    # ⚠️ 토큰 측정은 공식 파일 교체 '전에' 한다.
    #    나중에 하면 명령은 실패했는데 공식 파일은 이미 바뀌는 일이 생긴다.
    token_failed = False
    if args.tokenize:
        model = cfg.get("embedding_model")
        max_len = cfg.get("embedding_max_length")
        if not model or not max_len:
            die("--tokenize 를 쓰려면 base.yaml 의 embedding_model 과 "
                "embedding_max_length 가 채워져 있어야 합니다. "
                "임의 기본값으로 진행하지 않습니다.", code=3)
        stats["token_len"] = tokenize_stats(all_chunks, errors, model, int(max_len))
        # 명시적으로 요청한 측정이 실패했으면 성공으로 끝내지 않는다.
        token_failed = not stats["token_len"]

    if token_failed:
        (out_dir / "errors.jsonl").write_text(
            "\n".join(json.dumps(e, ensure_ascii=False) for e in errors) + "\n",
            encoding="utf-8")
        die("--tokenize 를 요청했으나 토큰 길이를 측정하지 못했습니다. "
            "기존 산출물을 보존하고 중단합니다. errors.jsonl 확인.", code=3)

    # 여기까지 오면 모든 검증이 끝났다. 이제 공식 파일을 교체한다.
    tmp_path = chunks_path.with_suffix(".jsonl.tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        for c in merged:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    tmp_path.replace(chunks_path)

    (out_dir / "stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    with (out_dir / "errors.jsonl").open("w", encoding="utf-8") as f:
        for e in errors:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    (out_dir / "VERSION.txt").write_text(render_version(cfg, gi, stats, registry_path,
                                                        corpus_dir, table_dir),
                                          encoding="utf-8")

    print(f"청크 {len(all_chunks)}개 생성 (파일 총 {written}개)")
    print(f"  일반 {stats['by_type'].get('text', 0)} / 표 {stats['by_type'].get('table', 0)}")
    print(f"  검색 대상 {stats['retrieval_eligible']} / 제외 {stats['retrieval_excluded']}")
    print(f"  degraded 표 {stats['table_degraded']} / 크기 초과 {stats['oversize_chunks']}")
    print(f"  길이(자) 중앙 {stats['char_len']['median']} p95 {stats['char_len']['p95']} "
          f"최대 {stats['char_len']['max']}")
    if stats.get("token_len"):
        tl = stats["token_len"]
        over = tl.get("over_limit")
        print(f"  토큰({tl['model']}) 중앙 {tl['median']} p95 {tl['p95']} 최대 {tl['max']}"
              + f" / {tl['max_input_length']}토큰 초과 {over}건")
    print(f"  소요 {stats['elapsed_sec']}초  →  {out_dir}")


def tokenize_stats(chunks, errors, model: str, max_len):
    """C-1 실행 체크리스트 — 실제 임베딩 입력 기준 토큰 길이.

    장절 접두를 포함한 search_text 로 잰다(C-3 ③).
    확정 모델이 OpenAI text-embedding-3-small 이므로 tiktoken 으로 센다.
    로컬 모델 가중치를 받지 않으므로 모델 캐시 권한과 무관하다.
    """
    try:
        import tiktoken
    except ImportError:
        errors.append({"level": "warn", "msg": "tiktoken 미설치 — 토큰 집계 생략"})
        return None
    try:
        enc = tiktoken.encoding_for_model(model)
    except Exception as e:                                     # noqa: BLE001
        errors.append({"level": "warn",
                       "msg": f"토크나이저 로드 실패 {model}: {e}"})
        return None

    lens = sorted(len(enc.encode(c["search_text"])) for c in chunks)
    if not lens:
        return None
    out = {
        "model": model,
        "mean": round(sum(lens) / len(lens), 1),
        "median": pct(lens, 0.50), "p95": pct(lens, 0.95), "max": lens[-1],
    }
    out["max_input_length"] = int(max_len)
    out["over_limit"] = sum(1 for x in lens if x > int(max_len))
    return out


def render_version(cfg, gi, stats, registry_path, corpus_dir, table_dir):
    lv = stats.get("location_verification")
    if lv:
        loc_line = (f"location 대조 : {lv['reference_chunking_version']} "
                    f"{lv['locations_prev']}청크 대조 이상 없음"
                    + (f" (개수 변화 {lv['count_changed']}건)"
                       if lv["count_changed"] else ""))
    else:
        loc_line = "location 대조 : 미수행 (--verify-location 미사용)"

    hr = stats.get("heading_recovery")
    if hr:
        head_line = (f"소실 헤딩 복원: {hr['heading_to_text'] + hr['heading_to_table']}건 "
                     f"(문단 {hr['heading_to_text']} / 표 search_text "
                     f"{hr['heading_to_table']}) · 미복원 {hr['heading_unrecovered']} "
                     f"· 별첨 제외 {hr['heading_skipped_boilerplate']}")
    else:
        head_line = "소실 헤딩 복원: 해당 없음"

    lines = [
        "# RFP 검색용 청크",
        f"chunking version   : {cfg['chunking_version']}",
        f"corpus version     : {cfg['corpus']}",
        f"preprocess version : {cfg['preprocess']}",
        f"table version      : {cfg['table']}",
        f"generated at       : {stats['generated_at']}",
        f"elapsed sec        : {stats['elapsed_sec']}",
        "",
        f"chunk_size              : {cfg['chunk_size']}   (baseline 시작값 — 최종 확정 아님)",
        f"chunk_overlap           : {cfg['chunk_overlap']}    (baseline 시작값 — 최종 확정 아님)",
        f"chunk_unit              : {cfg['chunk_unit']}",
        f"table_chunk_threshold   : {cfg['table_chunk_threshold']}",
        f"table_degraded_threshold: {cfg['table_degraded_threshold']}",
        f"table_format            : {cfg['table_format']}",
        f"table_empty_cell        : {cfg['table_empty_cell']}",
        "",
        f"git_commit  : {gi['git_commit']}",
        f"git_dirty   : {gi['git_dirty']}",
        f"git_branch  : {gi['git_branch']}",
        "",
        f"입력 코퍼스   : {corpus_dir}",
        f"입력 등록부   : {registry_path}",
        f"입력 추출표   : {table_dir}",
        "",
        f"문서 수       : {stats['documents_in_file']}" + (f"  (이번 실행 {stats['documents_this_run']}건)" if stats["mode"] == "partial" else ""),
        f"청크 수       : {stats['chunks_written_total']}" + (f"  (이번 실행 {stats['chunks_this_run']}개)" if stats["mode"] == "partial" else ""),
        f"실행 모드     : {stats['mode']}",
        f"검색 대상     : {stats['retrieval_eligible']}",
        f"검색 제외     : {stats['retrieval_excluded']}",
        f"degraded 표   : {stats['table_degraded']}",
        f"크기 초과     : {stats['oversize_chunks']}",
        (f"토큰 길이     : {stats['token_len']['model']} 중앙 "
         f"{stats['token_len']['median']} / p95 {stats['token_len']['p95']} / "
         f"최대 {stats['token_len']['max']}"
         if stats.get("token_len") else "토큰 길이     : 미측정 (--tokenize 미사용)"),
        loc_line,
        head_line,
        "",
        "⚠️ chunk_size·chunk_overlap 은 baseline 시작값이다.",
        "   최종 확정은 토큰 측정과 검색 평가 후 (C-1 · 4-7).",
        "",
        "적용 결정 (확정정리 C-1~C-5):",
        " - C-1 ① chunk_size 1500 (장절 접두 포함)",
        " - C-1 ② chunk_overlap 150",
        " - C-1 ④ 병합은 같은 장절 경로일 때만 (2026-09-01 개정)",
        " - C-1 ⑤ 표를 만나면 병합을 끊음",
        " - C-1 ⑥ 꼬리 구간은 문단 경계로 분할, 초과 문단은 그대로 두고 기록",
        " - C-2 ① 표 1500자 이하 통째 유지",
        " - C-2 ② degraded 판정은 검색 대상 표에만 적용",
        " - C-2 ③-a 머리글 첫 행 반복",
        " - C-2 ③-b 중첩 표는 바깥 기준으로만 분할",
        " - C-2 ③-c 한 행 초과는 그대로 두고 기록",
        " - C-2 ④ 검색 텍스트에서 빈 셀만 건너뜀 (행 구조 유지)",
        " - C-3 ① chunk_id = <document_id>-<4자리 순번>",
        " - C-3 ③ 접두 = 문서명 + 장절 전체 경로",
        " - C-3 ④ 문단 위치 = 절 안의 문단 순번",
        " - C-3 ⑤ 별첨·서식 구간은 청크 생성 후 검색 제외",
        " - C-5 ① 등록부 document_id 순서로 처리 (재현성)",
        " - C-5 ③ document_version 불일치 시 중단",
        " - 팀 결정 2026-09-03 location 전수 대조 (--verify-location)",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
