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
    python build_chunks.py --tokenize                        # 토큰 길이 집계까지

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
from datetime import datetime, timezone
from pathlib import Path

# ─────────────────────────────────────────────────────────────
# 설정 로드
# ─────────────────────────────────────────────────────────────

# 코드가 읽는 설정 키. base.yaml에 없으면 중단한다(값을 지어내지 않는다).
REQUIRED_KEYS = [
    "corpus", "preprocess", "table",
    "chunk_size", "chunk_overlap", "chunking_version", "chunk_unit",
    "table_chunk_threshold", "table_degraded_threshold",
    "table_format", "table_empty_cell",
]


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


def read_extraction_versions(table_dir: Path) -> dict:
    """추출표에서 document_id → document_version 을 뽑는다.

    컬럼이 없으면 경고만 남기고 계속한다(비교할 대상이 없으므로).
    """
    if not table_dir.exists():
        return {}
    for cand in sorted(table_dir.glob("*.csv")):
        try:
            with cand.open(newline="", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                if not reader.fieldnames:
                    continue
                if "document_id" in reader.fieldnames and "document_version" in reader.fieldnames:
                    out = {}
                    for r in reader:
                        out[r["document_id"]] = r["document_version"]
                    if out:
                        return out
        except Exception:
            continue
    return {}


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# ─────────────────────────────────────────────────────────────
# 블록 분해 — 헤딩 / 표 / 문단
# ─────────────────────────────────────────────────────────────

HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(.*\S)\s*$")
TABLE_OPEN_RE = re.compile(r"<table\b", re.I)
TABLE_CLOSE_RE = re.compile(r"</table\s*>", re.I)
TR_SPLIT_RE = re.compile(r"(?=<tr\b)", re.I)
CELL_RE = re.compile(r"<t[dh]\b[^>]*>(.*?)</t[dh]\s*>", re.S | re.I)
TAG_RE = re.compile(r"<[^>]+>")
PIPE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")


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


def table_rows(table_html: str):
    """<tr> 단위로 자른다. 첫 조각(<table ...> 머리)은 헤더 앞부분으로 붙인다."""
    parts = [p for p in TR_SPLIT_RE.split(table_html) if p.strip()]
    if not parts:
        return [], ""
    if not parts[0].lstrip().lower().startswith("<tr"):
        return parts[1:], parts[0]
    return parts, ""


def table_blank_ratio(table_html: str):
    cells = CELL_RE.findall(table_html)
    if not cells:
        return None, 0, 0
    blank = sum(1 for c in cells if cell_is_blank(c))
    return blank / len(cells), blank, len(cells)


def table_search_text(table_html: str) -> str:
    """검색용 텍스트.

    C-2 ④ — 빈 셀만 건너뛰고 행 구조는 유지한다.
    HTML 원본은 수정하지 않는다(이 함수는 별도 텍스트를 만든다).
    """
    lines = []
    rows, _ = table_rows(table_html)
    for row in rows:
        cells = CELL_RE.findall(row)
        vals = []
        for c in cells:
            if cell_is_blank(c):
                continue                        # 빈 셀만 건너뜀
            vals.append(re.sub(r"\s+", " ", TAG_RE.sub(" ", c)).strip())
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

def build_prefix(doc_title: str, path: tuple) -> str:
    """C-3 ③ — 문서명 + 전체 장절 경로. 기계 조립, LLM 생성 없음."""
    return " > ".join([doc_title] + list(path))


def para_label(path: tuple, doc_title: str, start: int, end: int) -> str:
    """C-3 ④ — 절 안의 문단 순번. 헤딩이 없으면 문서명을 0단 절로 쓴다."""
    where = path[-1] if path else doc_title
    if start == end:
        return f"{where} · 문단 {start}"
    return f"{where} · 문단 {start}-{end}"


def can_merge(path_a: tuple, path_b: tuple) -> bool:
    """C-1 ④ — 같은 상위 헤딩 아래에서만 병합.

    3.1과 3.2는 합치고, 3장과 4장은 합치지 않는다.
    """
    if path_a == path_b:
        return True
    if len(path_a) >= 2 and len(path_b) >= 2 and path_a[:-1] == path_b[:-1]:
        return True
    return False


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
                pieces.append("\n\n".join(cur))
                cur = []
            pieces.append(p)                      # 자르지 않는다
            oversize += 1
            continue
        cand = ("\n\n".join(cur + [p])) if cur else p
        if len(cand) <= budget:
            cur = cur + [p]
        else:
            pieces.append("\n\n".join(cur))
            tail = pieces[-1][-overlap:] if overlap > 0 else ""
            cur = ([tail, p] if tail else [p])
    if cur:
        pieces.append("\n\n".join(cur))
    return pieces, oversize


def split_table(table_html: str, budget: int):
    """C-2 — 행 경계로 분할하고 머리글을 반복한다.

    part/of는 글자 수가 아니라 행 경계로 계산한다.
    한 행이 budget을 넘으면 자르지 않고 초과를 기록한다.
    """
    rows, head = table_rows(table_html)
    if not rows:
        return [(table_html, 0, 0, 1, 1)], 0
    header = rows[0]
    body = rows[1:]
    if not body:
        return [(table_html, 0, 0, 1, 1)], 0

    parts, cur, cur_start, oversize = [], [], 1, 0
    base = len(head) + len(header) + len("</table>")
    for idx, row in enumerate(body, start=1):
        if base + len(row) > budget and not cur:
            parts.append(([row], idx, idx))       # 한 행이 초과 — 그대로 둔다
            oversize += 1
            cur_start = idx + 1
            continue
        cand_len = base + sum(len(r) for r in cur) + len(row)
        if cand_len <= budget or not cur:
            cur.append(row)
        else:
            parts.append((cur, cur_start, cur_start + len(cur) - 1))
            cur, cur_start = [row], idx
    if cur:
        parts.append((cur, cur_start, cur_start + len(cur) - 1))

    of = len(parts)
    out = []
    for i, (rowset, r0, r1) in enumerate(parts, start=1):
        html = head + header + "".join(rowset)
        if not html.rstrip().lower().endswith("</table>"):
            html += "</table>"
        out.append((html, r0, r1, i, of))
    return out, oversize


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

    # 장절 경로 스택과 절 안 문단 순번
    stack = []                                   # [(level, title)]
    para_no = defaultdict(int)
    table_idx = 0
    units = []                                   # 청크 후보 단위

    for kind, body, ln0, ln1 in blocks:
        if kind == "heading":
            m = HEADING_RE.match(body)
            level, title = len(m.group(1)), m.group(2).strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            continue

        path = tuple(t for _, t in stack)
        b_type, b_label = boiler_of(ln0)

        if kind in ("table", "pipe_table"):
            table_idx += 1
            units.append({
                "kind": "table", "html": body, "path": path,
                "line_start": ln0, "line_end": ln1,
                "table_idx": table_idx, "boiler_type": b_type, "boiler_label": b_label,
            })
        else:
            n_paras = len([p for p in re.split(r"\n\s*\n", body) if p.strip()])
            if n_paras == 0:
                continue
            start_no = para_no[path] + 1
            para_no[path] += n_paras
            units.append({
                "kind": "text", "text": body, "path": path,
                "line_start": ln0, "line_end": ln1,
                "para_start": start_no, "para_end": para_no[path],
                "boiler_type": b_type, "boiler_label": b_label,
            })

    # ── 병합: 표를 만나면 끊고(C-1 ⑤), 같은 상위 헤딩 아래에서만 합친다(C-1 ④)
    chunks_raw = []
    group = []

    def flush_group():
        nonlocal group
        if not group:
            return
        paths = [u["path"] for u in group]
        path = common_prefix(paths) if len(set(paths)) > 1 else paths[0]
        prefix = build_prefix(doc_title, path)
        budget = max(size - len(prefix) - 1, 200)
        text = "\n\n".join(u["text"] for u in group)
        pieces, oversize = split_by_paragraph(text, budget, overlap)
        for i, piece in enumerate(pieces):
            chunks_raw.append({
                "kind": "text", "path": path, "content": piece,
                "line_start": group[0]["line_start"], "line_end": group[-1]["line_end"],
                "para_start": group[0]["para_start"], "para_end": group[-1]["para_end"],
                "boiler_type": group[0]["boiler_type"], "boiler_label": group[0]["boiler_label"],
                "oversize": 1 if (len(piece) > budget) else 0,
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
                parts = [(u["html"], 1, len(table_rows(u["html"])[0]), 1, 1)]
                t_over = 0
            else:
                parts, t_over = split_table(u["html"], budget)
            for html, r0, r1, part, of in parts:
                chunks_raw.append({
                    "kind": "table", "path": path, "content": html,
                    "line_start": u["line_start"], "line_end": u["line_end"],
                    "table_idx": u["table_idx"], "row_start": r0, "row_end": r1,
                    "part": part, "of": of,
                    "blank_ratio": ratio, "blank_cells": blank, "total_cells": total,
                    "boiler_type": u["boiler_type"], "boiler_label": u["boiler_label"],
                    "oversize": t_over if of == 1 else 0,
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
    ap.add_argument("--tokenize", action="store_true",
                    help="토큰 길이 집계까지 수행 (transformers 필요)")
    ap.add_argument("--repo-root", type=Path, default=None)
    args = ap.parse_args()

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
    extraction_versions = read_extraction_versions(table_dir)
    errors = []

    if not extraction_versions:
        errors.append({"level": "warn",
                       "msg": "추출표에서 document_version을 찾지 못해 대조를 건너뜁니다"})

    # C-5 ③ 중단 조건 — document_version 대조
    mismatches = []
    for r in registry:
        ev = extraction_versions.get(r["document_id"])
        if ev is not None and str(ev) != str(r["document_version"]):
            mismatches.append({"document_id": r["document_id"],
                               "registry": r["document_version"], "extraction": ev})
    if mismatches:
        out_dir.mkdir(parents=True, exist_ok=True)
        with (out_dir / "errors.jsonl").open("w", encoding="utf-8") as f:
            for m in mismatches:
                f.write(json.dumps({"level": "error", "msg": "document_version 불일치",
                                    **m}, ensure_ascii=False) + "\n")
        die(f"등록부와 추출표의 document_version이 다릅니다 ({len(mismatches)}건). "
            f"errors.jsonl 확인.")

    only = set(x.strip() for x in args.only.split(",")) if args.only else None
    targets = [r for r in registry if (only is None or r["document_id"] in only)]
    if only and len(targets) != len(only):
        die(f"--only 에 등록부에 없는 document_id가 있습니다: "
            f"{sorted(only - {r['document_id'] for r in targets})}")

    started = datetime.now(timezone.utc)
    all_chunks, per_doc = [], {}
    for r in targets:
        chunks, stat = process_document(r, md_dir, sidecar_dir, cfg, errors)
        all_chunks.extend(chunks)
        per_doc[r["document_id"]] = stat

    out_dir.mkdir(parents=True, exist_ok=True)
    chunks_path = out_dir / "chunks.jsonl"

    if only:
        # C-5 — 해당 document_id의 청크만 교체한다.
        kept = []
        if chunks_path.exists():
            with chunks_path.open(encoding="utf-8") as f:
                for line in f:
                    obj = json.loads(line)
                    if obj["document_id"] not in only:
                        kept.append(obj)
        merged = kept + all_chunks
        merged.sort(key=lambda c: (c["document_id"], c["chunk_id"]))
        with chunks_path.open("w", encoding="utf-8") as f:
            for c in merged:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
        written = len(merged)
    else:
        with chunks_path.open("w", encoding="utf-8") as f:
            for c in all_chunks:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
        written = len(all_chunks)

    # ── 집계
    lens = sorted(c["char_len"] for c in all_chunks)
    stats = {
        "generated_at": started.isoformat(),
        "elapsed_sec": round((datetime.now(timezone.utc) - started).total_seconds(), 1),
        "config": {k: cfg[k] for k in REQUIRED_KEYS},
        "git": gi,
        "documents_processed": len(targets),
        "chunks_written_total": written,
        "chunks_this_run": len(all_chunks),
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

    if args.tokenize:
        stats["token_len"] = tokenize_stats(all_chunks, errors)

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
    print(f"  소요 {stats['elapsed_sec']}초  →  {out_dir}")
    err_n = sum(1 for e in errors if e["level"] == "error")
    if err_n:
        print(f"[경고] error {err_n}건 — errors.jsonl 확인", file=sys.stderr)
        sys.exit(2)


def tokenize_stats(chunks, errors):
    """C-1 실행 체크리스트 — 512·8,192 두 기준 모두로 초과 건수를 센다.

    장절 접두를 포함한 search_text 기준으로 잰다(C-3 ③).
    """
    try:
        from transformers import AutoTokenizer
    except ImportError:
        errors.append({"level": "warn", "msg": "transformers 미설치 — 토큰 집계 생략"})
        return None

    out = {}
    for name in ("BAAI/bge-m3", "nlpai-lab/KURE-v1"):
        try:
            tok = AutoTokenizer.from_pretrained(name)
        except Exception as e:
            errors.append({"level": "warn", "msg": f"토크나이저 로드 실패 {name}: {e}"})
            continue
        lens = sorted(len(tok(c["search_text"], add_special_tokens=True)["input_ids"])
                      for c in chunks)
        out[name] = {
            "mean": round(sum(lens) / len(lens), 1) if lens else None,
            "median": pct(lens, 0.50), "p95": pct(lens, 0.95),
            "max": lens[-1] if lens else None,
            "over_512": sum(1 for x in lens if x > 512),
            "over_8192": sum(1 for x in lens if x > 8192),
        }
    return out


def render_version(cfg, gi, stats, registry_path, corpus_dir, table_dir):
    lines = [
        "# RFP 검색용 청크",
        f"chunking version   : {cfg['chunking_version']}",
        f"corpus version     : {cfg['corpus']}",
        f"preprocess version : {cfg['preprocess']}",
        f"table version      : {cfg['table']}",
        f"generated at       : {stats['generated_at']}",
        f"elapsed sec        : {stats['elapsed_sec']}",
        "",
        f"chunk_size              : {cfg['chunk_size']}",
        f"chunk_overlap           : {cfg['chunk_overlap']}",
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
        f"문서 수       : {stats['documents_processed']}",
        f"청크 수       : {stats['chunks_written_total']}",
        f"검색 대상     : {stats['retrieval_eligible']}",
        f"검색 제외     : {stats['retrieval_excluded']}",
        f"degraded 표   : {stats['table_degraded']}",
        f"크기 초과     : {stats['oversize_chunks']}",
        "",
        "적용 결정 (확정정리 C-1~C-5):",
        " - C-1 ① chunk_size 1500 (장절 접두 포함)",
        " - C-1 ② chunk_overlap 150",
        " - C-1 ④ 병합은 같은 상위 헤딩 아래에서만",
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
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
