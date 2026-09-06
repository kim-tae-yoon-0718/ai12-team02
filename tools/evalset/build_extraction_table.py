#!/usr/bin/env python3
#@title 1-12-2 추출 테이블 생성기 — 문서 100건 × 필드 12개 = 1,200행
#@markdown 공식 전처리 문서·sidecar·문서 등록부·원본 메타데이터를 **읽기만** 해서
#@markdown `extraction_table_v4.csv/.json` 을 만든다. 원본과 공식 자료는 절대 건드리지 않는다.
#@markdown
#@markdown 추출은 두 갈래를 합친다.
#@markdown   (1) 규칙: 항목명·동의어를 찾아 값을 떼어낸다 (코드가 한다)
#@markdown   (2) 의미 판단: 규칙이 애매한 행만 **고정된 결정 파일**에서 읽어 덮어쓴다
#@markdown 같은 결정 파일이면 결과 CSV·JSON 이 매번 바이트까지 같다(생성 시각은 본문에 넣지 않는다).
#@markdown
#@markdown 실행: python3 build_extraction_table.py [--out 폴더]

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import unicodedata
from datetime import datetime, timezone
import os as _os
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import extraction_rules_v3 as R          # 순수 규칙 (파일을 모른다)

# ── 공식 입력 (전부 읽기 전용) ────────────────────────────────────
REGISTRY_DIR = Path("/srv/rfp/shared_data/processed/document_registry_v2")
CORPUS_DIR = Path("/srv/rfp/shared_data/processed/corpus_v2")
METADATA_CSV = Path("/srv/rfp/shared_data/raw/data_list.csv")
# ★[2026-09-04 §13] 개인 계정 절대경로를 없앴다. 보정 CSV·규칙 YAML 은 저장소가 아니라
#   산출물 폴더에 있으므로 **반드시 인자로 받는다**(없으면 어느 파일이 없는지 찍고 멈춘다).
_CAL_ENV, _RULES_ENV = "RFP_CALIBRATION_CSV", "RFP_RULES_YAML"
CALIBRATION_CSV = Path(_os.environ.get(_CAL_ENV, "")) if _os.environ.get(_CAL_ENV) else None
RULES_YAML = Path(_os.environ.get(_RULES_ENV, "")) if _os.environ.get(_RULES_ENV) else None

SCHEMA_VERSION = "1-12-2/v3"  # 스키마는 v3 그대로(열 구조 불변)
EXTRACTION_VERSION = "v4"
PENDING_STATUSES = frozenset({"extraction_failed", "review_required"})

# 결정 파일 — 이 세션에서 내린 의미 판단을 **고정**해 둔 자료
DECISIONS_CSV = "semantic_decisions_v4.csv"
DECISION_FIELDS = ["document_id", "field_name", "status", "answer_raw", "answer_normalized",
                   "matched_expression", "representative_source_type",
                   "representative_location", "source_excerpt_redacted",
                   "additional_locations", "confidence", "review_reason", "decision_note",
                   "source_review_completed", "source_review_method", "reviewer",
                   "source_context_note", "discarded_candidates", "discarded_reason"]

# 최종 테이블 열
COLUMNS = [
    "schema_version", "extraction_version", "corpus_version", "registry_version",
    "document_id", "document_version", "source_filename_nfc", "active", "retrieval_eligible",
    "field_name", "status", "answer_raw", "answer_normalized", "matched_expression",
    "representative_source_type", "representative_location", "source_excerpt_redacted",
    "additional_locations", "extraction_method", "confidence", "review_reason",
    "processed_sha256", "rules_sha256",
]

MAX_BLOCK_LINES = 40        # 값 블록으로 볼 최대 줄 수
MAX_VALUE_CHARS = 1500      # 값 문자열 상한 (원문 의미를 해치지 않는 선)
MAX_EXCERPT_CHARS = 400     # 근거 발췌 상한
FRONT_RATIO = 0.12          # 앞머리로 볼 문서 앞부분 비율


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def jdump(value) -> str:
    """CSV 칸에 목록·객체를 넣을 때 쓰는 표준 JSON 문자열(정렬 고정)."""
    if value in ("", None, [], {}):
        return ""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


# ══════════════════════════════════════════════════════════════════
# 1. 입력 읽기
# ══════════════════════════════════════════════════════════════════

def load_registry() -> list[dict]:
    with (REGISTRY_DIR / "document_registry_v2.csv").open(encoding="utf-8-sig",
                                                            newline="") as fh:
        return [dict(r) for r in csv.DictReader(fh)]


def load_metadata() -> dict[str, dict]:
    """원본 메타데이터를 {파일명 stem(NFC): 행} 으로. 공고일의 단일 공식 출처다."""
    out = {}
    with METADATA_CSV.open(encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            name = (row.get("파일명") or "").strip()
            if name:
                out[R.nfc(os.path.splitext(name)[0])] = row
    return out


def load_sidecar(name: str) -> dict:
    p = CORPUS_DIR / "sidecar" / (name + ".sections.json")
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def load_decisions(path: Path) -> dict[tuple[str, str], dict]:
    """고정된 의미 판단. 없으면 규칙 결과만 쓴다(판단을 지어내지 않는다)."""
    if not path.exists():
        return {}
    with path.open(encoding="utf-8-sig", newline="") as fh:
        rows = [{k: (v or "").strip() for k, v in r.items()}
                for r in csv.DictReader(fh) if r.get("document_id")]
    out = {}
    for row in rows:
        key = (row["document_id"], row["field_name"])
        if key in out:
            raise SystemExit(f"결정 파일 중복 키: {key[0]} / {key[1]}")
        if row.get("source_review_completed", "").lower() != "true":
            raise SystemExit(f"원문 검토 완료가 아닌 결정을 적용할 수 없음: {key[0]} / {key[1]}")
        if not row.get("source_review_method") or not row.get("source_context_note"):
            raise SystemExit(f"원문 검토 방법·문맥이 없는 결정: {key[0]} / {key[1]}")
        out[key] = row
    return out


# ══════════════════════════════════════════════════════════════════
# 2. 문서 안에서 필드 위치 찾기
# ══════════════════════════════════════════════════════════════════

TABLE_LINE = re.compile(r"^\s*(\||<t[dhr]|<table)")
HEADING = re.compile(r"^\s{0,3}#{1,6}\s")
MARKDOWN_HEADING = re.compile(r"^\s{0,3}(#{1,6})\s+(.*?)\s*#*\s*$")

# 전처리된 문서에는 원본 표에서 복원된 제목 일부가 Markdown ``#`` 없이
# 남아 있다. 계층 구간을 끊을 때 이 제목들도 함께 알아보되, ``1. 사업명: 값``
# 같은 실제 값 줄은 제목으로 보지 않는다.
ROMAN_NUMBER = "ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩIVXLCDM"
PSEUDO_HEAD = [
    (re.compile(rf"^\s*(?:제\s*\d+\s*장|[{ROMAN_NUMBER}]+[.)])\s*\S"), 1),
    # ``1.`` 아래 ``가.`` 아래 ``1)`` 순서를 서로 다른 단계로 본다.
    (re.compile(r"^\s*\d+(?:\.\d+)*\.\s*\S"), 2),
    # 표 복원 과정에서 ``4. 기대효과``의 점만 사라지는 경우가 있다.
    (re.compile(r"^\s*\d+(?:\.\d+)*\s+[가-힣A-Za-z]\S*"), 2),
    (re.compile(r"^\s*[가-하]\.\s*\S"), 3),
    (re.compile(r"^\s*\d+\)\s*\S"), 4),
    (re.compile(r"^\s*\(\d+\)\s*\S"), 4),
    (re.compile(r"^\s*[가-하]\)\s*\S"), 5),
    (re.compile(r"^\s*[①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮]\s*\S"), 5),
]


def structured_heading_level(text: str) -> int | None:
    """장/절 번호 자체가 나타내는 계층(마크다운 깊이보다 우선)."""
    if len(text) > 120 or re.search(r"[:：]|[.!?。]$", text):
        return None
    for pat, level in PSEUDO_HEAD:
        if pat.match(text):
            return level
    return None


def heading_level(raw: str) -> int | None:
    """Markdown 제목 또는 짧은 복원 제목의 계층 단계를 돌려준다."""
    m = MARKDOWN_HEADING.match(raw)
    if m:
        # PDF의 글머리표가 Markdown 제목으로 승격된 경우다. ``### ㅇ ...``와
        # ``## ☐ ...``는 부모 제목의 하위 내용이지 새 동급 절이 아니다.
        if re.match(r"\s*[-◦ㅇ○●□■▢▣☐☑❍※]\s*", m.group(2)):
            return None
        # 변환기가 같은 번호 단계에 서로 다른 ``#`` 깊이를 붙이는 경우가
        # 있으므로 장/절 번호가 있으면 Markdown 깊이보다 번호 계층을 쓴다.
        title = strip_markup(m.group(2))
        numbered = structured_heading_level(title)
        if numbered is not None:
            return numbered
        return len(m.group(1))
    # 장 제목이던 표 행을 전처리가 그대로 보존한 경우. 이 행 아래의 ``#``
    # 불릿들을 자식으로 읽고 다음 번호 표 행에서 멈추기 위해 0단계로 둔다.
    if re.search(r"<tr><t[hd][^>]*>\s*\d+\s*</t[hd]>.*<t[hd][^>]*>[^<]{2,80}</t[hd]>",
                 raw, flags=re.IGNORECASE):
        return 0
    plain = strip_markup(raw)
    # 값 구분자가 있거나 완전한 문장인 줄은 번호가 있어도 제목이 아니다.
    if len(plain) > 120 or re.search(r"[:：]", plain) or re.search(r"[.!?。]$", plain):
        return None
    return structured_heading_level(plain)


# 뒤쪽 첨부·서식 묶음이 시작되는 제목 (붙임/별첨/별지/별표/서식/양식/참고자료)
ATTACH_HEAD = re.compile(
    r"^\s{0,3}#{1,6}\s*.{0,8}?(붙\s*임|별\s*첨|별\s*지|별\s*표|서\s*식|양\s*식|참고\s*자료)")
# 문서 앞쪽에 있는 서식 언급은 경계가 아니다. 뒤쪽 40% 이후만 인정한다.
ATTACH_MIN_RATIO = 0.40


def attachment_start(side: dict, total: int, lines: list[str] | None = None) -> int:
    """뒤쪽 첨부·빈 서식이 시작되는 줄. 그 뒤는 보조 출처로만 쓴다.

    sidecar 의 sections 는 '첨부 경계'가 아니라 빈칸 서식이 흩어져 있는 위치이므로
    그 최솟값을 경계로 쓰면 본문 앞부분까지 첨부로 잘못 판정된다.
    실제 경계는 문서 자신의 '붙임/별지/서식' 제목에서 찾는다.
    """
    floor = int(total * ATTACH_MIN_RATIO)
    if lines:
        for i, raw in enumerate(lines):
            if i >= floor and ATTACH_HEAD.match(raw):
                return i + 1
    # 제목이 없으면 뒤쪽 절반에 몰려 있는 빈칸 서식 구간을 보조로 쓴다
    marks = [x["start_line"] for x in side.get("sections", [])
             if x.get("start_line") and x["start_line"] >= total * 0.5]
    if len(marks) >= 3:
        return min(marks)
    return total + 1


def source_type_of(idx: int, line: str, att_start: int, front_limit: int) -> str:
    """이 위치가 어떤 종류의 출처인가."""
    if idx >= att_start:
        return "attachment_form"
    if TABLE_LINE.match(line):
        return "body_table"
    if idx <= front_limit:
        return "front_matter"
    return "body_sentence"


def strip_markup(text: str) -> str:
    """표 기호·태그를 걷어내 사람이 읽을 문장으로 만든다(값 자체는 바꾸지 않는다)."""
    t = re.sub(r"<[^>]{1,200}>", " ", text)
    t = t.replace("|", " ").replace("\t", " ")
    t = re.sub(r"^[\s#>*\-–—·ㆍ□■▢▣○◯〇◦⚬●❍◉◎▪▫◆◇☐☑✓✔▶※ㅇ]+", "", t)
    return re.sub(r"\s{2,}", " ", t).strip()


def looks_like_toc_heading(raw: str, line_number: int) -> bool:
    """문서 앞쪽의 목차 제목을 실제 본문 장·절로 저장하지 않는다."""
    if line_number > 140:
        return False
    text = strip_markup(raw)
    if re.search(r"(?:목\s*차|차\s*례)$", text):
        return True
    return bool(re.search(r"(?:\t|\s{2,}|\]\s*)\d{1,4}\s*\**$", raw))


def build_section_index(lines: list[str]) -> list[dict]:
    """각 원문 줄에 실제 장·절 경로와 문단 위치를 붙인다.

    답을 찾은 문장과 장·절 제목은 서로 다른 정보다. 검색 규칙은 기존의
    ``heading``(근거 문장)을 계속 쓸 수 있지만, 사용자에게 보여줄 위치에는
    이 함수가 만든 실제 제목 경로만 저장한다.
    """
    stack: list[dict] = []
    result: list[dict] = []
    block_index = 0
    previous_blank = True
    for line_number, raw in enumerate(lines, start=1):
        plain = strip_markup(raw)
        markdown_heading = MARKDOWN_HEADING.match(raw) is not None
        level = heading_level(raw)
        toc_like = bool(level is not None and looks_like_toc_heading(raw, line_number))
        if level is not None and not toc_like:
            # 표의 첫 열을 임시 부모 제목(level 0)으로 쓴 뒤 실제 Markdown
            # 장·절이 시작되면 그 임시 부모는 끝난 것이다. 남겨 두면 수백 줄
            # 전의 요구사항 표 제목이 뒤쪽 제안 안내 절의 부모로 잘못 붙는다.
            stack = [item for item in stack
                     if item["level"] < level
                     and not (markdown_heading and item["level"] == 0)]
            stack.append({"level": level, "line": line_number, "title": plain})
            block_index = 0
        elif level is not None and toc_like and re.search(r"(?:목\s*차|차\s*례)$", plain):
            stack = []
            block_index = 0
        elif plain and previous_blank:
            block_index += 1

        if level is not None:
            block_type = "heading"
        elif TABLE_LINE.match(raw):
            block_type = "table"
        else:
            block_type = "paragraph"
        result.append({
            "section_path": [dict(item) for item in stack],
            "section_heading": stack[-1]["title"] if stack else "",
            "block_type": block_type,
            "block_index": block_index,
        })
        previous_blank = not bool(plain)
    return result


def location_payload(hit: dict) -> dict:
    """한 근거 후보를 사용자용 위치 JSON 구조로 바꾼다."""
    line = int(hit.get("line") or 0)
    # level 0은 표의 첫 열을 검색용 임시 부모로 사용한 값이지 실제 장·절이
    # 아니다. 사용자용 위치에는 실제 제목(level 1 이상)만 남긴다.
    visible_path = [item for item in hit.get("section_path", [])
                    if int(item.get("level") or 0) > 0]
    location_heading = (visible_path[-1].get("title", "") if line > 0 and visible_path
                        else hit.get("heading", "") if line <= 0 else "")
    payload = {
        "source_type": hit.get("source_type", ""),
        "line": line,
        "line_start": int(hit.get("line_start") or line),
        "line_end": int(hit.get("line_end") or line),
        "heading": R.redact(location_heading)[:120],
        "section_path": [
            {"level": item.get("level"), "line": item.get("line"),
             "title": R.redact(item.get("title", ""))[:120]}
            for item in visible_path
        ],
        "block_type": hit.get("block_type", "") or ("metadata" if line == 0 else ""),
        "block_index": hit.get("block_index", 0),
    }
    return payload


def enrich_saved_location(raw, section_index: list[dict], document_id: str,
                          filename: str, source_type: str = ""):
    """고정 의미 결정에 저장된 옛 위치도 현재 위치 구조로 다시 만든다.

    예전 결정 파일은 답 문장을 ``heading``으로 저장했다. 그 값을 그대로
    복사하지 않고 줄 번호를 기준으로 실제 장·절 경로를 다시 붙인다.
    """
    if raw in ("", None, [], {}):
        return ""
    value = raw
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return raw
    if isinstance(value, list):
        return [enrich_saved_location(item, section_index, document_id, filename)
                for item in value]
    if not isinstance(value, dict):
        return value
    line = int(value.get("line_start") or value.get("line") or 0)
    if line <= 0:
        return {
            **value, "document_id": document_id, "file": filename,
            "line": 0, "line_start": 0, "line_end": 0,
            "section_path": [], "block_type": "metadata", "block_index": 0,
        }
    section = section_index[line - 1] if line <= len(section_index) else {}
    hit = {
        "source_type": value.get("source_type") or source_type,
        "line": line,
        "line_start": line,
        "line_end": int(value.get("line_end") or line),
        "section_heading": section.get("section_heading", ""),
        "section_path": section.get("section_path", []),
        "block_type": section.get("block_type", ""),
        "block_index": section.get("block_index", 0),
    }
    return {"document_id": document_id, "file": filename, **location_payload(hit)}


# 항목명 앞에 올 수 있는 장식(글머리표·번호·표 기호)만 모은 것
DECOR_ONLY = re.compile(r"^[\s#>*\-–—·ㆍ□■▢▣○◯〇◦⚬●❍◉◎▪▫◆◇☐☑✓✔▶※ㅇ\|:：\(\)\[\]【】<>0-9a-zA-Z."
                        r"ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ가나다라마바사아자차카타파하①②③④⑤⑥⑦⑧⑨⑩]*$")


# 값이 아니라 표의 '열 이름 줄'인지 본다(연번/사업명/계약금액 … 처럼 항목명만 나열)
HEADER_WORDS = re.compile(
    r"(연번|번호|구\s*분|항\s*목|사업명|계약기간|계약금액|발주자|수급형태|지분율|"
    r"이행완료|비\s*고|순\s*번|평가항목|배\s*점|세부내용)")
# 값이 아니라 법령·예규를 그대로 옮겨 적은 줄
STATUTE_LINE = re.compile(
    r"(^[\s]*[①-⑮]|제\s*\d+\s*조|제\s*\d+\s*항|시행령|시행규칙|예규|고시)\s*")


def looks_header_row(value: str, matched: str, heading: str = "") -> bool:
    """이 값이 실제 내용이 아니라 표의 열 이름 줄인가.

    항목명 뒤쪽만 보면 '사업분야 | 계약기간 | 계약금액' 처럼 잘려서
    열 이름 줄인 줄 모른다. 그래서 줄 전체를 함께 본다.
    """
    whole = f"{heading} {value}" if heading else value
    words = HEADER_WORDS.findall(whole)
    # 항목명 종류가 3가지 이상 늘어서 있고 숫자 값이 거의 없으면 머리글로 본다
    return len(set(words)) >= 3 and len(re.findall(r"\d", whole)) <= 2


def looks_statute(value: str) -> bool:
    """이 값이 발주 조건이 아니라 법령 조문 인용인가."""
    if not STATUTE_LINE.match(value):
        return False
    # 단순 번호 목록(① 온라인으로 제출하여야 한다 등)은 법령 인용이 아니다.
    if not re.search(r"(법률|시행령|시행규칙|예규|고시|제\s*\d+\s*조)", value):
        return False
    return bool(re.search(r"(한다|하여야|본다|따른다|같다)\s*[.。]?\s*$", value.strip()))


# 항목명 뒤에 올 수 있는 조사(이것까지는 같은 낱말이 아니다)
JOSA = re.compile(r"^(은|는|이|가|을|를|의|에서|에|와|과|도|으로|로|만|및)?")


def word_complete(plain: str, m: re.Match) -> bool:
    """항목명이 낱말 하나로 끝났는가.

    '공동수급'이 '공동수급체'의 앞부분일 때 여기서 잘라 쓰면
    값이 '체 구성원 수는…'처럼 낱말 중간부터 시작해 뜻이 망가진다.
    """
    tail = plain[m.end():]
    # 조사까지 항목명으로 허용하면 ``사업 내용이 변경될 경우``의 ``사업
    # 내용``을 라벨로 오인하고 값이 ``이 변경될 경우``에서 시작한다.
    # 문장형 언급은 unanchored 후보로 남겨 의미 검사를 받게 한다.
    if tail and "가" <= tail[0] <= "힣":
        return False
    rest = tail.lstrip()
    # 공백 뒤에 제목 꼬리가 이어진 경우도 짧은 별칭이 완결된 것이 아니다.
    # ``사업기간 계약일로부터`` 같은 실제 값 문장은 유지한다.
    if re.match(r"(?:세부|및|한도|기준|방법|사항|항목|조건|증명서류|배점표)\b", rest):
        return False
    return True


def label_anchored(plain: str, m: re.Match, raw: str) -> bool:
    """이 등장이 '항목명 자리'인가, 아니면 문장 한가운데 스친 것인가.

    문장 한가운데의 단어를 항목명으로 오해하면 뒤따르는 문단 전체를 값으로
    끌어와 엉뚱한 답이 된다. 그래서 대표값은 항목명 자리에서만 읽는다.
    """
    before = plain[:m.start()]
    if DECOR_ONLY.match(before):        # 줄 첫머리(장식 제외)에 있다
        return True
    if "|" in raw:                      # 표의 한 칸이 통째로 항목명이다
        for cell in raw.split("|"):
            c = strip_markup(cell)
            if c and m.group(0) in c and len(c) <= len(m.group(0)) + 6:
                return True
    if "<tr" in raw:
        for cell in re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", raw,
                               flags=re.IGNORECASE):
            c = strip_markup(cell)
            if c and m.group(0) in c and len(c) <= len(m.group(0)) + 8:
                return True
    return False


def value_after_label(line: str, m: re.Match) -> str:
    """항목명 바로 뒤에 값이 붙어 있으면 그 부분만 떼어낸다."""
    tail = line[m.end():]
    tail = re.sub(r"^[\s\)\]\}:：·\-–—|]+", "", tail)
    return strip_markup(tail)


PAGE_ONLY = re.compile(
    rf"^\s*(?:p(?:age)?\.?\s*)?(?:\d{{1,4}}|[{ROMAN_NUMBER}]+)(?:\s*[*]+)?\s*$",
    re.IGNORECASE)
TITLE_SUFFIX_ONLY = re.compile(
    r"^\s*(?:및\s*)?(?:일정|기한|안내|방법|내용|목적|예산|유의사항|제출서류|"
    r"평가방법|작성방법)(?:\s*및\s*(?:방법|서류|기한|일정|유의사항))*\s*$")
LABEL_REMAINDER_ONLY = re.compile(
    r"^\s*(?:(?:항목|기준|방법)\s*)?(?:및\s*)?"
    r"(?:배점(?:\s*(?:한도|기준|현황|총괄표))?|기준|방법|협상|항목)"
    r"(?:\s*및\s*(?:배점(?:\s*(?:한도|기준))?|기준|방법|협상|항목))*\s*$")
TOC_WORDS = re.compile(
    r"(목\s*차|사업\s*(?:개요|내용|범위)|주요\s*사업\s*내용|"
    r"추진\s*(?:배경|방안|내용)|제안\s*요청|과업\s*(?:내용|범위)|"
    r"입찰\s*(?:안내|참가)|제안서\s*(?:작성|제출)|평가\s*(?:방법|기준|항목|배점)|"
    r"별지\s*서식)")
TOC_GROUPS = re.compile(
    r"(목\s*차|사업|추진|제안\s*요청|과업|입찰|제안서|평가|별지)")


def page_or_title_only(text: str) -> bool:
    """쪽수·목차 잔여어·기호뿐인 조각인지 판정한다."""
    t = strip_markup(text).strip("* _\t")
    return (not t or bool(PAGE_ONLY.fullmatch(t)) or bool(TITLE_SUFFIX_ONLY.fullmatch(t))
            or bool(LABEL_REMAINDER_ONLY.fullmatch(t)))


def looks_toc_line(lines: list[str], idx: int, plain: str, inline: str) -> bool:
    """목차의 제목/쪽수 줄을 실제 필드 값과 구분한다.

    목차는 문서 앞부분뿐 아니라 중간에 한 번 더 삽입되기도 하므로 위치만으로
    단정하지 않는다. 가까운 ``목차`` 표식, 줄 끝 쪽수, 한 줄에 몰린 여러 장
    제목을 함께 사용한다.
    """
    raw = lines[idx]
    near = " ".join(strip_markup(x) for x in lines[max(0, idx - 45):idx + 1])
    near_toc = bool(re.search(r"(?:^|\s)목\s*차(?:\s|$)", near))
    tail_page = bool(re.search(r"(?:\t|\s{2,}|\|)\s*\d{1,4}\s*\**\s*$", raw))
    neighbor_pages = sum(bool(re.search(r"(?:\t|\s{2,}|\|)\s*\d{1,4}\s*\**\s*$", x))
                         for x in lines[max(0, idx - 6):min(len(lines), idx + 7)])
    many_titles = len(TOC_WORDS.findall(plain)) >= 3
    many_pages = len(re.findall(r"(?<!\d)\d{1,3}(?:\*+)?(?!\d)", plain)) >= 5
    if many_titles and many_pages:
        return True
    if page_or_title_only(inline) and (tail_page or (near_toc and neighbor_pages >= 3
                                                      and idx < 160)):
        return True
    if idx < 180 and tail_page and neighbor_pages >= 3:
        return True
    if (idx < 180 and re.search(r"\s\d{1,3}\s*\**$", plain)
            and TOC_WORDS.search(plain) and not re.search(r"\d\s*(?:점|%)", plain)):
        return True
    # 표 한 칸의 항목명과 마지막 쪽수 셀 조합
    if ("<tr" in raw or "|" in raw) and tail_page and idx < 180:
        return True
    return False


def looks_toc_value(text: str) -> bool:
    """여러 목차 항목을 한 값으로 끌어온 경우를 잡는다."""
    vals = [x.strip() for x in (text or "").splitlines() if x.strip()]
    page_lines = sum(bool(re.search(r"(?:\s|-{2,})\d{1,3}\s*\**$", x)) for x in vals)
    title_kinds = len(set(TOC_WORDS.findall(text or "")))
    title_groups = len(set(TOC_GROUPS.findall(text or "")))
    roman_heads = len(re.findall(rf"(?m)^\s*[{ROMAN_NUMBER}]+[.)\s]", text or ""))
    # 평가표의 점수가 줄 끝 숫자로 남아도 목차 쪽수로 세지 않는다. 서로
    # 다른 장(사업/입찰/평가 등)이 둘 이상 있을 때만 쪽수 묶음으로 본다.
    return ((page_lines >= 3 and title_kinds >= 2 and title_groups >= 2)
            or (title_kinds >= 4 and roman_heads >= 2)
            or (len(vals) >= 8 and page_lines >= max(3, len(vals) // 2)))


def block_after(lines: list[str], idx: int, parent_level: int | None = None,
                parent_is_markdown: bool = False,
                parent_is_structured: bool = False) -> tuple[str, int]:
    """항목명 아래의 계층 구간을 모은다.

    Markdown 제목 아래에서는 더 낮은 단계(더 많은 ``#``)의 하위 제목과 그
    내용을 포함하고, 같은 단계 또는 더 높은 단계 제목에서 멈춘다. ``#``가
    유실된 짧은 번호 제목에도 같은 원칙을 적용한다.
    """
    out, i, blanks = [], idx, 0
    while i < len(lines) and len(out) < MAX_BLOCK_LINES:
        raw = lines[i]
        level = heading_level(raw)
        if level is not None:
            # Markdown 제목 아래의 번호·가나다 항목은 변환 과정에서 ``#``가
            # 빠진 하위 내용일 수 있다. 다음 Markdown 동급/상위 제목까지만
            # 읽고, 그 하위 번호 제목은 구간에 포함한다.
            is_markdown = bool(MARKDOWN_HEADING.match(raw))
            child_structured = structured_heading_level(strip_markup(raw)) is not None
            should_stop = (parent_level is None or level <= parent_level)
            # 번호가 있는 부모 아래에서 PDF 줄바꿈 조각이 같은 ``###``로
            # 변환돼도 새 절이 아니다. 다음 번호 동급/상위에서만 끊는다.
            if parent_is_structured and is_markdown and not child_structured:
                should_stop = False
            if should_stop:
                break
        s = strip_markup(raw)
        if not s:
            blanks += 1
            if blanks >= 3 and out:
                break
            i += 1
            continue
        blanks = 0
        out.append(s)
        i += 1
    return "\n".join(out)[:MAX_VALUE_CHARS], i


REQUIRED_DOC_NUMBERED_ITEM = re.compile(
    r"^\s*(?:[①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳]|(?:\d{1,2}|[가-하])[.)])\s*\S")


def numbered_required_docs_value(inline: str, block: str) -> str:
    """같은 제출서류 목록의 ①, ② ... 연속 항목을 한 값으로 보존한다."""
    first = (inline or "").strip()
    if not REQUIRED_DOC_NUMBERED_ITEM.match(first):
        return ""
    items = [first]
    for raw in (block or "").splitlines():
        item = raw.strip()
        if not item:
            continue
        if not REQUIRED_DOC_NUMBERED_ITEM.match(item):
            break
        items.append(item)
    return "\n".join(items)[:MAX_VALUE_CHARS] if len(items) >= 2 else ""


REQUIRED_DOC_HEADING = re.compile(
    r"^\s*(?:\d+[.)]?|[가-하][.)]?|[①-⑳])?\s*"
    r"(?:기타\s+|입찰\s*(?:관련|참가)?\s*)?"
    r"(?:필수\s*)?(?:제출\s*서류|구비\s*서류|붙임\s*서류)"
    r"(?:\s*(?:목록|명세|및\s*문의처))?\s*$", re.IGNORECASE)


def required_docs_table_after(lines: list[str], label_index: int) -> tuple[str, int]:
    """제출서류 제목 바로 아래 표 전체를 한 답 후보로 읽는다."""
    table_start = None
    for idx in range(label_index, min(len(lines), label_index + 7)):
        if "<table" in lines[idx] or "<tr" in lines[idx]:
            table_start = idx
            break
        if strip_markup(lines[idx]) and idx > label_index + 3:
            break
    if table_start is None:
        return "", label_index + 1
    values = []
    end = table_start
    table_closed_at = None
    for idx in range(table_start, min(len(lines), table_start + 100)):
        value = strip_markup(lines[idx])
        if values and value and REQUIRED_DOC_STOP.search("\n" + value):
            break
        if value:
            values.append(value)
        end = idx + 1
        if "</table>" in lines[idx]:
            table_closed_at = idx
            break
    # 변환된 표가 중간에서 닫히고 ⑤~⑬ 같은 나머지 항목이 표 밖의
    # 일반 문단으로 이어지는 문서가 있다. 첫 후속 내용이 실제 번호 항목일
    # 때만 같은 목록으로 합치고, 문의·유의사항이 시작되면 멈춘다.
    if table_closed_at is not None:
        first_following = None
        for idx in range(table_closed_at + 1, min(len(lines), table_closed_at + 15)):
            following = strip_markup(lines[idx])
            if following:
                first_following = (idx, following)
                break
        if first_following and re.match(r"^\s*[-*]?\s*[⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳]", first_following[1]):
            for idx in range(first_following[0], min(len(lines), table_closed_at + 80)):
                following = strip_markup(lines[idx])
                if following and REQUIRED_DOC_STOP.search("\n" + following):
                    break
                if following:
                    values.append(following)
                end = idx + 1
    return "\n".join(values)[:MAX_VALUE_CHARS], end


def find_hits(lines: list[str], field: str, att_start: int, front_limit: int,
              section_index: list[dict] | None = None) -> list[dict]:
    """한 필드의 등장 위치를 문서 전체에서 모은다(대표 하나만 남기지 않는다)."""
    pat = R.ALIAS_RE[field]
    hits = []
    sections = section_index if section_index is not None else build_section_index(lines)
    for i, raw in enumerate(lines):
        if not raw.strip():
            continue
        plain = strip_markup(raw)
        if not plain:
            continue
        m = pat.search(plain)
        if not m and field == "평가 배점":
            m = FULL_EVALUATION_SENTENCE.search(plain)
        if not m:
            continue
        # 제안업체 자신에 관한 서식은 발주 사업의 값이 아니다
        window = " ".join(strip_markup(x) for x in lines[max(0, i - 6):i + 3])
        proposer = bool(R.PROPOSER_CONTEXT.search(window))
        anchored = label_anchored(plain, m, raw) and word_complete(plain, m)
        if field == "필수 제출 서류" and REQUIRED_DOC_HEADING.fullmatch(plain):
            anchored = True
        inline = value_after_label(plain, m)
        level = heading_level(raw)
        toc = looks_toc_line(lines, i, plain, inline)
        line_end = i + 1
        if anchored:
            effective_level = level if level is not None else 2
            block, block_stop = block_after(
                lines, i + 1, effective_level,
                parent_is_markdown=bool(MARKDOWN_HEADING.match(raw)),
                parent_is_structured=(structured_heading_level(plain) is not None))
            qualification_frame = (field == "참가 자격(면허·실적)" and bool(re.fullmatch(
                r"(?:아래\s*)?(?:조건을\s*모두\s*충족하는\s*업체|조건|"
                r"증명서류(?:\s*\([^)]*\))?(?:\s*일체\s*\d+부)?|및\s*입찰방식)",
                inline or "")))
            qualification_ref_with_body = (
                field == "참가 자격(면허·실적)" and R.REFER_ONLY.search(inline or "")
                and QUALIFICATION_EVIDENCE.search(block))
            document_frame = (
                field == "필수 제출 서류"
                and (len(list(DOCUMENT_LIST_EVIDENCE.finditer(block))) >= 3
                     or (bool(R.REFER_ONLY.search(inline or ""))
                         and bool(DOCUMENT_LIST_EVIDENCE.search(block))))
                and (not DOCUMENT_LIST_EVIDENCE.search(inline or "")
                     or bool(R.REFER_ONLY.search(inline or "")))
                and not REQUIRED_DOC_REFERENCE_GUIDANCE.search(inline or ""))
            continued_documents = (numbered_required_docs_value(inline, block)
                                   if field == "필수 제출 서류" else "")
            use_inline = bool(inline and not R.looks_blank(inline)
                              and not page_or_title_only(inline) and not qualification_frame
                              and not qualification_ref_with_body and not document_frame)
            table_value, table_stop = ((required_docs_table_after(lines, i)
                                        if field == "필수 제출 서류" else ("", i + 1)))
            value = table_value or continued_documents or (inline if use_inline else block)
            if table_value:
                line_end = table_stop
            elif continued_documents or not use_inline:
                line_end = max(line_end, i + 1, block_stop)
            if field == "필수 제출 서류" and not table_value:
                for stop_idx in range(i + 1, min(line_end, len(lines))):
                    if REQUIRED_DOC_STOP.search("\n" + strip_markup(lines[stop_idx])):
                        last = stop_idx - 1
                        while last > i and not strip_markup(lines[last]):
                            last -= 1
                        line_end = last + 1
                        break
            if field == "과업 범위":
                for stop_idx in range(i + 1, min(line_end, len(lines))):
                    if re.search(r"[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]+\s*(?:운영\s*현황|사업\s*현황|추진\s*현황)",
                                 strip_markup(lines[stop_idx])):
                        last = stop_idx - 1
                        while last > i and not strip_markup(lines[last]):
                            last -= 1
                        line_end = last + 1
                        break
        else:
            # 항목명 자리가 아니면 그 줄 자체만 근거로 남기고 뒤 문단은 끌어오지 않는다
            value = plain
        # 짧은 정책 줄에서 항목명 뒤를 떼면 값이 '불가/불허'만 남는다.
        # 이 경우 문장 전체가 의미 단위이므로 제목과 값을 다시 합치지 말고
        # 원문 정책 문장 자체를 대표값으로 보존한다.
        if field == "컨소시엄 요건" and consortium_policy_kind(plain):
            value = plain
        section = sections[i] if i < len(sections) else {}
        hits.append({
            "line": i + 1,
            "line_start": i + 1,
            "line_end": line_end,
            "heading": plain[:120],
            "section_heading": section.get("section_heading", ""),
            "section_path": section.get("section_path", []),
            "block_type": section.get("block_type", ""),
            "block_index": section.get("block_index", 0),
            "matched": m.group(0),
            "source_type": source_type_of(i + 1, raw, att_start, front_limit),
            "value": value,
            "inline": inline,
            "anchored": anchored,
            "hierarchical": level is not None,
            "proposer_context": proposer,
            "toc": toc or looks_toc_value(value),
            "context": window[:1200],
            "raw": raw,
        })
    return hits


# 알려진 항목명 정규식으로 먼저 검색하면 목록 밖 표현은 구조적으로 발견할 수
# 없다. 제목과 표의 왼쪽 항목 칸을 독립적으로 모은 뒤에만 12필드 의미를 붙인다.
DISCOVERY_HINTS: list[tuple[str, re.Pattern]] = [
    ("사업 개요", re.compile(r"(?:과업|용역)\s*명$")),
    ("공고일", re.compile(r"(게시|공고)\s*(일시|게시일)$")),
    ("사업기간", re.compile(r"(납품|완료)\s*기한$|용역\s*수행\s*일정$")),
    ("예산", re.compile(r"(예정|추정|사업)\s*(금액|대가)$")),
    ("참가 자격(면허·실적)", re.compile(r"(지원|신청|응모|입찰자)\s*자격$")),
    ("지역제한", re.compile(r"(소재지|관할\s*지역)\s*(요건|조건)$")),
    ("컨소시엄 요건", re.compile(r"공동\s*(참여|계약)\s*(조건|사항|허용\s*여부)$")),
    ("평가 배점", re.compile(r"(심사|평가)\s*(점수|비율)$")),
    ("제출 방식", re.compile(r"(제안서|입찰서류)\s*접수$|접수\s*(경로|장소와\s*방법)$")),
    ("필수 제출 서류", re.compile(r"(입찰|참가)\s*(등록|신청)\s*(문서|서류)$|구비\s*문서$|"
                                r"제출할\s*도서의\s*품목\s*,?\s*매체\s*,?\s*수량$")),
    ("과업 범위", re.compile(r"(업무|구축|개발)\s*범위$|대상\s*업무$|세부\s*과업$|"
                         r"주요\s*(?:과업\s*내용|사업\s*(?:내용|범위))$|"
                         r"사업\s*주요\s*내용$|추진\s*목표$|"
                         r"사업\s*추진\s*방향\s*및\s*규모$|"
                         r"사업\s*최종목표\s*시스템\s*및\s*사업\s*범위$")),
]


def structural_labels(raw: str) -> list[str]:
    """Markdown 제목과 표 왼쪽 항목 칸에서 짧은 라벨을 수집한다."""
    candidates: list[str] = []
    m = MARKDOWN_HEADING.match(raw)
    if m:
        candidates.append(strip_markup(m.group(2)))
    if raw.lstrip().startswith("|"):
        cells = [strip_markup(x) for x in raw.split("|") if strip_markup(x)]
        if cells:
            candidates.append(cells[0])
    elif "<tr" in raw:
        cells = [strip_markup(x) for x in re.findall(
            r"<t[hd][^>]*>(.*?)</t[hd]>", raw, flags=re.IGNORECASE)]
        if cells:
            candidates.append(cells[0])

    out = []
    for value in candidates:
        value = re.sub(rf"^\s*(?:제\s*\d+\s*장|[{ROMAN_NUMBER}]+[.)]?|"
                       r"\d+(?:\.\d+)*[.)]|[가-하][.)])\s*", "", value)
        value = value.strip(" *#|[]【】()<>·-–—\t")
        if ":" in value or "：" in value:
            value = re.split(r"[:：]", value, 1)[0].strip()
        if 2 <= len(value) <= 50 and not re.search(r"[.!?。]$", value):
            out.append(value)
    return out


def known_expression(field: str, expression: str) -> bool:
    """공백·자간·문장부호 또는 일반적인 제목 꼬리만 다른 기존 표현인가."""
    key = R.expression_key(expression)
    if key in R.DISCOVERY_BASELINE_EXPRESSION_KEYS[field]:
        return True
    generic = ("여부", "안내", "사항", "세부", "일정", "기한", "방법", "요건", "관련")
    for base in R.DISCOVERY_BASELINE_EXPRESSION_KEYS[field]:
        if key.startswith(base) and key[len(base):] in generic:
            return True
    return False


def discover_aliases(lines: list[str], reg_row: dict, att_start: int,
                     front_limit: int) -> list[dict]:
    """기존 목록 밖의 구조적 항목명 중 12필드 의미가 분명한 것만 돌려준다."""
    found = []
    for i, raw in enumerate(lines):
        for label in structural_labels(raw):
            for field, hint in DISCOVERY_HINTS:
                if not hint.fullmatch(re.sub(r"\s+", " ", label).strip()):
                    continue
                if known_expression(field, label):
                    continue
                source_type = source_type_of(i + 1, raw, att_start, front_limit)
                heading = strip_markup(raw)
                local_context = " ".join(strip_markup(x)
                                         for x in lines[max(0, i - 5):i + 3])
                # 첨부 서식의 칸 이름과 평가 보조표의 역할/가감점 칸은 발주
                # 사업 필드의 새로운 표현이 아니다.
                if source_type == "attachment_form":
                    continue
                if field == "사업분야" and re.search(r"(적용\s*기준|대금\s*지급|실적|경력)", heading):
                    continue
                if field == "예산" and re.search(r"(수행|이행)\s*실적|평가\s*기준", local_context):
                    continue
                if field == "평가 배점" and re.search(r"(가감점|위원별|심사위원)", heading):
                    continue
                if field == "과업 범위" and re.search(
                        r"(제안사|제안업체|수행실적|사업실적|분야별로\s*구분|작성\s*(?:지침|방법))",
                        f"{heading} {local_context}"):
                    continue
                if (i < max(100, int(len(lines) * 0.08))
                        and re.search(r"\s\d{1,3}\s*\**$", heading)):
                    continue
                found.append({
                    "document_id": reg_row["document_id"], "field_name": field,
                    "matched_expression": R.nfc(label),
                    "source_type": source_type,
                    "location": f"{reg_row['source_filename_nfc']}#L{i + 1}",
                    "heading": R.redact(heading)[:120],
                    "basis": ("공식 원문의 Markdown 제목/표 왼쪽 항목 칸을 먼저 수집하고 "
                              "발주 사업의 기존 12필드 역할을 직접 확인함"),
                    "confidence": "source_reviewed_new_alias",
                })
                break
    return found


# ══════════════════════════════════════════════════════════════════
# 3. 위치들로부터 상태와 값을 정한다
# ══════════════════════════════════════════════════════════════════

# 대표 출처를 고를 때의 우선순위 (필드 성격에 따라 다르다)
PREF_BODY_FIRST = {"과업 범위", "평가 배점", "참가 자격(면허·실적)", "필수 제출 서류",
                   "컨소시엄 요건", "제출 방식", "지역제한"}
LIST_FIELDS = {"참가 자격(면허·실적)", "필수 제출 서류", "평가 배점", "과업 범위"}
# 발주 사업의 값이 아니라서 제안업체 문맥을 반드시 걸러야 하는 필드
PROPOSER_SENSITIVE = {"사업분야", "사업 개요", "사업기간", "예산", "제출 방식"}

# 필드 값 자체가 아닌 주변 문맥. 단어 하나만 보고 버리지 않고, 대표 후보의
# 값/주변 제목과 함께 적용한다.
HISTORY_PERIOD = re.compile(
    r"(과거|기존\s*시스템|개발\s*(?:이력|실적|경험)|유사\s*(?:사업|용역)|"
    r"수행\s*실적|사업\s*실적|용역\s*실적|실적\s*(?:인정\s*)?(?:기간|증명)|최근\s*\d+\s*년|"
    r"제안업체|참여인력|경력\s*(?:기간|사항)|[’'`]\d{2}년부터\s*단계)")
NON_PROJECT_PERIOD_CONTEXT = re.compile(
    r"(소프트웨어사업\s*영향평가|개인정보\s*처리업무|위탁업무\s*기간|"
    r"보안\s*확약|비밀\s*유지|계약의\s*효력|영구적으로\s*그\s*효력|"
    r"행사\s*기간|협상\s*기간|공고\s*기간)")
BLANK_DATE_FORM = re.compile(
    r"(?:20\s*[○〇O_ ]+|년\s*월\s*일|\d*\s*[.년]\s*[.월]\s*[.일])"
    r"[^\n]{0,25}(?:[~∼-])[^\n]{0,25}(?:20\s*[○〇O_ ]+|년\s*월\s*일)")
SUBMISSION_MODE = re.compile(
    r"(나라장터|e-?발주|전자\s*(?:제출|입찰)|(?<![가-힣])온라인\s*(?:제출|접수)|"
    r"직접\s*(?:방문|제출|접수)|방문\s*(?:제출|접수)|밀봉|봉합|접수처)",
    re.IGNORECASE)
SUBMISSION_QUESTION = re.compile(r"(서면\s*질의|질의서|질의\s*접수|문의처)")
SUBMISSION_PLACE_EVIDENCE = re.compile(
    r"(제안서\s*제출[^\n]{0,70}(?:본관|회관|사무국|총무팀|기획\w*팀|\d+층|\d+호)|"
    r"^(?:\([^)]*\)|[가-힣A-Za-z0-9·\s]+)?(?:본관|회관|사무국|총무팀|"
    r"기획\w*팀|\d+층|\d+호)[가-힣A-Za-z0-9·\s()/-]*$|"
    r"(?:재단|공사|대학교|연구원)[^\n]{0,35}(?:팀|실|처|본부)\)?)")
SUBMISSION_ACTION = re.compile(
    r"((?:온라인|전자|나라장터|e-?발주|전자조달시스템)[^\n]{0,45}(?:제출|접수)|"
    r"(?:제출|접수)[^\n]{0,45}(?:온라인|전자|나라장터|e-?발주|전자조달시스템)|"
    r"(?:직접|방문)[^\n]{0,15}(?:제출|접수)|(?:제출|접수)[^\n]{0,15}(?:직접|방문))",
    re.IGNORECASE)


def has_submission_mode(text: str) -> bool:
    """제안서의 실제 제출 수단이 명시됐는지 본다(연락처/불가 수단 제외)."""
    if SUBMISSION_MODE.search(text or ""):
        return True
    for m in re.finditer(r"(우편|이메일|e-?mail|전자우편|팩스|택배)", text or "",
                         flags=re.IGNORECASE):
        around = (text or "")[max(0, m.start() - 20):m.end() + 25]
        if re.search(r"(제출|접수|송부)", around) and not re.search(
                r"(불가|금지|받지\s*않|인정하지\s*않)", around):
            return True
    return False
BID_CONTRACT_ONLY = re.compile(
    r"^\s*(?:입찰\s*방법\s*[:：-]?\s*)?(?:제한|일반)?\s*(?:\([^)]*\))?\s*"
    r"(?:총액\s*)?경쟁\s*입찰(?:\s*[/,]\s*협상에\s*의한\s*계약)?\s*$")
QUALIFICATION_PUNISHMENT = re.compile(
    r"(입찰\s*참가\s*자격\s*(?:제한|정지)|참가\s*자격\s*제한|행정\s*처분|"
    r"부정당\s*업자|처벌|제재|불이익)")
QUALIFICATION_EVIDENCE = re.compile(
    r"(자격을\s*(?:갖추|갖춘|구비|보유)|자격\s*요건|면허|"
    r"등록(?:한|되어\s*있는)\s*업체|지정되지\s*않은\s*업체|"
    r"직접생산확인|중소기업|실적을\s*보유|입찰에\s*참가할\s*수\s*있는)")
QUALIFICATION_DOCUMENT_ONLY = re.compile(
    r"(참가\s*자격\s*확인\s*서류|입찰참가자격\s*\d+\s*식|"
    r"증명서류|재직증명서|위임장|사업자등록증\s*사본)")
QUALIFICATION_VAGUE = re.compile(
    r"^(?:다\.\s*)?공동수급체[^\n]{0,35}입찰참가자격[^\n]{0,20}모든\s*자격을\s*갖추어야\s*함$")
REQUIRED_DOC_BAD_CONTEXT = re.compile(
    r"(하도급|착수\s*(?:계|보고)|완료\s*보고|중간\s*보고|"
    r"사업\s*수행\s*중|계약\s*(?:체결|이행|수행)\s*(?:후|중)|산출물\s*(?:계획|제출)|"
    r"보안서약.*참여인력|유의사항만|실적을\s*확인|실적증명\s*자료|"
    r"유사\s*사업\s*실적|사업관리자.*변경|투입인력.*증빙)")
REQUIRED_DOC_GOOD_CONTEXT = re.compile(
    r"(입찰\s*(?:참가|등록|서류)|제안서?\s*(?:제출|접수)|가격\s*제안서|"
    r"정량적\s*(?:평가|제안)|정성적\s*(?:평가|제안)|구비\s*서류)")
DOCUMENT_LIST_EVIDENCE = re.compile(
    r"(입찰\s*참가\s*(?:표지|신청서|서류|등록\s*서류|자격을\s*증명하는\s*서류)|"
    r"용역\s*참여\s*신청서|전자\s*파일\s*형태의\s*제안서|사업자\s*등록증|"
    r"법인\s*(?:등기부|인감)|인감\s*증명서|확인서|"
    r"사용\s*인감계|위임장|서약서|확약서|제안서[^\n]{0,20}\d+\s*부|"
    r"제안서\s*(?:,|및)\s*(?:발표자료|(?:제안\s*)?요약서)|제안서\s*(?:원본|PDF)|"
    r"(?:기술\s*)?제안서\s*제출\s*공문|기술\s*제안서\s*본문|"
    r"일반\s*현황\s*및\s*연혁|조직\s*및\s*인력\s*현황|"
    r"제안\s*요약서\s*원본|발표\s*자료\s*파일|"
    r"정량(?:적)?\s*제안서|정성(?:적)?\s*제안서|가격\s*제안서|입찰\s*참가\s*구비\s*서류|"
    r"USB|CD\s*\d*\s*(?:장|매|부)|입찰\s*참가\s*공문|제안\s*요약서|"
    r"중소기업\s*확인서|청렴\s*계약\s*이행\s*서약서|신용\s*평가\s*등급|직접\s*생산\s*확인|"
    r"소프트웨어\s*사업자\s*일반현황|수행\s*실적\s*증명서)", re.IGNORECASE)
REQUIRED_DOC_REFERENCE_GUIDANCE = re.compile(
    r"(?:입찰\s*참가\s*신청서\s*(?:\[[^\]]+\])?\s*(?:상|에)?\s*(?:의|의한)?\s*)?"
    r"(?:붙임|첨부)\s*자료\s*(?:에\s*)?(?:의함|따름|참조|확인)", re.IGNORECASE)
GENERIC_APPLICATION_FORM = re.compile(
    r"(?:입찰|경쟁\s*입찰)(?:\s*\([^)]*\))?\s*에\s*참가하고자"
    r"[^\n]{0,500}(?:입찰\s*유의서|입찰\s*공고\s*사항)[^\n]{0,500}"
    r"(?:붙임|별첨)\s*서류",
    re.IGNORECASE)
GENERIC_REQUIRED_DOC_STATEMENT = re.compile(
    r"입찰에\s*참여하는\s*사업자는[^\n]{0,120}(?:서류|구비서류)[^\n]{0,120}"
    r"모두\s*제출[^\n]{0,80}(?:무효|사유)", re.IGNORECASE)
REQUIRED_DOC_EXTERNAL_WORDING = re.compile(
    r"(?:입찰\s*)?공고(?:문|서|사항)?(?:으로써|에서)?\s*(?:참조|정한\s*서류)|"
    r"공고로써\s*정한\s*서류|제안\s*요청서에\s*요구한\s*서류",
    re.IGNORECASE)
REQUIRED_DOC_DIRECT_REFERENCE = re.compile(
    r"(?:입찰\s*참가\s*)?(?:필수\s*)?(?:제출|구비)\s*서류\s*(?:제출)?\s*[:：-]?\s*"
    r"[‘’“\"']?입찰\s*공고(?:문|서|사항)?[‘’”\"']?\s*(?:에\s*)?(?:참조|따름)",
    re.IGNORECASE)
REQUIRED_DOC_STOP = re.compile(
    r"\n\s*(?:【\s*(?:서식|양식)|\[\s*(?:별지\s*)?(?:제\s*)?\d*\s*호?\s*(?:서식|양식)\s*\]|"
    r"나[.)]\s*제안서\s*발표|\d+[.)]\s*제안\s*(?:관련\s*)?문의|"
    r"\d+[.)]\s*제안서의\s*효력|\d+[.)]\s*제안\s*설명회|"
    r"제안\s*관련\s*문의사항|제출\s*방법\s*및\s*문의사항|문\s*의\s*처|문의처|"
    r"제안\s*요청\s*설명회|제안서\s*(?:발표회|보상)|"
    r"(?:[가-하]\s*[.)]\s*)?제안서\s*제출\s*일정\s*및\s*방법|"
    r"제출\s*방법\s*[,·]|제출\s*장소(?=\s*(?:\n|$))|제출\s*규격|제안서\s*작성\s*(?:기준|유의사항|목차)|"
    r"(?:\d+\s*)?계약\s*체결\s*관련\s*서류|"
    r"(?:[가-하]\s*[.)]\s*)?유의사항|제출\s*기한\s*내에\s*미제출|"
    r"특이사항|입찰\s*참가\s*신청자|신\s*청\s*인\b|"
    r"[-–—]{5,}\s*(?:\([^)]*\)\s*)?절\s*취\s*선)",
    re.IGNORECASE)
REQUIRED_DOC_TRAILING_FORM = re.compile(
    r"\s+(?:20\s*\d{0,2}\s*[년.]\s*(?:\d{0,2}\s*[월.]\s*)?\d{0,2}\s*(?:일|\.)?|"
    r"년\s*월\s*일)\s+(?:상호\s*[:：]|신청인\s*[:：(]|대표자\s*[:：])",
    re.IGNORECASE)
CONSORTIUM_VALUE = re.compile(
    r"((?:공동\s*(?:수급|이행|계약|도급)|컨소시엄)[^\n]{0,80}"
    r"(?:허용|가능|불가|불허|금지|않음|구성원|대표|지분|\d+\s*개사|방식)|"
    r"단독\s*(?:입찰|참여)|공동\s*이행[^\n]{0,50}분담\s*이행)")


def consortium_policy_kind(text: str) -> str:
    """공동수급의 실제 발주 정책만 분류한다(평가·서식 언급은 빈 문자열)."""
    value = text or ""
    if re.search(r"(하도급\s*계획서|입찰\s*금액\s*대비\s*하도급|"
                 r"공동수급체의\s*경우[^\n]{0,35}(?:점수|실적)[^\n]{0,25}지분율)", value):
        return ""
    subject = r"(?:공동\s*(?:수급(?:체)?|이행(?:\s*방식)?|계약|도급)|컨소시엄)"
    conditional = re.search(
        r"공동\s*이행[^\n]{0,30}(?:가능|허용)[^\n]{0,70}"
        r"분담\s*이행[^\n]{0,30}(?:불가|불허|금지|않)|"
        r"분담\s*이행[^\n]{0,30}(?:불가|불허|금지|않)[^\n]{0,70}"
        r"공동\s*이행[^\n]{0,30}(?:가능|허용)|"
        r"공동\s*(?:수급|계약)[^\n]{0,25}공동\s*이행\s*방식[^\n]{0,20}"
        r"(?:원칙|허용)[^\n]{0,70}분담\s*이행[^\n]{0,25}(?:불가|불허|않)", value)
    if conditional:
        return "conditional_allow"
    deny = None
    if not re.search(r"(중복|이중|구성원\s*변경|낙찰자로\s*결정된\s*이후|대기업\s*간)", value):
        combined = re.search(
            r"공동\s*(?:수급|계약)[^\n]{0,35}(?:및|,|과|와)\s*(?:재)?하도급"
            r"[^\n]{0,15}(?:불가|불허|금지)", value)
        direct = re.search(
            rf"{subject}[^\n]{{0,80}}?(?:허용\s*하지\s*않\w*|불가(?!피)|불허\w*|금지\w*)|"
            rf"(?:복수의\s*업체로\s*)?컨소시엄[^\n]{{0,35}}구성[^\n]{{0,25}}"
            rf"참여를\s*제한|공동\s*수급을\s*제외한\s*단독\s*입찰", value)
        if direct:
            between = direct.group(0)
            # '공동수급인 경우 ... 하도급 불허'는 하도급 정책일 뿐이다.
            scoped = re.search(r"(하도급|협정서\s*제출|구성원\s*변경|중복|이중)", between)
            deny = direct if (combined or not scoped) else None
    allow = None if re.search(r"허용\s*여부", value) else re.search(
        rf"{subject}[^\n]{{0,35}}(?:가능|허용)(?!\s*(?:하지|되지|않|아니))|"
        rf"{subject}[^\n]{{0,30}}참(?:가|여)할\s*수\s*있|"
        rf"(?:가능|허용)[^\n]{{0,20}}{subject}|"
        rf"단독\s*또는\s*{subject}[^\n]{{0,45}}(?:입찰|참가|가능|허용)", value)
    if deny:
        return "deny"
    if allow:
        return "allow"
    constraint = re.search(
        rf"{subject}[^\n]{{0,100}}(?:구성원[^\n]{{0,35}}\d+\s*개사|"
        rf"최소\s*지분율|구성하여\s*참여해야|표준협정서[^\n]{{0,20}}제출)|"
        rf"{subject}\s*를?\s*구성하여\s*참여해야", value)
    return "constraint" if constraint else ""


def consortium_hit_kind(hit: dict) -> str:
    """후보 시작부의 정책만 본다. 긴 상위 블록 뒤쪽 정책을 끌어오지 않는다."""
    value = (hit.get("value") or "")[:500]
    kind = consortium_policy_kind(value)
    return kind or (consortium_policy_kind(hit.get("heading", "")) if len(value) < 4 else "")
SCOPE_EVIDENCE = re.compile(
    r"(구축|개발|재개발|고도화|개선|운영|유지\s*관리|유지보수|연계|통합|"
    r"도입|설치|전환|분석|설계|기능|시스템|서비스|웹사이트|애플리케이션|"
    r"플랫폼|데이터|DB|품질\s*검사|상담|지원)")
SCOPE_BAD = re.compile(
    r"(사업\s*내용이?\s*변경|과업\s*변경\s*절차|협상\s*범위|"
    r"제안사(?:의|가)?[^\n]{0,35}(?:사업\s*내용|과업\s*내용)[^\n]{0,30}"
    r"(?:작성|기술|표시|제시)|제안서[^\n]{0,35}(?:작성|기술|제시)[^\n]{0,20}"
    r"(?:사업\s*내용|과업\s*범위)|제안사(?:는|가)?[^\n]{0,90}"
    r"(?:목적|범위|전제조건)[^\n]{0,60}(?:기술|작성|제시))")
EVALUATION_SCORE_EVIDENCE = re.compile(
    r"(\d+(?:\.\d+)?\s*(?:점|%)|(?:기술|가격|정량|정성)\s*평가[^\n]{0,35}"
    r"\d+(?:\.\d+)?\s*%|(?:기술|가격|정량|정성)(?:\s*능력)?\s*평\s*가\s*"
    r"\(?\s*\d+(?:\.\d+)?\s*\)?|평가\s*비율|"
    r"배점\s*(?:한도|표)?[^\n]{0,20}\d)")
EVALUATION_BAD_CONTEXT = re.compile(
    r"(기능\s*(?:구축|개발|구현)|관리할\s*수\s*있|요구사항\s*(?:명칭|상세)|"
    r"시스템으로\s*구축|위원회\s*운영|취약점\s*분석.?평가|동점자\s*처리|"
    r"하도급\s*계약.*평가\s*점수)")
EVALUATION_COMPLETE = re.compile(
    r"(?s)(?=.*(?:기술|기술\s*능력)\s*평\s*가[^\n]{0,45}"
    r"\(?\s*\d+(?:\.\d+)?\s*(?:점|%)?\s*\)?)(?=.*(?:입찰\s*)?가격\s*평\s*가"
    r"[^\n]{0,45}\(?\s*\d+(?:\.\d+)?\s*(?:점|%)?\s*\)?)|"
    r"기술\s*능력\s*대\s*가격\s*점수\s*비중\s*\d+\s*[:：]\s*\d+|"
    r"종합\s*평가\s*점수\s*\(?\s*100\s*%?\s*\)?\s*=\s*기술\s*능력\s*평가\s*"
    r"\(?\s*100\s*%?\s*\)?")
FULL_EVALUATION_SENTENCE = re.compile(
    r"기술\s*(?:능력\s*)?평가(?:\s*점수)?[^\n]{0,55}?"
    r"\(?\s*\d+(?:\.\d+)?\s*(?:점|%)?\s*\)?[^\n]{0,90}?"
    r"(?:입찰\s*)?가격\s*(?:평가|점수)[^\n]{0,35}?"
    r"\(?\s*\d+(?:\.\d+)?\s*(?:점|%)?\s*\)?")


def evaluation_complete(text: str) -> bool:
    return bool(EVALUATION_COMPLETE.search(text or "")
                or FULL_EVALUATION_SENTENCE.search(text or ""))


def evaluation_ratio_pair(text: str) -> bool:
    """기술/가격에 직접 붙은 두 배점이 합계 100인지 확인한다."""
    tech = [float(x) for x in re.findall(
        r"기술\s*(?:능력\s*)?(?:평가(?:\s*점수)?|제안서)\s*"
        r"\(?\s*(\d+(?:\.\d+)?)\s*(?:점|%)?\s*\)?", text or "")]
    price = [float(x) for x in re.findall(
        r"(?:입찰\s*)?가격\s*(?:평가(?:\s*점수)?|제안서|점수)\s*"
        r"\(?\s*(\d+(?:\.\d+)?)\s*(?:점|%)?\s*\)?", text or "")]
    return any(abs(t + p - 100) < 0.001 for t in tech for p in price)
QUALIFICATION_GUIDANCE = re.compile(
    r"(제안서\s*작성[^\n]{0,40}(?:자격|요건)|사업\s*수행[^\n]{0,25}"
    r"자격\s*요건[^\n]{0,35}(?:제시|작성)|투입\s*인력\s*자격|"
    r"핵심\s*인력\s*자격\s*요건|PM[^\n]{0,20}자격\s*요건)")
AMBIGUOUS_FIELD_VALUE = re.compile(
    r"^\s*(?:별도\s*협의|추후\s*결정|추후\s*안내|미정)\s*[.]?\s*$")

# 예산은 단어가 아니라 현재 발주 사업의 금액 선언이어야 한다. 예산 관리·위험,
# 제안업체 실적 금액, 빈 계약서 서식처럼 금액 필드와 다른 역할은 제외한다.
BUDGET_AMOUNT_EVIDENCE = re.compile(
    r"(?:[₩￦]\s*\d[\d,.]*|\d[\d,.]*\s*(?:원|천\s*원|만\s*원|백\s*만\s*원|억(?:\s*원)?))")
BUDGET_BAD_CONTEXT = re.compile(
    r"(예산\s*초과[^\n]{0,40}(?:리스크|위험|예방)|예산\s*(?:관리|편성|집행)\s*(?:기능|업무)|"
    r"사업비\s*(?:관리|산정|조정|검토|협의|분담)|계약금액\s*(?:조정|변경|증감)|"
    r"총사업비\s*(?:협의자료|관리지침)|개략사업비\s*(?:비교|검토|산정)|"
    r"(?:기초금액|추정가격)[^\n]{0,100}(?:예시|실적평가|동등이상|유사분야)|"
    r"(?:설계비\s*)?추정가격[^\n]{0,50}\d[^\n]{0,20}(?:이상|미만)[^\n]{0,40}(?:건축물|발주)|"
    r"(?:총\s*)?사업예산[^\n]{0,30}\d[^\n]{0,15}미만[^\n]{0,80}(?:보상|대상|해당)|"
    r"(?:최근|공고일)[^\n]{0,80}(?:사업|용역)?\s*실적[^\n]{0,80}(?:계약금액|금액)|"
    r"하도급[^\n]{0,80}(?:계약금액|예정액)|계약금액의[^\n]{0,35}(?:부과|위약)|"
    r"부과\s*한도|위약금|페널티)")
BUDGET_NON_CURRENT_CONTEXT = re.compile(
    r"(SW\s*사업정보\s*제출|총\s*사업비[^\n]{0,30}이상의[^\n]{0,50}사업은|"
    r"現[^\n]{0,100}구축\s*개요|"
    r"(?:現|기존|과거)[^\n]{0,40}(?:시스템|사업)[^\n]{0,30}(?:구축\s*개요|총\s*사업비)|"
    r"추진\s*일정[^\n]{0,120}단계[^\n]{0,80}사업비)")

# 일반 표현보다 해당 필드를 직접 선언하는 제목을 먼저 쓴다.
CORE_LABEL_KEYS = {
    "과업 범위": {R.expression_key(x) for x in (
        "과업범위", "과업의 범위", "사업범위", "용역범위",
        "과업의 범위 및 내용",
        "주요 과업내용", "주요 사업내용", "주요 사업범위", "사업 주요내용",
        "추진목표", "사업 추진 방향 및 규모",
        "사업 최종목표 시스템 및 사업 범위")},
    "평가 배점": {R.expression_key(x) for x in (
        "평가배점", "배점한도", "평가비율", "평가 비율", "전체평가비율",
        "전체 평가비율", "평가방식", "평가기준", "평가방법", "평가항목",
        "제안서 평가", "제안서평가", "제안서 평가 방법", "제안서 평가항목 및 배점",
        "제안서 평가항목 및 배점한도",
        "평가항목 및 배점", "평가항목 및 배점한도", "평가항목 및 비율",
        "평가기준 및 방법", "배점기준", "종합평가점수",
        "기술·가격평가 비중", "기술ㆍ가격평가 비중", "심사표")},
    "참가 자격(면허·실적)": {R.expression_key(x) for x in (
        "입찰참가자격", "입찰 참가자격", "입찰참가 자격", "입찰참가자격 요건",
        "제안(입찰)참가자격", "입찰 참가 자격 조건",
        "입찰 참가자격 및 입찰방식", "입찰 자격 요건")},
    "제출 방식": {R.expression_key(x) for x in (
        "제출방법", "제안서 제출방법", "접수방법", "제출기한 및 방법",
        "제출방법 및 유의사항", "제출처", "제출장소")},
    "필수 제출 서류": {R.expression_key(x) for x in (
        "제출서류", "구비서류", "필수 제출서류", "제출서류 명세",
        "제안서 제출서류", "제출서류 및 자료", "제출서류목록", "붙임서류",
        "제출할 도서의 품목, 매체, 수량")},
}

SCOPE_LABEL_PRIORITY = {
    R.expression_key(x): rank
    for rank, names in enumerate((
        ("과업범위", "과업의 범위", "과업의 범위 및 내용", "사업범위", "용역범위", "주요 사업범위",
         "사업 최종목표 시스템 및 사업 범위"),
        ("주요 과업내용", "주요 사업내용", "사업 주요내용", "과업내용", "사업내용"),
        ("사업 추진 방향 및 규모", "추진목표"),
        ("제안요청내용", "추진내용"),
    ))
    for x in names
}


def overview_value(text: str) -> str:
    """사업 개요 구간에서 실제 사업명 또는 첫 목적·개요 내용을 고른다."""
    vals = [x.strip() for x in (text or "").splitlines() if x.strip()]
    for pos, val in enumerate(vals):
        m = re.search(r"(?:^|[□◦ㅇ○\s])\(?(?:사\s*업|용\s*역|과\s*업)\s*명\)?\s*[:：)]\s*(.+)", val)
        if m and len(re.sub(r"[^가-힣A-Za-z]", "", m.group(1))) >= 4:
            return m.group(1).strip()[:MAX_VALUE_CHARS]
        if re.fullmatch(r"(?:사업|용역|과업)\s*명", val) and pos + 1 < len(vals):
            nxt = vals[pos + 1]
            if len(re.sub(r"[^가-힣A-Za-z]", "", nxt)) >= 4:
                return nxt[:MAX_VALUE_CHARS]
    meaningful = [x for x in vals
                  if not page_or_title_only(x) and not looks_toc_value(x)
                  and len(re.sub(r"[^가-힣A-Za-z]", "", x)) >= 6]
    return "\n".join(meaningful[:6])[:MAX_VALUE_CHARS]


def field_hit_usable(field: str, hit: dict) -> bool:
    """대표 후보가 발주 사업의 해당 필드 값으로 쓰일 수 있는지 본다."""
    value = (hit.get("value") or "").strip()
    context = f"{hit.get('context', '')} {hit.get('heading', '')} {value}"
    direct_eval_section = (field == "평가 배점" and hit.get("anchored")
                           and hit.get("hierarchical")
                           and R.expression_key(hit.get("matched", ""))
                           in CORE_LABEL_KEYS["평가 배점"]
                           and EVALUATION_SCORE_EVIDENCE.search(value))
    document_evidence_count = len(list(DOCUMENT_LIST_EVIDENCE.finditer(value)))
    canonical_attachment_list = (
        field == "필수 제출 서류"
        and document_evidence_count >= 4
        and bool(GENERIC_APPLICATION_FORM.search(value)))
    actual_document_list = (
        field == "필수 제출 서류"
        and document_evidence_count >= 2
        and (not hit.get("toc")
             or (hit.get("source_type") == "body_table"
                 and document_evidence_count >= 4)
             or canonical_attachment_list)
        and (bool(hit.get("anchored")) or canonical_attachment_list)
        and (hit.get("source_type") != "attachment_form"
             or REQUIRED_DOC_GOOD_CONTEXT.search(context)
             or canonical_attachment_list))
    direct_consortium_policy = (field == "컨소시엄 요건"
                                and bool(consortium_hit_kind(hit)))
    if ((hit.get("toc") or looks_toc_value(value))
            and not direct_eval_section and not actual_document_list
            or (page_or_title_only(value) and not direct_consortium_policy)):
        return False
    if (field in PROPOSER_SENSITIVE and hit.get("proposer_context")
            and not actual_document_list):
        # 제안서 제출 방법 문장 주변에는 자연스럽게 '제안업체'가 나온다.
        # 제출 방식은 후보 줄 자체가 제안업체 실적서/첨부서식일 때만 제외한다.
        if field != "제출 방식" or (hit.get("source_type") == "attachment_form"
                                      or R.PROPOSER_CONTEXT.search(
                                          f"{hit.get('heading', '')} {value}")):
            return False
    if field == "사업 개요":
        bad = re.search(r"(하도급|실적\s*증명|유사\s*업무|참여\s*사업명|"
                        r"사업기간\s*시작\s*사업부서|추진배경\s*검토)", context)
        return not bad and not looks_header_row(value, hit.get("matched", ""),
                                                hit.get("heading", "")) and bool(overview_value(value))
    if field == "사업기간":
        # 앞뒤 몇 줄을 넣은 ``context``에는 바로 다음 동급 제목(예: 기존
        # 시스템의 개발기간)이 걸칠 수 있다. 현재 후보의 값과 그 후보가
        # 속한 제목만으로 과거 이력 여부를 판단해야 정상 사업기간까지 같이
        # 버리지 않는다.
        period_identity_context = f"{hit.get('heading', '')} {value}"
        old_development = (R.expression_key(hit.get("matched", "")) == R.expression_key("개발기간")
                           and re.search(r"(시스템\s*(?:구성|개요|현황)|라이프\s*사이클|노후|"
                                         r"적정\s*(?:사업|개발)\s*기간|전체\s*개발기간|투입기간)",
                                         context))
        generic_period_label = R.expression_key(hit.get("matched", "")) == R.expression_key("기간")
        generic_period = (generic_period_label and (not hit.get("anchored") or not re.search(
            r"(?:계약|체결|착수|완료|\d{4}\s*[.\-/년])", value)
            or not re.search(r"(사업|용역|과업)\s*개요", context)))
        sentence_current = bool(re.search(
            r"(?:본|당해)\s*(?:사업|용역|과업)[^\n]{0,15}기간\s*(?:은|:)", value))
        explicit_inline_current = bool(re.search(
            r"(?:사업|용역|과업)\s*기간\s*[:：]\s*(?:계약|착수)[^\n]{0,30}"
            r"(?:\d+\s*(?:개월|일)|\d{4}\s*[.\-/년])", value))
        unanchored_mention = (not hit.get("anchored") and not sentence_current
                              and not explicit_inline_current)
        return (has_period(value) and not old_development and not generic_period
                and not unanchored_mention
                and not NOT_PROJECT_PERIOD.search(value)
                and not HISTORY_PERIOD.search(period_identity_context)
                and not NON_PROJECT_PERIOD_CONTEXT.search(period_identity_context)
                and not BLANK_DATE_FORM.search(value)
                and not re.search(r"년\s*월\s*[~∼-]\s*\d{4}\s*년\s*월|"
                                  r"\d{4}\s*년\s*월\s*[~∼-]\s*년\s*월", value))
    if field == "제출 방식":
        if R.REFER_ONLY.search(value):
            return True
        if SUBMISSION_QUESTION.search(value[:180]):
            return False
        place_label = R.expression_key(hit.get("matched", "")) in {
            R.expression_key("제출처"), R.expression_key("제출장소")}
        return ((has_submission_mode(value) or (place_label and len(value) >= 3)
                 or bool(SUBMISSION_PLACE_EVIDENCE.search(value)))
                and not bool(BID_CONTRACT_ONLY.fullmatch(value)))
    if field == "참가 자격(면허·실적)":
        if QUALIFICATION_GUIDANCE.search(value):
            return False
        if QUALIFICATION_VAGUE.search(value.strip()):
            return False
        if (QUALIFICATION_DOCUMENT_ONLY.search(value)
                and not QUALIFICATION_EVIDENCE.search(value)):
            return False
        if QUALIFICATION_PUNISHMENT.search(context) and not QUALIFICATION_EVIDENCE.search(value):
            return False
        return bool(QUALIFICATION_EVIDENCE.search(value))
    if field == "사업분야":
        bad = re.search(r"(사업\s*분야[^\n]{0,20}실적|유사\s*사업|직원|참여인력|경력|"
                        r"실적표|사업수행\s*경험|전문가\s*평가|SDGs)", context)
        return not bad and not looks_header_row(value, hit.get("matched", ""),
                                                hit.get("heading", ""))
    if field == "지역제한":
        if re.search(r"(참여\s*지분율|평점|평가항목|가감점)", context):
            return False
        return bool(re.search(
            r"(서울|부산|대구|인천|광주|대전|울산|세종|경기|강원|충북|충남|"
            r"전북|전남|경북|경남|제주|[가-힣]+(?:광역)?시|[가-힣]+도)"
            r"[^\n]{0,40}(?:소재|본점|영업소|지역\s*제한)|"
            r"(?:소재지|본점|영업소)[^\n]{0,50}(?:서울|부산|대구|인천|광주|대전|"
            r"울산|세종|경기|강원|충북|충남|전북|전남|경북|경남|제주|[가-힣]+도)|"
            r"지역\s*제한\s*\([^)]{2,30}\)|지역\s*제한\s*(?:없음|미적용)", value))
    if field == "컨소시엄 요건":
        if (hit.get("source_type") == "attachment_form"
                and not re.search(r"본\s*사업[^\n]{0,80}(?:공동\s*(?:수급|계약)|컨소시엄)",
                                  f"{hit.get('heading', '')} {value}")):
            return False
        return bool(consortium_hit_kind(hit))
    if field == "예산":
        if (re.search(r"(?:(?:사업금액|추정가격)\s*)?\d[^\n]{0,20}미만\s*사업", value)
                and re.search(r"(입찰\s*참가|대기업|중견기업|중소기업|사업금액의\s*하한)",
                              context)):
            return False
        if R.REFER_ONLY.search(value):
            return True
        if re.fullmatch(r"\s*비\s*공개(?:\s*입찰)?\s*", value):
            return True
        if R.NOT_DISCLOSED.search(f"{hit.get('matched', '')} {value}"):
            return True
        if hit.get("source_type") == "attachment_form" and not hit.get("anchored"):
            return False
        identity = f"{hit.get('heading', '')} {value}"
        if BUDGET_BAD_CONTEXT.search(identity):
            return False
        if BUDGET_NON_CURRENT_CONTEXT.search(context):
            return False
        direct_united_number = (hit.get("anchored") and bool(re.search(
            r"(?:금\s*)?\d[\d,]{4,}(?:\s*\([^\n]{0,40}(?:만원|억원|VAT|부가))?", value)))
        return bool(BUDGET_AMOUNT_EVIDENCE.search(value) or direct_united_number)
    if field == "필수 제출 서류":
        # 목차에 서식 이름이 여럿 보여도 실제 제출 목록은 아니다.
        if hit.get("toc") and not actual_document_list:
            return False
        if actual_document_list:
            return True
        if GENERIC_APPLICATION_FORM.search(value):
            # 빈 신청서의 공통 문구라도 뒤에 실제 장문 목록이 이어질 수 있다.
            # 쓸 수 있는 후보로는 남기고, 외부참조-only 여부는 별도로 판정한다.
            return True
        if GENERIC_REQUIRED_DOC_STATEMENT.search(value):
            return bool(required_docs_referral_only(value))
        if REQUIRED_DOC_DIRECT_REFERENCE.search(value):
            return True
        if (submission_referral_only(value)
                or REQUIRED_DOC_REFERENCE_GUIDANCE.search(value)):
            return True
        # "확인서를 제출서류에 포함" 같은 설명 문장 하나를 전체 목록으로
        # 승격하지 않는다. 둘 이상의 구체 문서 또는 직접 항목명이 필요하다.
        if not hit.get("anchored") and document_evidence_count < 2:
            return False
        if REQUIRED_DOC_BAD_CONTEXT.search(context) and not REQUIRED_DOC_GOOD_CONTEXT.search(value):
            return False
        return bool(REQUIRED_DOC_GOOD_CONTEXT.search(context)
                    and DOCUMENT_LIST_EVIDENCE.search(value))
    if field == "과업 범위":
        if (SCOPE_BAD.search(context) or looks_header_row(value, hit.get("matched", ""),
                                                          hit.get("heading", ""))):
            return False
        letters = len(re.sub(r"[^가-힣A-Za-z]", "", value))
        return letters >= 8 and bool(SCOPE_EVIDENCE.search(value))
    if field == "평가 배점":
        if submission_referral_only(value):
            return True
        # 기술·가격의 전체 비율이 한 후보에 함께 있으면 그 자체로 이 필드의
        # 직접 증거다. 주변 기능요구사항 문구가 context 창에 걸쳤다는 이유로
        # 완결된 90/10·80/20 문장을 버리지 않는다.
        if evaluation_ratio_pair(value):
            return True
        if re.search(r"제\s*\d+\s*조", hit.get("heading", "")):
            return False
        # 시스템 기능 요구사항 안에 우연히 나온 점수/평가 용어는 이 입찰의
        # 제안서 배점이 아니다.
        if EVALUATION_BAD_CONTEXT.search(context) and not evaluation_ratio_pair(value):
            return False
        return bool(EVALUATION_SCORE_EVIDENCE.search(value))
    return True


def submission_referral_only(text: str) -> bool:
    """제출 방식이 다른 문서 참조뿐이고 실제 방법·장소는 없는가."""
    if not R.REFER_ONLY.search(text or ""):
        return False
    remainder = R.REFER_ONLY.sub(" ", text or "")
    remainder = re.sub(r"(?s)(?:문의처|문의\s*사항|담당\s*[:：]).*$", " ", remainder)
    # 참조 안내와 함께 실제 온라인/방문 제출을 명시했다면 그 수단은 값이다.
    if has_submission_mode(remainder) and not SUBMISSION_QUESTION.search(remainder):
        return False
    remainder = re.sub(
        r"(?m)^\s*(?:[가-하0-9]+[.)]?\s*)?(?:입찰참가자격\s*및\s*)?"
        r"(?:제안서\s*)?(?:제출|접수|발표)\s*"
        r"(?:처|장소|기한|기간|일정|방법|서류)?\s*[:：-]?\s*", " ", remainder)
    remainder = re.sub(r"(제안서|제출|접수|기한|기간|일정|방법|발표|해당\s*없음)", " ", remainder)
    return len(re.sub(r"[^가-힣A-Za-z0-9]", "", remainder)) < 8


def required_docs_referral_only(text: str) -> bool:
    """다른 문서/붙임을 가리킬 뿐 이 위치에 서류 목록이 없는가."""
    source = text or ""
    direct_reference = REQUIRED_DOC_DIRECT_REFERENCE.search(source)
    if direct_reference:
        remainder = REQUIRED_DOC_DIRECT_REFERENCE.sub(" ", source)
        return len(list(DOCUMENT_LIST_EVIDENCE.finditer(remainder))) < 1
    generic = GENERIC_APPLICATION_FORM.search(source)
    if generic:
        evidence = len(list(DOCUMENT_LIST_EVIDENCE.finditer(source)))
        if evidence < 4:
            return True
    if REQUIRED_DOC_EXTERNAL_WORDING.search(source):
        remainder = REQUIRED_DOC_EXTERNAL_WORDING.sub(" ", source)
        return len(list(DOCUMENT_LIST_EVIDENCE.finditer(remainder))) < 2
    guidance = REQUIRED_DOC_REFERENCE_GUIDANCE.search(source)
    if guidance:
        # 이 문구 안의 '입찰참가신청서'는 실제 제출서류가 아니라 참조할
        # 신청서 이름이다. 해당 안내만 지운 뒤 실제 목록이 남는지 본다.
        remainder = REQUIRED_DOC_REFERENCE_GUIDANCE.sub(" ", source)
        return len(list(DOCUMENT_LIST_EVIDENCE.finditer(remainder))) < 1
    if not R.external_reference_target(source):
        return False
    # '제안서 10부, USB 1매 ... 입찰서류는 공고문 참조'처럼 실제 목록이
    # 함께 있으면 참조-only가 아니다. 다만 '입찰참가신청서의 붙임자료에
    # 의함' 안의 신청서 명칭은 실제 제출서류 한 항목으로 세지 않는다.
    return len(list(DOCUMENT_LIST_EVIDENCE.finditer(source))) < 1


def required_docs_answer(text: str) -> str:
    """제출서류 후보에서 안내문·빈 신청서 꼬리를 빼고 실제 목록만 남긴다."""
    value = (text or "").strip()
    stop = REQUIRED_DOC_STOP.search("\n" + value)
    if stop:
        value = ("\n" + value)[:stop.start()].lstrip()
    marker = re.search(r"(?:붙임|제출)\s*서류\s*[:：]?\s*", value)
    before_evidence = DOCUMENT_LIST_EVIDENCE.search(value)
    if marker and (before_evidence is None or marker.start() <= before_evidence.start()):
        value = value[marker.end():]
    first = DOCUMENT_LIST_EVIDENCE.search(value)
    if first and first.start() > 0:
        value = value[first.start():]
    trailing = REQUIRED_DOC_TRAILING_FORM.search(value)
    if trailing:
        value = value[:trailing.start()]
    value = re.sub(r"\s+(?:20\s*\d{0,2}\s*[년.]\s*(?:\d{0,2}\s*[월.]\s*)?"
                   r"\d{0,2}\s*(?:일|\.)?|년\s*월\s*일)\s*$", "", value)
    value = re.split(r"\s+(?:신청인\s*[:：]|[가-힣A-Za-z]+장\s*귀하)\b",
                     value, maxsplit=1)[0]
    return value.strip(" \n·․-–—")[:MAX_VALUE_CHARS]


def external_reference_answer(field: str, text: str) -> str:
    """외부 자료 안내를 사용자에게 보여줄 짧은 답으로 보존한다."""
    value = (text or "").strip()
    if field != "필수 제출 서류":
        return value[:MAX_VALUE_CHARS]
    direct_reference = REQUIRED_DOC_DIRECT_REFERENCE.search(value)
    if direct_reference:
        return direct_reference.group(0).strip()
    direct = re.search(
        r"(?:제출\s*서류\s*(?:는|은)?\s*[:：]?\s*)?(?:[“\"]?입찰\s*공고(?:문|서|사항)?[”\"]?"
        r"\s*(?:에\s*)?(?:참조|따름)|붙임\s*참조|공고로써\s*정한\s*서류)",
        value, flags=re.IGNORECASE)
    if direct:
        return direct.group(0).strip()
    return required_docs_answer(value)


def external_reference_only(field: str, hit: dict) -> bool:
    """후보가 실제 값 없이 명시적인 외부 자료만 가리키는지 판정한다."""
    value = hit.get("value", "") or ""
    explicit_required_docs_reference = (field == "필수 제출 서류" and bool(
        REQUIRED_DOC_EXTERNAL_WORDING.search(value)
        or REQUIRED_DOC_DIRECT_REFERENCE.search(value)
        or GENERIC_APPLICATION_FORM.search(value)
        or REQUIRED_DOC_REFERENCE_GUIDANCE.search(value)))
    if not R.external_reference_target(value) and not explicit_required_docs_reference:
        return False
    if field == "제출 방식":
        return submission_referral_only(value)
    if field == "필수 제출 서류":
        direct_start = re.match(
            r"^\s*(?:[#>*□◦ㅇ○●❍-]+\s*)*(?:[가-하0-9]+[.)]?\s*)?"
            r"(?:입찰\s*참가\s*)?(?:필수\s*)?(?:제출|구비)\s*서류\s*(?:제출)?"
            r"(?:\s*(?:는|은|및|:|：|-)|\s*$)",
            hit.get("raw", "") or "")
        generic_reference = bool(GENERIC_APPLICATION_FORM.search(value)
                                 or REQUIRED_DOC_EXTERNAL_WORDING.search(value))
        return bool((hit.get("anchored") or direct_start or generic_reference)
                    and required_docs_referral_only(value))
    if field == "예산":
        return not BUDGET_AMOUNT_EVIDENCE.search(value)
    if field == "참가 자격(면허·실적)":
        return not QUALIFICATION_EVIDENCE.search(value)
    if field == "평가 배점":
        direct_label = R.expression_key(hit.get("matched", "")) in CORE_LABEL_KEYS["평가 배점"]
        return bool(direct_label and not EVALUATION_SCORE_EVIDENCE.search(value))
    return bool(hit.get("anchored"))


def conflict_result(field: str, rep: dict, candidates: list[dict], reason: str) -> dict:
    """충돌하는 모든 문구와 위치를 버리지 않고 하나의 결과로 만든다."""
    ordered = [rep] + [h for h in candidates if h is not rep]
    distinct = []
    seen = set()
    for hit in ordered:
        value = (hit.get("value") or "").strip()
        key = (hit.get("line"), value)
        if value and key not in seen:
            distinct.append(hit)
            seen.add(key)
    answer = "\n---\n".join(h["value"].strip() for h in distinct)[:MAX_VALUE_CHARS]
    additional = [location_payload(h) for h in distinct[1:]][:12]
    return {"status": "conflict", "answer_raw": answer,
            "answer_normalized": normalize_value(field, answer),
            "matched_expression": rep["matched"], "rep": rep,
            "additional": additional, "confidence": "rule_medium",
            "review_reason": reason}


def pick_representative(field: str, hits: list[dict]) -> tuple[dict | None, list[dict]]:
    """대표 출처 하나를 고르고 **나머지는 버리지 않고** 함께 돌려준다."""
    # 첨부 서식이라고 무조건 버리면 안 된다. 제출서류 목록처럼 뒤쪽 서식에만
    # 내용이 실려 있는 문서가 있다. 배제 대상은 '제안사가 채울 빈칸'뿐이다.
    usable = [h for h in hits
              if (h["source_type"] != "attachment_form" or not R.looks_blank(h["value"]))
              and field_hit_usable(field, h)]
    # 항목명 자리에 있는 등장을 먼저 쓴다. 하나도 없으면 그때만 문장 속 등장을 쓴다.
    anchored = [h for h in usable if h.get("anchored")]
    if field == "컨소시엄 요건":
        policy = [h for h in usable if consortium_hit_kind(h)]
        pool = policy or anchored or usable
    elif field == "제출 방식":
        exact_methods = [h for h in usable if R.expression_key(h.get("matched", "")) in {
            R.expression_key(x) for x in ("제출방법", "제안서 제출방법", "접수방법",
                                           "제출기한 및 방법")}]
        pool = exact_methods or anchored or usable
    elif field == "평가 배점":
        # 제목에 매달린 협상 문턱보다 기술/가격 수치가 함께 적힌 완결 문장을
        # 전체 후보에서 먼저 고른다(문장형 표현은 항목명이 아닐 수 있다).
        ratios = [h for h in usable if evaluation_ratio_pair(h["value"])]
        complete = [h for h in usable if evaluation_complete(h["value"])]
        pool = ratios or complete or anchored or usable
    elif field == "필수 제출 서류":
        # 본문이 '붙임 참조'로만 연결되더라도 같은 파일 뒤쪽 붙임에 실제
        # 입찰서류 목록이 있으면 그 목록을 답으로 쓴다.
        actual_docs = [h for h in usable if not required_docs_referral_only(h["value"])]
        pool = actual_docs or anchored or usable
    else:
        pool = anchored or usable
    if not pool:
        return None, hits
    # 표 머리글·법령 인용만 남은 경우가 아니면 그것들을 대표로 쓰지 않는다
    def junk(h):
        return (1 if (looks_header_row(h["value"], h["matched"], h["heading"])
                      or looks_statute(h["value"])) else 0)
    solid = [h for h in pool if not junk(h)]
    pool = solid or pool
    # '입찰공고문 참조'만 적힌 위치보다 실제 값이 적힌 위치를 먼저 쓴다
    def referral(h):
        v = h["value"]
        referred = (bool(REQUIRED_DOC_REFERENCE_GUIDANCE.search(v))
                    if field == "필수 제출 서류" else submission_referral_only(v))
        return 1 if referred else 0
    if field != "제출 방식":
        solid2 = [h for h in pool if not referral(h)]
        pool = solid2 or pool

    if field == "사업기간":
        # '사업기간 산정서' 같은 제목만 있는 위치를 대표로 쓰지 않는다
        real = [h for h in pool if field_hit_usable(field, h)]
        pool = real or pool

    if field == "평가 배점":
        # 85% 협상 문턱이나 하위 3점 항목보다 기술/가격의 전체 배점이
        # 함께 적힌 후보를 우선한다. 길이가 아니라 답의 완결성 기준이다.
        ratios = [h for h in pool if evaluation_ratio_pair(h["value"])]
        complete = [h for h in pool if evaluation_complete(h["value"])]
        pool = ratios or complete or pool

    # 같은 조건이면 본문을 첨부 서식보다 먼저 쓴다
    def att(h):
        return 1 if h["source_type"] == "attachment_form" else 0
    if field == "컨소시엄 요건":
        def consortium_policy(h):
            v = h["value"]
            kind = consortium_hit_kind(h)
            subject = re.search(r"(공동\s*(?:수급|이행|계약|도급)|컨소시엄)", v)
            return (0 if kind in {"allow", "deny", "conditional_allow"} else 1,
                    att(h), len(v), subject.start() if subject else 9999, h["line"])
        pool = sorted(pool, key=consortium_policy)
    elif field == "제출 방식":
        def submission_policy(h):
            v = h["value"]
            place_label = R.expression_key(h.get("matched", "")) in {
                R.expression_key("제출처"), R.expression_key("제출장소")}
            actual_mode = has_submission_mode(v) and not submission_referral_only(v)
            actual_action = actual_mode and bool(SUBMISSION_ACTION.search(v))
            strong_action = actual_mode and bool(re.search(
                r"(?:통해\s*)?온라인으로\s*제출|직접\s*제출|방문\s*(?:접수|제출)", v))
            actual_place = place_label and not submission_referral_only(v)
            referred = submission_referral_only(v)
            method_label = R.expression_key(h.get("matched", "")) in {
                R.expression_key(x) for x in ("제출방법", "제안서 제출방법", "접수방법",
                                               "제출기한 및 방법")}
            rank = (0 if method_label and strong_action
                    else 1 if strong_action
                    else 2 if method_label and actual_action
                    else 3 if actual_action else 4 if method_label and referred
                    else 5 if actual_mode else 6 if referred
                    else 7 if actual_place else 8)
            return (rank, att(h), len(v), h["line"])
        pool = sorted(pool, key=submission_policy)
    elif field == "필수 제출 서류":
        def required_document_policy(h):
            value = h.get("value", "")
            evidence = len(list(DOCUMENT_LIST_EVIDENCE.finditer(value)))
            generic = bool(GENERIC_APPLICATION_FORM.search(value)
                           or GENERIC_REQUIRED_DOC_STATEMENT.search(value))
            referred = required_docs_referral_only(value)
            bad = bool(REQUIRED_DOC_BAD_CONTEXT.search(
                f"{h.get('context', '')} {value}")
                and not REQUIRED_DOC_GOOD_CONTEXT.search(value))
            if evidence >= 2 and not generic and not bad and not referred:
                kind = 0
            elif referred and not generic:
                kind = 1
            elif referred:
                kind = 2
            else:
                kind = 3
            direct_section = (0 if h.get("anchored") and h.get("hierarchical")
                              and R.expression_key(h.get("matched", ""))
                              in CORE_LABEL_KEYS["필수 제출 서류"] else 1)
            return (kind, direct_section, -evidence, att(h),
                    not h.get("anchored"), h["line"])
        pool = sorted(pool, key=required_document_policy)
    elif field in PREF_BODY_FIRST:
        # 문장 길이는 정확도의 근거가 아니다. 필드를 직접 선언한 제목/표
        # 항목 → 그 계층 본문 → 앞머리 명시값 → 표 값 → 문장 언급 순이다.
        core = CORE_LABEL_KEYS.get(field, set())

        def explicit_policy(h):
            key = R.expression_key(h.get("matched", ""))
            label_rank = SCOPE_LABEL_PRIORITY.get(key, 9) if field == "과업 범위" else 0
            if h.get("anchored") and key in core and h.get("hierarchical"):
                semantic_rank = 0
            elif h.get("anchored") and key in core:
                semantic_rank = 1
            elif h.get("anchored") and h.get("hierarchical"):
                semantic_rank = 2
            elif h.get("anchored"):
                semantic_rank = 3
            elif h["source_type"] == "front_matter":
                semantic_rank = 4
            elif h["source_type"] == "body_table":
                semantic_rank = 5
            else:
                semantic_rank = 6
            return (semantic_rank, label_rank, att(h), h["line"])

        pool = sorted(pool, key=explicit_policy)
    else:
        # 값 하나짜리는 앞머리 개요와 본문을 먼저 본다
        pool = sorted(pool, key=lambda h: (R.looks_blank(h["value"]), att(h), h["line"]))
    rep = pool[0]
    required_docs_original = None
    if field == "필수 제출 서류":
        required_docs_original = rep
        rep = dict(rep)
        rep["value"] = required_docs_answer(rep["value"])
    if field == "제출 방식" and re.fullmatch(
            r"(?:밀봉|봉합|날인|간인|밀봉\s*및\s*봉합날인|밀봉\s*후\s*제출)[가-힣\s·ㆍ/]*",
            rep.get("value", "")) and re.search(r"가격\s*제안서", rep.get("context", "")):
        rep = dict(rep)
        rep["value"] = "가격제안서: " + rep["value"].strip()
    if field == "사업 개요":
        # 장 전체 대신 그 안의 실제 사업명·목적·개요 내용을 대표값으로 쓴다.
        rep = dict(rep)
        rep["value"] = overview_value(rep["value"])
    if field == "과업 범위":
        rep = dict(rep)
        rep["value"] = re.split(
            r"\n\s*[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]+\s*(?:운영\s*현황|사업\s*현황|추진\s*현황)",
            rep["value"], maxsplit=1)[0].strip()
    others = [h for h in hits if h is not (required_docs_original or rep)]
    return rep, others


def normalize_value(field: str, text: str) -> str:
    """조건 검색에 필요한 최소한의 정규화만 한다(뜻을 바꾸지 않는다)."""
    t = re.sub(r"\s+", " ", R.nfc(text)).strip()
    if field in LIST_FIELDS:
        parts = [p.strip(" ·-–—ㆍ") for p in re.split(r"[\n]+|(?<=[.。])\s+", text)]
        parts = [p for p in (re.sub(r"\s+", " ", x).strip() for x in parts) if p]
        return jdump(parts[:60])
    return t[:MAX_VALUE_CHARS]


# 실제 '기간'을 말하는 표현(날짜 구간·개월수·일수·계약일 기준)
PERIOD_RE = re.compile(
    r"(\d{4}\s*[.\-/년]\s*\d{1,2}|\d{1,3}\s*(개월|일간|일\s*간|일\s*이내|주간|년간)|"
    r"계약\s*(?:체결\s*)?(?:일|일자)?\s*(?:로)?부터|계약\s*(?:체결\s*)?후\s*\d{1,3}\s*일|"
    r"체결일로부터|착수\s*일로부터|~\s*\d{4})")


def has_period(text: str) -> bool:
    """이 값이 실제 기간을 말하고 있는가(서식 제목·목차 언급과 구분)."""
    return bool(PERIOD_RE.search(text))


# 사업기간이 아닌 다른 기간(하자보수·무상보수 등)
NOT_PROJECT_PERIOD = re.compile(r"(하도급\s*(?:계약\s*)?기간|하자\s*(보수|담보)|"
                                r"무상\s*(보수|하자)|보증\s*기간|안정화\s*지원계획)")
# 여러 차수·단계로 나뉜 기간
PHASED = re.compile(r"([1-9]\s*차|[1-9]\s*단계)")


def period_tokens(text: str) -> frozenset:
    """기간을 '표기 차이 없는 조각'으로 바꾼다.

    2025.1.1. 과 2025.01.01. 은 같은 날이고, 4월(120일) 과 4개월도 같은 기간이다.
    앞의 0을 떼고 단위를 통일해야 표기만 다른 값을 충돌로 오해하지 않는다.
    """
    out = set()
    date_re = re.compile(
        r"(\d{2,4})\s*[.\-/년]\s*(\d{1,2})\s*[.\-/월]?\s*(\d{0,2})")
    date_matches = list(date_re.finditer(text))
    for match in date_matches:
        y, mo, d = match.groups()
        yy = int(y)
        yy = yy + 2000 if yy < 100 else yy
        out.add(f"D{yy}-{int(mo)}" + (f"-{int(d)}" if d else ""))
    for match in re.finditer(r"(\d+(?:\.\d+)?)\s*(개월|달|월|주|년|일)", text):
        # 날짜의 2027년·9월·30일을 각각 2027년/9개월/30일의 기간으로
        # 다시 세지 않는다.
        if any(match.start() < dm.end() and dm.start() < match.end()
               for dm in date_matches):
            continue
        n, u = match.groups()
        v = float(n)
        if u in ("개월", "달", "월"):
            out.add(f"M{v:g}")
        elif u == "년":
            out.add(f"M{v * 12:g}")
        elif u == "주":
            out.add(f"M{v * 7 / 30:.0f}")
        elif u == "일" and v >= 20:
            out.add(f"M{v / 30:.0f}")       # 120일 == 4개월
    return frozenset(out)


def comparable_period_difference(a: frozenset, b: frozenset) -> bool:
    """두 기간 증거가 같은 종류이고 실제로 다를 때만 참이다."""
    ad = {x for x in a if x.startswith("D")}
    bd = {x for x in b if x.startswith("D")}
    if ad and bd:
        # 2027-09와 2027-09-30처럼 한쪽이 일자를 생략한 표기는 양립한다.
        compatible = any(x == y or x.startswith(y + "-") or y.startswith(x + "-")
                         for x in ad for y in bd)
        if not compatible:
            return True
    am = {x for x in a if x.startswith("M")}
    bm = {x for x in b if x.startswith("M")}
    return bool(am and bm and am.isdisjoint(bm))


def substance(field: str, text: str) -> str:
    """두 값이 '같은 말'인지 비교하려고 실질만 뽑는다(표현 차이는 무시)."""
    if field == "예산":
        v = R.budget_value(text, "")
        return str(v.get("krw") or "")
    if field == "사업기간":
        if not has_period(text) or NOT_PROJECT_PERIOD.search(text):
            return ""          # 기간 표현이 없거나 사업기간이 아니면 비교 대상이 아니다
        return "|".join(sorted(period_tokens(text)))
    return re.sub(r"[^가-힣]", "", text)[:40]


def decide(field: str, hits: list[dict], doc_text: str) -> dict:
    """규칙만으로 낼 수 있는 결론. 애매하면 원문 추가 확인 대상으로 넘긴다."""
    rep, others = pick_representative(field, hits)
    add = [location_payload(h) for h in others][:12]

    # (가) 어디에도 없다 → 없음. **여기서 '제한 없음'·'단독 가능' 으로 바꾸지 않는다.**
    if not hits:
        return {"status": "field_absent", "answer_raw": "", "answer_normalized": "",
                "matched_expression": "", "rep": None, "additional": [],
                "confidence": "rule_high",
                "review_reason": ""}

    # 현재 문서에 실제 값은 없고 확인할 외부 자료만 명시된 경우는 기술적
    # 실패가 아니라 정상적인 사용자 안내 상태다. 같은 문서의 다른 후보에
    # 실제 값이 있으면 그 값을 우선하고 참조 위치는 additional에 남긴다.
    referral_hits = [h for h in hits if external_reference_only(field, h)]
    actual_hits = [h for h in hits if field_hit_usable(field, h)
                   and not external_reference_only(field, h)]
    if referral_hits and not actual_hits:
        reference = rep if rep in referral_hits else sorted(
            referral_hits, key=lambda h: (not h.get("anchored"), h["line"]))[0]
        target = R.external_reference_target(reference.get("value", ""))
        additional = [location_payload(h)
                      for h in referral_hits if h is not reference][:12]
        return {"status": "external_reference",
                "answer_raw": external_reference_answer(field, reference["value"]),
                "answer_normalized": normalize_value(
                    field, external_reference_answer(field, reference["value"])),
                "matched_expression": reference["matched"], "rep": reference,
                "additional": additional, "confidence": "rule_high",
                "review_reason": (f"현재 문서에는 실제 값 대신 {target} 확인 안내만 있음; "
                                  f"{target}에서 실제 값을 확인해야 함")}

    # (나) 문서가 그 필드의 값을 밝히지 않겠다고 했다.
    #     문서 아무 데나 있는 '비공개'가 아니라, **대표 위치의 값 안에서** 확인한다.
    scope = " ".join(h["value"] for h in hits if h["value"])[:4000]
    # '사업예산 : 비공개' 처럼 주어가 항목명 쪽에 있으면 값만 봐서는 알 수 없다.
    # 항목명과 값을 붙여서 함께 본다.
    nd = None
    if rep is not None:
        nd = R.NOT_DISCLOSED.search(rep["value"])
        # 값 칸이 사실상 '비공개' 한 마디뿐일 때만 항목명을 주어로 붙여 본다.
        # 값에 실제 금액·내용이 이미 있으면 비공개가 아니다.
        short_and_empty = (len(rep["value"]) <= 20
                           and not re.search(r"\d", rep["value"]))
        if nd is None and short_and_empty:
            nd = R.NOT_DISCLOSED.search(f'{rep["matched"]} {rep["value"]}')
        if nd is None and field == "예산" and short_and_empty:
            nd = re.search(r"비\s*공개", rep["value"])
    if nd is not None and not (field == "평가 배점"
                               and evaluation_complete(rep["value"])):
        m = nd
        return {"status": "not_disclosed", "answer_raw": rep["value"][:MAX_VALUE_CHARS],
                "answer_normalized": normalize_value(field, rep["value"]),
                "matched_expression": rep["matched"], "rep": rep, "additional": add,
                "confidence": "rule_high",
                "review_reason": f"문서가 비공개/미포함이라고 명시: {m.group(0)}"}

    # (다) 쓸 수 있는 위치가 첨부 빈 서식·제안업체 서식뿐이다 → 값으로 쓰지 않는다
    if rep is None:
        ambiguous = next((h for h in hits if h.get("anchored")
                          and AMBIGUOUS_FIELD_VALUE.fullmatch(
                              (h.get("value") or "").strip())), None)
        if ambiguous is not None:
            return {"status": "review_required",
                    "answer_raw": ambiguous["value"][:MAX_VALUE_CHARS],
                    "answer_normalized": "", "matched_expression": ambiguous["matched"],
                    "rep": ambiguous, "additional": add, "confidence": "rule_medium",
                    "review_reason": "원문 문구가 애매하여 해당 필드의 실제 값으로 자동 확정하기 어려움"}
        only_att = all(h["source_type"] == "attachment_form" for h in hits)
        only_toc = all(h.get("toc") or looks_toc_value(h.get("value", "")) for h in hits)
        irrelevant_period = (field == "사업기간"
                             and all(not field_hit_usable(field, h) for h in hits))
        bad_qualification = (field == "참가 자격(면허·실적)" and all(
            QUALIFICATION_PUNISHMENT.search(
                f"{h.get('context', '')} {h.get('heading', '')} {h.get('value', '')}")
            for h in hits))
        qualification_ref_only = (field == "참가 자격(면허·실적)"
                                  and any(R.REFER_ONLY.search(h.get("value", ""))
                                          for h in hits))
        irrelevant_qualification = (field == "참가 자격(면허·실적)"
                                    and not qualification_ref_only
                                    and all(not field_hit_usable(field, h) for h in hits))
        bad_field_context = (field in {"사업분야", "필수 제출 서류"} and all(
            h.get("proposer_context") or REQUIRED_DOC_BAD_CONTEXT.search(
                f"{h.get('context', '')} {h.get('value', '')}") for h in hits))
        irrelevant_documents = (field == "필수 제출 서류"
                                and all(not field_hit_usable(field, h) for h in hits))
        irrelevant_consortium = (field == "컨소시엄 요건"
                                 and all(not field_hit_usable(field, h) for h in hits))
        irrelevant_evaluation = (field == "평가 배점"
                                 and all(not field_hit_usable(field, h) for h in hits))
        irrelevant_scope = (field == "과업 범위"
                            and all(not field_hit_usable(field, h) for h in hits))
        irrelevant_budget = (field == "예산"
                             and all(not field_hit_usable(field, h) for h in hits))
        direct_region_declaration = (field == "지역제한" and any(
            re.search(r"입찰\s*방식[^\n]{0,35}지역\s*제한",
                      f"{h.get('heading', '')} {h.get('value', '')}")
            for h in hits))
        irrelevant_region = (field == "지역제한" and not direct_region_declaration
                             and all(not field_hit_usable(field, h) for h in hits))
        irrelevant_submission = (field == "제출 방식"
                                 and all(not field_hit_usable(field, h) for h in hits))
        if only_toc:
            reason = "목차·쪽수·제목 줄에서만 항목명이 보이며 실제 값 구간은 없음"
        elif irrelevant_period:
            reason = "하도급·보증·과거 이력·빈 날짜 서식만 있고 현재 발주 사업기간은 없음"
        elif bad_qualification:
            reason = "참가자격 제한·처벌 문장만 있고 실제 입찰 참가 자격은 없음"
        elif irrelevant_qualification:
            reason = "인력 자격·제출서류·부수 문맥뿐이며 실제 입찰 참가 자격은 없음"
        elif bad_field_context or irrelevant_documents:
            reason = "제안업체 실적·계약수행·하도급 문맥뿐이며 발주 단계의 값은 없음"
        elif irrelevant_consortium:
            reason = "공동수급 서식·부수적 언급뿐이며 이 사업의 컨소시엄 조건은 없음"
        elif irrelevant_evaluation:
            reason = "평가 기능·위원회 운영·법령 언급뿐이며 이 입찰의 실제 배점은 없음"
        elif irrelevant_scope:
            reason = "변경 절차·제안서 작성 안내·표 머리글뿐이며 실제 과업 범위는 없음"
        elif irrelevant_budget:
            reason = "기업 참여제한용 사업금액 구간만 있고 실제 예산 금액은 없음"
        elif irrelevant_region:
            reason = "평가표의 지역 관련 항목뿐이며 실제 입찰 지역 제한은 없음"
        elif irrelevant_submission:
            reason = "입찰·계약 방식 또는 부수 문맥뿐이며 실제 제안서 제출 방식은 없음"
        else:
            reason = ("뒤쪽 첨부·빈 서식에서만 항목명이 보임" if only_att
                      else "항목명은 있으나 발주 사업의 해당 필드 값으로 쓸 수 없음")
        # 명세 §6: 사업분야·사업 개요는 제안업체 실적표·일반현황의 같은 이름 칸을
        # 값으로 쓰지 않는다. 발주 사업 쪽에 명시가 없으면 '없음'이 정확한 답이다.
        if (field in {"사업분야", "사업 개요"} or irrelevant_period
                or bad_qualification or irrelevant_qualification
                or bad_field_context or irrelevant_documents
                or irrelevant_consortium or irrelevant_evaluation or irrelevant_scope
                or irrelevant_budget or irrelevant_region or irrelevant_submission):
            return {"status": "field_absent", "answer_raw": "", "answer_normalized": "",
                    "matched_expression": hits[0]["matched"], "rep": None,
                    "additional": [location_payload(h) for h in hits][:12],
                    "confidence": "rule_high",
                    "review_reason": reason}
        return {"status": "extraction_failed", "answer_raw": "", "answer_normalized": "",
                "matched_expression": hits[0]["matched"], "rep": None,
                "additional": [location_payload(h) for h in hits][:12],
                "confidence": "rule_medium",
                "review_reason": reason + " — 발주 사업의 값으로 쓸 수 없음"}

    # (라) 항목명은 있는데 값이 비어 있거나 '공고문 참조' 뿐이다
    if looks_header_row(rep["value"], rep["matched"], rep["heading"]):
        return {"status": "extraction_failed", "answer_raw": "", "answer_normalized": "",
                "matched_expression": rep["matched"], "rep": rep, "additional": add,
                "confidence": "rule_medium",
                "review_reason": "표의 열 이름 줄만 있고 실제 값이 없음"}
    current_consortium_rule = (field == "컨소시엄 요건"
                               and CONSORTIUM_VALUE.search(rep["value"]))
    if looks_statute(rep["value"]) and field == "지역제한":
        # 명세 §6: 지역 제한이 실제로 명시되지 않았으면 field_absent
        return {"status": "field_absent", "answer_raw": "", "answer_normalized": "",
                "matched_expression": rep["matched"], "rep": None, "additional": add,
                "confidence": "rule_high",
                "review_reason": "법령 조문 인용만 있고 이 사업의 지역 제한 명시 없음"}
    if looks_statute(rep["value"]) and not current_consortium_rule:
        return {"status": "extraction_failed",
                "answer_raw": external_reference_answer(field, rep["value"]),
                "answer_normalized": "",
                "matched_expression": rep["matched"], "rep": rep, "additional": add,
                "confidence": "rule_medium",
                "review_reason": "법령·예규 조문 인용이며 이 사업의 조건이 아님"}
    if R.looks_blank(rep["value"]):
        return {"status": "extraction_failed", "answer_raw": "", "answer_normalized": "",
                "matched_expression": rep["matched"], "rep": rep, "additional": add,
                "confidence": "rule_medium",
                "review_reason": "항목명은 있으나 값 칸이 비어 있음"}
    # '공고문 참조' 가 붙어 있어도 함께 적힌 실제 값(예: 온라인 제출)이 있으면 값이다.
    referral_only = (required_docs_referral_only(rep["value"])
                     if field == "필수 제출 서류"
                     else submission_referral_only(rep["value"]))
    if referral_only:
        target = R.external_reference_target(rep["value"])
        return {"status": "external_reference",
                "answer_raw": rep["value"][:MAX_VALUE_CHARS], "answer_normalized": "",
                "matched_expression": rep["matched"], "rep": rep, "additional": add,
                "confidence": "rule_high",
                "review_reason": (f"현재 문서에는 실제 값 대신 {target} 확인 안내만 있음; "
                                  f"{target}에서 실제 값을 확인해야 함")}

    # (마) 컨소시엄 요건: 동일한 현재 사업에 허용·불허 정책이 각각 있으면
    # 실제 충돌로 남긴다. 평가 산식·실적서식·구성원 언급은 위의 정책 분류에서
    # 제외되며, 공동이행 허용/분담이행 불허는 conditional_allow 하나다.
    if field == "컨소시엄 요건":
        internal_conflict = None
        for h in hits:
            one = (h.get("value") or "")[:500]
            if consortium_policy_kind(one) == "conditional_allow":
                continue
            allow_word = re.search(
                r"(?:공동\s*(?:수급|이행|계약|도급)|컨소시엄)[^\n]{0,35}"
                r"(?:가능|허용)(?!\s*(?:하지|되지|않|아니))", one)
            deny_word = re.search(
                r"(?:공동\s*(?:수급|이행|계약|도급)|컨소시엄)[^\n]{0,35}"
                r"(?:불가(?!피)|불허|금지|허용\s*하지\s*않)", one)
            if deny_word and re.search(r"(하도급|협정서\s*제출)", deny_word.group(0)):
                deny_word = None
            if allow_word and deny_word:
                internal_conflict = (allow_word.group(0), deny_word.group(0))
                break
        if internal_conflict:
            return conflict_result(
                field, rep, [rep],
                f"한 정책 문맥에 허용({internal_conflict[0]})과 "
                f"불허({internal_conflict[1]})가 함께 있음")
        policies = [(h, consortium_hit_kind(h)) for h in hits
                    if h["source_type"] != "attachment_form"
                    and field_hit_usable(field, h)]
        allows = [h for h, kind in policies if kind in {"allow", "conditional_allow"}]
        denies = [h for h, kind in policies if kind == "deny"]
        if allows and denies:
            return conflict_result(
                field, rep, [allows[0], denies[0]],
                f"현재 사업의 허용 정책 L{allows[0]['line']}과 "
                f"불허 정책 L{denies[0]['line']}이 서로 다름")

    # (바) 서로 다른 실제 값이 여러 위치에 있으면 충돌로 남긴다
    # 글자가 다른 것만으로는 충돌이 아니다(같은 값을 줄여 쓴 것일 수 있다).
    # 숫자·단위 같은 '실질'이 서로 다를 때만 충돌로 본다.
    real = [h for h in hits if h["source_type"] != "attachment_form"
            and h.get("anchored") and not R.looks_blank(h["value"])
            and field_hit_usable(field, h)]
    if field == "사업기간" and len(real) > 1:
        cand = [h for h in real
                if has_period(h["value"]) and not NOT_PROJECT_PERIOD.search(h["value"])]
        toks = [period_tokens(h["value"]) for h in cand]
        toks = [x for x in toks if x]
        # 종료일만 적은 값과 개월수만 적은 값은 계약 시작일 없이는 서로
        # 다른지 판단할 수 없다. 날짜끼리 또는 기간 길이끼리처럼 같은 종류의
        # 증거가 모두 있는데 값이 다를 때에만 충돌로 본다.
        disagreement = any(comparable_period_difference(a, b)
                           for i, a in enumerate(toks) for b in toks[i + 1:])
        phased = any(PHASED.search(h["value"]) for h in cand)
        if len(toks) > 1 and disagreement and not phased:
            sig = {"|".join(sorted(x)) for x in toks}
            return conflict_result(
                field, rep, cand,
                f"서로 다른 기간이 {len(sig)}가지 있음 — 용도 구분 필요")
    if field in {"예산", "지역제한"} and len(real) > 1:
        sig = {v for v in (substance(field, h["value"]) for h in real) if v}
        if len(sig) > 1:
            return conflict_result(
                field, rep, real,
                f"서로 다른 값이 {len(sig)}곳에 있음 — 용도 구분 필요")

    return {"status": "value_present", "answer_raw": rep["value"][:MAX_VALUE_CHARS],
            "answer_normalized": normalize_value(field, rep["value"]),
            "matched_expression": rep["matched"], "rep": rep, "additional": add,
            "confidence": "rule_high", "review_reason": ""}


# ══════════════════════════════════════════════════════════════════
# 4. 한 문서 처리
# ══════════════════════════════════════════════════════════════════

def process_document(reg_row: dict, meta_row: dict | None, decisions: dict,
                     alias_seen: list, corpus_version: str, registry_version: str,
                     rules_sha: str) -> list[dict]:
    name = reg_row["source_filename_nfc"]
    text = (CORPUS_DIR / "md" / reg_row["output_filename"]).read_text(encoding="utf-8")
    lines = text.split("\n")
    section_index = build_section_index(lines)
    side = load_sidecar(name)
    att = attachment_start(side, len(lines), lines)
    front_limit = max(40, int(len(lines) * FRONT_RATIO))
    alias_seen.extend(discover_aliases(lines, reg_row, att, front_limit))

    rows = []
    for field in R.FIELDS:
        if field == "공고일":
            # 공고일은 원본 메타데이터가 단일 공식 출처다. 본문 날짜를 섞지 않는다.
            raw = (meta_row or {}).get("공개 일자", "").strip()
            if raw:
                out = {"status": "value_present", "answer_raw": raw,
                       "answer_normalized": raw.split(" ")[0],
                       "matched_expression": "공개 일자",
                       "rep": {"source_type": "metadata_csv",
                               "line": 0, "heading": "data_list.csv 공개 일자",
                               "value": raw},
                       "additional": [], "confidence": "rule_high", "review_reason": ""}
            else:
                out = {"status": "field_absent", "answer_raw": "", "answer_normalized": "",
                       "matched_expression": "", "rep": None, "additional": [],
                       "confidence": "rule_high",
                       "review_reason": ""}
            method = "metadata_csv"
        else:
            hits = find_hits(lines, field, att, front_limit, section_index)
            out = decide(field, hits, text)
            method = "rule_alias_match"

        rep = out.get("rep")
        rep_type = rep["source_type"] if rep else ""
        rep_loc = ""
        if rep:
            rep_loc = jdump({"document_id": reg_row["document_id"], "file": name,
                             **location_payload(rep)})
        excerpt = R.redact((rep or {}).get("value", ""))[:MAX_EXCERPT_CHARS]

        row = {
            "schema_version": SCHEMA_VERSION, "extraction_version": EXTRACTION_VERSION,
            "corpus_version": corpus_version, "registry_version": registry_version,
            "document_id": reg_row["document_id"],
            "document_version": reg_row["document_version"],
            "source_filename_nfc": name, "active": reg_row["active"],
            "retrieval_eligible": reg_row["retrieval_eligible"],
            "field_name": field, "status": out["status"],
            "answer_raw": R.redact(out["answer_raw"]),
            "answer_normalized": R.redact(out["answer_normalized"]),
            "matched_expression": out["matched_expression"],
            "representative_source_type": rep_type,
            "representative_location": rep_loc,
            "source_excerpt_redacted": excerpt,
            "additional_locations": jdump(out["additional"]),
            "extraction_method": method,
            "confidence": out["confidence"], "review_reason": out["review_reason"],
            "processed_sha256": reg_row["processed_sha256"],
            "rules_sha256": rules_sha,
        }

        # 고정된 의미 판단이 있으면 그 행만 덮어쓴다.
        # 결정 파일에 '열이 있고 값이 비어 있음'은 '비우라'는 뜻이다.
        # 열 자체가 없을 때만 기존 값을 그대로 둔다.
        d = decisions.get((row["document_id"], field))
        if d:
            for k in ("status", "answer_raw", "answer_normalized", "matched_expression",
                      "representative_source_type", "representative_location",
                      "source_excerpt_redacted", "additional_locations",
                      "confidence", "review_reason"):
                if k in d:
                    row[k] = d[k]
            if not row["status"]:
                raise SystemExit(f"결정 파일에 status 가 비어 있음: "
                                 f"{row['document_id']} / {field}")
            # 값이 없는 상태로 바꿨다면 값 칸도 실제로 비운다
            if row["status"] == "field_absent":
                row["answer_raw"] = row["answer_normalized"] = ""
            row["extraction_method"] = "semantic_decision(fixed_file)"
        # 규칙 결과와 고정 의미 결정 모두 같은 위치 계약을 사용한다.
        row["representative_location"] = jdump(enrich_saved_location(
            row["representative_location"], section_index, row["document_id"], name,
            row["representative_source_type"]))
        row["additional_locations"] = jdump(enrich_saved_location(
            row["additional_locations"], section_index, row["document_id"], name))
        rows.append(row)
    return rows


# ══════════════════════════════════════════════════════════════════
# 5. 저장
# ══════════════════════════════════════════════════════════════════

def write_outputs(out_dir: Path, rows: list[dict], aliases: list[dict], meta: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "extraction_table_v4.csv").open("w", encoding="utf-8-sig",
                                                      newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS, quoting=csv.QUOTE_ALL,
                           lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in COLUMNS})

    def unpack(r: dict) -> dict:
        """JSON 쪽에는 문자열 대신 실제 목록·객체를 넣는다.

        비어 있을 때는 CSV 와 JSON 이 같은 뜻이 되도록 빈 문자열로 맞춘다.
        (CSV 의 빈 칸을 JSON 에서 [] 로 바꾸면 두 파일이 어긋난 것처럼 보인다.)
        """
        o = dict(r)
        for k in ("additional_locations", "representative_location"):
            o[k] = json.loads(r[k]) if r.get(k) else ""
        return o

    payload = {"schema_version": SCHEMA_VERSION, "extraction_version": EXTRACTION_VERSION,
               "corpus_version": meta["corpus_version"],
               "registry_version": meta["registry_version"],
               "document_count": len({r["document_id"] for r in rows}),
               "field_count": len(R.FIELDS), "row_count": len(rows),
               "fields": R.FIELDS, "rows": [unpack(r) for r in rows]}
    (out_dir / "extraction_table_v4.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")

    afields = ["document_id", "field_name", "matched_expression", "source_type",
               "location", "heading", "basis", "confidence"]
    seen, uniq = set(), []
    for a in sorted(aliases, key=lambda x: (x["field_name"], x["matched_expression"],
                                            x["document_id"])):
        key = (a["field_name"], R.expression_key(a["matched_expression"]), a["document_id"])
        if key in seen:
            continue
        seen.add(key)
        uniq.append(a)
    with (out_dir / "field_alias_candidates_v4.csv").open("w", encoding="utf-8-sig",
                                                            newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=afields, quoting=csv.QUOTE_ALL,
                           lineterminator="\n")
        w.writeheader()
        w.writerows({k: a.get(k, "") for k in afields} for a in uniq)


def main() -> int:
    global REGISTRY_DIR, CORPUS_DIR, METADATA_CSV, RULES_YAML, CALIBRATION_CSV
    ap = argparse.ArgumentParser(description="1-12-2 추출 테이블 생성")
    ap.add_argument("--out", default=str(HERE))
    ap.add_argument("--decisions", default=None,
                    help="고정된 의미 판단 파일(기본: --out 폴더의 semantic_decisions_v4.csv)")
    ap.add_argument("--registry-dir", default=str(REGISTRY_DIR),
                    help="읽기 전용 문서 등록부 폴더")
    ap.add_argument("--corpus-dir", default=str(CORPUS_DIR),
                    help="읽기 전용 공식 코퍼스 폴더")
    ap.add_argument("--metadata-csv", default=str(METADATA_CSV),
                    help="공고일 등이 담긴 읽기 전용 메타데이터 CSV")
    ap.add_argument("--rules-yaml", default=(str(RULES_YAML) if RULES_YAML else None),
                    help=f"전처리 규칙 파일(해시 대조용). 환경변수 {_RULES_ENV} 로도 지정")
    ap.add_argument("--calibration-csv",
                    default=(str(CALIBRATION_CSV) if CALIBRATION_CSV else None),
                    help=f"보정 CSV. 환경변수 {_CAL_ENV} 로도 지정")
    args = ap.parse_args()
    REGISTRY_DIR = Path(args.registry_dir)
    CORPUS_DIR = Path(args.corpus_dir)
    METADATA_CSV = Path(args.metadata_csv)
    # ★필수 입력이 없으면 **어느 파일이 없는지** 이름을 찍고 멈춘다(조용히 넘어가지 않는다).
    missing = [name for name, value in (("--rules-yaml", args.rules_yaml),
                                        ("--calibration-csv", args.calibration_csv))
               if not value]
    if missing:
        raise SystemExit(f"❌ 필수 입력이 없습니다: {', '.join(missing)} "
                         f"(환경변수 {_RULES_ENV}/{_CAL_ENV} 로도 지정할 수 있습니다)")
    RULES_YAML = Path(args.rules_yaml)
    CALIBRATION_CSV = Path(args.calibration_csv)
    for label, path in (("--rules-yaml", RULES_YAML), ("--calibration-csv", CALIBRATION_CSV),
                        ("--registry-dir", REGISTRY_DIR), ("--corpus-dir", CORPUS_DIR),
                        ("--metadata-csv", METADATA_CSV)):
        if not path.exists():
            raise SystemExit(f"❌ 필수 입력이 없습니다 — {label}: {path}")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    registry = load_registry()
    # 발주기관 이름을 등록부 파일명에서 뽑는다. '○○연구원'의 '연구원'이 직위로 보여
    # 사람 이름으로 잘못 걸리는 것을 막기 위한 것이고, 값을 지어내지 않는다.
    org_names = {r["source_filename_nfc"].split("_")[0].strip() for r in registry}
    R.set_org_allowlist(org_names)
    active = [r for r in registry if r["active"] == "true"]
    metadata = load_metadata()
    decisions_path = (Path(args.decisions) if args.decisions
                      else out_dir / DECISIONS_CSV)
    decisions = load_decisions(decisions_path)
    reg_meta = json.loads((REGISTRY_DIR / "registry_metadata.json").read_text(encoding="utf-8"))
    corpus_version = reg_meta.get("corpus_version", "")
    registry_version = reg_meta.get("registry_version", "")
    if corpus_version != "v2" or registry_version != "v2":
        raise SystemExit(
            "지원하지 않는 입력 버전: "
            f"corpus={corpus_version!r}, registry={registry_version!r} "
            "(기대값: 모두 'v2')"
        )
    rules_sha = sha256_of(RULES_YAML)

    rows, aliases = [], []
    for reg_row in sorted(active, key=lambda r: r["document_id"]):
        stem = R.nfc(os.path.splitext(reg_row["source_filename_nfc"])[0])
        rows += process_document(reg_row, metadata.get(stem), decisions, aliases,
                                 corpus_version, registry_version, rules_sha)

    write_outputs(out_dir, rows, aliases, {"corpus_version": corpus_version,
                                           "registry_version": registry_version})

    # 실행마다 달라지는 값은 본문이 아니라 메타데이터 파일에만 적는다
    (out_dir / "extraction_metadata.json").write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "generator": "tools/evalset/build_extraction_table.py",
        "schema_version": SCHEMA_VERSION, "extraction_version": EXTRACTION_VERSION,
        "corpus_version": corpus_version, "registry_version": registry_version,
        "corpus_dir": str(CORPUS_DIR), "registry_dir": str(REGISTRY_DIR),
        "metadata_csv": str(METADATA_CSV), "calibration_csv": str(CALIBRATION_CSV),
        "rules_sha256": rules_sha,
        "document_count": len(active), "field_count": len(R.FIELDS),
        "row_count": len(rows),
        "decisions_file": decisions_path.name,
        "decisions_applied": len(decisions),
        "note": "테이블 본문에는 생성 시각이 없다. 같은 결정 파일이면 결과가 매번 같다.",
    }, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    from collections import Counter
    print(json.dumps({"documents": len(active), "rows": len(rows),
                      "status": dict(Counter(r["status"] for r in rows).most_common()),
                      "alias_candidates": len({(a['field_name'], R.expression_key(a['matched_expression']))
                                               for a in aliases}),
                      "decisions_applied": len(decisions)},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
