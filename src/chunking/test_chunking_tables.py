"""
청킹 표 처리 검증 — PR #7 리뷰 지적 2건에 대한 회귀 테스트

    ① 파이프 표의 검색 본문이 비었다  (검색 대상 13,778개 중 1,384개)
    ② 중첩 표 분할이 HTML 구조를 깬다 (269개 청크 / 58문서)

실행:
    python3 -m pytest test_chunking_tables.py -v
    python3 test_chunking_tables.py          # pytest 없이도 동작
"""
import os
import re
import sys

os.environ.setdefault("RAG_ROOT", "/tmp")     # import 시점 sys.exit 방지
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from build_chunks import (                                    # noqa: E402
    CELL_RE, TABLE_CLOSE_RE, TABLE_OPEN_RE,
    find_tables, split_blocks, split_table,
    table_blank_ratio, table_kind, table_parts,
    table_row_count, table_search_text,
)

# ─────────────────────────────────────────────────────────────
# 픽스처 — 실제 코퍼스에서 관찰된 형태를 그대로 본뜬다
# ─────────────────────────────────────────────────────────────

PIPE_SIMPLE = """| 구분 | 역할 |
| --- | --- |
| 중소벤처기업부 | ㆍ사업총괄 운영ㆍ관리 감독 |
| 벤처기업확인기관 | ㆍ벤처확인제도 운영기관 |"""

PIPE_WITH_BLANKS = """| 추진<br>목표 |  | ◈ 시스템 고도화 |
| --- | --- | --- |
|  |  |  |
| 항목 |  | 값 |"""

PIPE_LONG = "| 번호 | 요구사항 |\n| --- | --- |\n" + "\n".join(
    f"| {i} | 요구사항 상세 설명 {'가' * 40} |" for i in range(1, 41)
)

HTML_SIMPLE = (
    '<table><tr><th>구분</th><th>값</th></tr>'
    '<tr><td>사업기간</td><td>12개월</td></tr>'
    '<tr><td>예산</td><td>3억원</td></tr></table>'
)

# 바깥 표 3행. 두 번째 행 안에 2행짜리 표가 들어 있다.
HTML_NESTED = (
    '<table>'
    '<tr><th>대분류</th><th>내용</th></tr>'
    '<tr><td>세부요구</td><td>'
    '<table><tr><td>안쪽1</td><td>값1</td></tr>'
    '<tr><td>안쪽2</td><td>값2</td></tr></table>'
    '</td></tr>'
    '<tr><td>비고</td><td>없음</td></tr>'
    '</table>'
)

HTML_NESTED_BIG = (
    '<table>'
    '<tr><th>대분류</th><th>내용</th></tr>'
    + "".join(
        '<tr><td>항목%d</td><td><table><tr><td>안%d-1</td><td>%s</td></tr>'
        '<tr><td>안%d-2</td><td>%s</td></tr></table></td></tr>'
        % (i, i, "나" * 120, i, "다" * 120)
        for i in range(1, 8)
    )
    + '</table>'
)

HTML_ROWSPAN = (
    '<table><tr><th rowspan="2">구분</th><th>항목</th></tr>'
    '<tr><td>값</td></tr>'
    '<tr><td colspan="2">비고</td></tr></table>'
)


def tag_balance(html: str):
    return (len(TABLE_OPEN_RE.findall(html)),
            len(TABLE_CLOSE_RE.findall(html)))


def _check(cond, msg):
    if not cond:
        raise AssertionError(msg)


# ─────────────────────────────────────────────────────────────
# ① 파이프 표 — 검색 본문
# ─────────────────────────────────────────────────────────────

def test_pipe_search_text_not_empty():
    """리뷰 지적 ① — 파이프 표의 검색 본문이 비면 안 된다."""
    out = table_search_text(PIPE_SIMPLE)
    _check(out.strip() != "", "파이프 표 검색 본문이 비었다")
    for token in ("구분", "역할", "중소벤처기업부", "벤처기업확인기관"):
        _check(token in out, f"검색 본문에 '{token}'이 없다")


def test_pipe_separator_excluded():
    """마크다운 구분선(| --- |)은 내용이 아니므로 검색 본문에 없어야 한다."""
    out = table_search_text(PIPE_SIMPLE)
    _check("---" not in out, "구분선이 검색 본문에 들어갔다")


def test_pipe_blank_cells_skipped_rows_kept():
    """C-2 ④ — 빈 셀만 건너뛰고 행 구조는 유지한다."""
    out = table_search_text(PIPE_WITH_BLANKS)
    _check("추진 목표" in out or "추진목표" in out.replace("<br>", ""),
           "머리글 내용이 빠졌다")
    _check("◈ 시스템 고도화" in out, "본문 셀이 빠졌다")
    for line in out.split("\n"):
        _check(not line.startswith("|"), f"빈 셀 자리가 남았다: {line!r}")
    # 전부 빈 행은 줄 자체가 생기지 않는다
    _check(len(out.split("\n")) == 2, f"빈 행이 줄로 남았다: {out!r}")


def test_pipe_blank_ratio_computed():
    """이전에는 <td>가 없어 None이었고 degraded 판정이 아예 안 됐다."""
    ratio, blank, total = table_blank_ratio(PIPE_WITH_BLANKS)
    _check(ratio is not None, "파이프 표의 빈 셀 비율이 None이다")
    _check(total == 9, f"셀 수가 9가 아니다: {total}")   # 구분선 제외 3행 × 3열
    _check(blank == 5, f"빈 셀 수가 5가 아니다: {blank}")
    _check(abs(ratio - 5 / 9) < 1e-9, f"비율이 틀렸다: {ratio}")


def test_pipe_kind_detected():
    _check(table_kind(PIPE_SIMPLE) == "pipe", "파이프 표를 html로 봤다")
    _check(table_kind(HTML_SIMPLE) == "html", "html 표를 pipe로 봤다")


# ─────────────────────────────────────────────────────────────
# ② 중첩 표 — 행 분리와 구조 보존
# ─────────────────────────────────────────────────────────────

def test_nested_outer_row_count():
    """리뷰 지적 ② — 안쪽 표의 <tr>을 바깥 행으로 세면 안 된다."""
    kind, head, header, body, tail = table_parts(HTML_NESTED)
    _check(kind == "html", "kind가 html이 아니다")
    _check(len(header) == 1, f"머리글 행이 1개가 아니다: {len(header)}")
    _check(len(body) == 2, f"바깥 본문 행이 2개가 아니어야 한다: {len(body)}")
    # 안쪽 표는 통째로 한 바깥 행 안에 들어 있어야 한다
    _check(tag_balance(body[0]) == (1, 1),
           f"안쪽 표가 한 행 안에 온전히 있지 않다: {tag_balance(body[0])}")
    _check(tail.lower().startswith("</table>"), f"tail이 이상하다: {tail!r}")


def test_nested_split_keeps_tags_balanced():
    """분할된 모든 조각의 <table> 여는/닫는 태그 수가 같아야 한다."""
    parts, oversize = split_table(HTML_NESTED_BIG, budget=800)
    _check(len(parts) > 1, "분할이 일어나지 않아 검증이 무의미하다")
    for html, r0, r1, part, of in parts:
        opens, closes = tag_balance(html)
        _check(opens == closes,
               f"part {part}/{of}에서 태그 짝이 안 맞는다: {opens} vs {closes}")


def test_nested_split_header_repeated():
    """C-2 ③-a — 각 조각에 머리글이 반복돼야 한다."""
    parts, _ = split_table(HTML_NESTED_BIG, budget=800)
    for html, *_ in parts:
        _check("대분류" in html, "조각에 머리글이 없다")


def test_split_row_ranges_contiguous():
    """row_start/row_end가 겹치지 않고 빠짐없이 이어져야 한다."""
    parts, _ = split_table(HTML_NESTED_BIG, budget=800)
    total = table_row_count(HTML_NESTED_BIG)
    seen = []
    for _, r0, r1, _, _ in parts:
        seen.extend(range(r0, r1 + 1))
    _check(seen == sorted(seen), "row 범위 순서가 어긋난다")
    _check(len(seen) == len(set(seen)), "row 범위가 겹친다")
    _check(seen == list(range(1, total + 1)),
           f"본문 행이 빠졌다: {len(seen)} / {total}")


def test_split_part_of_consistent():
    """part/of의 of는 실제 조각 수와 같아야 한다."""
    for fixture, budget in ((HTML_NESTED_BIG, 800), (PIPE_LONG, 600)):
        parts, _ = split_table(fixture, budget)
        of_values = {of for *_, of in parts}
        _check(of_values == {len(parts)},
               f"of가 조각 수와 다르다: {of_values} vs {len(parts)}")
        _check([p for *_, p, _ in parts] == list(range(1, len(parts) + 1)),
               "part 번호가 1..of가 아니다")


# ─────────────────────────────────────────────────────────────
# 파이프 표 분할
# ─────────────────────────────────────────────────────────────

def test_pipe_split_repeats_header_and_separator():
    parts, _ = split_table(PIPE_LONG, budget=600)
    _check(len(parts) > 1, "긴 파이프 표가 분할되지 않았다")
    for text, *_ in parts:
        lines = text.split("\n")
        _check(lines[0].startswith("| 번호"), f"머리글이 없다: {lines[0]!r}")
        _check(set(lines[1].replace("|", "").strip()) <= set("-: "),
               f"구분선이 반복되지 않았다: {lines[1]!r}")


def test_pipe_split_pieces_searchable():
    """분할된 조각 하나하나가 검색 본문을 가져야 한다."""
    parts, _ = split_table(PIPE_LONG, budget=600)
    for text, _, _, part, of in parts:
        out = table_search_text(text)
        _check(out.strip() != "", f"part {part}/{of}의 검색 본문이 비었다")


def test_pipe_split_no_row_lost():
    parts, _ = split_table(PIPE_LONG, budget=600)
    joined = "\n".join(t for t, *_ in parts)
    for i in range(1, 41):
        _check(f"| {i} |" in joined, f"{i}번 행이 사라졌다")


# ─────────────────────────────────────────────────────────────
# 회귀 — HTML 경로가 기존과 같아야 한다
# ─────────────────────────────────────────────────────────────

def test_html_blank_ratio_unchanged():
    """임계값 0.6은 '중첩 셀 포함 <td|th> 전수' 방식으로 잰 분포가 근거다.

    세는 방식을 바꾸면 C-2 ②의 근거가 사라지므로 기존 계산과 동일해야 한다.
    """
    for fixture in (HTML_SIMPLE, HTML_NESTED, HTML_ROWSPAN, HTML_NESTED_BIG):
        cells = CELL_RE.findall(fixture)
        blank_ref = sum(
            1 for c in cells
            if re.sub(r"<[^>]+>", "", c).replace("&nbsp;", "").strip() == ""
        )
        ratio, blank, total = table_blank_ratio(fixture)
        _check(total == len(cells), f"셀 수가 달라졌다: {total} vs {len(cells)}")
        _check(blank == blank_ref, f"빈 셀 수가 달라졌다: {blank} vs {blank_ref}")


def test_html_search_text_still_works():
    out = table_search_text(HTML_SIMPLE)
    for token in ("구분", "사업기간", "12개월", "3억원"):
        _check(token in out, f"HTML 표 검색 본문에 '{token}'이 없다")


def test_rowspan_colspan_preserved():
    """A-5·1-13-1 — 병합 속성이 보존돼야 한다."""
    parts, _ = split_table(HTML_ROWSPAN, budget=10_000)
    joined = "".join(t for t, *_ in parts)
    _check('rowspan="2"' in joined, "rowspan이 사라졌다")
    _check('colspan="2"' in joined, "colspan이 사라졌다")


def test_small_table_not_split():
    parts, oversize = split_table(HTML_SIMPLE, budget=10_000)
    _check(len(parts) == 1, "작은 표가 분할됐다")
    _check(parts[0][4] == 1, "of가 1이 아니다")


def test_oversize_row_kept_whole():
    """C-2 ③-c — 한 행이 budget을 넘어도 자르지 않는다."""
    parts, oversize = split_table(HTML_NESTED_BIG, budget=100)
    _check(oversize > 0, "초과 행이 기록되지 않았다")
    for html, *_ in parts:
        opens, closes = tag_balance(html)
        _check(opens == closes, "초과 행 처리에서 태그가 깨졌다")


# ─────────────────────────────────────────────────────────────
# 블록 분해 — 표 인식
# ─────────────────────────────────────────────────────────────

def test_find_tables_handles_nesting():
    doc = f"# 제목\n\n{HTML_NESTED}\n\n본문\n"
    spans = find_tables(doc)
    _check(len(spans) == 1, f"중첩 표를 여러 개로 셌다: {len(spans)}")
    s, e = spans[0]
    _check(tag_balance(doc[s:e]) == (2, 2), "잘라낸 범위의 태그가 안 맞는다")


def test_split_blocks_classifies_both_table_types():
    doc = f"# 1. 개요\n\n본문 문단.\n\n{PIPE_SIMPLE}\n\n{HTML_SIMPLE}\n"
    kinds = [k for k, *_ in split_blocks(doc)]
    _check("pipe_table" in kinds, f"파이프 표를 인식 못 했다: {kinds}")
    _check("table" in kinds, f"HTML 표를 인식 못 했다: {kinds}")
    _check("heading" in kinds, "헤딩을 인식 못 했다")


# ─────────────────────────────────────────────────────────────
# 불변식 — 무작위 조합에서도 깨지지 않아야 한다
# ─────────────────────────────────────────────────────────────

def test_invariants_over_budgets():
    """여러 budget에서 (태그 균형 · 행 보존 · 검색 본문) 세 불변식을 확인한다."""
    fixtures = {
        "pipe_long": PIPE_LONG,
        "html_nested_big": HTML_NESTED_BIG,
        "html_simple": HTML_SIMPLE,
        "pipe_simple": PIPE_SIMPLE,
    }
    for name, fx in fixtures.items():
        total = table_row_count(fx)
        for budget in (150, 300, 600, 900, 1500, 5000):
            parts, _ = split_table(fx, budget)
            rows = []
            for text, r0, r1, part, of in parts:
                if table_kind(fx) == "html":
                    o, c = tag_balance(text)
                    _check(o == c, f"{name}/budget={budget} part{part}: 태그 불균형")
                _check(table_search_text(text).strip() != "",
                       f"{name}/budget={budget} part{part}: 검색 본문 빔")
                rows.extend(range(r0, r1 + 1))
            _check(rows == list(range(1, total + 1)),
                   f"{name}/budget={budget}: 행 보존 실패 {len(rows)}/{total}")


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
