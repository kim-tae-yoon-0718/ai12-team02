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
    can_merge, clean_heading, common_prefix,
    find_tables, outer_cells, split_blocks, split_table,
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


INNER_TAGS = ("table", "tr", "td", "th", "thead", "tbody")


def inner_tag_balance(html: str):
    """table 뿐 아니라 tr·td·th·thead·tbody 짝도 센다.

    바깥 <table>만 세면 행·셀이 깨져도 통과한다 (리뷰 지적).
    """
    out = {}
    for tag in INNER_TAGS:
        opens = len(re.findall(rf"<{tag}\b", html, re.I))
        closes = len(re.findall(rf"</{tag}\s*>", html, re.I))
        out[tag] = (opens, closes)
    return out


def assert_tags_balanced(html: str, where: str):
    for tag, (o, c) in inner_tag_balance(html).items():
        _check(o == c, f"{where}: <{tag}> 짝 불일치 {o} vs {c}")


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
    for html, r0, r1, part, of, over in parts:
        assert_tags_balanced(html, f"part {part}/{of}")


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
    for _, r0, r1, _, _, _ in parts:
        seen.extend(range(r0, r1 + 1))
    _check(seen == sorted(seen), "row 범위 순서가 어긋난다")
    _check(len(seen) == len(set(seen)), "row 범위가 겹친다")
    _check(seen == list(range(1, total + 1)),
           f"본문 행이 빠졌다: {len(seen)} / {total}")


def test_split_part_of_consistent():
    """part/of의 of는 실제 조각 수와 같아야 한다."""
    for fixture, budget in ((HTML_NESTED_BIG, 800), (PIPE_LONG, 600)):
        parts, _ = split_table(fixture, budget)
        of_values = {of for *_, of, _ in parts}
        _check(of_values == {len(parts)},
               f"of가 조각 수와 다르다: {of_values} vs {len(parts)}")
        _check([t[3] for t in parts] == list(range(1, len(parts) + 1)),
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
    for text, _, _, part, of, _ in parts:
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

def test_blank_ratio_counts_outer_cells_only():
    """행에 직접 속한 셀만 센다. 중첩 표의 셀은 바깥 셀 내용에 흡수된다."""
    ratio, blank, total = table_blank_ratio(HTML_SIMPLE)
    _check(total == 6, f"단순 표 셀 수가 6이 아니다: {total}")
    _check(blank == 0, f"빈 셀이 없어야 한다: {blank}")

    # 바깥 3행 × 2열 = 6. 안쪽 표의 4셀은 바깥 셀 내용으로 흡수.
    ratio, blank, total = table_blank_ratio(HTML_NESTED)
    _check(total == 6, f"중첩 표 바깥 셀 수가 6이 아니다: {total}")
    _check(blank == 0, f"중첩 표에 빈 셀이 없어야 한다: {blank}")


def test_nested_in_cell_content_not_lost():
    """리뷰 후속 — <th><table>…</table><br>텍스트</th> 형태에서
    non-greedy 매칭이 안쪽 </th>를 바깥 셀 끝으로 봐 내용을 통째로 잃었다."""
    fx = ('<table><tr><th><table><tr><th></th><th></th></tr></table>'
          '<br><u>제 안 서</u><br>용역명 : 철도인프라 디지털트윈 ISP 수립 용역'
          '<br>업체명 : OO건설</th></tr></table>')
    out = table_search_text(fx)
    _check(out.strip() != "", "중첩 셀이 든 표의 검색 본문이 비었다")
    for token in ("제 안 서", "용역명", "철도인프라", "업체명"):
        _check(token in out, f"검색 본문에 '{token}'이 없다")

    ratio, blank, total = table_blank_ratio(fx)
    _check(total == 1, f"바깥 셀은 1개여야 한다: {total}")
    _check(blank == 0, f"내용이 있으므로 빈 셀이 아니어야 한다: {blank}")
    _check(ratio == 0.0, f"빈 셀 비율이 0이어야 한다: {ratio}")


def test_empty_table_search_text_is_empty():
    """내용이 정말 없는 표는 검색 본문이 비는 게 맞다 (버그 아님)."""
    fx = "|   |   |\n| --- | --- |"
    _check(table_search_text(fx).strip() == "", "빈 표에서 내용이 나왔다")
    ratio, blank, total = table_blank_ratio(fx)
    _check(ratio == 1.0, f"빈 표의 비율이 1.0이 아니다: {ratio}")


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
    _check(parts[0][5] is False, "작은 표가 초과로 표시됐다")


def test_oversize_row_kept_whole():
    """C-2 ③-c — 한 행이 budget을 넘어도 자르지 않는다."""
    parts, oversize = split_table(HTML_NESTED_BIG, budget=100)
    _check(oversize > 0, "초과 행이 기록되지 않았다")
    for i, (html, *_) in enumerate(parts, start=1):
        assert_tags_balanced(html, f"초과 행 처리 part{i}")


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
            for text, r0, r1, part, of, _ in parts:
                if table_kind(fx) == "html":
                    assert_tags_balanced(text, f"{name}/budget={budget} part{part}")
                _check(table_search_text(text).strip() != "",
                       f"{name}/budget={budget} part{part}: 검색 본문 빔")
                rows.extend(range(r0, r1 + 1))
            _check(rows == list(range(1, total + 1)),
                   f"{name}/budget={budget}: 행 보존 실패 {len(rows)}/{total}")


# ─────────────────────────────────────────────────────────────
# 헤딩 정제 · 병합 경로 보존 (PR #7 후속 지적)
# ─────────────────────────────────────────────────────────────

def test_clean_heading_strips_tags():
    """chapter_title_recovered가 붙인 HTML 태그가 출처 표기에 나가면 안 된다."""
    _check(clean_heading("<u>일반현황 및 연혁</u>") == "일반현황 및 연혁",
           f"태그가 남았다: {clean_heading('<u>일반현황 및 연혁</u>')!r}")
    _check(clean_heading("3.1  기술평가") == "3.1 기술평가", "공백 정리가 안 됐다")
    _check(clean_heading("<br>") == "", "태그만 있는 제목이 비지 않았다")


def test_heading_tags_not_in_section_path():
    doc = "# <u>Ⅰ. 사업안내</u>\n\n본문입니다.\n"
    kinds = [(k, b) for k, b, *_ in split_blocks(doc)]
    heading = [b for k, b in kinds if k == "heading"][0]
    m = re.match(r"^(#{1,6})[ \t]+(.*\S)\s*$", heading)
    _check("<" not in clean_heading(m.group(2)), "장절 경로에 태그가 남는다")


def test_merge_rule_siblings_only():
    """C-1 ④ — 3.1과 3.2는 합치고 3장과 4장은 합치지 않는다."""
    _check(can_merge(("3장", "3.1"), ("3장", "3.2")), "형제 절이 안 합쳐진다")
    _check(not can_merge(("3장",), ("4장",)), "다른 장이 합쳐진다")
    _check(can_merge(("3장",), ("3장",)), "같은 경로가 안 합쳐진다")
    _check(not can_merge(("3장",), ("3장", "3.1")), "깊이가 다른데 합쳐진다")


def test_common_prefix_is_shared_parent():
    paths = [("Ⅰ. 사업안내", "1. 사업개요"),
             ("Ⅰ. 사업안내", "2. 사업목표"),
             ("Ⅰ. 사업안내", "3. 사업유형")]
    _check(common_prefix(paths) == ("Ⅰ. 사업안내",),
           f"공통 부모가 틀렸다: {common_prefix(paths)}")
    # section_path만 두면 여기서 1·2·3 구분이 사라진다.
    # 그래서 청크에 section_paths(구성원 경로 전부)를 함께 남긴다.


def test_outer_cells_depth_aware():
    row = ('<tr><th><table><tr><th>안</th><th></th></tr></table>'
           '<br>용역명 : OO사업</th><td>값</td></tr>')
    cells = outer_cells(row)
    _check(len(cells) == 2, f"바깥 셀이 2개여야 한다: {len(cells)}")
    _check("용역명" in cells[0], "중첩 표 뒤 텍스트가 첫 셀에 없다")
    _check(cells[1].strip() == "값", f"두 번째 셀이 틀렸다: {cells[1]!r}")


def test_oversize_flag_only_on_uncut_rows():
    """C-2 ③-c — 자르지 못해 남긴 조각에만 표시한다.

    ⚠️ len(조각) > budget 으로 재면 머리글 반복분 때문에 정상 분할된 조각도
       넘어 오탐이 난다(실제 코퍼스에서 950자 조각이 초과로 찍혔다).
    """
    # 모든 행이 작은 표 — 분할은 되지만 초과는 하나도 없어야 한다
    fx = ('<table><tr><th>구분</th><th>값</th></tr>'
          + "".join(f'<tr><td>항목{i}</td><td>{"가" * 60}</td></tr>' for i in range(1, 21))
          + '</table>')
    parts, oversize = split_table(fx, budget=400)
    _check(len(parts) > 1, "분할이 안 일어나 검증이 무의미하다")
    _check(oversize == 0, f"초과가 없어야 하는데 {oversize}건")
    for html, r0, r1, part, of, over in parts:
        _check(over is False, f"part {part}/{of}가 초과로 잘못 표시됐다")

    # 한 행이 혼자 budget을 넘는 경우 — 그 조각만 표시
    big = "나" * 900
    fx2 = ('<table><tr><th>구분</th><th>값</th></tr>'
           '<tr><td>작은행</td><td>값</td></tr>'
           f'<tr><td>큰행</td><td>{big}</td></tr>'
           '<tr><td>작은행2</td><td>값</td></tr></table>')
    parts, oversize = split_table(fx2, budget=400)
    _check(oversize == 1, f"초과가 1건이어야 한다: {oversize}")
    flagged = [t for t in parts if t[5]]
    _check(len(flagged) == 1, f"표시된 조각이 1개여야 한다: {len(flagged)}")
    _check(big in flagged[0][0], "초과 표시가 엉뚱한 조각에 붙었다")


def test_paragraph_oversize_flag():
    from build_chunks import split_by_paragraph
    text = "짧은 문단.\n\n" + "다" * 900 + "\n\n또 짧은 문단."
    pieces, oversize = split_by_paragraph(text, budget=300, overlap=50)
    _check(oversize == 1, f"초과 문단이 1건이어야 한다: {oversize}")
    flagged = [p for p, over in pieces if over]
    _check(len(flagged) == 1, f"표시된 조각이 1개여야 한다: {len(flagged)}")
    for piece, over in pieces:
        if not over:
            _check(len(piece) <= 300 + 50, f"정상 조각이 budget을 넘는다: {len(piece)}")


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
