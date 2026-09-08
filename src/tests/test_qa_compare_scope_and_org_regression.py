#@title 비교 과차단·기관명 오판 회귀 검사
#@markdown 정상 비교와 정상 기관 인식을 되살리면서, 미확인 대상·유사한 다른 기관은 계속 막는지 검사합니다.
"""두 회귀의 재발 방지 검사.

결함 A — `comparison_scope_issue()`가 이미 확정된 문서의 **축약 사업명**을 미확인
비교 대상으로 오해해 정상 비교를 거절했다.
결함 B — 긴 기관명의 뒤쪽 토큰이 따로 잘려 미등록 기관으로 판정되고, 그 오판이
문서 특정 중단으로 이어졌다. 일반 명사 '발주처'도 같은 경로로 후속 질문을 막았다.

정상 사례(복구해야 할 것)와 차단 사례(계속 막아야 할 것)를 함께 본다.
⚠️ 네트워크·유료 API를 호출하지 않는다(`run_answer`는 호출 시 실패하는 대역을 쓴다).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from answer_pipeline import SessionState
from doc_resolver import (
    comparison_scope_issue, detect_unknown_orgs, resolve_document,
    resolve_documents_for_compare, unexplained_comparison_targets,
)
from test_selection_fix import cfg, present, run_answer, write_identity, write_table

# ---------------------------------------------------------------------------
# 합성 세계 — 공식 자료 없이도 두 결함의 구조를 그대로 재현한다
# ---------------------------------------------------------------------------

_ROWS = [
    ("RFP-000981", "가온재단", "하늘 관측망 구축 사업", "2025-01-01 17:00:00"),
    ("RFP-000982", "별빛공사", "바다 관측망 구축 사업", "2025-01-01 17:00:00"),
    ("RFP-000983", "초록협회", "숲길 안내 시스템 구축 사업", "2025-01-01 17:00:00"),
    ("RFP-000984", "초록협회", "산책로 정비 사업", "2025-01-01 17:00:00"),
    # 공백으로 나뉜 긴 기관명 + 긴 공식 사업명 — 축약 호출과 토큰 분리를 함께 본다.
    ("RFP-000985", "서울특별시 여성안전재단",
     "여성안전 통합지원 플랫폼 구축(2차) 사업 용역", "2025-01-01 17:00:00"),
]


@pytest.fixture
def world(tmp_path):
    table = write_table(tmp_path, {
        "RFP-000981": {"예산": present("1억원"), "사업기간": present("90일")},
        "RFP-000982": {"예산": present("2억원"), "사업기간": present("120일")},
        "RFP-000983": {"예산": present("3억원")},
        "RFP-000984": {"예산": present("4억원")},
        "RFP-000985": {"예산": present("5억원"), "사업기간": present("150일")},
    })
    return table, write_identity(tmp_path, _ROWS)


# ---------------------------------------------------------------------------
# 결함 A — 정상 비교는 복구, 미확인 대상은 계속 차단
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("question, expected", [
    # 기관명 + 공식 사업명을 줄여 부른 표현(회귀의 실제 형태)
    ("서울특별시 여성안전재단 여성안전 통합지원 플랫폼 사업이랑 "
     "가온재단 하늘 관측망 구축 사업 예산 비교해줘",
     {"RFP-000985", "RFP-000981"}),
    # 괄호 주석이 붙은 공식 사업명을 앞부분만 부른 표현
    ("서울특별시 여성안전재단 여성안전 통합지원 플랫폼 구축(2차) 사업이랑 "
     "별빛공사 바다 관측망 사업 예산 비교해줘",
     {"RFP-000985", "RFP-000982"}),
    # 문서 ID로 명시한 기존 정상 비교
    ("RFP-000981과 RFP-000982의 예산 비교해줘", {"RFP-000981", "RFP-000982"}),
    # 공식 사업명을 정확히 그대로 쓴 정상 비교
    ("하늘 관측망 구축 사업이랑 바다 관측망 구축 사업 예산 비교해줘",
     {"RFP-000981", "RFP-000982"}),
    # 기관명만 부른 정상 비교
    ("가온재단과 별빛공사의 예산 비교해줘", {"RFP-000981", "RFP-000982"}),
])
def test_resolvable_comparison_is_not_blocked(world, question, expected):
    table, identity = world
    assert comparison_scope_issue(
        question, identity,
        resolved_doc_ids=resolve_documents_for_compare(question, identity)[0]) is None
    result = run_answer(question, table, identity=identity)
    assert result.abstained is False, result
    assert set(result.selected_document_ids) == expected, result
    assert result.failure is None, result


@pytest.mark.parametrize("question", [
    # 비교 대상 셋 중 하나가 미등록 사업명
    "하늘 관측망 구축 사업이랑 바다 관측망 구축 사업이랑 우주 관측망 구축 사업 예산 비교해줘",
    # 기관에 문서가 하나뿐이어도 전혀 다른 사업명은 인정하지 않는다
    "가온재단 우주 관측망 구축 사업이랑 별빛공사 예산 비교해줘",
    # 알려진 기관 둘 + 새로운 미등록 기관
    "가온재단과 별빛공사와 미래재단의 예산 비교해줘",
    # 기관명과 사업명이 서로 다른 문서를 가리킨다
    "가온재단의 바다 관측망 구축 사업과 별빛공사의 예산 비교해줘",
    # 한 기관에 문서가 여러 건이라 대상을 확정할 수 없다
    "초록협회와 가온재단의 예산 비교해줘",
    # 명시한 문서 ID가 등록부에 없다
    "RFP-000981과 RFP-999999의 예산 비교해줘",
    # 명시한 문서 ID의 형식이 잘못됐다
    "RFP-000981과 RFP-0009810의 예산 비교해줘",
])
def test_unresolved_comparison_target_is_still_blocked(world, question):
    table, identity = world
    assert comparison_scope_issue(
        question, identity,
        resolved_doc_ids=resolve_documents_for_compare(question, identity)[0]) is not None
    result = run_answer(question, table, identity=identity)
    assert result.abstained is True, result
    assert result.selected_document_ids == [], result
    assert result.citations == [], result
    assert result.failure is None, result


def test_comparison_keeps_text_structure_documents_and_sources(world):
    """정상 비교가 복구될 때 본문·구조화 값·문서 목록·출처가 함께 남는지."""
    table, identity = world
    question = ("서울특별시 여성안전재단 여성안전 통합지원 플랫폼 사업이랑 "
                "가온재단 하늘 관측망 구축 사업, 예산이랑 사업기간만 비교해줘")
    result = run_answer(question, table, identity=identity)
    assert result.abstained is False and result.failure is None, result
    assert set(result.selected_document_ids) == {"RFP-000985", "RFP-000981"}, result
    assert isinstance(result.structured_answer, dict), result
    assert set(result.structured_answer) == {"RFP-000985", "RFP-000981"}, result
    for doc_id in ("RFP-000985", "RFP-000981"):
        assert set(result.structured_answer[doc_id]) == {"예산", "사업기간"}, result
    assert "5억원" in result.text and "1억원" in result.text, result
    assert "150일" in result.text and "90일" in result.text, result
    assert result.citations, result
    assert {c["document"] for c in result.citations} == {"RFP-000985", "RFP-000981"}, result
    assert {c["field"] for c in result.citations} == {"예산", "사업기간"}, result
    assert result.used_sources, result


def test_abbreviation_is_only_accepted_for_a_document_actually_resolved(world):
    """축약형 인정은 '확정된 문서의 공식 사업명 안에 이어져 있을 때'로 제한된다."""
    _, identity = world
    question = ("서울특별시 여성안전재단 여성안전 통합지원 플랫폼 사업이랑 "
                "가온재단 하늘 관측망 구축 사업 예산 비교해줘")
    # 두 문서를 모두 확정했으면 축약형은 새 대상이 아니다
    assert unexplained_comparison_targets(
        question, identity, ["RFP-000985", "RFP-000981"]) == []
    # 같은 질문·같은 축약형이라도 그 문서를 확정하지 못했으면 미확인 대상으로 남는다
    only_one = unexplained_comparison_targets(question, identity, ["RFP-000981"])
    assert "여성안전 통합지원 플랫폼" in only_one, only_one
    # 공식 사업명에 이어져 있지 않은 제목은 문서를 확정했어도 새 대상이다
    different_title = unexplained_comparison_targets(
        "서울특별시 여성안전재단 여성안전 우주 플랫폼 사업이랑 "
        "가온재단 하늘 관측망 구축 사업 예산 비교해줘",
        identity, ["RFP-000985", "RFP-000981"])
    assert "여성안전 우주 플랫폼" in different_title, different_title


# ---------------------------------------------------------------------------
# 결함 B — 정상 기관명은 인정, 유사한 다른 기관은 계속 미등록
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("org", [
    "서울특별시 여성안전재단",     # 공백으로 나뉜 긴 기관명
    "서울특별시여성안전재단",       # 공백 없이 이어 쓴 같은 기관
    "재단법인 서울특별시 여성안전재단",  # 법인 접두어
    "(재)서울특별시 여성안전재단",
])
def test_long_multi_token_org_is_recognized(world, org):
    table, identity = world
    question = f"{org}의 예산 알려줘"
    assert detect_unknown_orgs(question, identity) == []
    result = run_answer(question, table, identity=identity)
    assert result.abstained is False, result
    assert result.selected_document_ids == ["RFP-000985"], result
    assert result.failure is None, result


@pytest.mark.parametrize("org", [
    "새가온재단", "미래가온재단", "신별빛공사",
    "미래여성안전재단",           # 등록 기관명의 일부를 품은 다른 긴 이름
    "부산 여성안전재단",           # 앞 토큰이 다른 같은 꼬리
])
@pytest.mark.parametrize("with_active", [False, True])
def test_similar_but_different_org_stays_unknown(world, org, with_active):
    table, identity = world
    question = f"{org}의 예산 알려줘"
    assert detect_unknown_orgs(question, identity) == [org.split()[-1]
                                                       if " " in org else org]
    session = SessionState(active_document_id="RFP-000981") if with_active else None
    result = run_answer(question, table, identity=identity, session=session)
    assert result.abstained is True, result
    assert result.selected_document_ids == [], result
    assert result.citations == [], result
    assert result.failure is None, result


def test_full_name_and_a_different_org_with_the_same_tail_are_separated(world):
    """이름 경계를 본다 — 정식 이름 구간 안의 조각만 인정한다."""
    _, identity = world
    got = detect_unknown_orgs(
        "서울특별시 여성안전재단이랑 부산 여성안전재단의 예산 비교해줘", identity)
    assert got == ["여성안전재단"], got


@pytest.mark.parametrize("separator", ["과 ", "과", " 및 ", " 그리고 "])
def test_known_org_conjunction_without_space_is_not_unknown(world, separator):
    _, identity = world
    assert detect_unknown_orgs(
        f"가온재단{separator}별빛공사의 예산 비교해줘", identity) == []


def test_known_and_unknown_org_together_are_both_reported(world):
    _, identity = world
    got = detect_unknown_orgs("가온재단 사업이랑 가짜미래재단 사업 비교해줘", identity)
    assert got == ["가짜미래재단"], got


# ---------------------------------------------------------------------------
# 결함 B — 일반 명사 때문에 후속 질문이 끊기지 않는지
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("question", [
    "그 사업의 발주처와 예산을 알려줘",
    "해당 공고의 발주처가 어디야? 예산도 알려줘",
    "거기 발주처랑 사업기간 알려줘",
])
def test_generic_org_noun_does_not_break_the_active_document(world, question):
    table, identity = world
    assert detect_unknown_orgs(question, identity) == []
    assert resolve_document(question, identity,
                            active_document_id="RFP-000981").document_id == "RFP-000981"
    session = SessionState(active_document_id="RFP-000981")
    result = run_answer(question, table, identity=identity, session=session)
    assert result.abstained is False, result
    assert result.selected_document_ids == ["RFP-000981"], result
    assert result.failure is None, result


@pytest.mark.parametrize("question, expected", [
    ("그 사업의 예산 알려줘", "RFP-000981"),          # 이전 문서를 이어 묻는다
    ("거기 사업기간 알려줘", "RFP-000981"),
    ("RFP-000983의 예산 알려줘", "RFP-000983"),        # 문서 ID를 직접 명시
])
def test_followup_and_explicit_document_still_work(world, question, expected):
    table, identity = world
    session = SessionState(active_document_id="RFP-000981")
    result = run_answer(question, table, identity=identity, session=session)
    assert result.abstained is False, result
    assert result.selected_document_ids == [expected], result
    assert result.failure is None, result


def test_named_new_org_overrides_the_active_document(world):
    table, identity = world
    session = SessionState(active_document_id="RFP-000981")
    result = run_answer("그 사업 말고 별빛공사의 예산 알려줘", table,
                        identity=identity, session=session)
    assert result.abstained is False, result
    assert result.selected_document_ids == ["RFP-000982"], result


def test_unknown_new_org_never_falls_back_to_the_active_document(world):
    table, identity = world
    session = SessionState(active_document_id="RFP-000981")
    result = run_answer("그 사업 말고 미래재단의 예산 알려줘", table,
                        identity=identity, session=session)
    assert result.abstained is True, result
    assert result.selected_document_ids == [], result
    assert result.citations == [], result


# ---------------------------------------------------------------------------
# 공식 자료로만 확인할 수 있는 부분
# ---------------------------------------------------------------------------
# ⚠️ skip 조건은 기존 tests/test_official_inputs.py 와 같은 형태다(새 기준을
#    만들지 않는다). 공식 자료가 있는 서버에서는 전부 실행된다.

_RAG_ROOT = Path(os.environ.get("RAG_ROOT_OFFICIAL", "/srv/rfp"))
_PROCESSED = _RAG_ROOT / "shared_data" / "processed"
_IDENTITY_CSV = _PROCESSED / "document_registry_v2" / "document_identity_v2.csv"
# ⚠️ 공식 추출표 v4·평가셋 v2 는 저장소(data/)에 있다. 서버 공용 경로만 보면
#    자료가 있는데도 "없는 환경"으로 판정해 이 파일의 공식 검사가 통째로 skip 된다.
from conftest import (resolve_official_extraction_dir,      # noqa: E402
                      resolve_official_evalset_items)

_EXTRACTION_DIR = resolve_official_extraction_dir("v5")
_CHUNKS_DIR = _PROCESSED / "chunks_v3"
_PRACTICE_ITEMS = _RAG_ROOT / "evalset" / "practice_items.jsonl"
_OFFICIAL_ITEMS = resolve_official_evalset_items("v3")

_MISSING_OFFICIAL = [
    name for name, ok in (
        ("identity", _IDENTITY_CSV.exists()),
        ("추출표 v5", _EXTRACTION_DIR is not None),
        ("청크 v3", _CHUNKS_DIR.exists()),
    ) if not ok
]

official = pytest.mark.skipif(
    bool(_MISSING_OFFICIAL),
    reason="공식 입력 자료 없음: " + ", ".join(_MISSING_OFFICIAL),
)

_OFFICIAL_CFG = dict(
    cfg(), corpus="v2", preprocess="v2", table="v5",
    document_registry_version="v2", chunking_version="v3", extraction_version="v5",
    schema_version="1-12-2/v3",
)

# 결함 B로 문서 특정이 끊겼던 공식 기관명 → 실제 identity 기준 문서.
_BROKEN_ORGS = {
    "2025 구미 아시아육상경기선수권대회 조직위원회": "RFP-000005",
    "문화체육관광부 국립민속박물관": "RFP-000034",
    "서울특별시 여성가족재단": "RFP-000041",
    "재단법인 광주광역시 광주문화재단": "RFP-000053",
}


@pytest.fixture(scope="module")
def official_identity():
    from identity_metadata import load_identity
    return load_identity(_IDENTITY_CSV)


@pytest.fixture(scope="module")
def official_table():
    from table_query import load_extraction_table
    return load_extraction_table(
        _EXTRACTION_DIR / "extraction_table_v5.json", cfg=_OFFICIAL_CFG,
        metadata_path=_EXTRACTION_DIR / "extraction_metadata.json", official=True)


def _official_question(item_id: str) -> str:
    for path in (_OFFICIAL_ITEMS, _PRACTICE_ITEMS):
        if path is None or not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                item = json.loads(line)
                if item["id"] == item_id:
                    return item["question"]
    raise AssertionError(f"공식 평가셋에서 {item_id}를 찾지 못했다")


@official
@pytest.mark.parametrize("org, expected", sorted(_BROKEN_ORGS.items()))
def test_official_multi_token_org_resolves_again(official_identity, org, expected):
    question = f"{org} 사업 예산 알려줘"
    assert detect_unknown_orgs(question, official_identity) == []
    resolution = resolve_document(question, official_identity)
    assert resolution.document_id == expected, resolution
    assert resolution.unknown_orgs == [], resolution


@official
def test_no_official_display_name_is_reported_unknown(official_identity):
    """공식 기관 표시명 **전수** 확인.

    분모를 구분해 기록한다 — 표시 문자열 개수와 검색 키 개수는 다르다
    (별칭 키가 따로 있는 행이 있어 검색 키가 더 많다).
    """
    display_names = sorted({official_identity.get(doc_id).buyer_org
                            for doc_id in official_identity.records})
    search_keys = sorted(official_identity.by_org_key)
    assert len(display_names) >= 80 and len(search_keys) >= len(display_names)
    offenders = {name: detect_unknown_orgs(f"{name} 사업 예산 알려줘", official_identity)
                 for name in display_names}
    assert {k: v for k, v in offenders.items() if v} == {}


@official
def test_every_official_search_key_still_resolves_a_document(official_identity):
    """검색 키 기준 전수 — 오판 때문에 문서 특정이 끊기는 키가 없어야 한다."""
    broken = []
    for key, docs in official_identity.by_org_key.items():
        record = official_identity.get(docs[0])
        question = f"{record.buyer_org} 사업 예산 알려줘"
        resolution = resolve_document(question, official_identity)
        if resolution.unknown_orgs or (
                resolution.document_id is None and not resolution.candidates):
            broken.append((key, record.buyer_org, resolution.method,
                           resolution.unknown_orgs))
    assert broken == []


@official
@pytest.mark.parametrize("item_id, expected", [
    ("QA-005", {"RFP-000014", "RFP-000020"}),
    ("PRAC-QA-003", {"RFP-000038", "RFP-000043"}),
])
def test_official_comparison_items_are_answered(official_table, official_identity,
                                                item_id, expected):
    """실제 두 비교 문항 — 함수 반환이 아니라 answer() 경로까지 확인한다."""
    question = _official_question(item_id)
    result = run_answer(question, official_table, identity=official_identity,
                        config=_OFFICIAL_CFG)
    assert result.abstained is False, result
    assert result.failure is None and result.error_stage is None, result
    assert set(result.selected_document_ids) == expected, result
    assert isinstance(result.structured_answer, dict), result
    assert set(result.structured_answer) == expected, result
    for doc_id in expected:
        assert set(result.structured_answer[doc_id]) == {"예산", "사업기간"}, result
    # 근거는 개수뿐 아니라 해당 문서·필드의 실제 근거여야 한다.
    assert {c["document"] for c in result.citations} == expected, result
    assert {c["field"] for c in result.citations} == {"예산", "사업기간"}, result
    assert len(result.citations) == 4, result
    assert result.used_sources, result
