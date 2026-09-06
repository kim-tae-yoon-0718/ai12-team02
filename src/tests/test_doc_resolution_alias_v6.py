#@title 줄여 부른 사업명·기관 별칭으로도 문서를 정확히 찾는가 (§9)
#@markdown 공식 원자료에서 파생한 기관 별칭과 제목 핵심 낱말 순서 매칭만 쓴다 —
#@markdown 편집거리·임베딩 유사도로 문서를 확정하지 않는다. 합성 자료로 반례까지 검사한다.
"""EXT-05("KOICA 우즈베키스탄 국회 방송시스템 구축 사업") 재현과 반례."""
from __future__ import annotations

import pytest

from doc_resolver import (
    match_docs_by_title_terms, match_org_alias_keys, org_alias_map, resolve_document,
)
from test_selection_fix import present, write_identity, write_table

OFFICIAL_TITLE = ("[긴급] [지문] [국제] 우즈베키스탄 열린 의정활동 상하원 국회 "
                  "방송시스템 구축 및 지역의회 연계 개선 PMC 용역")


@pytest.fixture
def koica_world(tmp_path):
    """공식 자료와 같은 모양의 합성 세계 — 기관명에 약어가 들어 있다."""
    table = write_table(tmp_path, {
        "RFP-000901": {"예산": present("6,758,571,493원")},
        "RFP-000902": {"예산": present("1억원")},
    })
    identity = write_identity(tmp_path, [
        ("RFP-000901", "KOICA 전자조달", OFFICIAL_TITLE, "2025-01-01 17:00:00"),
        ("RFP-000902", "가온재단", "하늘 관측망 구축 사업", "2025-01-01 17:00:00"),
    ])
    return table, identity


# ── 별칭은 공식 원자료에서만 만든다 ──────────────────────────────────
def test_alias_comes_from_official_org_name_only(koica_world):
    _, identity = koica_world
    alias = org_alias_map(identity)
    assert alias.get("koica") == "koica전자조달"
    # 일반어(전자조달)는 별칭이 되지 않는다
    assert "전자조달" not in alias
    assert match_org_alias_keys("KOICA 사업 예산 알려줘", identity) == ["koica전자조달"]


def test_alias_is_dropped_when_two_orgs_share_the_word(tmp_path):
    from identity_metadata import load_identity  # noqa: F401  (write_identity 가 씀)
    identity = write_identity(tmp_path, [
        ("RFP-000901", "서울 미래재단", "가 사업", "2025-01-01 17:00:00"),
        ("RFP-000902", "서울 도시공사", "나 사업", "2025-01-01 17:00:00"),
    ])
    assert "서울" not in org_alias_map(identity)


# ── 줄여 부른 제목으로 확정 ──────────────────────────────────────────
@pytest.mark.parametrize("question", [
    "KOICA 우즈베키스탄 국회 방송시스템 구축 사업 예산 규모가 어느 정도인지 확인하고 싶어.",
    "우즈베키스탄 국회 방송시스템 구축 사업 예산 알려줘",
    "KOICA 국회 방송시스템 사업 예산 알려줘",
    "KOICA 우즈베키스탄 국회 방송시스템 구축 사업의 사업기간은?",
])
def test_abbreviated_title_resolves_to_the_right_document(koica_world, question):
    _, identity = koica_world
    assert resolve_document(question, identity).document_id == "RFP-000901"


# ── 반례: 확정하면 안 되는 질문 ──────────────────────────────────────
@pytest.mark.parametrize("question", [
    "우즈베키스탄 도로 건설 사업 예산 알려줘",          # 같은 국가명만 있는 가짜 사업
    "방송시스템 구축 사업 예산 알려줘",                 # 일반적인 일부 단어만
    "KOICA 몽골 도서관 정보화 사업 예산 알려줘",        # 존재하지 않는 그 기관의 사업
    "국회 예산 알려줘",                                # 한 낱말·기관 단서 없음
])
def test_fake_or_generic_titles_are_not_confirmed(koica_world, question):
    _, identity = koica_world
    r = resolve_document(question, identity)
    assert r.document_id is None, r


def test_name_conflict_between_org_and_title_is_blocked(koica_world):
    _, identity = koica_world
    r = resolve_document("가온재단 우즈베키스탄 국회 방송시스템 구축 사업 예산 알려줘", identity)
    assert r.document_id is None


def test_two_candidates_ask_back_instead_of_guessing(tmp_path):
    identity = write_identity(tmp_path, [
        ("RFP-000901", "누리진흥원", "스마트 관제 시스템 구축 사업 1차", "2025-01-01 17:00:00"),
        ("RFP-000902", "누리진흥원", "스마트 관제 시스템 구축 사업 2차", "2025-01-01 17:00:00"),
    ])
    r = resolve_document("누리진흥원 스마트 관제 시스템 구축 사업 예산 알려줘", identity)
    assert r.document_id is None
    assert len(r.candidates) == 2 and r.is_ambiguous


def test_single_term_needs_an_org_clue(tmp_path):
    identity = write_identity(tmp_path, [
        ("RFP-000901", "국민안전공단", "2024년 이러닝시스템 운영 용역", "2025-01-01 17:00:00"),
        ("RFP-000902", "가온재단", "하늘 관측망 구축 사업", "2025-01-01 17:00:00"),
    ])
    # 기관을 함께 부르면 한 낱말(복합명사)로도 확정한다
    assert resolve_document("국민안전공단 이러닝시스템 운영 용역 예산 알려줘",
                            identity).document_id == "RFP-000901"
    # 기관 단서가 없으면 확정하지 않는다
    assert resolve_document("이러닝시스템 운영 용역 예산 알려줘", identity).document_id is None


def test_terms_must_appear_in_the_same_order(koica_world):
    _, identity = koica_world
    docs, terms = match_docs_by_title_terms(
        "방송시스템 국회 우즈베키스탄 사업 예산 알려줘", identity)
    assert terms and docs == []          # 낱말은 다 있지만 순서가 다르다


def test_explicit_document_id_still_wins(koica_world):
    _, identity = koica_world
    assert resolve_document("RFP-000902의 예산 알려줘", identity).document_id == "RFP-000902"


def test_fake_project_does_not_fall_back_to_the_orgs_only_document(tmp_path):
    """§9 전수 점검에서 드러난 결함 — 기관은 실재하고 사업명은 가짜인 질문.

    기관명이 제목 앞에 붙으면 예전 보호 규칙이 제목을 알아보지 못해, 그 기관의
    유일한 문서로 대신 답했다("기초과학연구원 화성기지 관제시스템 구축 사업").
    """
    identity = write_identity(tmp_path, [
        ("RFP-000901", "누리연구원", "2025년도 중이온가속기용 극저온시스템 운전 용역",
         "2025-01-01 17:00:00"),
    ])
    assert resolve_document("누리연구원 화성기지 관제시스템 구축 사업 예산 알려줘",
                            identity).document_id is None
    # 기관만 부른 질문은 기존대로 그 기관의 문서로 답한다
    assert resolve_document("누리연구원 사업 예산 알려줘",
                            identity).document_id == "RFP-000901"
    # 실제 제목을 부르면 당연히 찾는다
    assert resolve_document("누리연구원 중이온가속기용 극저온시스템 운전 용역 예산 알려줘",
                            identity).document_id == "RFP-000901"


def test_alias_never_uses_words_that_appear_in_project_titles(tmp_path):
    """제목에도 쓰이는 낱말은 기관 별칭이 될 수 없다(전수 점검에서 44건 재현)."""
    identity = write_identity(tmp_path, [
        ("RFP-000901", "한국발명진흥회 입찰공고", "가 사업", "2025-01-01 17:00:00"),
        ("RFP-000902", "가온재단", "2024년 벤처확인 기능 고도화 용역사업 입찰공고",
         "2025-01-01 17:00:00"),
    ])
    alias = org_alias_map(identity)
    assert "입찰공고" not in alias          # 다른 문서의 공식 제목에 들어 있다
    assert alias.get("한국발명진흥회") == "한국발명진흥회입찰공고"
    # 정식 사업명을 그대로 적은 질문이 별칭 때문에 막히지 않는다
    assert resolve_document("2024년 벤처확인 기능 고도화 용역사업 입찰공고 예산 알려줘",
                            identity).document_id == "RFP-000902"


def test_numeric_tokens_are_not_aliases(tmp_path):
    identity = write_identity(tmp_path, [
        ("RFP-000901", "2025 구미대회 조직위원회", "가 사업", "2025-01-01 17:00:00"),
    ])
    assert "2025" not in org_alias_map(identity)
