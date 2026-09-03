#@title 추출·QA 문서 특정과 질문 분기 다각도 검사
#@markdown 인사말, 잘못된 문서 번호, 이전 대화, 상충하는 기관·사업명이 다른 답을 만들지 않는지 합성 자료로 검사합니다.
"""네트워크 없이 실제 answer()와 문서 특정·분기 함수를 검사한다."""
from __future__ import annotations

import pytest

from answer_pipeline import SessionState
from doc_resolver import detect_document_ids, resolve_document
from router import route
from test_selection_fix import cfg, present, run_answer, write_identity, write_table


@pytest.fixture
def routing_world(tmp_path):
    # 문서마다 서로 다른 예산을 두어 잘못된 문서 선택이 드러나게 한다.
    table = write_table(tmp_path, {
        "RFP-000981": {"예산": present("1억원"), "지역제한": present("서울 소재")},
        "RFP-000982": {"예산": present("2억원"), "지역제한": present("부산 소재")},
        "RFP-000983": {"예산": present("3억원")},
    })
    identity = write_identity(tmp_path, [
        ("RFP-000981", "가온재단", "하늘 관측망 구축 사업", "2025-01-01 17:00:00"),
        ("RFP-000982", "별빛공사", "바다 관측망 구축 사업", "2025-01-01 17:00:00"),
        ("RFP-000983", "초록협회", "숲길 안내 시스템 구축 사업", "2025-01-01 17:00:00"),
    ])
    return table, identity


@pytest.mark.parametrize("greeting", [
    "안녕하세요,", "안녕!", "감사합니다.", "고마워요. ", "반가워요!", "하이,", "헬로!",
])
def test_greeting_does_not_erase_actual_document_question(routing_world, greeting):
    table, identity = routing_world
    result = run_answer(f"{greeting} RFP-000981의 예산 알려줘", table, identity=identity)
    assert result.abstained is False, result
    assert result.selected_document_ids == ["RFP-000981"], result
    assert "1억원" in result.text, result
    assert result.failure is None, result


@pytest.mark.parametrize("question", [
    "RFP-000981 시스템 사용법 알려줘",
    "RFP-000981의 프로그램 이용 방법을 설명해줘",
    "그 사업의 서비스 사용 방법 알려줘",
    "해당 문서의 플랫폼은 어떻게 사용하나요?",
    "입찰 사이트 이용 방법을 설명해줘",
    "RFP-000981에서 무엇을 할 수 있다고 요구하나요?",
])
def test_document_content_usage_is_not_chatbot_help(question):
    assert route(question, cfg()).task_type != "no_search_needed"


@pytest.mark.parametrize("question", [
    "안녕하세요", "고마워요", "하이", "이 시스템은 어떻게 사용하나요?",
    "사용법 알려줘", "도움말", "무엇을 물어볼 수 있나요?",
])
def test_genuine_smalltalk_or_help_is_preserved(question):
    assert route(question, cfg()).task_type == "no_search_needed"


@pytest.mark.parametrize("bad_id", [
    "RFP-0009810", "RFP-000981A", "RFP-000981_1", "RFP-981", "RFP-00098",
])
@pytest.mark.parametrize("with_active", [False, True])
def test_malformed_id_never_substitutes_an_existing_document(routing_world, bad_id, with_active):
    table, identity = routing_world
    session = SessionState(active_document_id="RFP-000982") if with_active else None
    result = run_answer(f"{bad_id}의 예산 알려줘", table, identity=identity, session=session)
    assert result.abstained is True, result
    assert result.selected_document_ids == [], result
    assert result.citations == [], result
    assert result.failure is None, result
    if session:
        assert session.active_document_id == "RFP-000982"


@pytest.mark.parametrize("question", [
    "RFP-0009810의 예산 알려줘", "RFP-000981A의 예산 알려줘",
    "RFP-000981_1의 예산 알려줘", "XRFP-000981의 예산 알려줘",
])
def test_document_id_has_ascii_boundaries(question):
    assert detect_document_ids(question) == []


@pytest.mark.parametrize("separator", ["와", ", ", " 그리고 ", " 및 "])
def test_two_explicit_documents_are_not_silently_reduced_to_first(routing_world, separator):
    table, identity = routing_world
    question = f"RFP-000981{separator}RFP-000982의 예산 알려줘"
    result = run_answer(question, table, identity=identity)
    # 지원하는 비교 경로로 양쪽을 답하거나, 둘을 함께 보여주며 되묻는 것은 허용한다.
    assert result.abstained or set(result.selected_document_ids) == {"RFP-000981", "RFP-000982"}, result
    if not result.abstained:
        assert "1억원" in result.text and "2억원" in result.text
    assert result.failure is None, result


@pytest.mark.parametrize("question", [
    "RFP-000981과 RFP-999999의 예산 알려줘",
    "RFP-999999의 예산 알려줘",
])
def test_unknown_explicit_document_is_not_recorded_as_resolved(routing_world, question):
    _, identity = routing_world
    result = resolve_document(question, identity, active_document_id="RFP-000982")
    assert result.document_id is None, result


@pytest.mark.parametrize("question", [
    "그 사업 말고 별빛공사의 예산 알려줘",
    "그 문서 대신 별빛공사 예산 알려줘",
    "해당 사업 말고 별빛공사의 마감일 알려줘",
    "거기 대신 별빛공사의 지역제한 알려줘",
])
def test_named_new_org_overrides_stale_anaphora(routing_world, question):
    table, identity = routing_world
    session = SessionState(active_document_id="RFP-000981")
    result = run_answer(question, table, identity=identity, session=session)
    assert result.abstained is False, result
    assert result.selected_document_ids == ["RFP-000982"], result
    assert session.active_document_id == "RFP-000982"
    assert result.failure is None, result


@pytest.mark.parametrize("question", [
    "그 사업 말고 미래재단의 예산 알려줘",
    "그 문서 대신 미래재단의 지역제한 알려줘",
])
def test_unknown_named_org_cannot_reuse_active_document(routing_world, question):
    table, identity = routing_world
    result = run_answer(question, table, identity=identity,
                        session=SessionState(active_document_id="RFP-000981"))
    assert result.abstained is True, result
    assert result.selected_document_ids == [], result
    assert result.citations == [], result
    assert result.failure is None, result


@pytest.mark.parametrize("question", [
    "가온재단의 바다 관측망 구축 사업 예산 알려줘",
    "별빛공사의 하늘 관측망 구축 사업 예산 알려줘",
    "초록협회의 바다 관측망 구축 사업 지역제한 알려줘",
])
def test_org_and_project_from_different_documents_must_not_pick_one(routing_world, question):
    table, identity = routing_world
    result = run_answer(question, table, identity=identity)
    assert result.abstained is True, result
    assert result.citations == [], result
    assert result.failure is None, result


@pytest.mark.parametrize("question", [
    "그 사업 예산 알려줘", "해당 문서의 예산은?", "거기 지역제한 알려줘",
    "마감일 알려줘",
])
def test_actual_followup_still_uses_active_document(routing_world, question):
    table, identity = routing_world
    result = run_answer(question, table, identity=identity,
                        session=SessionState(active_document_id="RFP-000981"))
    assert result.abstained is False, result
    assert result.selected_document_ids == ["RFP-000981"], result
    assert result.failure is None, result


@pytest.mark.parametrize("question", [
    "RFP-000981의 예산 알려줘", "(RFP-000981) 예산 알려줘",
    "문서 ID: RFP-000981, 예산 알려줘", "RFP-000981의 예산과 RFP-000981의 예산 알려줘",
])
def test_exact_and_repeated_same_id_preserve_single_document(routing_world, question):
    table, identity = routing_world
    result = run_answer(question, table, identity=identity)
    assert result.abstained is False, result
    assert result.selected_document_ids == ["RFP-000981"], result
    assert result.failure is None, result


def test_real_two_document_comparison_is_preserved(routing_world):
    table, identity = routing_world
    result = run_answer("RFP-000981과 RFP-000982의 예산 비교해줘", table, identity=identity)
    assert result.task_type == "compare" and result.abstained is False, result
    assert set(result.selected_document_ids) == {"RFP-000981", "RFP-000982"}, result
    assert "1억원" in result.text and "2억원" in result.text
    assert result.failure is None, result


def _answer_with_synthetic_search(question, table, identity, session=None):
    """실제 검색·실행부는 그대로 두고 외부 API 두 개만 결정적인 대역으로 바꾼다."""
    from answer_pipeline import answer
    from conftest import FakeEmbed, FakeGen
    from vector_store import ChunkMetadata, VectorStore
    store = VectorStore()
    text = "정량평가는 수치 기준이며 정성평가는 제안 내용의 전문성 기준입니다."
    meta = ChunkMetadata(
        chunk_id="RFP-000981-C01", document_id="RFP-000981", document_version="1",
        processed_sha256="a", sidecar_sha256="b", corpus_version="v2", text=text,
        document_name="하늘 관측망 구축 사업", chapter="3. 제안 평가",
        section_path=[{"level": 1, "line": 10, "title": "3. 제안 평가"}],
        section_paths=[[{"level": 1, "line": 10, "title": "3. 제안 평가"}]],
        location_label="3. 제안 평가 · 문단 1", md_line_start=10, md_line_end=12,
    )
    store.upsert_document("RFP-000981", [(text, [1.0, 0.0])], [meta])
    embed, generate = FakeEmbed(), FakeGen(text)
    result = answer(question, store, lambda: embed, lambda: generate, table, cfg(),
                    identity=identity, session=session)
    return result, embed, generate


@pytest.mark.parametrize("question", [
    "RFP-000981에서 정량평가와 정성평가의 차이를 설명해줘",
    "RFP-000981의 정량평가와 정성평가 차이가 나는 이유는?",
    "이 문서에서 정량평가와 정성평가의 차이를 설명해줘",
    "해당 사업의 정량평가와 정성평가 차이를 자세히 알려줘",
])
def test_single_document_concept_comparison_uses_its_own_evidence(routing_world, question):
    table, identity = routing_world
    result, embed, generate = _answer_with_synthetic_search(
        question, table, identity, SessionState(active_document_id="RFP-000981"))
    assert result.task_type == "qa" and result.abstained is False, result
    assert embed.calls >= 1 and generate.calls == 1
    assert result.citations and {c["document"] for c in result.citations} == {"RFP-000981"}
    assert result.failure is None, result


@pytest.mark.parametrize("question", [
    "RFP-000981과 RFP-000982의 예산 차이를 자세히 설명해줘",
    "두 사업 예산의 차이를 설명해줘",
    "RFP-000981과 다른 사업의 예산을 비교해줘",
])
def test_real_multi_document_comparison_does_not_become_single_document_qa(question):
    assert route(question, cfg()).task_type == "compare"


@pytest.mark.parametrize("question", [
    "RFP-000981, RFP-000982, RFP-999999의 예산 비교해줘",
    "RFP-000981, RFP-000982, RFP-0009830의 예산 비교해줘",
    "가온재단과 별빛공사와 미래재단의 예산 비교해줘",
    "하늘 관측망 구축 사업과 바다 관측망 구축 사업과 우주 관측망 구축 사업 예산 비교해줘",
    "가온재단의 바다 관측망 구축 사업과 초록협회의 예산 비교해줘",
])
def test_unresolved_third_or_conflicting_comparison_target_is_not_dropped(routing_world, question):
    table, identity = routing_world
    result = run_answer(question, table, identity=identity)
    assert result.abstained is True, result
    assert result.citations == [], result
    assert result.failure is None, result


def test_ambiguous_third_org_is_not_silently_omitted(tmp_path):
    specs = {f"RFP-{number:06d}": {"예산": present(f"{number}만원")}
             for number in (981, 982, 983, 984)}
    table = write_table(tmp_path, specs)
    identity = write_identity(tmp_path, [
        ("RFP-000981", "가온재단", "하늘 구축 사업", "2025-01-01 17:00:00"),
        ("RFP-000982", "별빛공사", "바다 구축 사업", "2025-01-01 17:00:00"),
        ("RFP-000983", "초록협회", "숲길 구축 사업", "2025-01-01 17:00:00"),
        ("RFP-000984", "초록협회", "산책로 구축 사업", "2025-01-01 17:00:00"),
    ])
    result = run_answer("가온재단과 별빛공사와 초록협회 예산 비교해줘", table, identity=identity)
    assert result.abstained is True and result.citations == [], result
    assert result.failure is None, result


@pytest.mark.parametrize("org", ["새가온재단", "미래가온재단", "신별빛공사"])
@pytest.mark.parametrize("with_active", [False, True])
def test_unknown_longer_org_is_not_registered_shorter_org(routing_world, org, with_active):
    table, identity = routing_world
    result = run_answer(f"{org}의 예산 알려줘", table, identity=identity,
                        session=SessionState(active_document_id="RFP-000983") if with_active else None)
    assert result.abstained is True and result.selected_document_ids == [], result
    assert result.citations == [] and result.failure is None, result


@pytest.mark.parametrize("org", ["가온재단", "재단법인 가온재단", "(재)가온재단", "가온 재단"])
def test_exact_legal_prefix_and_spacing_still_match_registered_org(routing_world, org):
    table, identity = routing_world
    result = run_answer(f"{org}의 예산 알려줘", table, identity=identity)
    assert result.abstained is False and result.selected_document_ids == ["RFP-000981"], result
    assert result.failure is None, result


@pytest.mark.parametrize("separator", ["과 ", "과", " 및 ", " 그리고 "])
def test_known_org_conjunctions_do_not_become_unknown_org(routing_world, separator):
    from doc_resolver import detect_unknown_orgs
    _, identity = routing_world
    assert detect_unknown_orgs(f"가온재단{separator}별빛공사의 예산 비교해줘", identity) == []
