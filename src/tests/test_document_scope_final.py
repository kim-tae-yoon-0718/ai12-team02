#@title 문서 범위 최종 회귀 검증
#@markdown 제목의 구두점·조사·공백 차이와 대화 상태 때문에 다른 문서를 답하지 않는지 검사합니다.
"""합성 자료만으로 실제 answer() 경로의 단일 문서 범위를 검사한다."""
from __future__ import annotations

import pytest

from doc_resolver import looks_like_specific_document, resolve_document
from test_selection_fix import present, run_answer, write_identity, write_registry, write_table


@pytest.fixture
def scope_world(tmp_path):
    specs = {
        "RFP-000951": {"예산": present("6억원"), "필수 제출 서류": present("제안서, 신청서")},
        "RFP-000952": {"예산": present("2억원")},
        "RFP-000953": {"예산": present("3억원")},
    }
    table = write_table(tmp_path, specs)
    scope = write_registry(tmp_path, [
        {"document_id": doc_id, "active": True, "retrieval_eligible": True}
        for doc_id in specs
    ])
    identity = write_identity(tmp_path, [
        ("RFP-000951", "가온철도공사", "모바일현장 시스템 고도화 용역", "2025-01-01 17:00:00"),
        ("RFP-000952", "새벽문화재단", "새벽문화재단 공연장 정보화 사업", "2025-01-01 17:00:00"),
        ("RFP-000953", "달빛연구원", "달빛연구원 관측망 구축", "2025-01-01 17:00:00"),
    ])
    return table, scope, identity


@pytest.mark.parametrize("question", [
    "은하수정보원 우주관측 데이터 플랫폼 구축 용역, 사업 예산이 얼마나 되는지 알려줄 수 있어?",
    "은하수정보원 우주관측 데이터 플랫폼 구축 용역의 사업 예산이 얼마나 되는지 알려줄 수 있어?",
    "은하수정보원 우주관측 데이터 플랫폼 구축 용역 사업 예산이 얼마나 되는지 알려줄 수 있어?",
    "은하수정보원 우주관측 데이터 플랫폼 구축 용역은 사업 예산이 얼마야?",
    "은하수정보원 우주관측 데이터 플랫폼 구축 용역: 예산 알려줘",
    "은하수정보원 우주관측 데이터 플랫폼 구축 용역의 필수 제출 서류 목록을 알려줘",
    "우주관측플랫폼구축사업의 예산이 얼마야?",
    "우주관측플랫폼 구축 사업에 지역 제한이 따로 있어?",
])
@pytest.mark.parametrize("with_active", [False, True])
def test_unknown_title_never_returns_global_or_previous_document(scope_world, question, with_active):
    from answer_pipeline import SessionState
    table, scope, identity = scope_world
    assert looks_like_specific_document(question)
    session = SessionState(active_document_id="RFP-000951") if with_active else None
    result = run_answer(question, table, identity=identity, registry_scope=scope, session=session)
    assert result.task_type == "extract", result
    assert result.abstained is True, result
    assert result.selected_document_ids == [], result
    assert result.citations == [], result
    assert result.failure is None, result


@pytest.mark.parametrize("question", [
    "모바일현장 시스템 고도화 용역의 예산이 얼마야?",
    "모바일현장 시스템 고도화 용역 예산 알려줘",
    "모바일현장 시스템 고도화 용역의 필수 제출 서류 목록을 알려줘",
])
def test_known_title_still_answers_only_named_document(scope_world, question):
    from answer_pipeline import SessionState
    table, scope, identity = scope_world
    result = run_answer(question, table, identity=identity, registry_scope=scope,
                        session=SessionState(active_document_id="RFP-000952"))
    assert result.task_type == "extract", result
    assert result.abstained is False, result
    assert result.selected_document_ids == ["RFP-000951"], result
    assert result.failure is None, result
    assert result.citations and all(c["document"] == "RFP-000951" for c in result.citations)


@pytest.mark.parametrize("question", [
    "전체 공고에서 예산 5억 이상인 공고 목록 줘",
    "이 사업과 별개로 전체 공고에서 예산 5억 이상인 공고를 찾아줘",
    "모바일현장 시스템 고도화 용역과 별개로 전체 공고에서 예산 5억 이상인 공고를 찾아줘",
    "예산이 기재된 공고들을 찾아줘",
    "제출 서류 목록이 명시된 사업들을 찾아줘",
])
def test_explicit_multi_document_scope_preserved(scope_world, question):
    from answer_pipeline import SessionState
    table, scope, identity = scope_world
    result = run_answer(question, table, identity=identity, registry_scope=scope,
                        session=SessionState(active_document_id="RFP-000952"))
    assert result.task_type == "select", result
    assert result.abstained is False, result
    expected = ["RFP-000951", "RFP-000952", "RFP-000953"] if question == "예산이 기재된 공고들을 찾아줘" else ["RFP-000951"]
    assert set(result.selected_document_ids) == set(expected), result
    assert result.failure is None, result


@pytest.mark.parametrize("question", [
    "그 사업 예산이 얼마야?",
    "이 사업의 전체 사업비가 기재되어 있어?",
    "해당 사업의 제출 서류 목록을 알려줘",
])
def test_genuine_follow_up_keeps_active_document(scope_world, question):
    from answer_pipeline import SessionState
    table, scope, identity = scope_world
    assert looks_like_specific_document(question) is False
    result = run_answer(question, table, identity=identity, registry_scope=scope,
                        session=SessionState(active_document_id="RFP-000951"))
    assert result.task_type == "extract", result
    assert result.selected_document_ids == ["RFP-000951"], result
    assert result.failure is None, result


@pytest.mark.parametrize("question", [
    "그 사업 말고 은하수정보원 우주관측 데이터 플랫폼 구축 용역의 예산 알려줘",
    "해당 사업 대신 우주관측플랫폼구축사업의 예산을 알려줘",
])
def test_explicit_new_title_overrides_stale_anaphora(scope_world, question):
    table, scope, identity = scope_world
    resolution = resolve_document(question, identity, active_document_id="RFP-000951")
    assert resolution.document_id is None, resolution
    assert resolution.candidates == [], resolution


@pytest.mark.parametrize("question", [
    "예산이 기재된 공고의 목록을 알려줘",
    "공동수급이 가능한 사업 예산을 알려줘",
    "지역 제한이 없는 공고를 알려줘",
    "제출 서류 목록이 기재된 사업을 찾아줘",
    "그 사업의 예산을 알려줘",
    "공고의 예산을 알려줘",
    "전체 사업의 예산을 알려줘",
    "그런 공고 있으면 알려줘",
    "이런 사업의 예산을 알려줘",
])
def test_condition_or_generic_scope_is_not_an_unregistered_title(question):
    assert looks_like_specific_document(question) is False


def test_unregistered_title_cannot_reuse_active_without_identity():
    resolution = resolve_document(
        "그 사업 말고 우주관측플랫폼구축사업의 예산 알려줘", None,
        active_document_id="RFP-000951",
    )
    assert resolution.document_id is None


def test_explicit_document_id_still_has_priority(scope_world):
    table, scope, identity = scope_world
    resolution = resolve_document("RFP-000952의 예산 알려줘", identity, "RFP-000951")
    assert resolution.document_id == "RFP-000952"


@pytest.mark.parametrize("question", [
    "새벽문화재단 우주관측 데이터 플랫폼 구축 사업의 예산 알려줘",
    "새벽문화재단 우주관측 데이터 플랫폼 구축 사업 예산 알려줘",
    "새벽문화재단 우주관측 데이터 플랫폼 구축 사업, 예산 알려줘",
    "새벽 문화 재단 우주관측 데이터 플랫폼 구축 사업의 예산 알려줘",
    "재단법인 새벽문화재단 우주관측 데이터 플랫폼 구축 사업의 예산 알려줘",
])
@pytest.mark.parametrize("with_active", [False, True])
def test_known_org_does_not_substitute_its_other_project(scope_world, question, with_active):
    from answer_pipeline import SessionState
    table, scope, identity = scope_world
    result = run_answer(question, table, identity=identity, registry_scope=scope,
                        session=SessionState(active_document_id="RFP-000951") if with_active else None)
    assert result.task_type == "extract", result
    assert result.abstained is True, result
    assert result.selected_document_ids == [], result
    assert result.citations == [], result
    assert result.failure is None, result


@pytest.mark.parametrize("question", [
    "새벽문화재단 사업 예산 알려줘",
    "새벽문화재단의 예산을 알려줘",
    "재단법인 새벽문화재단 사업의 예산 알려줘",
])
def test_org_only_question_keeps_unique_org_resolution(scope_world, question):
    table, scope, identity = scope_world
    result = run_answer(question, table, identity=identity, registry_scope=scope)
    assert result.task_type == "extract", result
    assert result.abstained is False, result
    assert result.selected_document_ids == ["RFP-000952"], result
    assert result.failure is None, result


def test_known_org_with_multiple_projects_does_not_hide_unknown_title(tmp_path):
    specs = {"RFP-000991": {"예산": present("1억원")},
             "RFP-000992": {"예산": present("2억원")}}
    table = write_table(tmp_path, specs)
    identity = write_identity(tmp_path, [
        ("RFP-000991", "별빛재단", "별빛재단 도서관 구축 사업", "2025-01-01 17:00:00"),
        ("RFP-000992", "별빛재단", "별빛재단 공연장 구축 사업", "2025-01-01 17:00:00"),
    ])
    result = run_answer("별빛재단 우주관측 데이터 플랫폼 구축 사업의 예산 알려줘", table,
                        identity=identity)
    assert result.abstained is True and result.selected_document_ids == [], result
    # 기관만 묻는 원래 다후보 질문은 후보를 보여주며 선택을 요청한다.
    generic = run_answer("별빛재단 사업 예산 알려줘", table, identity=identity)
    assert generic.abstained is True, generic
    assert set(generic.selected_document_ids) == set(specs), generic


@pytest.mark.parametrize("question", [
    "입찰 참가를 준비 중인데 사업 예산이 기재된 공고 찾아줘",
    "새 프로젝트를 검토하려고 사업 예산이 기재된 공고 찾아줘",
    "우리 팀이 바빠서 사업 예산이 기재된 공고를 찾아줘",
    "이번에 자료를 정리하고 사업 예산이 기재된 공고를 찾아줘",
    "입찰을 준비 중이라 사업 예산이 기재된 공고를 찾아줘",
])
def test_genuine_background_is_not_an_unknown_title(scope_world, question):
    table, scope, identity = scope_world
    assert looks_like_specific_document(question) is False
    result = run_answer(question, table, identity=identity, registry_scope=scope)
    assert result.task_type == "select", result
    assert result.abstained is False, result
    assert set(result.selected_document_ids) == {"RFP-000951", "RFP-000952", "RFP-000953"}, result
    assert result.failure is None, result


def test_deictic_such_projects_does_not_become_a_project_name(scope_world):
    table, scope, identity = scope_world
    question = "컨소시엄 요건은 있는데 평가 배점표는 안 보이는 공고가 있다던데, 그런 공고 있으면 알려줘."
    assert looks_like_specific_document(question) is False
    result = run_answer(question, table, identity=identity, registry_scope=scope)
    assert result.task_type == "select" and result.abstained is False, result
    # 합성 자료에는 컨소시엄 값이 있는 문서가 없으므로 확정 0건이다.
    assert result.selected_document_ids == [], result


def test_abbreviated_title_keeps_compatible_registered_project(tmp_path):
    specs = {"RFP-000981": {"예산": present("6억원")}}
    table = write_table(tmp_path, specs)
    identity = write_identity(tmp_path, [
        ("RFP-000981", "아침문화재단", "2024년 공연장 정보화 구축 및 행사지원 시스템 공급 용역 입찰공고", "2025-01-01 17:00:00"),
    ])
    result = run_answer("아침문화재단 공연장 정보화 사업의 예산 알려줘", table, identity=identity)
    assert result.abstained is False and result.selected_document_ids == ["RFP-000981"], result


def test_shared_generic_title_word_is_not_enough_to_accept_other_project(tmp_path):
    specs = {"RFP-000981": {"예산": present("6억원")}}
    table = write_table(tmp_path, specs)
    identity = write_identity(tmp_path, [
        ("RFP-000981", "아침문화재단", "재난 데이터 플랫폼 구축 사업", "2025-01-01 17:00:00"),
    ])
    result = run_answer("아침문화재단 우주관측 데이터 플랫폼 구축 사업의 예산 알려줘", table, identity=identity)
    assert result.abstained is True and result.selected_document_ids == [], result
