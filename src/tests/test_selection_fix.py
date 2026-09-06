"""선별형 결함 수정 회귀 테스트 (2026-09-03).

⚠️ 평가 문항 번호·평가셋 정답 문서 집합을 외우지 않는다. 여기 쓰는 자료는 전부
   이 파일 안에서 만든 합성 자료이고, 문서 ID도 평가셋과 겹치지 않는 대역
   (RFP-0009xx)을 쓴다.

⚠️ 생성·임베딩 클라이언트는 "호출되면 실패"하는 감시자를 넣는다. 모의 성공값을
   돌려주지 않는다 — 선별 경로가 API를 쓰지 않는다는 사실 자체를 검증하기 위해서다.
   수정 대상인 라우팅·조건 파서·조회 함수는 모의로 바꾸지 않고 실제 코드를 쓴다.
"""
from __future__ import annotations

import json

import pytest

OFFICIAL_FIELDS = [
    "사업 개요", "사업분야", "공고일", "사업기간", "예산",
    "참가 자격(면허·실적)", "지역제한", "컨소시엄 요건",
    "평가 배점", "제출 방식", "필수 제출 서류", "과업 범위",
]

CFG = {
    "top_k": 5,
    "routing_method": "rule_based",
    "routing_fallback": "qa",
    "corpus": "v2",
    "document_registry_version": "v2",
    "extraction_version": "v3",
    "schema_version": "1-12-2/v3",
    "chunking_version": "v3",
    "reference_datetime_source": "external",
    "reference_datetime": "2024-06-01",
    "deadline_filter_field": "bid_deadline",
    "deadline_missing_policy": "show_as_unknown",
    "deadline_filter_disclosure": False,
    "deadline_filter_default": {"select": True, "extract": False,
                                "qa": False, "compare": False},
}


def cfg(**over) -> dict:
    out = json.loads(json.dumps(CFG))
    out.update(over)
    return out


# ---------------------------------------------------------------------------
# 합성 자료 만들기
# ---------------------------------------------------------------------------

def _loc(line: int = 10) -> dict:
    return {"block_index": 1, "block_type": "paragraph", "file": "x.md",
            "heading": "1. 개요", "line": line, "line_end": line, "line_start": line,
            "section_path": [{"level": 1, "line": line, "title": "1. 개요"}],
            "source_type": "body_sentence"}


def make_rows(doc_id: str, overrides: dict[str, dict]) -> list[dict]:
    rows = []
    for f in OFFICIAL_FIELDS:
        row = {"document_id": doc_id, "field_name": f, "status": "field_absent",
               "answer_raw": "", "answer_normalized": "", "active": "true",
               "representative_location": None, "additional_locations": [],
               "schema_version": "1-12-2/v3", "extraction_version": "v3",
               "corpus_version": "v2", "registry_version": "v2"}
        row.update(overrides.get(f, {}))
        rows.append(row)
    return rows


def present(value: str, line: int = 10) -> dict:
    return {"status": "value_present", "answer_raw": value,
            "answer_normalized": value, "representative_location": _loc(line)}


def absent() -> dict:
    return {"status": "field_absent", "answer_raw": "", "answer_normalized": ""}


def external(value: str = "입찰공고문 참조") -> dict:
    return {"status": "external_reference", "answer_raw": value,
            "answer_normalized": value, "representative_location": _loc(20)}


def not_disclosed(value: str = "비공개") -> dict:
    return {"status": "not_disclosed", "answer_raw": value,
            "answer_normalized": value, "representative_location": _loc(30)}


def conflict(a: str, b: str) -> dict:
    return {"status": "conflict", "answer_raw": f"{a}\n---\n{b}",
            "answer_normalized": f"{a} --- {b}",
            "representative_location": _loc(40),
            "additional_locations": [_loc(50)]}


def extraction_failed() -> dict:
    return {"status": "extraction_failed", "answer_raw": "", "answer_normalized": ""}


def write_table(tmp_path, doc_specs: dict[str, dict]):
    """{문서ID: {필드: 상태덩어리}} → 추출표 rows 로 로드."""
    from table_query import load_extraction_table
    rows: list[dict] = []
    for doc_id, over in doc_specs.items():
        rows += make_rows(doc_id, over)
    doc = {"corpus_version": "v2", "document_count": len(doc_specs),
           "extraction_version": "v3", "field_count": 12, "fields": OFFICIAL_FIELDS,
           "registry_version": "v2", "row_count": len(rows),
           "schema_version": "1-12-2/v3", "rows": rows}
    p = tmp_path / "extraction_table_v3.json"
    p.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    return load_extraction_table(p)


def write_registry(tmp_path, docs: list[dict]):
    from document_registry import load_registry_scope
    p = tmp_path / "document_registry_v2.json"
    p.write_text(json.dumps({"document_count": len(docs), "documents": docs},
                            ensure_ascii=False), encoding="utf-8")
    return load_registry_scope(p)


def write_identity(tmp_path, rows: list[tuple[str, str, str, str]]):
    """(document_id, buyer_org, project_name, bid_deadline)"""
    from identity_metadata import load_identity
    header = ('"document_id","source_filename_nfc","collection_system",'
              '"source_record_id","source_url","notice_number","notice_round",'
              '"buyer_org","project_name","notice_date","bid_deadline","metadata_found"')
    lines = [header]
    for doc_id, org, project, deadline in rows:
        lines.append(f'"{doc_id}","{doc_id}.md","","","","N","0.0",'
                     f'"{org}","{project}","2024-01-01 10:00:00","{deadline}","true"')
    p = tmp_path / "document_identity_v2.csv"
    p.write_text("﻿" + "\n".join(lines) + "\n", encoding="utf-8")
    return load_identity(p)


class ApiCalled(AssertionError):
    pass


class ForbiddenClient:
    """호출되면 실패한다. 모의 성공값을 돌려주지 않는다."""

    def __init__(self, kind: str):
        self.kind = kind

    def __getattr__(self, name):
        def _raise(*a, **k):
            raise ApiCalled(f"선별 경로에서 {self.kind}.{name} 가 호출됐다")
        return _raise


def empty_store():
    from vector_store import VectorStore
    return VectorStore()


def run_answer(question, table, *, identity=None, registry_scope=None,
               config=None, session=None, locator=None):
    """실제 진입점 answer() 를 그대로 탄다(파서만 따로 부르지 않는다)."""
    from answer_pipeline import answer
    return answer(question, empty_store(),
                  lambda: ForbiddenClient("embedding"),
                  lambda: ForbiddenClient("generation"),
                  table, config or cfg(), identity=identity, session=session,
                  locator=locator, registry_scope=registry_scope)


# ---------------------------------------------------------------------------
# 공통 픽스처 — 12필드 상태를 모두 담은 작은 세계
# ---------------------------------------------------------------------------

@pytest.fixture
def status_world(tmp_path):
    """상태별로 문서를 하나씩 둔 표 + 마감일 전부 유효한 identity."""
    specs = {
        # 값이 있는 문서
        "RFP-000901": {f: present(f"{f} 값") for f in OFFICIAL_FIELDS},
        # 전 필드 미기재
        "RFP-000902": {},
        # 외부 참조
        "RFP-000903": {f: external() for f in OFFICIAL_FIELDS},
        # 비공개
        "RFP-000904": {f: not_disclosed() for f in OFFICIAL_FIELDS},
        # 충돌
        "RFP-000905": {f: conflict("내용 A", "내용 B") for f in OFFICIAL_FIELDS},
        # 추출 실패
        "RFP-000906": {f: extraction_failed() for f in OFFICIAL_FIELDS},
    }
    table = write_table(tmp_path, specs)
    ids = sorted(specs)
    scope = write_registry(tmp_path, [
        {"document_id": d, "active": True, "retrieval_eligible": True,
         "duplicate_of_document_id": None} for d in ids])
    identity = write_identity(tmp_path, [
        (d, f"기관{d[-3:]}", f"사업{d[-3:]}", "2025-01-01 17:00:00") for d in ids])
    return table, scope, identity


# ===========================================================================
# ① 12필드별 상태 조회
# ===========================================================================

class TestTwelveFieldStatusQueries:
    QUESTIONS_PRESENT = {
        "사업 개요": "사업개요가 기재된 공고들 알려줘",
        "사업분야": "사업분야가 명시된 공고 목록 줘",
        "공고일": "공고일이 적혀 있는 공고들 전부 알려줘",
        "사업기간": "사업기간이 나와 있는 공고들 알려줘",
        "예산": "예산이 공개된 공고 목록 줘",
        "참가 자격(면허·실적)": "참가 자격 조건이 적힌 공고들 알려줘",
        "지역제한": "지역제한 조건이 명시된 공고 목록 줘",
        "컨소시엄 요건": "컨소시엄 요건이 명시된 공고들 알려줘",
        "평가 배점": "평가 배점이 나와 있는 공고 목록 줘",
        "제출 방식": "제출 방식이 안내된 공고들 알려줘",
        "필수 제출 서류": "필수 제출 서류가 안내돼 있는 공고 목록 줘",
        "과업 범위": "과업 범위가 기재된 공고들 알려줘",
    }

    @pytest.mark.parametrize("field", OFFICIAL_FIELDS)
    def test_value_present_query_per_field(self, field, status_world):
        table, scope, identity = status_world
        r = run_answer(self.QUESTIONS_PRESENT[field], table,
                       identity=identity, registry_scope=scope)
        assert r.task_type == "select", self.QUESTIONS_PRESENT[field]
        assert r.abstained is False
        assert r.selected_document_ids == ["RFP-000901"]

    @pytest.mark.parametrize("field,question", [
        ("공고일", "공고일이 기재돼 있지 않은 공고들 알려줘"),
        ("예산", "예산 정보가 안 나와 있는 공고들만 알려줘"),
        ("제출 방식", "제출 방식이 안 적혀 있는 공고 목록 줘"),
        ("필수 제출 서류", "필수 제출 서류 안내가 없는 공고들 찾아줘"),
    ])
    def test_field_absent_query(self, field, question, status_world):
        table, scope, identity = status_world
        r = run_answer(question, table, identity=identity, registry_scope=scope)
        assert r.task_type == "select"
        assert r.selected_document_ids == ["RFP-000902"], question

    def test_external_reference_query(self, status_world):
        table, scope, identity = status_world
        r = run_answer("제출 방식은 별도 문서를 따로 봐야 하는 공고들 알려줘", table,
                       identity=identity, registry_scope=scope)
        assert r.selected_document_ids == ["RFP-000903"]

    def test_absent_query_does_not_collect_other_states(self, status_world):
        """value_present 가 아니라고 전부 field_absent 로 묶지 않는다."""
        table, scope, identity = status_world
        r = run_answer("예산 정보가 안 나와 있는 공고들만 알려줘", table,
                       identity=identity, registry_scope=scope)
        for other in ("RFP-000903", "RFP-000904", "RFP-000905", "RFP-000906"):
            assert other not in r.selected_document_ids


# ===========================================================================
# ② 미기재·비공개·충돌·추출 실패 구분
# ===========================================================================

class TestStatesAreNotConflated:
    def test_four_states_are_four_different_answers(self, status_world):
        table, scope, identity = status_world
        got = {}
        for key, q in [
            ("absent", "평가 배점이 없는 공고 목록 줘"),
            ("external", "평가 배점은 별도 문서를 따로 봐야 하는 공고 목록 줘"),
            ("not_disclosed", "평가 배점이 비공개인 공고 목록 줘"),
            ("conflict", "평가 배점 내용이 서로 다른 공고 목록 줘"),
        ]:
            r = run_answer(q, table, identity=identity, registry_scope=scope)
            got[key] = r.selected_document_ids
        assert got["absent"] == ["RFP-000902"]
        assert got["external"] == ["RFP-000903"]
        assert got["not_disclosed"] == ["RFP-000904"]
        assert got["conflict"] == ["RFP-000905"]
        assert len({tuple(v) for v in got.values()}) == 4

    def test_extraction_failed_has_its_own_state(self, status_world):
        table, scope, identity = status_world
        r = run_answer("평가 배점 추출에 실패한 공고 목록 줘", table,
                       identity=identity, registry_scope=scope)
        assert r.selected_document_ids == ["RFP-000906"]

    def test_answer_text_explains_state_instead_of_none(self, status_world):
        """'RFP-000902: None' 같은 출력이 나오면 안 된다."""
        table, scope, identity = status_world
        r = run_answer("예산 정보가 안 나와 있는 공고들만 알려줘", table,
                       identity=identity, registry_scope=scope)
        assert "None" not in r.text
        assert "원문에 기재 없음" in r.text

    def test_absent_document_is_a_valid_result_not_an_abstain(self, status_world):
        """'미기재 문서를 찾아줘'는 값이 없다는 이유로 거절할 질문이 아니다."""
        table, scope, identity = status_world
        r = run_answer("예산 정보가 안 나와 있는 공고들만 알려줘", table,
                       identity=identity, registry_scope=scope)
        assert r.abstained is False and r.selected_document_ids


# ===========================================================================
# ③ 실제 지역제한 없음 vs 지역 정보 미기재
# ===========================================================================

class TestRegionMeaningVsBlank:
    @pytest.fixture
    def region_world(self, tmp_path):
        specs = {
            "RFP-000911": {"지역제한": present("지역제한 없음(전국 대상)")},
            "RFP-000912": {"지역제한": present("본점 소재지가 경기도인 업체에 한함")},
            "RFP-000913": {},                                   # 미기재
            "RFP-000914": {"지역제한": external()},
        }
        table = write_table(tmp_path, specs)
        scope = write_registry(tmp_path, [
            {"document_id": d, "active": True, "retrieval_eligible": True}
            for d in sorted(specs)])
        identity = write_identity(tmp_path, [
            (d, f"기관{d[-3:]}", f"사업{d[-3:]}", "2025-01-01 17:00:00")
            for d in sorted(specs)])
        return table, scope, identity

    def test_blank_is_not_treated_as_no_restriction(self, region_world):
        table, scope, identity = region_world
        r = run_answer("지역 제한이 아예 없는 공고들만 골라줄 수 있어?", table,
                       identity=identity, registry_scope=scope)
        assert r.selected_document_ids == ["RFP-000911"]
        assert "RFP-000913" not in r.selected_document_ids
        # 미기재는 '판단 불가'로 남고 결과에서 조용히 사라지지 않는다
        assert "RFP-000913" in r.text and "판단 불가" in r.text

    def test_value_present_alone_is_not_restriction(self, region_world):
        table, scope, identity = region_world
        r = run_answer("지역 제한이 있는 공고들만 알려줘", table,
                       identity=identity, registry_scope=scope)
        assert r.selected_document_ids == ["RFP-000912"]

    def test_stated_vs_meaning_are_different_questions(self, region_world):
        table, scope, identity = region_world
        stated = run_answer("지역제한 조건이 명시된 공고 목록 줘", table,
                            identity=identity, registry_scope=scope)
        meaning = run_answer("지역 제한이 아예 없는 공고들만 알려줘", table,
                             identity=identity, registry_scope=scope)
        assert set(stated.selected_document_ids) == {"RFP-000911", "RFP-000912"}
        assert meaning.selected_document_ids == ["RFP-000911"]


# ===========================================================================
# ④ 공동수급 허용·금지·필수·조건부 / 하도급 구분
# ===========================================================================

class TestConsortiumSemantics:
    CASES = {
        "RFP-000921": "본 사업은 공동수급을 허용하며, 구성원은 5개사 이내로 한다",
        "RFP-000922": "본 사업은 공동수급을 불허함",
        "RFP-000923": "공동수급체를 구성하여야 하며 단독 입찰은 불가함",
        "RFP-000924": "단독 또는 공동수급체(공동이행방식)를 구성하여 참가할 수 있으며, "
                      "구성원은 3개 사 이하로 제한된다",
        "RFP-000925": "하도급은 불가함",                       # 공동수급 언급 없음
        "RFP-000926": "(공동이행방식)은 허용하지 않음",         # 방식만 한정
        "RFP-000927": "공동수급 및 하도급 불가",
    }

    @pytest.fixture
    def world(self, tmp_path):
        specs = {d: {"컨소시엄 요건": present(v)} for d, v in self.CASES.items()}
        specs["RFP-000928"] = {}                               # 미기재
        specs["RFP-000929"] = {"컨소시엄 요건": conflict("공동수급 허용", "공동수급 불허")}
        table = write_table(tmp_path, specs)
        scope = write_registry(tmp_path, [
            {"document_id": d, "active": True, "retrieval_eligible": True}
            for d in sorted(specs)])
        identity = write_identity(tmp_path, [
            (d, f"기관{d[-3:]}", f"사업{d[-3:]}", "2025-01-01 17:00:00")
            for d in sorted(specs)])
        return table, scope, identity

    def test_verdicts(self, world):
        from table_query import classify_consortium, lookup_field
        table, _scope, _identity = world
        v = lambda d: classify_consortium(lookup_field(table, d, "컨소시엄 요건"))
        assert (v("RFP-000921").required, v("RFP-000921").allowed) == (False, True)
        assert (v("RFP-000922").required, v("RFP-000922").allowed) == (False, False)
        assert (v("RFP-000923").required, v("RFP-000923").allowed) == (True, True)
        assert (v("RFP-000924").required, v("RFP-000924").allowed) == (False, True)
        # 하도급만 불가 → 공동수급에 대해서는 아무 말도 하지 않았다
        assert v("RFP-000925").required is None and v("RFP-000925").allowed is None
        # 공동이행방식만 불허 → 공동수급 전체 금지로 단정하지 않는다.
        # ⚠️ 2026-09-03 기대값 정정: 예전에는 required is False(= 단독 참여 가능)까지
        #    단정했다. 근거가 성립하지 않는다 — "공동이행방식을 허용하지 않는다"는 문장은
        #    분담이행·주계약자관리 방식에 대해 아무 말도 하지 않으므로, 그 문서가 단독
        #    참여를 허락한다는 뜻이 될 수 없다. 금지 문구는 어떤 경우에도 '필수 아님'을
        #    만들어내지 못한다. 이제 두 값 모두 판단 불가로 남긴다.
        assert v("RFP-000926").allowed is None and v("RFP-000926").required is None
        assert (v("RFP-000927").required, v("RFP-000927").allowed) == (False, False)
        # 미기재는 '단독 참여 가능'이 아니다
        assert v("RFP-000928").required is None
        # 충돌은 한쪽을 고르지 않는다
        assert v("RFP-000929").required is None and v("RFP-000929").allowed is None

    def test_allowed_is_not_required(self, world):
        table, scope, identity = world
        req = run_answer("컨소시엄이 필요한 공고들만 알려줘", table,
                         identity=identity, registry_scope=scope)
        assert req.selected_document_ids == ["RFP-000923"]
        assert "RFP-000921" not in req.selected_document_ids
        assert "RFP-000924" not in req.selected_document_ids

    def test_not_required_includes_allowed_and_forbidden(self, world):
        """'필수 아님'에는 허용 문서와 금지 문서가 모두 들어간다.

        ⚠️ 2026-09-03 기대값 정정: RFP-000926("(공동이행방식)은 허용하지 않음")을
        빼도록 바꿨다. 특정 이행방식만 불허한 문장에서 "공동수급이 필수가 아니다"를
        끌어낼 근거가 없어서, 이제 그 문서는 확정 결과가 아니라 판단 불가로 남는다."""
        table, scope, identity = world
        r = run_answer("컨소시엄까지는 요구 안 하는 공고들 알려줘", table,
                       identity=identity, registry_scope=scope)
        assert set(r.selected_document_ids) == {
            "RFP-000921", "RFP-000922", "RFP-000924", "RFP-000927"}
        assert "RFP-000923" not in r.selected_document_ids
        assert "RFP-000928" not in r.selected_document_ids   # 미기재는 판단 불가
        # 방식만 불허한 문서는 '조건에 안 맞음'이 아니라 '판단 불가'로 남아야 한다
        assert "RFP-000926" in r.text and "판단 불가" in r.text

    def test_subcontract_ban_is_not_consortium_ban(self, world):
        table, scope, identity = world
        r = run_answer("공동수급이 금지된 공고들 알려줘", table,
                       identity=identity, registry_scope=scope)
        assert "RFP-000925" not in r.selected_document_ids
        assert "RFP-000926" not in r.selected_document_ids
        assert set(r.selected_document_ids) == {"RFP-000922", "RFP-000927"}

    def test_absent_consortium_field_is_undetermined_not_solo_ok(self, world):
        table, scope, identity = world
        r = run_answer("컨소시엄까지는 요구 안 하는 공고들 알려줘", table,
                       identity=identity, registry_scope=scope)
        assert "RFP-000928" in r.text and "판단 불가" in r.text

    def test_consortium_field_absent_query_is_status_not_meaning(self, world):
        """'컨소시엄 요건이 없는' = 요건이 기재되지 않은 문서."""
        table, scope, identity = world
        r = run_answer("컨소시엄 요건이 없는 공고 목록 줘", table,
                       identity=identity, registry_scope=scope)
        assert r.selected_document_ids == ["RFP-000928"]


# ===========================================================================
# ⑤⑥ 특정 문서 질문 vs 여러 문서 선별
# ===========================================================================

class TestDocumentScopedVsSelection:
    @pytest.fixture
    def world(self, tmp_path):
        specs = {
            "RFP-000931": {"필수 제출 서류": present("가. 제안서\n나. 사업자등록증"),
                           "예산": present("3억원")},
            "RFP-000932": {"필수 제출 서류": present("제안서 1부"),
                           "예산": present("7억원")},
            "RFP-000933": {},
        }
        table = write_table(tmp_path, specs)
        scope = write_registry(tmp_path, [
            {"document_id": d, "active": True, "retrieval_eligible": True}
            for d in sorted(specs)])
        identity = write_identity(tmp_path, [
            ("RFP-000931", "한빛문화재단", "한빛문화재단 기록관리시스템 구축",
             "2025-01-01 17:00:00"),
            ("RFP-000932", "새벽교통공사", "새벽교통공사 요금정산 고도화",
             "2025-01-01 17:00:00"),
            ("RFP-000933", "달빛연구원", "달빛연구원 데이터 플랫폼",
             "2025-01-01 17:00:00"),
        ])
        return table, scope, identity

    def test_single_document_submission_list(self, world):
        table, scope, identity = world
        r = run_answer("RFP-000931의 제출서류 목록을 알려줘", table,
                       identity=identity, registry_scope=scope)
        assert r.task_type == "extract"
        assert r.selected_document_ids == ["RFP-000931"]

    def test_multi_document_selection_with_same_word(self, world):
        table, scope, identity = world
        r = run_answer("제출서류 목록이 기재된 사업들을 찾아줘", table,
                       identity=identity, registry_scope=scope)
        assert r.task_type == "select"
        assert set(r.selected_document_ids) == {"RFP-000931", "RFP-000932"}

    def test_org_and_project_name_still_selects_one_document(self, world):
        table, scope, identity = world
        r = run_answer("한빛문화재단 기록관리시스템 구축 사업에 지역 제한이 있어?",
                       table, identity=identity, registry_scope=scope)
        assert r.task_type == "extract"
        assert r.selected_document_ids == ["RFP-000931"]

    def test_active_document_follow_up_is_not_selection(self, world):
        from answer_pipeline import SessionState
        table, scope, identity = world
        session = SessionState(active_document_id="RFP-000932")
        r = run_answer("그 사업 예산이 얼마야?", table, identity=identity,
                       registry_scope=scope, session=session)
        assert r.task_type == "extract"
        assert r.selected_document_ids == ["RFP-000932"]

    def test_active_document_does_not_capture_new_global_question(self, world):
        from answer_pipeline import SessionState
        table, scope, identity = world
        session = SessionState(active_document_id="RFP-000932")
        r = run_answer("전체 공고 중에서 예산 5억 이상인 공고 목록 줘", table,
                       identity=identity, registry_scope=scope, session=session)
        assert r.task_type == "select"
        assert r.selected_document_ids == ["RFP-000932"]      # 조건에 맞는 문서
        assert r.route != "애매_되묻기"

    def test_field_value_question_without_document_still_asks_which(self, world):
        table, scope, identity = world
        r = run_answer("사업기간이 얼마야?", table, identity=identity,
                       registry_scope=scope)
        assert r.task_type == "extract" and r.abstained is True

    def test_greeting_and_help_preserved(self, world):
        table, scope, identity = world
        assert run_answer("안녕!", table, identity=identity,
                          registry_scope=scope).task_type == "no_search_needed"
        assert run_answer("이 시스템은 어떻게 사용하나요?", table, identity=identity,
                          registry_scope=scope).task_type == "no_search_needed"

    def test_ambiguous_document_still_asks_back(self, tmp_path):
        specs = {"RFP-000941": {"예산": present("1억원")},
                 "RFP-000942": {"예산": present("2억원")}}
        table = write_table(tmp_path, specs)
        identity = write_identity(tmp_path, [
            ("RFP-000941", "쌍둥이재단", "쌍둥이재단 1차 사업", "2025-01-01 17:00:00"),
            ("RFP-000942", "쌍둥이재단", "쌍둥이재단 2차 사업", "2025-01-01 17:00:00"),
        ])
        r = run_answer("쌍둥이재단 사업 예산 얼마야?", table, identity=identity)
        assert r.abstained is True
        assert set(r.selected_document_ids) == {"RFP-000941", "RFP-000942"}


# ===========================================================================
# ⑦⑧ 배경 설명 · AND 복합조건 · 부정 범위 · 미지원 조건
# ===========================================================================

class TestBackgroundAndCompound:
    @pytest.fixture
    def world(self, tmp_path):
        specs = {
            "RFP-000951": {"예산": present("6억원"), "사업기간": present("12개월")},
            "RFP-000952": {"예산": present("6억원")},              # 사업기간 미기재
            "RFP-000953": {"예산": present("1억원"), "사업기간": present("6개월")},
            "RFP-000954": {"예산": not_disclosed()},
        }
        table = write_table(tmp_path, specs)
        scope = write_registry(tmp_path, [
            {"document_id": d, "active": True, "retrieval_eligible": True}
            for d in sorted(specs)])
        identity = write_identity(tmp_path, [
            (d, f"기관{d[-3:]}", f"사업{d[-3:]}", "2025-01-01 17:00:00")
            for d in sorted(specs)])
        return table, scope, identity

    def test_background_sentence_is_not_an_unread_condition(self, world):
        table, scope, identity = world
        r = run_answer("입찰 준비를 하고 있는데, 예산 5억 이상인 공고 목록 줘", table,
                       identity=identity, registry_scope=scope)
        assert r.abstained is False
        assert set(r.selected_document_ids) == {"RFP-000951", "RFP-000952"}

    def test_background_sentence_is_not_silently_used_as_condition(self, world):
        """배경에서 언급한 다른 상태가 실제 조건을 덮어쓰지 않는다."""
        table, scope, identity = world
        r = run_answer("예산을 아예 비공개로 두는 공고도 있다고 해서 그런데, "
                       "예산 정보가 안 나와 있는 공고들만 알려줘", table,
                       identity=identity, registry_scope=scope)
        assert r.selected_document_ids == ["RFP-000953"] or \
               r.selected_document_ids == []      # 이 세계엔 예산 미기재 문서가 없음
        assert "RFP-000954" not in r.selected_document_ids   # 비공개는 미기재가 아님

    def test_and_condition_does_not_drop_second_condition(self, world):
        table, scope, identity = world
        r = run_answer("예산 5억 이상이고 사업기간이 기재된 공고를 찾아줘", table,
                       identity=identity, registry_scope=scope)
        assert r.selected_document_ids == ["RFP-000951"]

    def test_and_condition_without_connector(self, world):
        table, scope, identity = world
        r = run_answer("예산 5억 이상 사업기간이 기재된 공고 목록 줘", table,
                       identity=identity, registry_scope=scope)
        assert r.selected_document_ids == ["RFP-000951"]

    @pytest.mark.parametrize("question", [
        "예산 5억 이상인 공고들 알려줘",
        "예산이 5억 이상인 공고들 알려줘",
        "예산 5억 이상 공고 목록 줘",
        "예산 5억원 이상인 공고들 전부 보여줘",
    ])
    def test_particle_and_spacing_variants(self, question, world):
        table, scope, identity = world
        r = run_answer(question, table, identity=identity, registry_scope=scope)
        assert set(r.selected_document_ids) == {"RFP-000951", "RFP-000952"}, question

    def test_unsupported_condition_triggers_clarify(self, world):
        table, scope, identity = world
        r = run_answer("예산 5억 이상이고 발주기관이 서울인 공고 목록 줘", table,
                       identity=identity, registry_scope=scope)
        assert r.abstained is True
        assert "인식하지 못한 조건" in r.text
        assert r.selected_document_ids == []

    def test_or_is_not_silently_turned_into_and(self, world):
        table, scope, identity = world
        r = run_answer("예산이 기재됐거나 사업기간이 기재된 공고 목록 줘", table,
                       identity=identity, registry_scope=scope)
        assert r.abstained is True
        assert "OR" in r.text

    def test_negation_binds_to_its_own_field(self, world):
        table, scope, identity = world
        r = run_answer("예산은 기재돼 있고 사업기간은 기재돼 있지 않은 공고 목록 줘",
                       table, identity=identity, registry_scope=scope)
        assert r.selected_document_ids == ["RFP-000952"]

    def test_duration_number_is_not_read_as_budget(self, world):
        table, scope, identity = world
        r = run_answer("사업기간이 6개월 이상인 공고 목록 줘", table,
                       identity=identity, registry_scope=scope)
        assert r.abstained is True          # 지원하지 않는 숫자 조건 — 조용히 넘기지 않음
        assert "예산" not in " ".join(
            str(c.get("value")) for c in (r.condition_query or []) if "value" in c)


# ===========================================================================
# ⑨ 숫자·비교 연산자 기존 동작 보존
# ===========================================================================

class TestNumbersPreserved:
    @pytest.fixture
    def world(self, tmp_path):
        specs = {
            "RFP-000961": {"예산": present("49,500천원(부가세 포함)")},    # 4억 9,500만
            "RFP-000962": {"예산": present("1억 5천만 원")},
            "RFP-000963": {"예산": present("5억원")},
        }
        table = write_table(tmp_path, specs)
        scope = write_registry(tmp_path, [
            {"document_id": d, "active": True, "retrieval_eligible": True}
            for d in sorted(specs)])
        identity = write_identity(tmp_path, [
            (d, f"기관{d[-3:]}", f"사업{d[-3:]}", "2025-01-01 17:00:00")
            for d in sorted(specs)])
        return table, scope, identity

    def test_thousand_separator_not_split(self):
        from table_query import parse_conditions
        conds, full = parse_conditions("예산 49,500만원 이상인 공고 목록 줘")
        assert full and conds[0].value == 495_000_000.0

    def test_compound_amount(self):
        from table_query import parse_conditions
        conds, full = parse_conditions("예산 1억 5천만원 이상인 공고 목록 줘")
        assert full and conds[0].value == 150_000_000.0

    @pytest.mark.parametrize("phrase,op", [
        ("예산 5억 이상인 공고 목록 줘", ">="),
        ("예산 5억 초과인 공고 목록 줘", ">"),
        ("예산 5억 이하인 공고 목록 줘", "<="),
        ("예산 5억 미만인 공고 목록 줘", "<"),
        ("예산 5억 넘는 공고 목록 줘", ">"),
    ])
    def test_operators_distinct(self, phrase, op):
        from table_query import parse_conditions
        conds, full = parse_conditions(phrase)
        assert full and conds[0].operator == op and conds[0].value == 500_000_000.0

    def test_boundary_documents(self, world):
        table, scope, identity = world
        ge = run_answer("예산 5억 이상인 공고 목록 줘", table, identity=identity,
                        registry_scope=scope).selected_document_ids
        gt = run_answer("예산 5억 초과인 공고 목록 줘", table, identity=identity,
                        registry_scope=scope).selected_document_ids
        assert "RFP-000963" in ge and "RFP-000963" not in gt


# ===========================================================================
# ⑩⑪ 결과 전체 반환 / 마감 문서가 앞에 몰린 경우
# ===========================================================================

def _bulk_world(tmp_path, n=40, past_prefix=0):
    """n개 문서. 앞의 past_prefix개는 마감이 지난 문서."""
    specs = {}
    ident = []
    for i in range(n):
        doc_id = f"RFP-{900 + i:06d}"
        specs[doc_id] = {"예산": present("6억원"), "공고일": present("2024-01-01")}
        deadline = "2024-01-01 17:00:00" if i < past_prefix else "2025-01-01 17:00:00"
        ident.append((doc_id, f"기관{i:03d}", f"사업{i:03d}", deadline))
    table = write_table(tmp_path, specs)
    scope = write_registry(tmp_path, [
        {"document_id": d, "active": True, "retrieval_eligible": True}
        for d in sorted(specs)])
    identity = write_identity(tmp_path, ident)
    return table, scope, identity


class TestFullResultSet:
    @pytest.mark.parametrize("n", [30, 40, 60])
    def test_all_matching_documents_returned(self, tmp_path, n):
        table, scope, identity = _bulk_world(tmp_path, n=n)
        r = run_answer("예산 5억 이상인 공고 목록 줘", table,
                       identity=identity, registry_scope=scope)
        assert len(r.selected_document_ids) == n
        assert len(r.structured_answer) == n

    def test_deadline_filter_runs_after_full_condition_pass(self, tmp_path):
        """앞의 20개가 전부 마감 지난 문서여도 뒤의 유효 문서가 살아남아야 한다."""
        table, scope, identity = _bulk_world(tmp_path, n=40, past_prefix=25)
        r = run_answer("예산 5억 이상인 공고 목록 줘", table,
                       identity=identity, registry_scope=scope)
        assert len(r.selected_document_ids) == 15
        assert r.selected_document_ids[0] == "RFP-000925"
        assert "RFP-000900" not in r.selected_document_ids

    def test_count_matches_list_and_no_duplicates(self, tmp_path):
        table, scope, identity = _bulk_world(tmp_path, n=35)
        r = run_answer("예산 5억 이상인 공고 목록 줘", table,
                       identity=identity, registry_scope=scope)
        ids = r.selected_document_ids
        assert len(ids) == len(set(ids))
        assert f"{len(ids)}건" in r.text
        assert r.text.count("\n- RFP-") == len(ids)

    def test_structured_answer_and_selected_ids_agree(self, tmp_path):
        table, scope, identity = _bulk_world(tmp_path, n=33)
        r = run_answer("예산 5억 이상인 공고 목록 줘", table,
                       identity=identity, registry_scope=scope)
        assert r.structured_answer == r.selected_document_ids
        assert r.condition_result_doc_ids == r.selected_document_ids

    def test_display_limit_does_not_truncate_graded_list(self, tmp_path):
        table, scope, identity = _bulk_world(tmp_path, n=40)
        r = run_answer("예산 5억 이상인 공고 목록 줘", table, identity=identity,
                       registry_scope=scope, config=cfg(select_display_limit=5))
        assert len(r.selected_document_ids) == 40
        assert r.text.count("\n- RFP-") == 5
        assert "전체 40건" in r.text


# ===========================================================================
# ⑫ 등록부 범위 — 중복 제외·대표 유지·비활성 제외
# ===========================================================================

class TestRegistryScope:
    @pytest.fixture
    def world(self, tmp_path):
        specs = {d: {"예산": present("6억원")} for d in
                 ["RFP-000971", "RFP-000972", "RFP-000973", "RFP-000974"]}
        table = write_table(tmp_path, specs)
        scope = write_registry(tmp_path, [
            {"document_id": "RFP-000971", "active": True, "retrieval_eligible": True},
            {"document_id": "RFP-000972", "active": True, "retrieval_eligible": False,
             "duplicate_of_document_id": "RFP-000971",
             "relation_status": "duplicate_confirmed"},
            {"document_id": "RFP-000973", "active": False, "retrieval_eligible": True},
            {"document_id": "RFP-000974", "active": True, "retrieval_eligible": True},
        ])
        identity = write_identity(tmp_path, [
            (d, f"기관{d[-3:]}", f"사업{d[-3:]}", "2025-01-01 17:00:00")
            for d in sorted(specs)])
        return table, scope, identity

    def test_duplicate_copy_excluded_representative_kept(self, world):
        table, scope, identity = world
        r = run_answer("예산 5억 이상인 공고 목록 줘", table,
                       identity=identity, registry_scope=scope)
        assert "RFP-000972" not in r.selected_document_ids
        assert "RFP-000971" in r.selected_document_ids

    def test_inactive_document_excluded(self, world):
        table, scope, identity = world
        r = run_answer("예산 5억 이상인 공고 목록 줘", table,
                       identity=identity, registry_scope=scope)
        assert "RFP-000973" not in r.selected_document_ids
        assert set(r.selected_document_ids) == {"RFP-000971", "RFP-000974"}

    def test_no_hardcoded_document_ids_in_source(self):
        """제외 대상을 코드에 적어두지 않았는지 — 공식 중복 2건 ID로 확인."""
        import pathlib
        import table_query
        import document_registry
        import answer_pipeline
        for mod in (table_query, document_registry, answer_pipeline):
            text = pathlib.Path(mod.__file__).read_text(encoding="utf-8")
            for banned in ("RFP-000006", "RFP-000017", "RFP-000075", "RFP-000098"):
                assert banned not in text, f"{mod.__name__}에 {banned}가 하드코딩됨"

    def test_missing_registry_is_disclosed(self, world):
        table, _scope, identity = world
        r = run_answer("예산 5억 이상인 공고 목록 줘", table, identity=identity)
        assert "등록부" in r.text
        assert "RFP-000972" in r.selected_document_ids   # 범위 제한 없이 전부

    def test_scope_reported_in_condition_query(self, world):
        table, scope, identity = world
        r = run_answer("예산 5억 이상인 공고 목록 줘", table,
                       identity=identity, registry_scope=scope)
        diag = [c for c in r.condition_query
                if c.get("kind") == "selection_diagnostics"][0]
        assert diag["registry_scope"]["eligible_count"] == 2
        assert diag["registry_scope"]["excluded_count"] == 2
        assert diag["scanned_document_count"] == 2


# ===========================================================================
# ⑬ 마감일 정책
# ===========================================================================

class TestDeadlinePolicy:
    @pytest.fixture
    def world(self, tmp_path):
        specs = {d: {"예산": present("6억원")} for d in
                 ["RFP-000981", "RFP-000982", "RFP-000983", "RFP-000984"]}
        table = write_table(tmp_path, specs)
        scope = write_registry(tmp_path, [
            {"document_id": d, "active": True, "retrieval_eligible": True}
            for d in sorted(specs)])
        identity = write_identity(tmp_path, [
            ("RFP-000981", "기관1", "사업1", "2024-05-31 17:00:00"),   # 마감 지남
            ("RFP-000982", "기관2", "사업2", "2024-06-01"),            # 경계 = 기준일
            ("RFP-000983", "기관3", "사업3", "2024-12-31 17:00:00"),   # 마감 전
            ("RFP-000984", "기관4", "사업4", ""),                      # 미상
        ])
        return table, scope, identity

    def test_past_excluded_boundary_kept_future_kept(self, world):
        table, scope, identity = world
        r = run_answer("예산 5억 이상인 공고 목록 줘", table,
                       identity=identity, registry_scope=scope)
        assert "RFP-000981" not in r.selected_document_ids
        assert "RFP-000982" in r.selected_document_ids      # 경계는 포함(>=)
        assert "RFP-000983" in r.selected_document_ids

    def test_unknown_deadline_shown_not_dropped(self, world):
        table, scope, identity = world
        r = run_answer("예산 5억 이상인 공고 목록 줘", table,
                       identity=identity, registry_scope=scope)
        assert "RFP-000984" in r.selected_document_ids
        assert "마감일 미상" in r.text

    def test_filter_on_for_select_only(self, world):
        table, scope, identity = world
        off = cfg(deadline_filter_default={"select": False, "extract": False,
                                           "qa": False, "compare": False})
        r = run_answer("예산 5억 이상인 공고 목록 줘", table, identity=identity,
                       registry_scope=scope, config=off)
        assert "RFP-000981" in r.selected_document_ids

    def test_reference_datetime_is_external_not_now(self, world):
        table, scope, identity = world
        r = run_answer("예산 5억 이상인 공고 목록 줘", table,
                       identity=identity, registry_scope=scope)
        diag = [c for c in r.condition_query
                if c.get("kind") == "selection_diagnostics"][0]
        assert diag["reference_datetime"] == "2024-06-01"
        assert diag["deadline_filter_on"] is True


# ===========================================================================
# ⑭ 정상 0건 / 해석 실패 / 정보 부족
# ===========================================================================

def _diag(answer):
    """선별 진단 항목(condition_query 안) — 새 최상위 필드를 만들지 않는다."""
    for c in (answer.condition_query or []):
        if isinstance(c, dict) and c.get("kind") == "selection_diagnostics":
            return c
    return {}


def _first_sentence(answer) -> str:
    return (answer.text or "").split("\n")[0]


class TestZeroVsClarifyVsUndetermined:
    """확정 0건이 무엇을 뜻하는지 **첫 문장·abstained·집계**까지 검사한다.

    ⚠️ 2026-09-03 기대값 강화: 예전 test_undetermined_is_not_reported_as_zero 는
    본문 어딘가에 '판단 불가'라는 낱말이 있는지만 봤다. 그래서 첫 문장이
    "조건에 맞는 공고가 없습니다"라고 정반대로 단정해도 통과했다(실제로 SEL-011·020이
    그 상태로 통과했다). 이제 세 경우를 각각 다르게 검사한다."""

    # --- A. 확정 0 · 판단 불가 0 → 정상적인 0건 ---
    def test_A_true_zero_is_a_normal_answer(self, tmp_path):
        # 모든 문서의 예산을 읽을 수 있고, 그 값들이 조건에 **명시적으로** 못 미친다.
        # (읽을 수 없는 값이 하나라도 있으면 그건 A가 아니라 B다 — 아래 B 검사 참조)
        specs = {"RFP-000978": {"예산": present("1억원")},
                 "RFP-000979": {"예산": present("2억원")}}
        table = write_table(tmp_path, specs)
        scope = write_registry(tmp_path, [
            {"document_id": d, "active": True, "retrieval_eligible": True}
            for d in sorted(specs)])
        identity = write_identity(tmp_path, [
            (d, f"기관{d[-3:]}", f"사업{d[-3:]}", "2025-01-01 17:00:00")
            for d in sorted(specs)])
        r = run_answer("예산 5억 이상인 공고 목록 줘", table,
                       identity=identity, registry_scope=scope)
        assert r.selected_document_ids == []
        assert r.abstained is False
        assert _diag(r)["outcome_class"] == "empty"
        assert _diag(r)["undetermined_count"] == 0
        first = _first_sentence(r)
        assert "조건에 맞는 공고가 없습니다" in first
        assert "판단할 수 없" not in first

    # --- B. 확정 0 · 판단 불가 ≥1 → 없다고 단정하지 않고 보류 ---
    def test_B_undetermined_only_does_not_claim_none_exist(self, status_world):
        table, scope, identity = status_world
        r = run_answer("지역 제한이 아예 없는 공고들만 알려줘", table,
                       identity=identity, registry_scope=scope)
        assert r.selected_document_ids == []
        assert r.abstained is True
        d = _diag(r)
        assert d["outcome_class"] == "undetermined_only"
        assert d["undetermined_count"] >= 1
        first = _first_sentence(r)
        # 첫 문장이 "없다"고 단정하면 안 된다
        assert "조건 일치가 확인된 공고는 없으며" in first
        assert "정보 부족으로 판단할 수 없습니다" in first
        assert "검색 범위와 마감 정책에서 조건에 맞는 공고가 없습니다" not in first
        assert "판단 불가" in r.text

    def test_B_unreadable_values_are_undetermined_not_zero(self, status_world):
        """값이 있어도 숫자로 읽지 못하면 '조건 불일치'가 아니라 '판단 불가'다."""
        table, scope, identity = status_world
        r = run_answer("예산 5억 이상인 공고 목록 줘", table,
                       identity=identity, registry_scope=scope)
        assert r.selected_document_ids == []
        assert r.abstained is True
        assert _diag(r)["outcome_class"] == "undetermined_only"

    # --- C. 확정 ≥1 · 판단 불가 ≥1 → 확정분을 주되 전부라고 하지 않는다 ---
    def test_C_partial_returns_confirmed_and_flags_rest(self, tmp_path):
        specs = {
            "RFP-000971": {"지역제한": present("지역제한 없음(전국 대상)")},
            "RFP-000972": {},                                   # 미기재 → 판단 불가
            "RFP-000973": {},                                   # 미기재 → 판단 불가
        }
        table = write_table(tmp_path, specs)
        scope = write_registry(tmp_path, [
            {"document_id": d, "active": True, "retrieval_eligible": True}
            for d in sorted(specs)])
        identity = write_identity(tmp_path, [
            (d, f"기관{d[-3:]}", f"사업{d[-3:]}", "2025-01-01 17:00:00")
            for d in sorted(specs)])
        r = run_answer("지역 제한이 아예 없는 공고들만 알려줘", table,
                       identity=identity, registry_scope=scope)
        assert r.selected_document_ids == ["RFP-000971"]
        assert r.abstained is False
        d = _diag(r)
        assert d["outcome_class"] == "partial_undetermined"
        assert d["confirmed_count"] == 1 and d["undetermined_count"] == 2
        first = _first_sentence(r)
        assert "조건에 맞는 공고 1건" in first
        assert "전부라고 확정할 수는 없습니다" in first

    # --- 확정만 있는 경우는 '전부'라고 말해도 된다 ---
    def test_matched_only_makes_no_undetermined_claim(self, status_world):
        table, scope, identity = status_world
        r = run_answer("예산 정보가 안 나와 있는 공고들만 알려줘", table,
                       identity=identity, registry_scope=scope)
        assert r.selected_document_ids and r.abstained is False
        assert _diag(r)["outcome_class"] == "matched"
        assert "판단 불가" not in r.text

    # --- 상태 자체를 묻는 질문은 정상 선별이다(보류로 바꾸면 안 된다) ---
    def test_status_question_is_a_normal_selection_not_an_abstain(self, tmp_path):
        specs = {"RFP-000974": {}, "RFP-000975": {"지역제한": present("경기도 소재 업체")}}
        table = write_table(tmp_path, specs)
        scope = write_registry(tmp_path, [
            {"document_id": d, "active": True, "retrieval_eligible": True}
            for d in sorted(specs)])
        identity = write_identity(tmp_path, [
            (d, f"기관{d[-3:]}", f"사업{d[-3:]}", "2025-01-01 17:00:00")
            for d in sorted(specs)])
        r = run_answer("지역제한이 미기재된 공고 목록 줘", table,
                       identity=identity, registry_scope=scope)
        assert r.selected_document_ids == ["RFP-000974"]
        assert r.abstained is False
        assert _diag(r)["outcome_class"] == "matched"
        assert _diag(r)["undetermined_count"] == 0

    def test_unparsed_question_asks_back(self, status_world):
        table, scope, identity = status_world
        r = run_answer("괜찮은 공고들 좀 골라줘", table,
                       identity=identity, registry_scope=scope)
        assert r.abstained is True
        assert r.route == "애매_되묻기"
        assert _diag(r) == {}          # 조건 실행 자체를 하지 않았다

    def test_deadline_dropped_document_is_not_counted_as_undetermined(self, tmp_path):
        """마감으로 빠진 문서가 '판단 불가'로 둔갑하면 안 된다."""
        specs = {"RFP-000976": {"예산": present("6억원")},
                 "RFP-000977": {"예산": present("6억원")}}
        table = write_table(tmp_path, specs)
        scope = write_registry(tmp_path, [
            {"document_id": d, "active": True, "retrieval_eligible": True}
            for d in sorted(specs)])
        identity = write_identity(tmp_path, [
            ("RFP-000976", "기관976", "사업976", "2024-01-01 17:00:00"),   # 마감 지남
            ("RFP-000977", "기관977", "사업977", "2025-01-01 17:00:00"),
        ])
        r = run_answer("예산 5억 이상인 공고 목록 줘", table,
                       identity=identity, registry_scope=scope)
        assert r.selected_document_ids == ["RFP-000977"]
        d = _diag(r)
        assert d["outcome_class"] == "matched"
        assert d["undetermined_count"] == 0
        assert d["dropped_by_deadline_filter"] == 1


# ===========================================================================
# ⑮ 응답 계약·출처·재현성
# ===========================================================================

class TestResponseContract:
    REQUIRED = {"id", "answer", "structured_answer", "contexts", "retrieved",
                "citations", "selected_document_ids", "abstained", "route",
                "failure", "latency_ms", "cost_usd"}

    def test_response_envelope_unchanged(self, status_world):
        from answer_pipeline import answer_to_response
        table, scope, identity = status_world
        r = run_answer("예산이 공개된 공고 목록 줘", table,
                       identity=identity, registry_scope=scope)
        payload = answer_to_response("Q1", r)
        assert set(payload) == self.REQUIRED
        assert isinstance(payload["structured_answer"], list)
        assert isinstance(payload["selected_document_ids"], list)

    def test_citations_only_for_used_fields(self, status_world):
        table, scope, identity = status_world
        r = run_answer("예산이 공개된 공고 목록 줘", table,
                       identity=identity, registry_scope=scope)
        assert r.citations
        assert all(c["document"] in r.selected_document_ids for c in r.citations)

    def test_no_fabricated_citation_for_absent_field(self, status_world):
        """미기재 문서에는 존재하지 않는 문단을 만들어 인용하지 않는다."""
        table, scope, identity = status_world
        r = run_answer("예산 정보가 안 나와 있는 공고들만 알려줘", table,
                       identity=identity, registry_scope=scope)
        assert r.selected_document_ids == ["RFP-000902"]
        # [2026-09-04 §8] 미기재 근거는 Evidence 형으로 낸다 — 없는 문단을 지어내지
        # 않는다는 원래 뜻은 그대로 두고, "무엇을 근거로 미기재라고 했는지"까지 본다.
        assert len(r.citations) == 1
        cite = r.citations[0]
        assert cite["kind"] == "extraction_table"
        assert cite["document"] == "RFP-000902" and cite["field"] == "예산"
        assert cite["status"] == "field_absent"
        for forbidden in ("location", "line", "ref_no", "section", "block_index"):
            assert forbidden not in cite, forbidden

    def test_compound_condition_keeps_evidence_per_condition(self, tmp_path):
        specs = {"RFP-000991": {"예산": present("6억원", line=11),
                                "사업기간": present("12개월", line=22)}}
        table = write_table(tmp_path, specs)
        scope = write_registry(tmp_path, [
            {"document_id": "RFP-000991", "active": True, "retrieval_eligible": True}])
        # ⚠️ 기관명·사업명을 "기관"/"사업" 같은 일반 명사로 두면 질문 본문의 낱말과
        #    겹쳐서 문서 특정이 잘못 걸린다(테스트 자료 결함). 실제 이름처럼 바꾼다.
        identity = write_identity(tmp_path, [
            ("RFP-000991", "푸른바다연구원", "푸른바다연구원 해양관측 플랫폼 구축",
             "2025-01-01 17:00:00")])
        r = run_answer("예산 5억 이상이고 사업기간이 기재된 공고 목록 줘", table,
                       identity=identity, registry_scope=scope)
        fields = {c.get("field") for c in r.citations}
        assert {"예산", "사업기간"} <= fields

    def test_rerun_is_identical(self, tmp_path):
        from answer_pipeline import answer_to_response
        table, scope, identity = _bulk_world(tmp_path, n=30, past_prefix=10)
        outs = []
        for _ in range(3):
            r = run_answer("예산 5억 이상인 공고 목록 줘", table,
                           identity=identity, registry_scope=scope)
            payload = answer_to_response("Q", r)
            payload.pop("latency_ms", None)
            outs.append(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        assert len(set(outs)) == 1

    def test_route_records_actual_path(self, status_world):
        from answer_pipeline import ROUTE_SELECT, ROUTE_CLARIFY
        table, scope, identity = status_world
        assert run_answer("예산이 공개된 공고 목록 줘", table, identity=identity,
                          registry_scope=scope).route == ROUTE_SELECT
        assert run_answer("괜찮은 공고들 좀 골라줘", table, identity=identity,
                          registry_scope=scope).route == ROUTE_CLARIFY


# ===========================================================================
# ⑯ 선별 경로에서 API 호출 0회
# ===========================================================================

class TestNoApiCalls:
    QUESTIONS = [
        "예산 5억 이상인 공고 목록 줘",
        "예산 정보가 안 나와 있는 공고들만 알려줘",
        "지역 제한이 아예 없는 공고들만 알려줘",
        "컨소시엄이 필요한 공고들만 알려줘",
        "제출 방식은 별도 문서를 따로 봐야 하는 공고들 알려줘",
        "예산 5억 이상이고 발주기관이 서울인 공고 목록 줘",
        "괜찮은 공고들 좀 골라줘",
    ]

    @pytest.mark.parametrize("question", QUESTIONS)
    def test_selection_never_calls_api(self, question, status_world):
        """ForbiddenClient 는 호출되면 AssertionError를 던진다 — 통과 = 호출 0회."""
        table, scope, identity = status_world
        r = run_answer(question, table, identity=identity, registry_scope=scope)
        assert r.error_stage is None, r.error_detail


# ===========================================================================
# 라운드2 — 독립 검토에서 재현된 결함 4묶음
# ===========================================================================

class TestRound2ConditionsNotDroppedOrInverted:
    """결함 ① 질문 조건을 조용히 버리거나 반대로 해석하지 않는다.

    ⚠️ 문장을 예외 목록에 등록하는 방식이 아니라, 표현·조사·조건 순서를 바꿔도
       같은 결과가 나오는지를 함께 검사한다."""

    @pytest.fixture
    def world(self, tmp_path):
        specs = {
            # 예산은 조건을 만족하지만 지역은 서울이 아니다 — 조건을 버리면 이 문서가
            # 정답처럼 반환된다(실제 재현된 오답).
            "RFP-000801": {"예산": present("6억원"),
                           "지역제한": present("부산 소재 업체만 참여 가능")},
            "RFP-000802": {"예산": present("6억원"),
                           "지역제한": present("서울 소재 업체만 참여 가능")},
            "RFP-000803": {"예산": present("6천만원")},
            "RFP-000804": {"평가 배점": present("기술 90점 / 가격 10점"),
                           "예산": present("6억원")},
        }
        table = write_table(tmp_path, specs)
        scope = write_registry(tmp_path, [
            {"document_id": d, "active": True, "retrieval_eligible": True}
            for d in sorted(specs)])
        identity = write_identity(tmp_path, [
            (d, f"관측기관{d[-3:]}", f"관측사업{d[-3:]} 구축", "2025-01-01 17:00:00")
            for d in sorted(specs)])
        return table, scope, identity

    # --- 지역 조건을 배경 설명으로 버리지 않는다 ---
    @pytest.mark.parametrize("question", [
        "예산 5억 이상이고 서울 소재 업체만 참여 가능한 공고를 찾아줘.",
        "서울 소재 업체만 참여 가능하고 예산 5억 이상인 공고를 찾아줘.",
        "예산 5억 이상, 서울 소재 업체만 참여 가능한 공고 목록 줘.",
        "본점 소재지가 서울인 업체만 참여 가능한 공고 중 예산 5억 이상인 것 알려줘.",
    ])
    def test_region_condition_is_not_dropped(self, question, world):
        from table_query import parse_selection
        table, scope, identity = world
        p = parse_selection(question)
        assert p.fully_matched is False, question
        assert p.unresolved, question
        r = run_answer(question, table, identity=identity, registry_scope=scope)
        assert r.abstained is True
        assert r.route == "애매_되묻기"
        # 조건을 못 읽었으면 문서를 확정해 주면 안 된다
        assert r.selected_document_ids == []
        assert "RFP-000801" not in r.text.replace("RFP-000801", "", 1) or True

    def test_dropped_region_condition_never_returns_wrong_document(self, world):
        table, scope, identity = world
        r = run_answer("예산 5억 이상이고 서울 소재 업체만 참여 가능한 공고를 찾아줘.",
                       table, identity=identity, registry_scope=scope)
        assert "RFP-000801" not in r.selected_document_ids   # 부산 문서

    # --- 12필드 밖 수량 조건을 버리지 않는다 ---
    @pytest.mark.parametrize("question", [
        "평가 배점은 80점 이상이고 예산 5억 이상인 공고를 찾아줘.",
        "예산 5억 이상이면서 평가 배점이 80점 이상인 공고 목록 줘.",
        "예산 5억 이상, 평가 배점 80점 이상인 공고 알려줘.",
    ])
    def test_unsupported_numeric_condition_is_reported(self, question, world):
        from table_query import parse_selection
        table, scope, identity = world
        p = parse_selection(question)
        assert p.fully_matched is False, question
        r = run_answer(question, table, identity=identity, registry_scope=scope)
        assert r.abstained is True and r.selected_document_ids == []
        assert "인식하지 못한 조건" in r.text
        # '지원하지 않음'을 '조건에 맞는 공고 없음'으로 표현하지 않는다
        assert "조건에 맞는 공고가 없습니다" not in r.text

    def test_duration_number_still_not_read_as_budget(self, world):
        from table_query import parse_selection
        p = parse_selection("사업기간이 6개월 이상인 공고 목록 줘")
        assert p.fully_matched is False
        assert not any(c.field == "예산" for c in p.conditions)

    # --- 같은 필드 안의 OR ---
    @pytest.mark.parametrize("question", [
        "예산 5억 이상 또는 1억 이하인 공고를 찾아줘.",
        "예산이 5억 이상이거나 1억 이하인 공고 목록 줘.",
        "예산 5억 이상 혹은 1억 이하인 공고 알려줘.",
    ])
    def test_same_field_or_is_detected(self, question, world):
        from table_query import parse_selection
        table, scope, identity = world
        p = parse_selection(question)
        assert p.unsupported_or is True, question
        assert p.fully_matched is False
        r = run_answer(question, table, identity=identity, registry_scope=scope)
        assert r.abstained is True
        assert "OR" in r.text
        # AND로 바꿔 0건이라고 답하지 않는다
        assert "조건에 맞는 공고가 없습니다" not in r.text

    def test_cross_field_or_still_detected(self, world):
        from table_query import parse_selection
        p = parse_selection("예산이 기재됐거나 사업기간이 기재된 공고 목록 줘")
        assert p.unsupported_or is True

    def test_and_without_or_is_unaffected(self, world):
        """OR 감지가 정상 AND 질문을 건드리지 않는다."""
        from table_query import parse_selection
        table, scope, identity = world
        p = parse_selection("예산 5억 이상이고 사업기간이 기재된 공고 목록 줘")
        assert p.unsupported_or is False and p.fully_matched is True

    # --- 부정 범위 ---
    @pytest.mark.parametrize("question,operator", [
        ("공동수급이 허용되지 않는 공고를 찾아줘.", "consortium_forbidden"),
        ("공동수급을 허용하지 않는 공고 목록 줘.", "consortium_forbidden"),
        ("공동수급이 허용되는 공고를 찾아줘.", "consortium_allowed"),
        ("공동수급이 금지된 공고 목록 줘.", "consortium_forbidden"),
        ("컨소시엄이 필요하지 않은 공고 목록 줘.", "consortium_not_required"),
        ("컨소시엄이 필요한 공고 목록 줘.", "consortium_required"),
    ])
    def test_negation_binds_to_the_predicate(self, question, operator, world):
        from table_query import parse_selection
        p = parse_selection(question)
        ops = [c.operator for c in p.conditions]
        assert operator in ops, f"{question} -> {ops}"

    @pytest.mark.parametrize("question,operator", [
        ("지역 제한이 없는 공고 목록 줘", "no_restriction"),
        ("지역 제한이 있는 공고 목록 줘", "has_restriction"),
        ("예산이 기재된 공고 목록 줘", "status_value_present"),
        ("예산이 기재되지 않은 공고 목록 줘", "status_field_absent"),
    ])
    def test_positive_and_negative_pairs(self, question, operator, world):
        from table_query import parse_selection
        p = parse_selection(question)
        assert [c.operator for c in p.conditions] == [operator], question

    # --- 제외(빼기) 조건 ---
    @pytest.mark.parametrize("question", [
        "예산이 기재된 공고는 제외하고 찾아줘.",
        "예산이 기재된 공고는 빼고 알려줘.",
        "예산이 기재된 공고 말고 알려줘.",
    ])
    def test_exclusion_is_not_executed_backwards(self, question, world):
        table, scope, identity = world
        r = run_answer(question, table, identity=identity, registry_scope=scope)
        # 제외해야 할 문서(예산 기재)를 결과로 돌려주면 안 된다
        assert "RFP-000801" not in r.selected_document_ids
        assert "RFP-000802" not in r.selected_document_ids
        assert "RFP-000803" not in r.selected_document_ids

    def test_exclusion_returns_the_complement(self, tmp_path):
        specs = {"RFP-000805": {"예산": present("6억원")},
                 "RFP-000806": {},                       # 예산 미기재
                 "RFP-000807": {"예산": external()}}     # 외부 참조
        table = write_table(tmp_path, specs)
        scope = write_registry(tmp_path, [
            {"document_id": d, "active": True, "retrieval_eligible": True}
            for d in sorted(specs)])
        identity = write_identity(tmp_path, [
            (d, f"관측기관{d[-3:]}", f"관측사업{d[-3:]} 구축", "2025-01-01 17:00:00")
            for d in sorted(specs)])
        r = run_answer("예산이 기재된 공고는 제외하고 알려줘", table,
                       identity=identity, registry_scope=scope)
        # 상태는 문서마다 정확히 하나라서 여집합이 명확하다
        assert set(r.selected_document_ids) == {"RFP-000806", "RFP-000807"}
        assert r.abstained is False

    def test_exclusion_with_ambiguous_scope_asks_back(self, world):
        """조건이 둘 이상이면 '제외'가 어디에 걸리는지 확정할 수 없다 — 되묻는다."""
        table, scope, identity = world
        r = run_answer("예산 5억 이상이고 사업기간이 기재된 공고는 제외하고 알려줘",
                       table, identity=identity, registry_scope=scope)
        assert r.abstained is True
        assert "제외" in r.text

    def test_exclusion_on_semantic_condition_is_unsupported(self, world):
        """뜻 판정(지역제한·공동수급)의 여집합은 판단 불가와 섞여 정확하지 않다."""
        from table_query import parse_selection, validate_condition, InvalidQueryError
        p = parse_selection("지역 제한이 없는 공고는 제외하고 알려줘")
        for c in p.conditions:
            if c.negated and c.kind != "status":
                with pytest.raises(InvalidQueryError):
                    validate_condition(c)

    # --- 실제 배경 설명은 계속 배경으로 남는다(회귀 금지) ---
    @pytest.mark.parametrize("question,expected_ids", [
        ("입찰 준비를 하고 있는데, 예산 5억 이상인 공고 목록 줘.",
         {"RFP-000801", "RFP-000802", "RFP-000804"}),
        ("우리 회사는 소규모 업체인데, 예산 1억 이하인 공고들 알려줘.", {"RFP-000803"}),
        ("제안서 마감이 촉박해서 그런데, 예산 5억 이상인 공고 알려줘.",
         {"RFP-000801", "RFP-000802", "RFP-000804"}),
        ("내부 보고 자료를 만들어야 해서, 예산 5억 이상인 공고 전부 보여줘.",
         {"RFP-000801", "RFP-000802", "RFP-000804"}),
    ])
    def test_real_background_is_still_ignored(self, question, expected_ids, world):
        table, scope, identity = world
        r = run_answer(question, table, identity=identity, registry_scope=scope)
        assert r.abstained is False, question
        assert set(r.selected_document_ids) == expected_ids, question


class TestRound2ConsortiumClauseScope:
    """결함 ② 추출표의 공동수급 문장을 절 단위로 읽는다."""

    CASES = [
        ("공동수급은 허용하며 하도급은 불가합니다.", False, True),
        ("공동수급을 허용하며, 하도급 및 재하도급은 불허함", False, True),
        ("공동수급은 공동이행방식으로 허용하며 분담이행방식은 불허합니다.", False, True),
        ("공동수급은 필수가 아닙니다. 단독 참여가 가능합니다.", False, None),
        ("공동수급은 의무가 아니며 단독으로도 참여 가능합니다.", False, None),
        ("공동수급체 구성원은 반드시 관련 면허를 보유해야 합니다.", None, None),
        ("컨소시엄 구성원은 반드시 실적 요건을 충족하여야 한다.", None, None),
        ("(공동이행방식)은 허용하지 않음", None, None),
        ("본 입찰은 공동수급(분담이행방식)을 허용하지 않음", None, None),
        ("본 사업은 공동수급을 불허함", False, False),
        ("공동수급 및 하도급 불가", False, False),
        ("하도급은 불가함", None, None),
        ("단독 또는 공동수급체를 구성하여 입찰 참가할 수 있음", False, True),
        ("공동수급체를 구성하여야 하며 단독 입찰은 불가함", True, True),
        ("반드시 공동수급체를 구성하여 참여하여야 합니다.", True, True),
    ]

    @pytest.mark.parametrize("value,required,allowed", CASES)
    def test_clause_scoped_verdict(self, value, required, allowed):
        from table_query import classify_consortium
        v = classify_consortium({"status": "value_present", "answer_raw": value,
                                 "answer_normalized": value})
        assert (v.required, v.allowed) == (required, allowed), f"{value} -> {v.reason}"

    # --- 적대적 점검(라운드2 자체 검증)에서 실제 공식 행으로 재현된 오독 ---
    @pytest.mark.parametrize("value,required,allowed,note", [
        # RFP-000057 — 한 줄에 하도급 금지와 공동수급 허용이 같이 있다.
        # 양태마다 주체를 풀지 않으면 '불허'가 공동수급으로 옮겨붙는다.
        ("하도급/공동수급 여부 : 하도급 불허 / 공동수급 허용", False, True, "주체별 양태 해석"),
        # RFP-000027/000088 — '불가피한'의 '불가'를 금지로 읽으면 안 된다.
        ("공동수급체를 구성하지 못하는 불가피한 사정이 있는 경우 그 사유를 제시하여야 함",
         None, None, "불가피한 ≠ 불가"),
        # RFP-000030 — '구성하여 입찰'은 허용 표식이 아니고 '참여를 제한'은 금지다.
        ("복수의 업체로 컨소시엄을 구성하여 입찰하는 경우 참여를 제한함",
         False, False, "참여 제한은 금지"),
        # RFP-000015 — 구성원·지분율이 같이 적혀 있어도 허용 문구는 살아야 한다.
        ("단독 도는 공동수급(주계약자관리방식만 허용, 2개사 이내, 최소 지분율 25%이상)",
         False, True, "구성 조건이 허용을 삼키지 않음"),
        # RFP-000049 — 괄호 설명이 끼어도 나란한 주체는 함께 묶인다.
        ("공동계약(공동 및 분담 이행방식) 및 하도급 불가", False, False, "괄호 넘어 접속"),
        # 부정 어미 — '아닌', '없습니다'
        ("공동수급은 의무가 아닌 선택 사항입니다.", False, None, "의무가 아닌"),
        ("공동수급 구성은 의무 사항이 없습니다.", False, None, "의무 사항이 없음"),
        # 같은 절에 단독 문구와 공동수급 허용이 함께 있으면 허용은 살아야 한다
        ("단독 참여 가능하며 공동수급도 허용함", False, True, "단독 문구가 허용을 삼키지 않음"),
        # 구성 자체가 무조건 의무인 경우는 필수다(구성 '조건'과 구분)
        ("본 사업은 공동수급체를 구성하여 입찰에 참가하여야 한다.", True, True, "무조건 구성 의무"),
        ("공동수급체는 5개 이하로 구성하여야 하며, 구성원별 최소지분율은 10% 이상",
         None, None, "구성 조건은 의무가 아님"),
    ])
    def test_adversarial_findings(self, value, required, allowed, note):
        from table_query import classify_consortium
        v = classify_consortium({"status": "value_present", "answer_raw": value,
                                 "answer_normalized": value})
        assert (v.required, v.allowed) == (required, allowed), f"{note}: {v.reason}"

    def test_official_distribution_is_stable(self):
        """공식 추출표 100행 전수 분류 — 분포가 조용히 바뀌면 실패한다."""
        import collections
        import json
        from pathlib import Path
        from conftest import resolve_official_extraction_dir
        from table_query import classify_consortium
        # 저장소 data/preprocessed 우선 — 서버 경로만 보면 자료가 있는데도 skip 된다.
        official_dir = resolve_official_extraction_dir("v4")
        if official_dir is None:
            pytest.skip("공식 추출표 v4 를 저장소·서버 어디에서도 찾지 못함")
        official = official_dir / "extraction_table_v4.json"
        rows = json.loads(official.read_text(encoding="utf-8"))["rows"]
        dist = collections.Counter()
        for r in rows:
            if r["field_name"] != "컨소시엄 요건":
                continue
            if str(r.get("active", "true")).lower() == "false":
                continue
            v = classify_consortium(r)
            dist[(r["status"], v.required, v.allowed)] += 1
        assert dist[("value_present", False, True)] == 45      # 허용
        assert dist[("value_present", False, False)] == 28     # 금지
        assert dist[("value_present", None, None)] == 17       # 판단 불가
        assert dist[("field_absent", None, None)] == 9
        assert dist[("conflict", None, None)] == 1
        # 공식표에 공동수급을 무조건 필수로 요구하는 문서는 없다
        assert not any(k[1] is True for k in dist)

    def test_subcontract_ban_never_migrates_to_consortium(self):
        from table_query import classify_consortium
        v = classify_consortium({"status": "value_present",
                                 "answer_raw": "공동수급은 허용하며 하도급은 불가합니다.",
                                 "answer_normalized": "공동수급은 허용하며 하도급은 불가합니다."})
        assert v.allowed is True

    @pytest.mark.parametrize("value,required,allowed", [
        # '제외'의 목적어가 공동수급일 때만 금지다 (적대적 점검에서 잡힌 오탐)
        ("제안서에서 일부 항목을 제외한 나머지 서류는 공동수급 허용", False, True),
        ("본 사업은 공동수급을 제외한 단독입찰로 진행함", False, False),
        # 쉼표로 이어진 목적어 나열이 끊기면 안 된다
        ("공동수급, 하도급 불허", False, False),
        ("나. 공동수급, 하도급 불허", False, False),
        # 법령 인용문 안의 '공동계약'에 속으면 안 된다
        ("공동수급체는 5개 이하로 구성하여야 하며, 구성원별 계약참여 최소지분율은 10% "
         "이상으로 하여야 함「공동계약운용요령(기획재정부, 계약예규)」 제9조 제5항 참조",
         None, None),
    ])
    def test_modality_object_is_checked(self, value, required, allowed):
        from table_query import classify_consortium
        v = classify_consortium({"status": "value_present", "answer_raw": value,
                                 "answer_normalized": value})
        assert (v.required, v.allowed) == (required, allowed), f"{value} -> {v.reason}"

    def test_non_value_statuses_unchanged(self):
        from table_query import classify_consortium
        for status in ("field_absent", "external_reference", "not_disclosed",
                       "conflict", "extraction_failed"):
            v = classify_consortium({"status": status, "answer_raw": "공동수급 허용",
                                     "answer_normalized": "공동수급 허용"})
            assert v.required is None and v.allowed is None, status

    @pytest.fixture
    def world(self, tmp_path):
        specs = {f"RFP-0008{40 + i}": {"컨소시엄 요건": present(v)}
                 for i, (v, _r, _a) in enumerate(self.CASES)}
        table = write_table(tmp_path, specs)
        scope = write_registry(tmp_path, [
            {"document_id": d, "active": True, "retrieval_eligible": True}
            for d in sorted(specs)])
        identity = write_identity(tmp_path, [
            (d, f"수급기관{d[-3:]}", f"수급사업{d[-3:]} 구축", "2025-01-01 17:00:00")
            for d in sorted(specs)])
        ids = {v: f"RFP-0008{40 + i}" for i, (v, _r, _a) in enumerate(self.CASES)}
        return table, scope, identity, ids

    def test_selection_returns_allowed_documents(self, world):
        table, scope, identity, ids = world
        r = run_answer("공동수급이 허용되는 공고 목록 줘", table,
                       identity=identity, registry_scope=scope)
        expected = {ids[v] for v, _rq, al in self.CASES if al is True}
        assert set(r.selected_document_ids) == expected

    def test_selection_excludes_method_only_prohibition_from_forbidden(self, world):
        table, scope, identity, ids = world
        r = run_answer("공동수급이 금지된 공고 목록 줘", table,
                       identity=identity, registry_scope=scope)
        assert ids["(공동이행방식)은 허용하지 않음"] not in r.selected_document_ids
        assert ids["하도급은 불가함"] not in r.selected_document_ids
        assert ids["본 사업은 공동수급을 불허함"] in r.selected_document_ids

    def test_selection_required_is_only_explicit(self, world):
        table, scope, identity, ids = world
        r = run_answer("공동수급이 필수인 공고 목록 줘", table,
                       identity=identity, registry_scope=scope)
        assert set(r.selected_document_ids) == {
            ids["공동수급체를 구성하여야 하며 단독 입찰은 불가함"],
            ids["반드시 공동수급체를 구성하여 참여하여야 합니다."]}
        assert ids["공동수급체 구성원은 반드시 관련 면허를 보유해야 합니다."] \
            not in r.selected_document_ids


class TestRound2DocumentScopeVsSelection:
    """결함 ③ 한 사업의 서류 목록을 전체 선별로 오분류하지 않는다."""

    @pytest.fixture
    def world(self, tmp_path):
        specs = {
            "RFP-000861": {"필수 제출 서류": present("가. 제안서\n나. 사업자등록증"),
                           "예산": present("3억원")},
            "RFP-000862": {"필수 제출 서류": present("제안서 1부"), "예산": present("7억원")},
            "RFP-000863": {"필수 제출 서류": present("제안서, 산출내역서"),
                           "예산": present("8억원")},
        }
        table = write_table(tmp_path, specs)
        scope = write_registry(tmp_path, [
            {"document_id": d, "active": True, "retrieval_eligible": True}
            for d in sorted(specs)])
        identity = write_identity(tmp_path, [
            ("RFP-000861", "한빛문화재단", "한빛문화재단 디지털 아카이브 사업",
             "2025-01-01 17:00:00"),
            ("RFP-000862", "새벽교통공사", "새벽교통공사 요금정산 고도화",
             "2025-01-01 17:00:00"),
            ("RFP-000863", "쌍둥이연구원", "쌍둥이연구원 1차 실증", "2025-01-01 17:00:00"),
        ])
        return table, scope, identity

    @pytest.mark.parametrize("question", [
        "한빛문화재단 디지털 아카이브 사업의 필수 제출 서류를 알려줘",
        "한빛문화재단 디지털 아카이브 사업의 필수 제출 서류 목록을 알려줘",
        "한빛문화재단 디지털 아카이브 사업 제출서류 목록 전체를 알려줘",
        "한빛문화재단 디지털 아카이브 사업, 제출 서류 리스트 좀 줘",
    ])
    def test_named_document_document_wins_over_plural_word(self, question, world):
        table, scope, identity = world
        r = run_answer(question, table, identity=identity, registry_scope=scope)
        assert r.task_type == "extract", question
        assert r.selected_document_ids == ["RFP-000861"], question
        assert r.abstained is False

    def test_document_id_with_plural_word(self, world):
        table, scope, identity = world
        r = run_answer("RFP-000861의 제출서류 목록 알려줘", table,
                       identity=identity, registry_scope=scope)
        assert r.task_type == "extract" and r.selected_document_ids == ["RFP-000861"]

    @pytest.mark.parametrize("question", [
        "이 사업과 별개로 전체 공고에서 예산 5억 이상인 사업 골라줘.",
        "그 사업 말고 전체 공고 중에서 예산 5억 이상인 공고 목록 줘.",
        "이 공고 말고 모든 공고에서 예산 5억 이상인 것 알려줘.",
    ])
    def test_explicit_global_scope_beats_anaphora(self, question, world):
        from answer_pipeline import SessionState
        table, scope, identity = world
        r = run_answer(question, table, identity=identity, registry_scope=scope,
                       session=SessionState(active_document_id="RFP-000861"))
        assert r.task_type == "select", question
        assert set(r.selected_document_ids) == {"RFP-000862", "RFP-000863"}, question

    def test_anaphora_without_global_scope_still_uses_context(self, world):
        from answer_pipeline import SessionState
        table, scope, identity = world
        r = run_answer("이 사업의 제출 서류 목록을 알려줘", table, identity=identity,
                       registry_scope=scope,
                       session=SessionState(active_document_id="RFP-000861"))
        assert r.task_type == "extract"
        assert r.selected_document_ids == ["RFP-000861"]

    def test_ambiguous_org_uses_candidate_clarify_not_arbitrary_pick(self, tmp_path):
        specs = {"RFP-000864": {"필수 제출 서류": present("제안서")},
                 "RFP-000865": {"필수 제출 서류": present("제안서")}}
        table = write_table(tmp_path, specs)
        scope = write_registry(tmp_path, [
            {"document_id": d, "active": True, "retrieval_eligible": True}
            for d in sorted(specs)])
        identity = write_identity(tmp_path, [
            ("RFP-000864", "쌍둥이재단", "쌍둥이재단 1차 사업", "2025-01-01 17:00:00"),
            ("RFP-000865", "쌍둥이재단", "쌍둥이재단 2차 사업", "2025-01-01 17:00:00"),
        ])
        r = run_answer("쌍둥이재단 사업의 제출 서류 목록 알려줘", table,
                       identity=identity, registry_scope=scope)
        assert r.abstained is True
        assert set(r.selected_document_ids) == {"RFP-000864", "RFP-000865"}

    def test_true_multi_document_selection_unaffected(self, world):
        table, scope, identity = world
        r = run_answer("제출서류 목록이 기재된 사업들을 찾아줘", table,
                       identity=identity, registry_scope=scope)
        assert r.task_type == "select"
        assert set(r.selected_document_ids) == {
            "RFP-000861", "RFP-000862", "RFP-000863"}

    def test_router_and_executor_agree(self, world):
        """라우터(잠정)와 실행부(identity 포함)가 **같은 함수**를 쓴다.

        identity 를 넣으면 기관명·사업명 특정까지 반영되므로 판단이 더 정확해질 수는
        있어도, 두 판단이 서로 모순된 결과를 내면 안 된다."""
        from table_query import is_selection_question, parse_selection
        table, scope, identity = world
        # 이름으로 한 문서를 부르는 질문 — identity 유무와 무관하게 선별형이 아니다
        q = "한빛문화재단 디지털 아카이브 사업의 필수 제출 서류 목록을 알려줘"
        p = parse_selection(q)
        assert is_selection_question(q, p, identity=identity) is False
        r = run_answer(q, table, identity=identity, registry_scope=scope)
        assert r.task_type == "extract"
        # 조건이 읽히고 여러 문서를 원하는 질문 — 둘 다 선별형
        q2 = "제출서류 목록이 기재된 사업들을 찾아줘"
        p2 = parse_selection(q2)
        assert is_selection_question(q2, p2, identity=identity) is True
        assert is_selection_question(q2, p2, identity=None) is True
        assert run_answer(q2, table, identity=identity,
                          registry_scope=scope).task_type == "select"


# ===========================================================================
# 라운드2 자체 적대적 점검에서 잡힌 회귀 — 수정본이 만든 오탐/오독
# ===========================================================================

class TestRound2AdversarialRegressions:
    """수정 자체가 만들어낸 결함들. 전부 실제 입력으로 재현된 뒤 고쳤다.

    조용한 오답(정반대 결과)과 시끄러운 오탐(정상 질문이 되묻기로 바뀜)을 함께 막는다."""

    @pytest.fixture
    def world(self, tmp_path):
        specs = {
            "RFP-000B01": {"예산": present("6억원"), "필수 제출 서류": present("제안서"),
                           "지역제한": present("지역제한 없음")},
            "RFP-000B02": {"예산": present("1억원"), "필수 제출 서류": present("제안서")},
        }
        table = write_table(tmp_path, specs)
        scope = write_registry(tmp_path, [
            {"document_id": d, "active": True, "retrieval_eligible": True}
            for d in sorted(specs)])
        identity = write_identity(tmp_path, [
            ("RFP-000B01", "푸른숲연구원", "푸른숲연구원 관측망 구축", "2025-01-01 17:00:00"),
            ("RFP-000B02", "붉은노을공단", "붉은노을공단 요금정산 개선", "2025-01-01 17:00:00"),
        ])
        return table, scope, identity

    # --- ① '전체 사업비'의 '전체 사업'을 전체 범위로 읽으면 안 된다 ---
    @pytest.mark.parametrize("question,is_global", [
        ("이 사업의 전체 사업비가 기재되어 있어?", False),
        ("이 사업의 전체 사업기간이 어떻게 돼?", False),
        ("이 사업의 전체 사업금액 알려줘", False),
        ("이 사업과 별개로 전체 공고에서 예산 5억 이상인 사업 골라줘.", True),
        ("전체 공고 중에서 예산 5억 이상인 공고 목록 줘", True),
        ("모든 공고에서 예산 5억 이상인 것 알려줘", True),
    ])
    def test_global_scope_needs_a_scope_particle(self, question, is_global):
        from table_query import parse_selection
        assert parse_selection(question).has_global_scope is is_global, question

    def test_anaphoric_total_budget_question_stays_document_scoped(self, world):
        from answer_pipeline import SessionState
        table, scope, identity = world
        r = run_answer("이 사업의 전체 사업비가 기재되어 있어?", table, identity=identity,
                       registry_scope=scope,
                       session=SessionState(active_document_id="RFP-000B01"))
        assert r.task_type == "extract"
        assert r.selected_document_ids == ["RFP-000B01"]

    # --- ② 제외 표식이 다른 것에 붙으면 조건을 뒤집지 않는다 ---
    @pytest.mark.parametrize("question", [
        "평가 배점이 기재된 공고 중 마감 지난 건 빼고 알려줘",
        "예산이 기재된 공고 중 참여 불가한 곳 빼고 알려줘",
    ])
    def test_exclusion_attached_elsewhere_does_not_invert(self, question, world):
        from table_query import parse_selection
        table, scope, identity = world
        p = parse_selection(question)
        assert not any(c.negated for c in p.conditions), question
        assert p.unresolved, question
        r = run_answer(question, table, identity=identity, registry_scope=scope)
        assert r.abstained is True and r.selected_document_ids == []

    def test_malgoneun_is_not_an_exclusion(self):
        """"A 말고는 필요 없어" 는 "A만"이라는 뜻이라 뒤집으면 안 된다."""
        from table_query import parse_selection
        p = parse_selection("예산이 기재된 공고 말고는 필요 없어")
        assert not any(c.negated for c in p.conditions)

    def test_real_exclusion_still_works(self, world):
        from table_query import parse_selection
        p = parse_selection("예산이 기재된 공고는 제외하고 찾아줘.")
        assert [c.negated for c in p.conditions] == [True]

    # --- ③ 부정 꼬리가 대상 명사를 넘어가면 안 된다 ---
    @pytest.mark.parametrize("question,operator", [
        ("공동수급이 허용되는 공고인지 아닌지 알려줘", "consortium_allowed"),
        ("예산이 기재된 공고인지 아닌지 알려줘", "status_value_present"),
        ("공동수급이 허용되지 않는 공고를 찾아줘.", "consortium_forbidden"),
        ("예산이 기재되지 않은 공고 목록 줘", "status_field_absent"),
    ])
    def test_negation_tail_stops_at_target_noun(self, question, operator):
        from table_query import parse_selection
        ops = [c.operator for c in parse_selection(question).conditions]
        assert operator in ops, f"{question} -> {ops}"

    # --- ④ 접속이 아닌 '거나'에 OR이 걸리면 안 된다 ---
    @pytest.mark.parametrize("question,is_or", [
        ("서류를 빠뜨리거나 하면 안 되니까 필수 제출 서류가 기재된 공고 알려줘", False),
        ("어떻게 하거나 상관없어, 예산이 기재된 공고 알려줘", False),
        ("예산 5억 이상 또는 1억 이하인 공고를 찾아줘.", True),
        ("예산이 5억 이상이거나 1억 이하인 공고 목록 줘.", True),
        ("예산이 기재됐거나 사업기간이 기재된 공고 목록 줘", True),
    ])
    def test_or_marker_needs_a_condition_before_it(self, question, is_or):
        from table_query import parse_selection
        assert parse_selection(question).unsupported_or is is_or, question

    # --- ⑤ 말하는 사람 사정은 조건 신호가 아니다 ---
    @pytest.mark.parametrize("question", [
        "3개월 이내에 끝내야 해서, 예산이 기재된 공고 알려줘",
        "본사 소재지가 부산이라, 지역제한이 없는 공고 알려줘",
        "직원이 100명 이상이라, 예산이 기재된 공고 알려줘",
        "제출 서류 준비에 시간이 꽤 걸려서 지역제한이 없는 공고 알려줘",
        "우리 회사는 매출 100억 이상이라, 예산이 기재된 공고 알려줘",
    ])
    def test_speaker_context_is_background_not_a_condition(self, question):
        from table_query import parse_selection
        p = parse_selection(question)
        assert p.fully_matched is True, f"{question} -> {p.unresolved}"
        assert not p.unresolved, question

    @pytest.mark.parametrize("question", [
        "예산 5억 이상이고 서울 소재 업체만 참여 가능한 공고를 찾아줘.",
        "평가 배점은 80점 이상이고 예산 5억 이상인 공고를 찾아줘.",
    ])
    def test_real_document_conditions_are_still_reported(self, question):
        from table_query import parse_selection
        p = parse_selection(question)
        assert p.fully_matched is False and p.unresolved, question

    # --- ⑦ 다른 지표의 금액을 예산으로 읽지 않는다 ---
    @pytest.mark.parametrize("question,has_budget_numeric", [
        ("우리 회사는 매출 100억 이상이라, 예산이 기재된 공고 알려줘", False),
        ("자본금 50억 이상인 회사인데 예산이 기재된 공고 알려줘", False),
        ("예산 5억 이상인 공고 목록 줘", True),
        ("5억 이상인 사업 알려줘", True),
        ("사업비 3억 이상인 공고 알려줘", True),
    ])
    def test_amount_binds_to_its_own_noun(self, question, has_budget_numeric):
        from table_query import parse_selection
        p = parse_selection(question)
        got = any(c.field == "예산" and c.operator in (">=", "<=", ">", "<")
                  for c in p.conditions)
        assert got is has_budget_numeric, f"{question} -> {[(c.field,c.operator) for c in p.conditions]}"

    # --- ⑧ 조건 값에 들어간 기관명이 선별을 가로채면 안 된다 ---
    def test_org_name_as_condition_value_keeps_selection(self):
        from table_query import parse_selection, is_selection_question
        from identity_metadata import load_identity
        from pathlib import Path
        official = Path("/srv/rfp/shared_data/processed/document_registry_v2/"
                        "document_identity_v2.csv")
        if not official.exists():
            pytest.skip("공식 identity 없음")
        idx = load_identity(official)
        q = "지역제한이 서울특별시로 걸린 공고들 알려줘"
        p = parse_selection(q)
        assert p.conditions and is_selection_question(q, p, identity=idx) is True

    def test_named_document_without_multi_target_still_redirects(self, world):
        table, scope, identity = world
        r = run_answer("푸른숲연구원 관측망 구축 사업의 필수 제출 서류 목록을 알려줘",
                       table, identity=identity, registry_scope=scope)
        assert r.task_type == "extract"
        assert r.selected_document_ids == ["RFP-000B01"]


class TestRound2AdversarialRound2:
    """두 번째 적대적 점검에서 잡힌 것들 — 조용한 오답과 시끄러운 오탐 양쪽."""

    # --- OR: 조사 '-나', '중 하나' 도 잡는다 (조용히 AND로 실행하면 안 됨) ---
    @pytest.mark.parametrize("question,is_or", [
        ("예산이 기재된 공고나 사업기간이 기재된 공고 알려줘", True),
        ("예산이 기재됐거나 사업기간이 기재된 공고 목록 줘", True),
        ("예산 5억 이상 또는 1억 이하인 공고를 찾아줘.", True),
        ("서류를 빠뜨리거나 하면 안 되니까 필수 제출 서류가 기재된 공고 알려줘", False),
        ("예산이 기재된 공고 다 보여줘", False),
    ])
    def test_or_particle_forms(self, question, is_or):
        from table_query import parse_selection
        assert parse_selection(question).unsupported_or is is_or, question

    def test_listed_fields_are_not_silently_dropped(self):
        """"예산, 사업기간 중 하나가 기재된 공고" 에서 예산이 사라지면 안 된다."""
        from table_query import parse_selection
        p = parse_selection("예산, 사업기간 중 하나가 기재된 공고 알려줘")
        assert p.fully_matched is False
        assert any("예산" in u for u in p.unresolved)

    # --- 상태의 여집합은 하나로 정해지지 않는다 ---
    def test_negated_status_is_not_flipped_to_a_single_status(self):
        from table_query import parse_selection
        p = parse_selection("예산이 미기재가 아닌 공고들 알려줘")
        assert not any(c.operator == "status_value_present" for c in p.conditions)
        assert p.unresolved and "부정 표현" in p.unresolved[0]

    def test_directly_negated_status_word_still_works(self):
        from table_query import parse_selection
        p = parse_selection("예산이 기재되지 않은 공고 목록 줘")
        assert [c.operator for c in p.conditions] == ["status_field_absent"]

    # --- '빼지 말고' 는 빼라는 뜻이 아니다 ---
    @pytest.mark.parametrize("question", [
        "예산이 기재된 공고 알려줘, 마감 지난 것도 빼지 말고",
        "예산이 기재된 공고들 보여줘. 하나도 빼지 말고 전부.",
        "예산이 기재된 공고 다 보여줘, 빼먹지 말고",
    ])
    def test_anti_exclusion_is_not_an_exclusion(self, question):
        from table_query import parse_selection
        p = parse_selection(question)
        assert p.fully_matched is True, f"{question} -> {p.unresolved}"
        assert not any(c.negated for c in p.conditions)

    def test_exclusion_noun_phrase_is_not_an_operator(self):
        from table_query import parse_selection
        p = parse_selection("제외 사유가 기재된 공고들 알려줘")
        assert not any("제외" in u for u in p.unresolved)

    # --- 기관명이 선별을 가로채면 안 되는 경우들 ---
    @pytest.mark.parametrize("question,is_selection", [
        ("울산광역시 지역제한이 있는 공고 전부 알려줘", True),
        ("대검찰청 말고 예산 5억 이상인 것만 추려줘", True),
        ("지역제한이 서울특별시로 걸린 공고들 알려줘", True),
        ("기초과학연구원 중이온가속기용 극저온시스템 운전 용역, "
         "입찰할 때 꼭 내야 하는 서류 목록이 따로 정리돼 있어?", False),
        ("벤처기업협회의 2024년 벤처확인종합관리시스템 기능 고도화 용역사업에 "
         "입찰 참가 지역 제한이 있어?", False),
    ])
    def test_org_name_does_not_hijack_selection(self, question, is_selection):
        from pathlib import Path
        from identity_metadata import load_identity
        from table_query import parse_selection, is_selection_question
        official = Path("/srv/rfp/shared_data/processed/document_registry_v2/"
                        "document_identity_v2.csv")
        if not official.exists():
            pytest.skip("공식 identity 없음")
        idx = load_identity(official)
        p = parse_selection(question)
        assert is_selection_question(question, p, identity=idx) is is_selection, question

    # --- 0건의 원인이 마감이면 그렇게 말한다 ---
    def test_zero_caused_by_deadline_says_so(self, tmp_path):
        specs = {"RFP-000C01": {"예산": present("6억원")}}
        table = write_table(tmp_path, specs)
        scope = write_registry(tmp_path, [
            {"document_id": "RFP-000C01", "active": True, "retrieval_eligible": True}])
        identity = write_identity(tmp_path, [
            ("RFP-000C01", "저녁놀재단", "저녁놀재단 관측 사업", "2024-01-01 17:00:00")])
        r = run_answer("예산 5억 이상인 공고 목록 줘", table,
                       identity=identity, registry_scope=scope)
        first = (r.text or "").split("\n")[0]
        assert "마감이 지난 공고가 1건" in first
        assert _diag(r)["dropped_by_deadline_filter"] == 1


class TestRound2AdversarialVerified:
    """독립 재검증에서 **확정된** 잔여 결함 2건 — 둘 다 조용한 오답이었다."""

    # --- "X 말고 다른 건 필요 없어" 는 X만 달라는 뜻이다(제외가 아니다) ---
    @pytest.mark.parametrize("question,negated", [
        ("지역제한이 기재된 공고 말고 다른 건 필요 없어", False),
        ("예산이 기재된 공고 말고 다른 건 필요 없어", False),
        ("예산이 기재된 공고 빼고 다른 건 필요 없어", False),
        ("예산이 기재된 공고 말고 다른 공고는 필요 없어", False),
        # 진짜 제외는 그대로 동작해야 한다
        ("예산이 기재된 공고는 제외하고 찾아줘.", True),
        ("예산이 기재된 공고 말고 알려줘", True),
        ("예산이 기재된 공고는 빼고 알려줘", True),
    ])
    def test_only_request_is_not_an_exclusion(self, question, negated):
        from table_query import parse_selection
        p = parse_selection(question)
        assert any(c.negated for c in p.conditions) is negated, question

    def test_only_request_returns_the_positive_set(self, tmp_path):
        specs = {"RFP-000D01": {"예산": present("6억원")},
                 "RFP-000D02": {},                       # 예산 미기재
                 "RFP-000D03": {}}
        table = write_table(tmp_path, specs)
        scope = write_registry(tmp_path, [
            {"document_id": d, "active": True, "retrieval_eligible": True}
            for d in sorted(specs)])
        identity = write_identity(tmp_path, [
            (d, f"바람결재단{d[-2:]}", f"바람결재단{d[-2:]} 정보화 사업", "2025-01-01 17:00:00")
            for d in sorted(specs)])
        r = run_answer("예산이 기재된 공고 말고 다른 건 필요 없어", table,
                       identity=identity, registry_scope=scope)
        assert r.selected_document_ids == ["RFP-000D01"]     # 여집합(2건)이 아니다

    # --- 기관 범위 안 선별: 읽은 조건을 버리고 "어느 문서냐"고만 묻지 않는다 ---
    def test_org_scoped_selection_keeps_the_recognized_condition(self, tmp_path):
        specs = {"RFP-000D11": {"예산": present("6억원")},
                 "RFP-000D12": {"예산": present("7억원")},
                 "RFP-000D13": {"예산": present("1억원")}}
        table = write_table(tmp_path, specs)
        scope = write_registry(tmp_path, [
            {"document_id": d, "active": True, "retrieval_eligible": True}
            for d in sorted(specs)])
        # 같은 기관에 문서 3건 → 이름만으로는 한 건으로 좁혀지지 않는다
        identity = write_identity(tmp_path, [
            ("RFP-000D11", "너울정보원", "너울정보원 1차 고도화", "2025-01-01 17:00:00"),
            ("RFP-000D12", "너울정보원", "너울정보원 2차 고도화", "2025-01-01 17:00:00"),
            ("RFP-000D13", "너울정보원", "너울정보원 3차 고도화", "2025-01-01 17:00:00"),
        ])
        r = run_answer("너울정보원 예산 5억 이상인 것만 골라줘", table,
                       identity=identity, registry_scope=scope)
        assert r.abstained is True
        # 읽은 조건이 답변에 살아 있어야 한다
        assert "예산" in r.text and "이상" in r.text
        # 미지원이라는 사실도 밝혀야 한다
        assert "지원하지 않습니다" in r.text
        # 조건을 무시하고 전체 문서를 고르지 않는다
        assert r.selected_document_ids == []

    def test_single_named_document_still_redirects(self, tmp_path):
        specs = {"RFP-000D21": {"지역제한": present("서울시 소재 업체")},
                 "RFP-000D22": {"예산": present("2억원")}}
        table = write_table(tmp_path, specs)
        scope = write_registry(tmp_path, [
            {"document_id": d, "active": True, "retrieval_eligible": True}
            for d in sorted(specs)])
        identity = write_identity(tmp_path, [
            ("RFP-000D21", "하늘빛재단", "하늘빛재단 데이터 허브 구축", "2025-01-01 17:00:00"),
            ("RFP-000D22", "구름산공사", "구름산공사 관제 고도화", "2025-01-01 17:00:00"),
        ])
        r = run_answer("하늘빛재단 데이터 허브 구축 사업에 지역 제한이 있어?", table,
                       identity=identity, registry_scope=scope)
        assert r.task_type == "extract"
        assert r.selected_document_ids == ["RFP-000D21"]


# ===========================================================================
# 라운드3 — 남은 코드 결함 3묶음
# ===========================================================================

class TestRound3ConditionsNotDropped:
    """결함 ① 조건 순서가 달라져도 버리지 않고, 숫자 비교의 부정을 반대로 읽지 않는다."""

    @pytest.fixture
    def world(self, tmp_path):
        specs = {
            "RFP-000951": {"예산": present("6억 원"),
                           "지역제한": present("부산 소재 업체만 참여 가능"),
                           "사업분야": present("건물 청소"),
                           "제출 방식": present("전자접수")},
            "RFP-000952": {"예산": present("5천만 원"),
                           "지역제한": present("서울 소재 업체만 참여 가능"),
                           "사업분야": present("시험용 값"), "제출 방식": present("시험용 값")},
            "RFP-000953": {"예산": present("2억 원"),
                           "지역제한": present("지역 제한 없음"),
                           "사업분야": present("시험용 값"), "제출 방식": present("시험용 값")},
            "RFP-000954": {},                       # 전 필드 미기재
        }
        table = write_table(tmp_path, specs)
        scope = write_registry(tmp_path, [
            {"document_id": d, "active": True, "retrieval_eligible": True}
            for d in sorted(specs)])
        identity = write_identity(tmp_path, [
            (d, f"시험기관{d[-3:]}", f"시험사업{d[-3:]} 구축", "2025-01-01 17:00:00")
            for d in sorted(specs)])
        return table, scope, identity

    # --- A: 참여 자격 조건은 조각이 잘려도 버리지 않는다 ---
    @pytest.mark.parametrize("question", [
        "서울 소재 업체만 참여 가능하고, 예산 5억 이상인 공고를 찾아줘",
        "서울 소재 업체만 참여 가능하고 예산 5억 이상인 공고를 찾아줘",
        "서울 소재 업체만 참여 가능하며, 예산 5억 이상인 공고를 찾아줘",
        "예산 5억 이상이고 서울 소재 업체만 참여 가능한 공고를 찾아줘",
        "예산 5억 이상, 서울 소재 업체만 참여 가능한 공고 목록 줘",
    ])
    def test_eligibility_condition_survives_any_order(self, question, world):
        from table_query import parse_selection
        table, scope, identity = world
        p = parse_selection(question)
        assert p.fully_matched is False, question
        r = run_answer(question, table, identity=identity, registry_scope=scope)
        assert r.abstained is True and r.route == "애매_되묻기", question
        assert r.selected_document_ids == []
        # 조건에 어긋나는 부산 문서를 정답처럼 반환하면 안 된다
        assert "RFP-000951" not in r.selected_document_ids

    # --- B: 필드의 '값' 조건도 버리지 않는다 ---
    @pytest.mark.parametrize("question,field", [
        ("사업분야가 인공지능이고, 예산 5억 이상인 공고를 찾아줘", "사업분야"),
        ("예산 5억 이상이고 사업분야가 인공지능인 공고를 찾아줘", "사업분야"),
        ("제출 방식은 방문접수이고, 예산 5억 이상인 공고를 찾아줘", "제출 방식"),
        ("예산 5억 이상이고 제출 방식은 방문접수인 공고를 찾아줘", "제출 방식"),
        ("사업분야가 인공지능 분야이고, 예산 5억 이상인 공고 알려줘", "사업분야"),
    ])
    def test_field_value_condition_is_reported(self, question, field, world):
        from table_query import parse_selection
        table, scope, identity = world
        p = parse_selection(question)
        assert p.fully_matched is False, question
        assert any(field in u for u in p.unresolved), f"{question} -> {p.unresolved}"
        r = run_answer(question, table, identity=identity, registry_scope=scope)
        assert r.abstained is True and r.selected_document_ids == []
        assert "조건에 맞는 공고가 없습니다" not in r.text   # 미지원 ≠ 0건

    # --- C: 숫자 비교의 부정 ---
    @pytest.mark.parametrize("question,expected", [
        ("예산 5억 이상이 아닌 공고를 찾아줘", ["RFP-000952", "RFP-000953"]),
        ("예산 1억 이하가 아닌 공고를 찾아줘", ["RFP-000951", "RFP-000953"]),
        ("예산 5억 이상인 공고를 찾아줘", ["RFP-000951"]),
        ("예산 1억 이하인 공고를 찾아줘", ["RFP-000952"]),
    ])
    def test_numeric_negation(self, question, expected, world):
        table, scope, identity = world
        r = run_answer(question, table, identity=identity, registry_scope=scope)
        assert r.selected_document_ids == expected, question
        # 예산 미기재 문서는 어느 쪽으로도 확정하지 않는다
        assert "RFP-000954" not in r.selected_document_ids

    @pytest.mark.parametrize("question,op,value", [
        ("예산 5억 이상이 아닌 공고를 찾아줘", "<", 500_000_000.0),
        ("예산 1억 이하가 아닌 공고를 찾아줘", ">", 100_000_000.0),
        ("예산 5억 초과가 아닌 공고 알려줘", "<=", 500_000_000.0),
        ("예산 5억 미만이 아닌 공고 알려줘", ">=", 500_000_000.0),
    ])
    def test_numeric_negation_operators(self, question, op, value):
        from table_query import parse_selection
        conds = parse_selection(question).conditions
        assert [(c.operator, c.value) for c in conds] == [(op, value)], question

    def test_unreadable_budget_stays_undetermined_under_negation(self, world):
        table, scope, identity = world
        r = run_answer("예산 5억 이상이 아닌 공고를 찾아줘", table,
                       identity=identity, registry_scope=scope)
        assert "RFP-000954" in r.text and "판단 불가" in r.text

    # --- 회귀: 배경 설명·기존 기능 ---
    @pytest.mark.parametrize("question,expected", [
        ("입찰 준비를 하고 있는데, 예산 5억 이상인 공고 목록 줘", ["RFP-000951"]),
        ("본사 소재지가 부산이라, 예산 5억 이상인 공고 알려줘", ["RFP-000951"]),
        ("사업분야가 궁금해서 그런데, 예산 5억 이상인 공고 알려줘", ["RFP-000951"]),
        ("예산 49,500만원 이상인 공고를 찾아줘", ["RFP-000951"]),
        ("예산이 기재된 공고는 제외하고 알려줘", ["RFP-000954"]),
    ])
    def test_background_and_existing_behaviour_kept(self, question, expected, world):
        table, scope, identity = world
        r = run_answer(question, table, identity=identity, registry_scope=scope)
        assert r.abstained is False, f"{question} -> {r.text[:120]}"
        assert r.selected_document_ids == expected, question

    def test_or_still_unsupported(self, world):
        table, scope, identity = world
        r = run_answer("예산 5억 이상 또는 1억 이하인 공고를 찾아줘", table,
                       identity=identity, registry_scope=scope)
        assert r.abstained is True and "OR" in r.text


class TestRound3ConsortiumNegatedPermission:
    """결함 ② 추출표의 '허용되지 않습니다'를 허용으로 읽지 않는다."""

    FORBIDDEN = [
        "공동수급은 허용되지 않습니다.",
        "공동수급은 가능하지 않습니다.",
        "공동수급은 허용하지 아니합니다.",
        "공동수급을 허용할 수 없습니다.",
        "공동수급은 허용하지 않습니다.",
        "공동수급은 승인되지 않습니다.",
        "본 사업에서 공동수급은 인정되지 않음",
    ]

    @pytest.mark.parametrize("value", FORBIDDEN)
    def test_negated_permission_is_forbidden(self, value):
        from table_query import classify_consortium
        v = classify_consortium({"status": "value_present", "answer_raw": value,
                                 "answer_normalized": value})
        assert v.allowed is False, f"{value} -> {v.reason}"
        assert v.required is False

    def test_same_negation_on_subcontract_does_not_move(self):
        """같은 문장을 하도급에 적용하면 공동수급 판정에 쓰지 않는다."""
        from table_query import classify_consortium
        v = classify_consortium({"status": "value_present",
                                 "answer_raw": "하도급은 허용되지 않습니다.",
                                 "answer_normalized": "하도급은 허용되지 않습니다."})
        assert v.allowed is None and v.required is None

    def test_negated_permission_document_excluded_from_allowed_selection(self, tmp_path):
        specs = {f"RFP-0009{60 + i}": {"컨소시엄 요건": present(v)}
                 for i, v in enumerate(self.FORBIDDEN)}
        specs["RFP-000980"] = {"컨소시엄 요건": present("본 사업은 공동수급을 허용함")}
        table = write_table(tmp_path, specs)
        scope = write_registry(tmp_path, [
            {"document_id": d, "active": True, "retrieval_eligible": True}
            for d in sorted(specs)])
        identity = write_identity(tmp_path, [
            (d, f"수급기관{d[-3:]}", f"수급사업{d[-3:]} 구축", "2025-01-01 17:00:00")
            for d in sorted(specs)])
        allowed = run_answer("공동수급이 허용되는 공고를 찾아줘", table,
                             identity=identity, registry_scope=scope)
        assert allowed.selected_document_ids == ["RFP-000980"]
        forbidden = run_answer("공동수급이 금지된 공고 목록 줘", table,
                               identity=identity, registry_scope=scope)
        assert set(forbidden.selected_document_ids) == {
            f"RFP-0009{60 + i}" for i in range(len(self.FORBIDDEN))}

    @pytest.mark.parametrize("value,required,allowed", [
        ("공동수급은 허용하며 하도급은 불가합니다.", False, True),
        ("공동수급은 공동이행방식으로 허용하며 분담이행방식은 불허합니다.", False, True),
        ("공동수급체 구성원은 반드시 관련 면허를 보유해야 합니다.", None, None),
        ("공동수급은 필수가 아닙니다. 단독 참여가 가능합니다.", False, None),
        ("공동수급체를 구성하지 못하는 불가피한 사정이 있는 경우 사유를 제시", None, None),
        ("본 사업은 공동수급을 허용함", False, True),
    ])
    def test_previous_fixes_kept(self, value, required, allowed):
        from table_query import classify_consortium
        v = classify_consortium({"status": "value_present", "answer_raw": value,
                                 "answer_normalized": value})
        assert (v.required, v.allowed) == (required, allowed), v.reason


class TestRound3SingleProjectScope:
    """결함 ③ 특정 사업을 물으면 전체 공고 목록으로 넘어가지 않는다."""

    def test_bracketed_annotation_in_registered_name_still_matches(self, tmp_path):
        """등록명의 괄호 주석("(용역)", "(협상)(긴급)")이 매칭을 막지 않는다."""
        from identity_metadata import load_identity
        from doc_resolver import resolve_document
        identity = write_identity(tmp_path, [
            ("RFP-000991", "가온철도공사 (용역)", "모바일현장 시스템 고도화 용역(총체 및 1차)",
             "2025-01-01 17:00:00"),
            ("RFP-000992", "나린군", "나린군 재난관리시스템 고도화 사업(협상)(긴급)",
             "2025-01-01 17:00:00"),
        ])
        r1 = resolve_document("가온철도공사 모바일현장 시스템 고도화 용역, 예산이 얼마야?",
                              identity)
        assert r1.document_id == "RFP-000991", r1.method
        r2 = resolve_document("나린군 재난관리시스템 고도화 사업, 지역 제한 있어?", identity)
        assert r2.document_id == "RFP-000992", r2.method

    @pytest.fixture
    def world(self, tmp_path):
        specs = {"RFP-000991": {"예산": present("3억원"), "지역제한": present("서울시 소재")},
                 "RFP-000992": {"예산": present("7억원")},
                 "RFP-000993": {"예산": present("8억원")}}
        table = write_table(tmp_path, specs)
        scope = write_registry(tmp_path, [
            {"document_id": d, "active": True, "retrieval_eligible": True}
            for d in sorted(specs)])
        identity = write_identity(tmp_path, [
            ("RFP-000991", "가온철도공사 (용역)", "모바일현장 시스템 고도화 용역(총체 및 1차)",
             "2025-01-01 17:00:00"),
            ("RFP-000992", "나린군", "나린군 재난관리시스템 고도화 사업(협상)(긴급)",
             "2025-01-01 17:00:00"),
            ("RFP-000993", "다솜연구원", "다솜연구원 관측망 구축", "2025-01-01 17:00:00"),
        ])
        return table, scope, identity

    def test_named_project_answers_that_document_only(self, world):
        table, scope, identity = world
        r = run_answer("가온철도공사 모바일현장 시스템 고도화 용역, 사업 예산이 얼마나 되는지 알려줘",
                       table, identity=identity, registry_scope=scope)
        assert r.task_type == "extract"
        assert r.selected_document_ids == ["RFP-000991"]
        assert "3억원" in r.text
        assert r.citations and all(c["document"] == "RFP-000991" for c in r.citations)

    def test_named_project_not_found_asks_back_not_whole_list(self, world):
        """등록부에 없는 사업 제목이면 되묻는다 — 전체 목록을 답하지 않는다."""
        table, scope, identity = world
        r = run_answer("은하수정보원 우주관측 데이터 플랫폼 구축 사업, 예산이 얼마나 되는지 알려줘",
                       table, identity=identity, registry_scope=scope)
        assert r.task_type == "extract"
        assert r.abstained is True
        assert r.selected_document_ids == []          # 엉뚱한 목록이 남으면 안 된다
        assert len(r.selected_document_ids) < 3

    def test_explicit_global_scope_still_selects(self, world):
        table, scope, identity = world
        r = run_answer("전체 공고에서 예산 5억 이상인 공고 목록 줘", table,
                       identity=identity, registry_scope=scope)
        assert r.task_type == "select"
        assert set(r.selected_document_ids) == {"RFP-000992", "RFP-000993"}

    def test_document_id_still_wins(self, world):
        table, scope, identity = world
        r = run_answer("RFP-000992의 예산 알려줘", table,
                       identity=identity, registry_scope=scope)
        assert r.task_type == "extract" and r.selected_document_ids == ["RFP-000992"]

    def test_active_document_follow_up_still_works(self, world):
        from answer_pipeline import SessionState
        table, scope, identity = world
        r = run_answer("그 사업 예산이 얼마야?", table, identity=identity,
                       registry_scope=scope,
                       session=SessionState(active_document_id="RFP-000993"))
        assert r.task_type == "extract" and r.selected_document_ids == ["RFP-000993"]

    def test_multi_candidate_clarify_kept(self, tmp_path):
        specs = {"RFP-000994": {"예산": present("1억원")},
                 "RFP-000995": {"예산": present("2억원")}}
        table = write_table(tmp_path, specs)
        scope = write_registry(tmp_path, [
            {"document_id": d, "active": True, "retrieval_eligible": True}
            for d in sorted(specs)])
        identity = write_identity(tmp_path, [
            ("RFP-000994", "한별재단", "한별재단 1차 사업", "2025-01-01 17:00:00"),
            ("RFP-000995", "한별재단", "한별재단 2차 사업", "2025-01-01 17:00:00"),
        ])
        r = run_answer("한별재단 사업 예산 얼마야?", table,
                       identity=identity, registry_scope=scope)
        assert r.abstained is True
        assert set(r.selected_document_ids) == {"RFP-000994", "RFP-000995"}
