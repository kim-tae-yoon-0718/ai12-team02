"""실제 API 7문항 검증에서 나온 세 문제의 회귀 검사.

세 문제 (2026-09-03 실제 API 실행에서 재현됨)
  A. 제출 방식을 풀어 쓴 질문이 추출형으로 분기하지 못한다.
  B. 공식 사업명과 조사(`의`) 한 글자·꼬리말만 다른 이름으로 문서를 못 찾는다.
  C. 체크리스트 요청에서 근거에 없는 항목을 만들어 붙인다.

⚠️ 평가 문항 ID·정답을 외우는 검사가 아니다. 여기 쓰는 문서·사업명·값은
   전부 이 파일 안에서 만든 합성 자료이고, 검사 대상은 **일반 규칙**이다.
   (공식 파일을 읽는 검사는 하나도 없다.)
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from conftest import OFFICIAL_FIELDS, _loc, forbidden


# ---------------------------------------------------------------------------
# 합성 자료 만들기 — 공식 코퍼스·등록부·평가셋을 읽지 않는다
# ---------------------------------------------------------------------------

_ID_HEADER = ('"document_id","source_filename_nfc","collection_system",'
              '"source_record_id","source_url","notice_number","notice_round",'
              '"buyer_org","project_name","notice_date","bid_deadline","metadata_found"')


def _identity_csv(tmp_path: Path, rows: list[tuple[str, str, str, str]],
                  name: str = "identity_syn.csv") -> Path:
    """(document_id, source_filename_nfc, buyer_org, project_name) 네 칸만 받는다."""
    lines = [_ID_HEADER]
    for doc_id, filename, org, project in rows:
        lines.append(
            f'"{doc_id}","{filename}","","","","2024000","0.0",'
            f'"{org}","{project}","2024-01-01 10:00:00","2025-01-01 17:00:00","true"')
    p = tmp_path / name
    p.write_text("﻿" + "\n".join(lines) + "\n", encoding="utf-8")
    return p


def _load_identity(tmp_path: Path, rows, name="identity_syn.csv"):
    from identity_metadata import load_identity
    return load_identity(_identity_csv(tmp_path, rows, name))


def _table_with(tmp_path: Path, doc_ids: list[str], overrides: dict) -> list[dict]:
    """합성 추출표 rows — overrides는 {document_id: {field: {...}}}."""
    from table_query import load_extraction_table
    rows = []
    for doc_id in doc_ids:
        for f in OFFICIAL_FIELDS:
            base = {
                "document_id": doc_id, "field_name": f, "status": "field_absent",
                "answer_raw": "", "answer_normalized": "", "active": "true",
                "representative_location": None, "additional_locations": [],
                "schema_version": "1-12-2/v3", "extraction_version": "v3",
                "corpus_version": "v2", "registry_version": "v2",
            }
            base.update(overrides.get(doc_id, {}).get(f, {}))
            rows.append(base)
    doc = {
        "corpus_version": "v2", "document_count": len(doc_ids),
        "extraction_version": "v3", "field_count": 12, "fields": OFFICIAL_FIELDS,
        "registry_version": "v2", "row_count": len(rows),
        "schema_version": "1-12-2/v3", "rows": rows,
    }
    p = tmp_path / "extraction_syn.json"
    p.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    return load_extraction_table(p)


# ===========================================================================
# A. 제출 방식 — 풀어 쓴 표현 인식 (9건)
# ===========================================================================

class Test제출방식_바꿔말하기:
    """필드명이 글자 그대로 없어도 '제출 방식'을 묻는 질문을 잡는다."""

    def test_어떤_방식으로_제출해야_하나요(self):
        from table_query import detect_fields
        assert "제출 방식" in detect_fields("제안서를 어떤 방식으로 제출해야 하나요?")

    def test_어떤_방법으로_제출하나요(self):
        from table_query import detect_fields
        assert "제출 방식" in detect_fields("제안서는 어떤 방법으로 제출하나요?")

    def test_입찰서류를_어떻게_제출해야_하나요(self):
        from table_query import detect_fields
        assert "제출 방식" in detect_fields("입찰서류를 어떻게 제출해야 하나요?")
        # 같은 뜻인데 동작어가 앞에 오는 어순도 놓치지 않는다
        assert "제출 방식" in detect_fields("제안서 접수는 어떻게 하나요?")
        assert "제출 방식" in detect_fields("신청서 접수는 어떤 절차로 하나요?")

    def test_어떻게_내야_하나요(self):
        from table_query import detect_fields
        assert "제출 방식" in detect_fields("제안서를 어떻게 내야 하나요?")
        # ⚠️ "내다"가 다른 낱말 안에서 걸리면 안 된다
        assert "제출 방식" not in detect_fields("제안서 발표 자료는 어떻게 만들어 내나요?")
        assert "제출 방식" not in detect_fields("제안서를 어떻게 해내나요?")
        assert "제출 방식" not in detect_fields("제안서 관련 공지는 어떻게 안내는 되나요?")

    def test_제출_방법_띄어쓰기_두_형태(self):
        from table_query import detect_fields
        assert "제출 방식" in detect_fields("제출 방법 알려줘")
        assert "제출 방식" in detect_fields("제출방법 알려줘")

    def test_평가방식_수행방식_제출이유는_제출방식이_아니다(self):
        """일반적인 '어떤 방식'을 전부 제출 방식으로 잡으면 안 된다.

        ⚠️ 세 신호(대상·방법을 묻는 말·제출 동작)가 **다른 절**에 흩어져 있는
        경우까지 막는다. 표현 하나씩만 검사하면 이어 붙인 문장에서 새어나간다.
        """
        from table_query import detect_fields
        for q in (
            "제안서 평가 방식이 어떻게 되나요?",
            "사업을 어떤 방식으로 수행하나요?",
            "제안서를 왜 제출해야 하나요?",
            # 절 경계를 넘어 붙던 오탐
            "제안서 평가 방식이 어떻게 되고 제출 기한은 언제인가요?",
            "제안서 발표는 어떻게 진행되고 접수는 언제 끝나나요?",
            "제안서를 어떻게 평가해서 제출 여부를 정하나요?",
            "예산이 어떻게 되고 제안서 제출 서류는 무엇인가요?",
            "제안서를 어떤 방식으로 평가하는지와 제출 서류를 알려주세요",
            "제안서 제출 마감이 어떻게 되나요?",
            # 방법 명사 "식"이 다른 낱말 첫 글자로 걸리던 오탐
            "어떤 식자재 납품 서류를 제출하나요?",
            "어느 식당에서 제안서를 제출하나요?",
        ):
            assert "제출 방식" not in detect_fields(q), q

    def test_제출_대상의_서류를_필수제출서류로_중복인식하지_않는다(self):
        """'입찰서류를 어떻게 제출하나요?'의 서류는 제출 대상이지 서류 질문이 아니다."""
        from table_query import detect_fields
        for q in ("입찰서류를 어떻게 제출하나요?", "입찰서류를 어떻게 제출해야 하나요?"):
            fields = detect_fields(q)
            assert "제출 방식" in fields, q
            assert "필수 제출 서류" not in fields, q

    def test_두_항목을_실제로_함께_물으면_둘_다_보존한다(self):
        """⚠️ 중복 방지 규칙이 사용자가 물은 서류 목록까지 지우면 안 된다."""
        from table_query import detect_fields
        fields = detect_fields("어떤 서류가 필요하고 어떻게 제출하나요?")
        assert "제출 방식" in fields
        assert "필수 제출 서류" in fields
        # 수식어가 앞에 오는 흔한 어순에서도 서류 필드가 사라지면 안 된다
        for q in ("필요한 서류와 제출 방법을 알려주세요",
                  "제출 방법과 서류 양식을 알려주세요",
                  "제출 방법이랑 서류도 같이 알려줘"):
            assert "필수 제출 서류" in detect_fields(q), q


class Test제출방식_실제_answer_경로:
    def test_추출표_값을_돌려주고_API를_호출하지_않는다(self, tmp_path, base_cfg,
                                                    small_store):
        """라우터만 바뀌고 실행부가 필드를 못 찾는 상태를 만들지 않는다."""
        from answer_pipeline import answer, SessionState, ROUTE_EXTRACT_VALUE
        identity = _load_identity(tmp_path, [
            ("RFP-000001", "합성기관_합성 통합관리시스템 운영지원.md",
             "합성기관", "2024년 합성 통합관리시스템 운영지원"),
        ])
        table = _table_with(tmp_path, ["RFP-000001"], {
            "RFP-000001": {"제출 방식": {
                "status": "value_present",
                "answer_raw": "전자조달시스템(나라장터)을 통한 전자제출",
                "answer_normalized": "전자조달시스템(나라장터)을 통한 전자제출",
                "representative_location": _loc("Ⅳ. 제안 일반사항", 1042,
                                                "paragraph", 7)}},
        })
        result = answer(
            "합성기관 합성 통합관리시스템 운영지원 사업, 제안서를 어떤 방식으로 "
            "제출해야 하는지 알려줘.",
            small_store, forbidden("임베딩"), forbidden("생성"), table, base_cfg,
            identity=identity, session=SessionState(),
        )
        assert result.task_type == "extract"
        assert result.route == ROUTE_EXTRACT_VALUE
        assert result.selected_document_ids == ["RFP-000001"]
        assert "전자조달시스템" in result.text
        assert result.abstained is False
        assert result.citations and result.citations[0]["document"] == "RFP-000001"


# ===========================================================================
# B. 사업명 특정 — 조사 보조 키 + 공식 파일명 별칭 (8건)
# ===========================================================================

class Test사업명_조사보조키:
    def test_기반의_영상과_기반_영상이_일치한다(self, tmp_path):
        from doc_resolver import resolve_document
        identity = _load_identity(tmp_path, [
            ("RFP-000001", "합성수행사_합성시 AI 기반의 영상감시 시스템.md",
             "합성수행사", "합성시 AI 기반의 영상감시 시스템 및 홍수감시 연동 개발"),
        ])
        res = resolve_document("합성시 AI 기반 영상감시 시스템 사업, 추진배경이 뭐야?",
                               identity)
        assert res.document_id == "RFP-000001"

    def test_다른_핵심단어가_들어간_제목은_일치하지_않는다(self, tmp_path):
        from doc_resolver import resolve_document
        identity = _load_identity(tmp_path, [
            ("RFP-000001", "합성수행사_합성시 AI 기반의 영상감시 시스템.md",
             "합성수행사", "합성시 AI 기반의 영상감시 시스템 및 홍수감시 연동 개발"),
        ])
        res = resolve_document("합성시 AI 기반 음성감시 시스템 사업, 추진배경이 뭐야?",
                               identity)
        assert res.document_id is None
        # ⚠️ 별칭(=사업명 앞부분)만 겹치고 **뒷부분 핵심어가 다른** 이름도
        #    조용히 그 문서로 확정하면 안 된다(되묻어야 한다).
        identity2 = _load_identity(tmp_path, [
            ("RFP-000001", "합성공사_합성공원 다목적구장 홈페이지 및 회원 통합운영.md",
             "합성공사", "합성공원 다목적구장 홈페이지 및 회원 통합운영 관리 시스템 구축"),
        ], name="identity_syn2.csv")
        fake = ("합성공원 다목적구장 홈페이지 및 회원 통합운영 관리 장비 구축 "
                "사업의 마감일이 언제야?")
        assert resolve_document(fake, identity2).document_id is None
        # 같은 색인에서 진짜 이름은 그대로 찾아야 한다
        assert resolve_document(
            "합성공원 다목적구장 홈페이지 및 회원 통합운영 사업의 마감일이 언제야?",
            identity2).document_id == "RFP-000001"

    def test_단어_안의_의는_지우지_않는다(self):
        """'의정부'·'의료'의 '의'는 조사가 아니다."""
        from text_normalize import fold_possessive_particles as fold
        assert fold("의정부시 의료원 통합 시스템") == "의정부시 의료원 통합 시스템"
        assert fold("의료 데이터 플랫폼") == "의료 데이터 플랫폼"
        # 낱말 사이의 조사 '의' 하나만 접는다
        assert fold("의정부시의 의료 시스템") == "의정부시 의료 시스템"
        # ⚠️ '의'로 **끝나는** 2음절 낱말도 조사가 아니다 — 잘라내면 안 된다
        for text in ("제안서 평가 심의 위원회 일정",
                     "사업 착수 회의 자료 준비",
                     "제출 시 주의 사항 확인",
                     "용어 정의 및 약어 정리"):
            assert fold(text) == text, text
        # 반대로 진짜 조사는 접어야 한다(낱말 목록으로 막지 않는다)
        assert fold("통합의 관리시스템 구축") == "통합 관리시스템 구축"

    def test_조사를_접은_뒤_여러_문서가_같아지면_되묻는다(self, tmp_path):
        from doc_resolver import resolve_document
        identity = _load_identity(tmp_path, [
            ("RFP-000001", "가기관_가기관 통합의 관리시스템 구축.md",
             "가기관", "통합의 관리시스템 구축"),
            ("RFP-000002", "나기관_나기관 통합 관리시스템 구축.md",
             "나기관", "통합 관리시스템 구축"),
        ])
        res = resolve_document("통합 관리시스템 구축 사업의 예산 알려줘", identity)
        assert res.document_id is None
        assert set(res.candidates) == {"RFP-000001", "RFP-000002"}


class Test사업명_공식파일명_별칭:
    def test_파일명의_짧은_사업명으로_문서를_찾는다(self, tmp_path):
        from doc_resolver import resolve_document
        identity = _load_identity(tmp_path, [
            ("RFP-000001", "합성공사_합성근린공원 다목적구장 홈페이지 및 회원 통합운영.md",
             "합성공사",
             "합성근린공원 다목적구장 홈페이지 및 회원 통합운영 관리 시스템 구축[협상에 의한 계약]"),
            ("RFP-000002", "다른공사_전혀 다른 사업.md", "다른공사", "전혀 다른 사업"),
        ])
        res = resolve_document(
            "합성근린공원 다목적구장 홈페이지 및 회원 통합운영 사업의 마감일이 언제야?",
            identity)
        assert res.document_id == "RFP-000001"

    def test_낱말_중간에서_잘린_별칭은_낱말_끝까지_늘린다(self, tmp_path):
        """공식 파일명은 길이 제한으로 낱말 중간을 자른다. 그 조각을 그대로 쓰면
        전혀 다른 뒷말("자동분석 시스코")에도 붙는다."""
        from doc_resolver import resolve_document
        from identity_metadata import load_identity
        path = _identity_csv(tmp_path, [
            ("RFP-000001", "합성공단_운행정보기록 자동분석 시스.md",
             "합성공단", "운행정보기록 자동분석 시스템 고도화 용역"),
        ])
        index = load_identity(path)
        assert index.get("RFP-000001").filename_project_alias == \
            "운행정보기록 자동분석 시스템"
        assert resolve_document(
            "운행정보기록 자동분석 시스코 장비를 쓰나요?", index).document_id is None
        assert resolve_document(
            "운행정보기록 자동분석 시스템 사업의 예산 알려줘",
            index).document_id == "RFP-000001"

    def test_별칭이_다른_문서_공식명에도_있으면_후보로_남긴다(self, tmp_path):
        """별칭 소유 문서 하나로 조용히 확정하지 않는다(임의 선택 금지)."""
        from doc_resolver import resolve_document
        from identity_metadata import load_identity
        from text_normalize import normalize_project_key
        path = _identity_csv(tmp_path, [
            # 파일명이 잘려 별칭이 "…기능개선"이 되고, 그 이름이 아래 문서의
            # 정식 사업명 안에도 글자 그대로 들어 있다.
            ("RFP-000001", "가기관_합성 종합정보시스템 기능개선.md",
             "가기관", "합성 종합정보시스템 기능개선 사업"),
            ("RFP-000002", "나기관_합성 종합정보시스템 기능개선 사업(2차).md",
             "나기관", "합성 종합정보시스템 기능개선 사업(2차)"),
        ])
        index = load_identity(path)
        key = normalize_project_key("합성 종합정보시스템 기능개선")
        assert set(index.by_project_alias_key[key]) == {"RFP-000001", "RFP-000002"}
        res = resolve_document("합성 종합정보시스템 기능개선 관련 마감일은?", index)
        assert res.document_id is None
        assert set(res.candidates) == {"RFP-000001", "RFP-000002"}
        # 기관명까지 함께 부르면 기존 규칙(기관+사업명 교집합)이 한 건으로 좁힌다
        assert resolve_document(
            "나기관의 합성 종합정보시스템 기능개선 관련 마감일은?",
            index).document_id == "RFP-000002"

    def test_파일명_기관접두가_다르면_별칭으로_쓰지_않는다(self, tmp_path):
        """파일명 앞 기관명이 같은 행의 buyer_org와 다르면 공식 별칭이 아니다."""
        from identity_metadata import load_identity
        from text_normalize import normalize_project_key
        path = _identity_csv(tmp_path, [
            ("RFP-000001", "엉뚱기관_숨은 사업명 별칭.md", "합성기관", "합성기관 정식 사업명"),
        ])
        index = load_identity(path)
        alias_key = normalize_project_key("숨은 사업명 별칭")
        assert alias_key not in index.by_project_alias_key

    def test_별칭_색인에_같은_문서_ID가_중복되지_않는다(self, tmp_path):
        """같은 문서의 여러 공식 별칭이 한 키로 합쳐져도 ID는 한 번만 저장한다."""
        from identity_metadata import load_identity
        path = _identity_csv(tmp_path, [
            # 파일명 짧은 사업명과 정식 사업명이 같은 키로 접힌다
            ("RFP-000001", "합성기관_합성 통합관리시스템 구축.md",
             "합성기관", "합성 통합관리시스템 구축"),
            ("RFP-000002", "다른기관_다른 사업.md", "다른기관", "다른 사업"),
        ])
        index = load_identity(path)
        for name, table in (("by_project_key", index.by_project_key),
                            ("by_project_core_key", index.by_project_core_key),
                            ("by_project_alias_key", index.by_project_alias_key),
                            ("by_project_folded_key", index.by_project_folded_key)):
            for key, docs in table.items():
                assert len(docs) == len(set(docs)), f"{name}[{key}]에 중복 문서 ID"

    def test_정확한_전체_사업명_매칭은_그대로_동작한다(self, tmp_path):
        from doc_resolver import resolve_document
        identity = _load_identity(tmp_path, [
            ("RFP-000001", "합성기관_합성 통합관리시스템 구축.md",
             "합성기관", "합성 통합관리시스템 구축 용역"),
            ("RFP-000002", "다른기관_다른 사업.md", "다른기관", "다른 사업"),
        ])
        res = resolve_document("합성 통합관리시스템 구축 용역의 예산이 얼마야?", identity)
        assert res.document_id == "RFP-000001"


# ===========================================================================
# C. 근거 기반 체크리스트 (5건)
# ===========================================================================

class Test근거기반_체크리스트:
    def test_체크리스트_요청이면_강화된_형식_지시를_만든다(self):
        from answer_pipeline import format_instruction_for
        for q in ("사업범위 체크포인트로 뽑아줘", "사업범위 체크리스트로 정리해줘",
                  "점검 항목으로 정리해줘", "확인 목록으로 만들어줘"):
            instruction = format_instruction_for(q)
            assert "형식" in instruction
            assert "근거에 없는" in instruction

    def test_일반_QA는_기존_형식_지시를_유지한다(self):
        from answer_pipeline import format_instruction_for
        assert format_instruction_for("추진배경이 뭐야?") == "간결하고 명확하게 답하세요."

    def test_base_yaml이_generate_v3를_쓴다(self):
        import yaml
        root = Path(__file__).resolve().parent.parent.parent
        cfg = yaml.safe_load((root / "config" / "base.yaml").read_text(encoding="utf-8"))
        assert cfg["prompt_generate"] == "generate_v3.txt"
        assert (root / "src" / "prompts" / "generate_v3.txt").exists()
        # v2는 보존한다(팀 규약: 같은 파일을 고치지 않는다)
        assert (root / "src" / "prompts" / "generate_v2.txt").exists()

    def test_실제_생성_호출까지_강화된_형식_지시가_전달된다(self, tmp_path, base_cfg,
                                                     small_store, table_rows):
        from answer_pipeline import answer_qa_or_extract_by_search
        from conftest import FakeEmbed, FakeGen
        seen = {}

        class RecordingGen(FakeGen):
            def generate(self, question, context_chunks,
                         format_instruction="...", structured_context=None):
                seen["instruction"] = format_instruction
                return super().generate(question, context_chunks,
                                        format_instruction, structured_context)

        gen = RecordingGen()
        answer_qa_or_extract_by_search(
            "사업범위 체크포인트로 뽑아줘", small_store, FakeEmbed(), gen, base_cfg,
            document_id="RFP-000001",
        )
        assert gen.calls == 1
        assert "근거에 없는" in seen["instruction"]

    def test_v3에_v2의_근거_인용_규약이_그대로_있다(self):
        """규약 블록을 **통째로** 대조한다(일부 문장만 검사하면 빠뜨린다)."""
        root = Path(__file__).resolve().parent.parent.parent / "src" / "prompts"
        v2 = (root / "generate_v2.txt").read_text(encoding="utf-8")
        v3 = (root / "generate_v3.txt").read_text(encoding="utf-8")
        marker = "[근거 인용 규약 — 반드시 지킬 것]"
        assert marker in v2 and marker in v3
        assert v2[v2.index(marker):].strip() == v3[v3.index(marker):].strip()
        # 실제로 로드되는 시스템 프롬프트가 v3인지도 확인한다
        import yaml
        from generation_client import _load_system_prompt_template
        cfg = yaml.safe_load(
            (root.parent.parent / "config" / "base.yaml").read_text(encoding="utf-8"))
        loaded = _load_system_prompt_template(cfg["prompt_generate"])
        assert loaded == v3
        assert "형식 변환 규칙" in loaded
