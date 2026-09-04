"""사업 회차 **문맥** 판정 회귀 검사 (2026-09-04).

고친 것: 질문 전체에서 "N차"를 찾다 보니 공고 차수·대화 순서까지 사업 회차로 읽어,
공식 사업명에 그 회차가 없다는 이유로 **정상 문서가 후보에서 사라졌다**.
   "2차 공고인 사업" · "재공고 2차 사업" · "2차 회의에서 본 사업" · "앞서 말한 2차 질문"
이제 공식 사업명용 추출(project_rounds)과 질문용 추출(question_project_rounds)을 분리해,
질문 쪽은 **사업을 가리키는 문맥**의 회차만 읽는다.

⚠️ 합성 자료만 쓴다(공식 파일 미사용, 문항 ID·문서 번호 미사용).
   공식 자료 기준 확인은 tools/required_checks.py 가 따로 수행한다.
"""
from __future__ import annotations

from pathlib import Path

_ID_HEADER = ('"document_id","source_filename_nfc","collection_system",'
              '"source_record_id","source_url","notice_number","notice_round",'
              '"buyer_org","project_name","notice_date","bid_deadline","metadata_found"')


def _identity(tmp_path: Path, rows, name="identity_ctx.csv"):
    from identity_metadata import load_identity
    lines = [_ID_HEADER]
    for doc_id, filename, org, project in rows:
        lines.append(
            f'"{doc_id}","{filename}","","","","2024000","0.0",'
            f'"{org}","{project}","2024-01-01 10:00:00","2025-01-01 17:00:00","true"')
    p = tmp_path / name
    p.write_text("﻿" + "\n".join(lines) + "\n", encoding="utf-8")
    return load_identity(p)


_NAME = "합성 통합정보 기능개선 사업"          # 회차 없는 사업
_ROWS = [
    ("RFP-000001", "합성원_합성 통합정보 기능개선 사업.md", "합성원", _NAME),
    ("RFP-000002", "합성원_합성 통합정보 기능개선 사업(1차).md", "합성원", _NAME + "(1차)"),
    ("RFP-000003", "합성원_합성 통합정보 기능개선 사업(2차).md", "합성원", _NAME + "(2차)"),
]


class Test사업회차_정상처리:
    def test_괄호_회차는_해당_문서로_연결된다(self, tmp_path):
        from doc_resolver import resolve_document
        index = _identity(tmp_path, _ROWS)
        assert resolve_document(f"{_NAME}(1차) 예산이 얼마야?", index).document_id == "RFP-000002"
        assert resolve_document(f"{_NAME}(2차) 예산이 얼마야?", index).document_id == "RFP-000003"

    def test_제N차와_붙여쓴_N차사업도_연결된다(self, tmp_path):
        from doc_resolver import resolve_document
        index = _identity(tmp_path, [
            ("RFP-000001", "합성원_합성 통합정보 고도화 2차사업.md", "합성원", "합성 통합정보 고도화 2차사업"),
            ("RFP-000002", "합성원_합성 통합정보 고도화 사업.md", "합성원", "합성 통합정보 고도화 사업"),
        ])
        for q in ("합성 통합정보 고도화 2차사업 예산이 얼마야?",
                  "합성 통합정보 고도화 제2차 사업 예산이 얼마야?",
                  "합성 통합정보 고도화 2차 사업 예산이 얼마야?"):
            assert resolve_document(q, index).document_id == "RFP-000001", q

    def test_N차_구축_용역_형태도_연결된다(self, tmp_path):
        from doc_resolver import resolve_document
        index = _identity(tmp_path, [
            ("RFP-000001", "합성원_합성 안전정보시스템 1차 구축 용역.md",
             "합성원", "합성 안전정보시스템 1차 구축 용역"),
        ])
        assert resolve_document(
            "합성 안전정보시스템 1차 구축 용역 예산", index).document_id == "RFP-000001"

    def test_존재하지_않는_회차는_다른_문서로_연결하지_않는다(self, tmp_path):
        from doc_resolver import resolve_document
        index = _identity(tmp_path, _ROWS)
        res = resolve_document(f"{_NAME}(3차) 예산이 얼마야?", index)
        assert res.document_id is None and res.candidates == []

    def test_회차_없는_정확한_사업명은_기존대로_연결된다(self, tmp_path):
        from doc_resolver import resolve_document
        index = _identity(tmp_path, _ROWS)
        assert resolve_document(f"{_NAME} 예산이 얼마야?", index).document_id == "RFP-000001"

    def test_기관명과_회차를_함께_입력해도_연결된다(self, tmp_path):
        from doc_resolver import resolve_document
        index = _identity(tmp_path, _ROWS)
        assert resolve_document(
            f"합성원 {_NAME}(2차) 예산이 얼마야?", index).document_id == "RFP-000003"

    def test_같은_회차_후보가_여럿이면_되묻는다(self, tmp_path):
        from doc_resolver import resolve_document
        index = _identity(tmp_path, [
            ("RFP-000001", "가기관_합성 통합정보 기능개선 사업(2차).md", "가기관", _NAME + "(2차)"),
            ("RFP-000002", "나기관_합성 통합정보 기능개선 사업(2차).md", "나기관", _NAME + "(2차)"),
        ])
        res = resolve_document(f"{_NAME}(2차) 예산이 얼마야?", index)
        assert res.document_id is None
        assert set(res.candidates) == {"RFP-000001", "RFP-000002"}


class Test공고차수_오인방지:
    """정확한 사업명에 공고 차수를 덧붙여도 사업 회차로 걸러내면 안 된다."""

    def test_공고_차수_표현은_사업_회차가_아니다(self, tmp_path):
        from doc_resolver import resolve_document
        index = _identity(tmp_path, _ROWS)
        for tail in ("2차 공고로 올라온", "2차 재공고된", "재공고 2차",
                     "2차 입찰공고에 나온", "2차 정정공고의"):
            q = f"{tail} {_NAME}의 예산이 얼마야?"
            assert resolve_document(q, index).document_id == "RFP-000001", q

    def test_공고_차수_표현_자체는_회차로_읽지_않는다(self):
        from text_normalize import question_project_rounds as rounds
        for text in ("2차 공고", "2차 재공고", "재공고 2차", "2차 입찰공고",
                     "2차 정정공고", "(공고 2차)", "2차 공고인 사업", "재공고 2차 사업"):
            assert rounds(text) == set(), text


class Test대화표현_오인방지:
    def test_회의_질문_답변_검토_차수는_사업_회차가_아니다(self, tmp_path):
        from doc_resolver import resolve_document
        index = _identity(tmp_path, _ROWS)
        for tail in ("2차 회의에서 언급한", "앞서 말한 2차 질문의",
                     "2차 답변에서 나온", "2차 검토 대상인", "2차 평가를 받은"):
            q = f"{tail} {_NAME}의 예산이 얼마야?"
            assert resolve_document(q, index).document_id == "RFP-000001", q

    def test_대화_차수_표현_자체는_회차로_읽지_않는다(self):
        from text_normalize import question_project_rounds as rounds
        for text in ("2차 회의", "2차 질문", "2차 답변", "2차 검토", "2차 평가",
                     "2차 회의에서 본 사업", "앞서 말한 2차 질문"):
            assert rounds(text) == set(), text

    def test_일반_숫자_표현_오인_방지는_그대로다(self):
        from text_normalize import question_project_rounds as rounds
        for text in ("3차원 설계", "2차전지 소재", "1차년도 사업비", "1차산업 지원",
                     "왕복 2차로", "1차선 도색", "3차방정식", "2개 항목", "7일 이내",
                     "2억원 이상", "문단 2 인용"):
            assert rounds(text) == set(), text


class Test이름용과_질문용_분리:
    def test_공식_사업명용_추출은_이름_안의_회차를_읽는다(self):
        from text_normalize import project_rounds
        assert project_rounds("기능개선 사업(2차)") == {2}
        assert project_rounds("고도화 2차사업") == {2}
        assert project_rounds("안전정보시스템 1차 구축 용역") == {1}
        assert project_rounds("고도화 용역(총체 및 1차)") == {1}

    def test_질문용_추출은_사업_문맥일_때만_읽는다(self):
        from text_normalize import question_project_rounds as rounds
        assert rounds("기능개선 사업(2차)의 예산은?") == {2}
        assert rounds("기능개선 제2차 사업 예산은?") == {2}
        assert rounds("1차 구축 용역 예산은?") == {1}
        assert rounds("2차 공고인 사업의 예산은?") == set()


class Test기존_수정_재발방지:
    def test_상투어_보조키_오확정이_재발하지_않는다(self, tmp_path):
        from doc_resolver import resolve_document
        index = _identity(tmp_path, [
            ("RFP-000001", "합성원_합성 재난관리시스템 구축 위탁용역.md", "합성원",
             "(긴급)「2024년도 합성 재난관리시스템 구축」 위탁용역"),
        ])
        assert "위탁용역" not in index.by_project_key
        assert resolve_document("가상의 위탁용역 예산 알려줘", index).document_id is None
        assert resolve_document(
            "2024년도 합성 재난관리시스템 구축 위탁용역 예산 알려줘",
            index).document_id == "RFP-000001"

    def test_사업명_안의_기관형_표현_오류가_재발하지_않는다(self, tmp_path):
        from doc_resolver import detect_unknown_orgs, resolve_document
        index = _identity(tmp_path, [
            ("RFP-000001", "합성보호원_합성-NAVI 해외합성지원센터 사업관리 시스템 기능개선.md",
             "합성보호원", "합성-NAVI 해외합성지원센터 사업관리 시스템 기능개선"),
        ])
        q = "합성-NAVI 해외합성지원센터 사업관리 시스템 기능개선 사업의 예산 알려줘"
        assert detect_unknown_orgs(q, index) == []
        assert resolve_document(q, index).document_id == "RFP-000001"

    def test_주석_괄호_처리는_기존과_동일하다(self):
        from text_normalize import strip_bracketed
        assert strip_bracketed("합성군 재난통합관리시스템 고도화 사업(협상)(긴급)") == \
            "합성군 재난통합관리시스템 고도화 사업"
        assert strip_bracketed("합성공원 회원 통합운영 시스템 구축[협상에 의한 계약]") == \
            "합성공원 회원 통합운영 시스템 구축"
        assert strip_bracketed("합성철도공사 (용역)") == "합성철도공사"
        assert strip_bracketed("합성 버스정보시스템(BIS) 구축사업") == \
            "합성 버스정보시스템 구축사업"
