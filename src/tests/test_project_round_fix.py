"""사업 회차(N차) 판정 회귀 검사 (2026-09-04).

고친 것: 사업명 뒤 괄호를 전부 주석처럼 지우다 보니 "…기능개선 사업(7차)"처럼
사용자가 명시한 회차가 사라지고, 회차가 없는 **다른 문서**로 연결됐다.
이제 질문에 회차가 있으면 공식 사업명의 회차와 대조해 같은 회차만 남긴다.

⚠️ 여기 쓰는 문서·사업명은 전부 이 파일 안에서 만든 합성 자료다. 공식 파일을 읽지
   않고, 평가 문항 ID나 특정 문서 번호를 외우지 않는다.
   공식 100문서 기준 확인은 `tools/round_official_check.py` 가 따로 수행한다.
"""
from __future__ import annotations

from pathlib import Path

_ID_HEADER = ('"document_id","source_filename_nfc","collection_system",'
              '"source_record_id","source_url","notice_number","notice_round",'
              '"buyer_org","project_name","notice_date","bid_deadline","metadata_found"')


def _identity(tmp_path: Path, rows, name="identity_round.csv"):
    """(document_id, filename, buyer_org, project_name[, notice_round]) 을 받는다."""
    from identity_metadata import load_identity
    lines = [_ID_HEADER]
    for row in rows:
        doc_id, filename, org, project = row[:4]
        notice_round = row[4] if len(row) > 4 else "0.0"
        lines.append(
            f'"{doc_id}","{filename}","","","","2024000","{notice_round}",'
            f'"{org}","{project}","2024-01-01 10:00:00","2025-01-01 17:00:00","true"')
    p = tmp_path / name
    p.write_text("﻿" + "\n".join(lines) + "\n", encoding="utf-8")
    return load_identity(p)


# 회차만 다른 두 사업 + 회차가 없는 같은 계열 사업
_ROUNDS = [
    ("RFP-000001", "합성원_합성 통합정보 기능개선 사업.md",
     "합성원", "합성 통합정보 기능개선 사업"),
    ("RFP-000002", "합성원_합성 통합정보 기능개선 사업(1차).md",
     "합성원", "합성 통합정보 기능개선 사업(1차)"),
    ("RFP-000003", "합성원_합성 통합정보 기능개선 사업(2차).md",
     "합성원", "합성 통합정보 기능개선 사업(2차)"),
]


class Test회차_판정:
    def test_각_회차는_해당_문서로_연결된다(self, tmp_path):
        from doc_resolver import resolve_document
        index = _identity(tmp_path, _ROUNDS)
        assert resolve_document(
            "합성 통합정보 기능개선 사업(1차) 예산이 얼마야?", index).document_id == "RFP-000002"
        assert resolve_document(
            "합성 통합정보 기능개선 사업(2차) 예산이 얼마야?", index).document_id == "RFP-000003"

    def test_존재하지_않는_회차는_다른_회차_문서로_연결하지_않는다(self, tmp_path):
        from doc_resolver import resolve_document
        index = _identity(tmp_path, _ROUNDS)
        res = resolve_document("합성 통합정보 기능개선 사업(3차) 예산이 얼마야?", index)
        assert res.document_id is None
        assert res.candidates == []

    def test_질문에_회차가_있고_후보_이름에_없으면_확정하지_않는다(self, tmp_path):
        """회차 없는 사업으로 대신 답하면 안 된다 — 이번 오류의 핵심."""
        from doc_resolver import resolve_document
        index = _identity(tmp_path, [_ROUNDS[0]])          # 회차 없는 문서 하나뿐
        assert resolve_document(
            "합성 통합정보 기능개선 사업(7차) 예산이 얼마야?", index).document_id is None

    def test_회차가_없는_정확한_사업명은_기존대로_연결된다(self, tmp_path):
        from doc_resolver import resolve_document
        index = _identity(tmp_path, _ROUNDS)
        assert resolve_document(
            "합성 통합정보 기능개선 사업 예산이 얼마야?", index).document_id == "RFP-000001"

    def test_괄호_없는_회차_표현도_처리한다(self, tmp_path):
        from doc_resolver import resolve_document
        index = _identity(tmp_path, [
            ("RFP-000001", "합성원_합성 통합정보 고도화 2차사업.md",
             "합성원", "합성 통합정보 고도화 2차사업"),
            ("RFP-000002", "합성원_합성 통합정보 고도화 사업.md",
             "합성원", "합성 통합정보 고도화 사업"),
        ])
        for q in ("합성 통합정보 고도화 2차사업 예산이 얼마야?",
                  "합성 통합정보 고도화 제2차 사업 예산이 얼마야?",
                  "합성 통합정보 고도화 2차 사업 예산이 얼마야?"):
            assert resolve_document(q, index).document_id == "RFP-000001", q

    def test_기관명과_회차를_함께_물어도_정상_처리한다(self, tmp_path):
        from doc_resolver import resolve_document
        index = _identity(tmp_path, _ROUNDS)
        assert resolve_document(
            "합성원 합성 통합정보 기능개선 사업(2차) 예산이 얼마야?",
            index).document_id == "RFP-000003"
        # 그 기관에 없는 회차를 물으면 기관만 보고 아무거나 고르지 않는다
        assert resolve_document(
            "합성원 합성 통합정보 기능개선 사업(9차) 예산이 얼마야?",
            index).document_id is None

    def test_회차_필터는_다중_후보_되묻기를_유지한다(self, tmp_path):
        from doc_resolver import resolve_document
        index = _identity(tmp_path, [
            ("RFP-000001", "가기관_합성 통합정보 기능개선 사업(2차).md",
             "가기관", "합성 통합정보 기능개선 사업(2차)"),
            ("RFP-000002", "나기관_합성 통합정보 기능개선 사업(2차).md",
             "나기관", "합성 통합정보 기능개선 사업(2차)"),
        ])
        res = resolve_document("합성 통합정보 기능개선 사업(2차) 예산이 얼마야?", index)
        assert res.document_id is None
        assert set(res.candidates) == {"RFP-000001", "RFP-000002"}

    def test_공고_재공고_차수를_사업_회차로_쓰지_않는다(self, tmp_path):
        """identity_v2의 notice_round 는 공고 차수다 — 사업명 회차와 별개다."""
        from doc_resolver import resolve_document
        index = _identity(tmp_path, [
            ("RFP-000001", "합성원_합성 통합정보 기능개선 사업.md",
             "합성원", "합성 통합정보 기능개선 사업", "3.0"),
        ])
        assert resolve_document(
            "합성 통합정보 기능개선 사업(3차) 예산이 얼마야?", index).document_id is None


class Test회차_오인_방지:
    def test_일반_숫자_표현은_회차가_아니다(self):
        from text_normalize import project_rounds
        for text in ("2개 항목", "7일 이내", "2억원 이상", "문단 2 인용",
                     "3차원 설계 용역", "2차전지 소재", "1차년도 사업비",
                     "1차산업 지원", "왕복 2차로 확장", "1차선 도색", "3차방정식"):
            assert project_rounds(text) == set(), text

    def test_실제_회차_표현은_인식한다(self):
        from text_normalize import project_rounds
        assert project_rounds("기능개선 사업(2차)") == {2}
        assert project_rounds("제2차 사업") == {2}
        assert project_rounds("고도화 2차사업") == {2}
        assert project_rounds("산단 안전정보시스템 1차 구축 용역") == {1}
        assert project_rounds("사업(7차)") == {7}

    def test_일반_숫자가_섞여도_문서_연결이_바뀌지_않는다(self, tmp_path):
        from doc_resolver import resolve_document
        index = _identity(tmp_path, [_ROUNDS[0]])
        for q in ("합성 통합정보 기능개선 사업 예산이 2억원 이상이야?",
                  "합성 통합정보 기능개선 사업 문단 2 인용해줘",
                  "합성 통합정보 기능개선 사업 3차원 자료 있어?"):
            assert resolve_document(q, index).document_id == "RFP-000001", q


class Test기존_수정_재발_방지:
    def test_상투어만_남은_보조키_오확정이_재발하지_않는다(self, tmp_path):
        """오류 A 재발 방지 — 「…」 안 제목 보존 + 상투어 키 색인 제외."""
        from doc_resolver import resolve_document
        index = _identity(tmp_path, [
            ("RFP-000001", "합성원_합성 재난관리시스템 구축 위탁용역.md", "합성원",
             "(긴급)「2024년도 합성 재난관리시스템 구축」 위탁용역"),
        ])
        assert "위탁용역" not in index.by_project_key
        assert resolve_document(
            "2024년도 합성 재난관리시스템 구축 위탁용역 예산 알려줘",
            index).document_id == "RFP-000001"
        assert resolve_document("가상의 위탁용역 예산 알려줘", index).document_id is None

    def test_사업명_안의_기관형_표현_오류가_재발하지_않는다(self, tmp_path):
        """오류 B 재발 방지 — 공식 사업명 구간 안의 기관형 낱말은 미등록이 아니다."""
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
