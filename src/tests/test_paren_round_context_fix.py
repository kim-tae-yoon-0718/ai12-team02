"""괄호형 사업 회차 문맥 판정 회귀 검사 (2026-09-04).

고친 것: `(N차)`를 찾으면 괄호 **안**만 확인하고 괄호 바로 앞뒤 문맥을 보지 않아,
`(2차) 회의에서 언급한 …`·`공고(2차)로 나온 …`처럼 공고·대화 차수까지 사업 회차로
읽었다. 그 결과 정확한 공식 사업명을 적었는데도 정상 문서가 후보에서 사라졌다.

이제 괄호 **바로 앞/뒤의 가장 가까운 낱말**을 보고 판단한다.
  ① 앞 낱말이 사업 문맥이면 인정   ② 뒤 낱말이 사업 문맥이면 인정
  ③ 앞이나 뒤가 차단 낱말이면 제외  ④ 애매하면 쓰지 않는다
⚠️ "앞뒤 N자 안에 금지 낱말이 있으면 제외"하는 넓은 방식은 쓰지 않는다 —
   실제 사업명에 `평가`·`안내` 같은 낱말이 들어 있을 수 있기 때문이다.

⚠️ 합성 자료만 쓴다(공식 파일 미사용, 문항 ID·문서 번호 미사용).
"""
from __future__ import annotations

from pathlib import Path

_ID_HEADER = ('"document_id","source_filename_nfc","collection_system",'
              '"source_record_id","source_url","notice_number","notice_round",'
              '"buyer_org","project_name","notice_date","bid_deadline","metadata_found"')


def _identity(tmp_path: Path, rows, name="identity_paren.csv"):
    from identity_metadata import load_identity
    lines = [_ID_HEADER]
    for doc_id, filename, org, project in rows:
        lines.append(
            f'"{doc_id}","{filename}","","","","2024000","0.0",'
            f'"{org}","{project}","2024-01-01 10:00:00","2025-01-01 17:00:00","true"')
    p = tmp_path / name
    p.write_text("﻿" + "\n".join(lines) + "\n", encoding="utf-8")
    return load_identity(p)


_NAME = "합성 통합정보 기능개선 사업"
_ROWS = [
    ("RFP-000001", "합성원_합성 통합정보 기능개선 사업.md", "합성원", _NAME),
    ("RFP-000002", "합성원_합성 통합정보 기능개선 사업(1차).md", "합성원", _NAME + "(1차)"),
    ("RFP-000003", "합성원_합성 통합정보 기능개선 사업(2차).md", "합성원", _NAME + "(2차)"),
]

# 괄호형 공고·대화 차수 — 사업 회차가 아니다
_BLOCK_PREFIXES = [
    "(2차) 회의에서 언급한", "회의(2차)에서 언급한",
    "(2차) 공고로 나온", "공고(2차)로 나온",
    "(2차) 재공고된", "재공고(2차)로 나온",
    "(2차) 입찰공고에 나온", "(2차) 정정공고의",
    "(2차) 질문에서 말한", "질문(2차)에서 말한",
    "(2차) 답변에서 나온", "(2차) 검토 대상인", "(2차) 평가를 받은",
    "(2차) 심사 대상인", "(2차) 통보된", "(2차) 안내된", "(2차) 요청한",
]


class Test괄호형_차수_오인방지:
    def test_괄호형_공고_대화_차수는_회차로_읽지_않는다(self):
        from text_normalize import question_project_rounds as rounds
        for prefix in _BLOCK_PREFIXES:
            assert rounds(f"{prefix} 사업") == set(), prefix

    def test_괄호형_차수_때문에_정상_문서가_사라지지_않는다(self, tmp_path):
        from doc_resolver import resolve_document
        index = _identity(tmp_path, _ROWS)
        for prefix in _BLOCK_PREFIXES:
            q = f"{prefix} {_NAME}의 예산이 얼마야?"
            assert resolve_document(q, index).document_id == "RFP-000001", q

    def test_괄호가_붙어도_차단_낱말_단독형은_제외된다(self):
        from text_normalize import question_project_rounds as rounds
        for text in ("(2차) 회의", "회의(2차)", "(2차) 공고", "공고(2차)",
                     "(2차) 재공고", "재공고(2차)", "(2차) 입찰공고", "(2차) 정정공고",
                     "(2차) 질문", "질문(2차)", "(2차) 답변", "(2차) 검토",
                     "(2차) 평가", "(2차) 심사", "(2차) 통보", "(2차) 안내", "(2차) 요청"):
            assert rounds(text) == set(), text

    def test_괄호_앞뒤가_충돌하면_앞_문맥을_우선한다(self):
        """`X(N차) Y`에서 N차는 직접 앞의 X를 수식한다."""
        from text_normalize import question_project_rounds as rounds

        # 앞이 공고·대화 문맥이면 뒤에 '사업'이 와도 사업 회차가 아니다.
        for text in ("공고(2차) 사업", "재공고(2차) 사업", "입찰공고(2차) 사업",
                     "정정공고(2차) 사업", "회의(2차) 사업", "질문(2차) 사업",
                     "답변(2차) 사업", "검토(2차) 사업", "평가(2차) 사업"):
            assert rounds(text) == set(), text

        # 앞이 사업 문맥이면 뒤에 회의·공고가 와도 사업 회차다.
        for text in ("사업(2차) 회의", "용역(2차) 공고", "구축(2차) 평가",
                     "고도화(2차) 검토", "기능개선(2차) 보고"):
            assert rounds(text) == {2}, text

    def test_괄호_앞뒤_3x3_판단표를_전부_고정한다(self):
        """앞(사업/차단/없음) × 뒤(사업/차단/없음) 전체 조합."""
        from text_normalize import question_project_rounds as rounds

        cases = {
            "사업(2차) 용역": {2},       # 앞 사업 우선
            "사업(2차) 회의": {2},       # 앞 사업 우선
            "사업(2차)": {2},            # 앞 사업
            "공고(2차) 사업": set(),     # 앞 차단 우선
            "공고(2차) 회의": set(),     # 앞 차단
            "공고(2차)": set(),          # 앞 차단
            "(2차) 사업": {2},           # 앞이 없으면 뒤 사업
            "(2차) 회의": set(),          # 앞이 없으면 뒤 차단
            "(2차)": set(),             # 양쪽 애매 -> 미사용
        }
        for text, expected in cases.items():
            assert rounds(text) == expected, text

    def test_충돌_표현으로_정상_문서가_사라지지_않는다(self, tmp_path):
        """재공고 차수를 사업 회차로 쓰지 않아 정확한 사업명을 유지한다."""
        from doc_resolver import resolve_document
        index = _identity(tmp_path, _ROWS)
        for prefix in ("공고(2차) 사업", "재공고(2차) 사업", "회의(2차) 사업",
                       "질문(2차) 사업"):
            q = f"{prefix}에서 언급한 {_NAME}의 예산이 얼마야?"
            assert resolve_document(q, index).document_id == "RFP-000001", q


class Test괄호형_사업회차_정상인정:
    def test_사업_문맥이_앞이나_뒤에_있으면_인정한다(self):
        from text_normalize import question_project_rounds as rounds
        assert rounds("기능개선 사업(1차)") == {1}
        assert rounds("기능개선 사업 (2차)") == {2}
        assert rounds("용역(2차)") == {2}
        assert rounds("구축(2차)") == {2}
        assert rounds("기능개선(2차)") == {2}
        assert rounds("(2차) 사업") == {2}
        assert rounds("(2차) 구축 사업") == {2}
        assert rounds("고도화 용역(총체 및 1차)") == {1}
        assert rounds("수립(2차) 사업 용역") == {2}

    def test_괄호_없는_기존_형태도_그대로_인정한다(self):
        from text_normalize import question_project_rounds as rounds
        assert rounds("기능개선 제2차 사업") == {2}
        assert rounds("고도화 2차사업") == {2}
        assert rounds("안전정보시스템 1차 구축 용역") == {1}

    def test_괄호형_회차로_해당_문서를_찾는다(self, tmp_path):
        from doc_resolver import resolve_document
        index = _identity(tmp_path, _ROWS)
        assert resolve_document(f"{_NAME}(1차) 예산이 얼마야?", index).document_id == "RFP-000002"
        assert resolve_document(f"{_NAME}(2차) 예산이 얼마야?", index).document_id == "RFP-000003"
        assert resolve_document(f"{_NAME} 예산이 얼마야?", index).document_id == "RFP-000001"
        assert resolve_document(f"{_NAME}(3차) 예산이 얼마야?", index).document_id is None

    def test_사업명에_평가나_안내가_들어_있어도_회차를_잃지_않는다(self, tmp_path):
        """넓은 범위 금지어 검사를 쓰지 않았는지 확인한다."""
        from doc_resolver import resolve_document
        from text_normalize import question_project_rounds as rounds
        index = _identity(tmp_path, [
            ("RFP-000001", "합성원_합성 성과평가 안내시스템 구축 사업(2차).md",
             "합성원", "합성 성과평가 안내시스템 구축 사업(2차)"),
            ("RFP-000002", "합성원_합성 성과평가 안내시스템 구축 사업.md",
             "합성원", "합성 성과평가 안내시스템 구축 사업"),
        ])
        assert rounds("합성 성과평가 안내시스템 구축 사업(2차) 예산") == {2}
        assert resolve_document(
            "합성 성과평가 안내시스템 구축 사업(2차) 예산", index).document_id == "RFP-000001"
        assert resolve_document(
            "합성 성과평가 안내시스템 구축 사업 예산", index).document_id == "RFP-000002"


class Test기존_수정_재발방지:
    def test_비괄호_공고_대화_차수도_계속_제외된다(self):
        from text_normalize import question_project_rounds as rounds
        for text in ("2차 공고", "2차 재공고", "재공고 2차", "2차 입찰공고", "2차 정정공고",
                     "2차 회의", "2차 질문", "2차 답변", "2차 검토", "2차 평가",
                     "2차 공고인 사업", "재공고 2차 사업"):
            assert rounds(text) == set(), text

    def test_일반_숫자_오인_방지는_그대로다(self):
        from text_normalize import question_project_rounds as rounds
        for text in ("3차원 설계", "2차전지 소재", "1차년도 사업비", "1차산업 지원",
                     "왕복 2차로", "1차선 도색", "2개 항목", "7일 이내",
                     "2억원 이상", "문단 2 인용"):
            assert rounds(text) == set(), text

    def test_공식_사업명용_추출은_바뀌지_않았다(self):
        from text_normalize import project_rounds
        assert project_rounds("기능개선 사업(2차)") == {2}
        assert project_rounds("고도화 2차사업") == {2}
        assert project_rounds("안전정보시스템 1차 구축 용역") == {1}
        assert project_rounds("고도화 용역(총체 및 1차)") == {1}

    def test_상투어_보조키와_사업명_안_기관형_표현_오류가_재발하지_않는다(self, tmp_path):
        from doc_resolver import detect_unknown_orgs, resolve_document
        index = _identity(tmp_path, [
            ("RFP-000001", "합성원_합성 재난관리시스템 구축 위탁용역.md", "합성원",
             "(긴급)「2024년도 합성 재난관리시스템 구축」 위탁용역"),
            ("RFP-000002", "합성보호원_합성-NAVI 해외합성지원센터 사업관리 시스템 기능개선.md",
             "합성보호원", "합성-NAVI 해외합성지원센터 사업관리 시스템 기능개선"),
        ])
        assert "위탁용역" not in index.by_project_key
        assert resolve_document("가상의 위탁용역 예산 알려줘", index).document_id is None
        assert resolve_document(
            "2024년도 합성 재난관리시스템 구축 위탁용역 예산", index).document_id == "RFP-000001"
        q = "합성-NAVI 해외합성지원센터 사업관리 시스템 기능개선 사업의 예산 알려줘"
        assert detect_unknown_orgs(q, index) == []
        assert resolve_document(q, index).document_id == "RFP-000002"
