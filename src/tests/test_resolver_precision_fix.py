"""문서 특정 정밀도 회귀 검사 (2026-09-03 2차).

이번에 고친 것
  A. 사업 제목을 감싸는 「…」 안의 제목까지 지워져 "위탁용역" 같은 상투어만 보조 키로
     남았고, 그 결과 존재하지 않는 사업명도 한 문서로 확신 있게 확정됐다.
  B. 공식 사업명 **안에** 있는 기관형 표현("… 해외지식재산센터 …")을 별도의 미등록
     기관으로 오인해, 정확한 공식 사업명을 적은 질문까지 차단했다.
  C. (ZIP 제출 이후 추가분) 보조 키 대조가 여러 후보를 한 건으로 줄일 때 확정하지 않기,
     생성 프롬프트 이름이 비면 조용히 옛 버전으로 되돌아가지 않기.

⚠️ 여기 쓰는 문서·사업명·기관명은 전부 이 파일 안에서 만든 합성 자료다.
   공식 파일을 읽지 않고, 평가 문항 ID나 특정 문서 번호를 외우지 않는다.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

_ID_HEADER = ('"document_id","source_filename_nfc","collection_system",'
              '"source_record_id","source_url","notice_number","notice_round",'
              '"buyer_org","project_name","notice_date","bid_deadline","metadata_found"')


def _identity(tmp_path: Path, rows, name="identity_precision.csv"):
    """(document_id, source_filename_nfc, buyer_org, project_name) 네 칸만 받는다."""
    from identity_metadata import load_identity
    lines = [_ID_HEADER]
    for doc_id, filename, org, project in rows:
        lines.append(
            f'"{doc_id}","{filename}","","","","2024000","0.0",'
            f'"{org}","{project}","2024-01-01 10:00:00","2025-01-01 17:00:00","true"')
    p = tmp_path / name
    p.write_text("﻿" + "\n".join(lines) + "\n", encoding="utf-8")
    return load_identity(p)


# ===========================================================================
# A. 제목 인용 괄호 보존 + 상투어 보조 키 차단
# ===========================================================================

class Test제목괄호와_상투어:
    def test_제목을_감싸는_인용괄호_안의_제목은_보존한다(self):
        """「…」·『…』·【…】 안은 제목 자체다 — 지우면 상투어만 남는다."""
        from text_normalize import strip_bracketed
        assert strip_bracketed("(긴급)「2024년도 합성 재난관리시스템 구축」 위탁용역") == \
            "2024년도 합성 재난관리시스템 구축 위탁용역"
        assert strip_bracketed("『합성 통합포털 재구축』 용역") == "합성 통합포털 재구축 용역"
        assert strip_bracketed("【합성 데이터 플랫폼 고도화】") == "합성 데이터 플랫폼 고도화"

    def test_절차_회차_주석_제거는_그대로_동작한다(self):
        """(긴급)·(협상)·(용역)·[협상에 의한 계약] 같은 진짜 주석은 계속 제거한다."""
        from text_normalize import strip_bracketed
        assert strip_bracketed("합성군 재난통합관리시스템 고도화 사업(협상)(긴급)") == \
            "합성군 재난통합관리시스템 고도화 사업"
        assert strip_bracketed("합성공원 회원 통합운영 시스템 구축[협상에 의한 계약]") == \
            "합성공원 회원 통합운영 시스템 구축"
        assert strip_bracketed("합성철도공사 (용역)") == "합성철도공사"
        assert strip_bracketed("합성 버스정보시스템(BIS) 구축사업") == \
            "합성 버스정보시스템 구축사업"

    def test_조달_상투어만_남은_키는_일반_표현으로_본다(self):
        """여러 문서에 나올 수 있는 표현만으로는 문서를 특정하지 못한다."""
        from text_normalize import is_generic_project_key as generic
        for key in ("위탁용역", "구축사업", "개선사업", "고도화사업", "용역사업", "재공고"):
            assert generic(key), key
        # 분야 낱말이 섞인 실제 사업명은 일반 표현이 아니다
        for key in ("통합관리시스템구축", "합성재난관리시스템구축위탁용역",
                    "건설통합시스템고도화", "비교과시스템개발"):
            assert not generic(key), key

    def test_상투어만_남은_보조키는_색인에_넣지_않는다(self, tmp_path):
        index = _identity(tmp_path, [
            ("RFP-000001", "합성원_합성 재난관리시스템 구축 위탁용역.md", "합성원",
             "(긴급)「2024년도 합성 재난관리시스템 구축」 위탁용역"),
        ])
        rec = index.get("RFP-000001")
        # 주석을 뗀 보조 키가 제목을 담고 있어야 한다(상투어 네 글자가 아니라)
        assert rec.project_key_unbracketed == "2024년도합성재난관리시스템구축위탁용역"
        assert "위탁용역" not in index.by_project_key
        for table in (index.by_project_key, index.by_project_core_key,
                      index.by_project_alias_key, index.by_project_folded_key):
            assert "위탁용역" not in table
            assert "구축사업" not in table

    def test_상투어만_적은_질문은_문서를_확정하지_않는다(self, tmp_path):
        from doc_resolver import resolve_document
        index = _identity(tmp_path, [
            ("RFP-000001", "합성원_합성 재난관리시스템 구축 위탁용역.md", "합성원",
             "(긴급)「2024년도 합성 재난관리시스템 구축」 위탁용역"),
            ("RFP-000002", "다른원_전혀 다른 사업.md", "다른원", "전혀 다른 사업"),
        ])
        # 정확한 공식 사업명은 그대로 찾아야 한다
        assert resolve_document(
            "(긴급)「2024년도 합성 재난관리시스템 구축」 위탁용역 예산 알려줘",
            index).document_id == "RFP-000001"
        assert resolve_document(
            "2024년도 합성 재난관리시스템 구축 위탁용역 예산 알려줘",
            index).document_id == "RFP-000001"
        # 상투어만 같은 질문은 확정하면 안 된다
        for q in ("가상의 위탁용역 예산 알려줘",
                  "전혀 다른 이름의 도서관 열람실 개보수 위탁용역 예산 알려줘",
                  "구축사업 예산 알려줘"):
            assert resolve_document(q, index).document_id is None, q

    def test_핵심어나_연도가_다르면_상투어_꼬리말이_같아도_확정하지_않는다(self, tmp_path):
        from doc_resolver import resolve_document
        index = _identity(tmp_path, [
            ("RFP-000001", "합성원_합성 재난관리시스템 구축 위탁용역.md", "합성원",
             "(긴급)「2024년도 합성 재난관리시스템 구축」 위탁용역"),
        ])
        for q in ("2099년도 합성 재난관리시스템 구축 위탁용역 예산 알려줘",
                  "2024년도 합성 재난관리장비 구축 위탁용역 예산 알려줘"):
            assert resolve_document(q, index).document_id is None, q


# ===========================================================================
# B. 공식 사업명 안의 기관형 표현
# ===========================================================================

class Test사업명_안의_기관형_표현:
    ROWS = [
        ("RFP-000001", "합성보호원_합성-NAVI 해외합성지원센터 사업관리 시스템 기능개선.md",
         "합성보호원", "합성-NAVI 해외합성지원센터 사업관리 시스템 기능개선"),
        ("RFP-000002", "다른기관_전혀 다른 사업.md", "다른기관", "전혀 다른 사업"),
    ]

    def test_공식_사업명_안의_기관형_표현은_미등록기관이_아니다(self, tmp_path):
        from doc_resolver import detect_unknown_orgs, resolve_document
        index = _identity(tmp_path, self.ROWS)
        q = "합성-NAVI 해외합성지원센터 사업관리 시스템 기능개선 사업의 예산 알려줘"
        assert detect_unknown_orgs(q, index) == []
        assert resolve_document(q, index).document_id == "RFP-000001"

    def test_띄어쓰기만_다른_공식_사업명도_연결된다(self, tmp_path):
        from doc_resolver import resolve_document
        index = _identity(tmp_path, self.ROWS)
        for q in ("합성-NAVI  해외합성지원센터  사업관리  시스템  기능개선 사업의 예산 알려줘",
                  "합성-NAVI 해외합성지원센터 사업관리시스템 기능개선 사업의 예산 알려줘"):
            assert resolve_document(q, index).document_id == "RFP-000001", q

    def test_공식_사업명에_없는_가짜_기관은_문서로_확정하지_않는다(self, tmp_path):
        from doc_resolver import detect_unknown_orgs, resolve_document
        index = _identity(tmp_path, self.ROWS)
        q = "가짜해외합성지원센터 사업관리 시스템 기능개선 사업의 예산 알려줘"
        assert detect_unknown_orgs(q, index) == ["가짜해외합성지원센터"]
        assert resolve_document(q, index).document_id is None

    def test_사업명_밖의_미등록_기관은_조용히_무시하지_않는다(self, tmp_path):
        from doc_resolver import detect_unknown_orgs, resolve_document
        index = _identity(tmp_path, self.ROWS)
        q = ("합성-NAVI 해외합성지원센터 사업관리 시스템 기능개선 사업이랑 "
             "가온재단 사업 예산 비교해줘")
        assert detect_unknown_orgs(q, index) == ["가온재단"]
        res = resolve_document(q, index)
        assert res.document_id is None
        assert res.unknown_orgs == ["가온재단"]

    def test_공식_기관명은_여전히_자기_문서로_연결된다(self, tmp_path):
        """미등록 기관 차단을 느슨하게 만들지 않았는지 확인한다."""
        from doc_resolver import resolve_document
        index = _identity(tmp_path, self.ROWS)
        assert resolve_document("합성보호원이 낸 사업의 예산 알려줘",
                                index).document_id == "RFP-000001"
        assert resolve_document("다른기관이 낸 사업의 예산 알려줘",
                                index).document_id == "RFP-000002"


# ===========================================================================
# C. ZIP 제출 이후 추가된 변경 검증
# ===========================================================================

class Test보조키_대조_보수화:
    def test_후보를_한_건으로_줄여도_확정하지_않고_되묻는다(self, tmp_path):
        """제목 낱말 대조는 후보를 **줄이기만** 한다 — 줄여서 만든 단일 후보는 확정하지 않는다."""
        from doc_resolver import resolve_document
        index = _identity(tmp_path, [
            # 질문의 별칭과 겹치지만 제목 낱말이 안 맞는 문서
            ("RFP-000001", "가기관_합성박물관 전시해설 시스템 구축(협상에.md",
             "가기관", "합성박물관 전시해설 시스템 구축(협상에 의한 계약)"),
            # 상투어가 아닌 고유 꼬리말로 함께 걸리는 다른 문서
            ("RFP-000002", "나기관_합성 교육위탁 운영지원 체계.md",
             "나기관", "합성 교육위탁 운영지원 체계"),
        ])
        res = resolve_document(
            "합성박물관 전시해설 시스템 구축(협상에 합성 교육위탁 운영지원 체계 예산 알려줘",
            index)
        assert res.document_id is None
        assert set(res.candidates) == {"RFP-000001", "RFP-000002"}

    def test_충돌_없는_고유_별칭은_정상_확정된다(self, tmp_path):
        from doc_resolver import resolve_document
        index = _identity(tmp_path, [
            ("RFP-000001", "합성공사_합성근린공원 다목적구장 홈페이지 및 회원 통합운영.md",
             "합성공사", "합성근린공원 다목적구장 홈페이지 및 회원 통합운영 관리 시스템 구축"),
            ("RFP-000002", "다른공사_전혀 다른 사업.md", "다른공사", "전혀 다른 사업"),
        ])
        assert resolve_document(
            "합성근린공원 다목적구장 홈페이지 및 회원 통합운영 사업의 마감일 알려줘",
            index).document_id == "RFP-000001"

    def test_여러_후보가_실제로_남으면_되묻기를_유지한다(self, tmp_path):
        from doc_resolver import resolve_document
        index = _identity(tmp_path, [
            ("RFP-000001", "가기관_합성 통합회원 관리체계 구축.md",
             "가기관", "합성 통합회원 관리체계 구축"),
            ("RFP-000002", "나기관_합성 통합회원 관리체계 구축.md",
             "나기관", "합성 통합회원 관리체계 구축"),
        ])
        res = resolve_document("합성 통합회원 관리체계 구축 사업의 예산 알려줘", index)
        assert res.document_id is None
        assert set(res.candidates) == {"RFP-000001", "RFP-000002"}


class Test생성프롬프트_되돌림_차단:
    def test_프롬프트_이름이_비면_조용히_옛_버전으로_돌아가지_않는다(self):
        from generation_client import _load_system_prompt_template
        for empty in (None, ""):
            with pytest.raises(RuntimeError, match="prompt_generate"):
                _load_system_prompt_template(empty)

    def test_정상_설정에서는_v3가_실제로_로딩된다(self):
        import yaml
        from generation_client import _load_system_prompt_template
        root = Path(__file__).resolve().parent.parent.parent
        cfg = yaml.safe_load((root / "config" / "base.yaml").read_text(encoding="utf-8"))
        loaded = _load_system_prompt_template(cfg["prompt_generate"])
        assert loaded == (root / "src" / "prompts" / "generate_v3.txt").read_text(
            encoding="utf-8")

    def test_설정_오류는_API_호출_전에_드러난다(self, base_cfg):
        """GenerationClient 생성 시점(=첫 API 호출 전)에 실패해야 한다."""
        from generation_client import GenerationClient
        cfg = dict(base_cfg)
        cfg.pop("prompt_generate", None)
        with pytest.raises(RuntimeError, match="prompt_generate"):
            GenerationClient(cfg)

    def test_클라이언트가_실제_로딩한_프롬프트_파일명을_기록한다(self, base_cfg):
        from generation_client import GenerationClient
        cfg = dict(base_cfg)
        cfg["prompt_generate"] = "generate_v3.txt"
        client = GenerationClient(cfg)
        assert client.prompt_file == "generate_v3.txt"
        assert "형식 변환 규칙" in client._system_prompt_template

    def test_평가_산출물이_설정값과_로딩값을_함께_남긴다(self):
        """summary.json에 두 값이 기록되어야 설정과 실행이 어긋난 것을 알 수 있다."""
        import inspect
        import run_eval
        src = inspect.getsource(run_eval.main)
        assert '"prompt_generate_configured"' in src
        assert '"prompt_generate_loaded"' in src
