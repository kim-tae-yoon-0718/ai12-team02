"""
4-10-2 — 선별형 마감 필터용 메타데이터 로더.
공식 data_list.csv(UTF-8 BOM, 12컬럼, 100행)를 document_registry_v2.json과
연결해 document_id -> 마감일 매핑을 만든다.

⚠️ 이 데이터는 청킹·임베딩 대상이 아니다(4-9-8 확정: "숫자·날짜·부정 조건은
벡터 검색이 아니라 검증된 구조화 질의"). document_id로 바로 조회하는
구조화 조회 대상이라 G-2와 같은 성격 — 그냥 로더 하나면 된다.

연결 키: CSV의 '파일명'(확장자 .hwp 등)과 registry의 'output_filename'
(확장자 .md)이 확장자만 빼면 동일하다(100/100 매핑 확인 완료, NFC 정규화 필요
— 체크리스트의 "파일명은 NFD 저장" 경고와 일치).

deadline_filter_field(base.yaml, "입찰 참여 마감일") 컬럼명은 코드에 다시
하드코딩하지 않고 cfg에서 그대로 가져온다 — 컬럼명이 바뀌면 base.yaml만
고치면 된다.
"""
from __future__ import annotations
import csv
import json
import os
import re
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any


def _stem_nfc(filename: str) -> str:
    """확장자를 떼고 NFC로 정규화 — CSV·registry 파일명을 같은 기준으로 비교."""
    return unicodedata.normalize("NFC", os.path.splitext(filename)[0])


def _load_csv_rows_by_document_id(csv_path: Path, registry_path: Path) -> dict[str, dict]:
    """document_id -> 원본 CSV 행(딕셔너리) 매핑. 마감일·기관명 등 여러 용도의
    공용 조인 — CSV·registry 파싱을 여기서만 한다(같은 로직 두 번 안 짬)."""
    with open(registry_path, "r", encoding="utf-8") as f:
        registry_doc = json.load(f)
    if "documents" not in registry_doc:
        raise ValueError(f"{registry_path}: 'documents' 키가 없습니다.")
    stem_to_doc_id = {
        _stem_nfc(row["output_filename"]): row["document_id"]
        for row in registry_doc["documents"]
    }

    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    result: dict[str, dict] = {}
    unmatched = 0
    for row in rows:
        doc_id = stem_to_doc_id.get(_stem_nfc(row["파일명"]))
        if doc_id is None:
            unmatched += 1
            continue
        result[doc_id] = row
    if unmatched:
        print(f"⚠️  CSV {unmatched}행이 registry 문서와 매핑되지 않았습니다.")
    return result


def load_deadline_by_document_id(
    csv_path: Path, registry_path: Path, cfg: dict[str, Any],
) -> dict[str, datetime | None]:
    """document_id -> 마감일(datetime) 매핑. 마감일이 빈 값(미상)이면 None.
    CSV·registry 어느 한쪽에도 없는 문서는 매핑에서 아예 빠진다(호출측이
    '미상'과 '매핑 자체가 없음'을 구분할 수 있게)."""
    deadline_field = cfg.get("deadline_filter_field")
    if not deadline_field:
        raise RuntimeError(
            "base.yaml에 deadline_filter_field가 비어 있습니다. "
            "CSV의 실제 컬럼명을 채워야 합니다."
        )

    rows_by_doc = _load_csv_rows_by_document_id(csv_path, registry_path)
    if rows_by_doc and deadline_field not in next(iter(rows_by_doc.values())):
        raise KeyError(f"{csv_path}: 컬럼 '{deadline_field}'가 없습니다.")

    mapping: dict[str, datetime | None] = {}
    for doc_id, row in rows_by_doc.items():
        raw = (row.get(deadline_field) or "").strip()
        if not raw:
            mapping[doc_id] = None  # 마감일 미상 — deadline_missing_policy로 처리
            continue
        try:
            mapping[doc_id] = datetime.strptime(raw, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            # 형식이 다른 값이 섞여 있으면 조용히 무시하지 않고 미상으로 남기되 알린다
            print(f"⚠️  {doc_id}: 마감일 형식을 못 읽었습니다({raw!r}) — 미상으로 처리")
            mapping[doc_id] = None

    return mapping


def load_org_index(csv_path: Path, registry_path: Path) -> dict[str, list[str]]:
    """발주 기관 이름(원문 그대로) -> document_id 목록. org_only 질문
    ("발주기관명만으로 묻는 질문")에서 문서를 특정하는 데 씀. 청킹·임베딩
    대상 아님 — CSV를 그대로 읽는 구조화 조회."""
    rows_by_doc = _load_csv_rows_by_document_id(csv_path, registry_path)
    index: dict[str, list[str]] = {}
    for doc_id, row in rows_by_doc.items():
        org = (row.get("발주 기관") or "").strip()
        if org:
            index.setdefault(org, []).append(doc_id)
    return index


def _filter_subsumed_org_names(matched_orgs: list[str]) -> list[str]:
    """긴 기관명에 포함되는 짧은 기관명은 배제한다. 예: 질문에 "서울특별시
    교육청"만 썼는데 "서울특별시"(다른 문서)까지 부분 문자열로 같이
    매칭되는 걸 막는다 — 체크리스트가 경고한 "부분일치 허용 시 실패"의
    구체 사례가 실제 데이터로 재현됐다(서울특별시 vs 서울특별시교육청)."""
    return [
        org for org in matched_orgs
        if not any(other != org and org in other for other in matched_orgs)
    ]


def match_org_names(question: str, org_index: dict[str, list[str]]) -> list[str]:
    """질문에 부분 문자열로 들어있는 기관명을 감지 — 서로 포함 관계인
    기관명은 긴 쪽만 남긴다(_filter_subsumed_org_names)."""
    matched = [org for org in org_index if org and org in question]
    return _filter_subsumed_org_names(matched)


_ORG_SUFFIX_RE = re.compile(
    r"[가-힣]{2,20}(?:재단|공사|공단|진흥원|연구원|협회|위원회|청|처|공단|센터|대학교|시청|도청)"
)


def detect_unknown_org(question: str, org_index: dict[str, list[str]]) -> str | None:
    """질문에서 기관명처럼 생긴 표현(재단/청/공사/진흥원 등으로 끝나는 고유
    명사)을 뽑아서, 그게 실제 org_index(공식 CSV 기관명)에 하나도 없으면
    그 표현을 반환한다 — 존재하지 않는 사업을 물어볼 때(QA-004류) 검색·
    생성을 태우기 전에 걸러내는 용도. 유사도 임계값 대신 구조화된 데이터
    (실제 발주기관 목록)로 판단한다(4-9-8 원칙 — 검증된 데이터 우선).
    기관명 표현 자체가 없으면(일반 질문) None — 이 경우는 정상적으로
    검색 경로를 탄다."""
    candidates = _ORG_SUFFIX_RE.findall(question)
    if not candidates:
        return None
    known_orgs = set(org_index.keys())
    for candidate in candidates:
        # 후보가 알려진 기관명의 부분 문자열이거나, 알려진 기관명이 후보의
        # 부분 문자열이면(표기 경계 차이) 알려진 기관으로 인정한다.
        if any(candidate in org or org in candidate for org in known_orgs):
            return None
    return candidates[0]


def find_documents_by_org_mention(question: str, org_index: dict[str, list[str]]) -> list[str]:
    """질문에 명시된 기관명이 부분 문자열로 들어있으면 매칭되는 document_id를
    전부 반환(중복 제거, 등장 순서 유지). 여러 문서가 매칭되면 호출측이
    모호성을 처리해야 한다 — 여기서 임의로 하나를 고르지 않는다."""
    matched: list[str] = []
    for org in match_org_names(question, org_index):
        for d in org_index[org]:
            if d not in matched:
                matched.append(d)
    return matched


def format_deadline_answer(
    document_id: str, deadline_map: dict[str, datetime | None],
) -> tuple[str, bool]:
    """추출형 "마감일이 언제야?" 질문용 — (답변 문구, 기권 여부) 반환.
    선별형 필터(is_before_deadline)와 달리 여기는 값 자체를 그대로 보여주는
    용도라 별도 함수로 둔다."""
    if document_id not in deadline_map:
        return f"{document_id}의 마감일 정보를 찾을 수 없습니다(CSV 매핑 실패).", True
    deadline = deadline_map[document_id]
    if deadline is None:
        return f"{document_id}의 마감일은 원문에 명시돼 있지 않습니다(미상).", True
    return f"{document_id}의 입찰 참여 마감일: {deadline.strftime('%Y-%m-%d %H:%M')}", False


def is_before_deadline(
    document_id: str, deadline_map: dict[str, datetime | None],
    reference_datetime: datetime, missing_policy: str = "show_as_unknown",
) -> tuple[bool | None, str]:
    """(마감 전인지, 상태 문구) 반환. 마감 전이면 True, 지났으면 False,
    미상이거나 매핑 자체가 없으면 None + 안내 문구(4-10-2 확정: 미상은
    제외하지 않고 '미상'으로 표시하고 통과시킨다 — 호출측이 그대로 노출)."""
    if document_id not in deadline_map:
        return None, "마감일 정보 없음(문서-CSV 매핑 실패)"
    deadline = deadline_map[document_id]
    if deadline is None:
        return None, "마감일 미상"
    return (deadline >= reference_datetime), ""
