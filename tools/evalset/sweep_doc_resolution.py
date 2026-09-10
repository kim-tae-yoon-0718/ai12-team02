"""§9 공식 100문서 전수 문서 특정 점검 — 수정 전후 비교용.

각 문서마다 다음 질문 형태를 만들어 문서 특정 결과를 기록한다.
  ① 기관명만            ② 정식 사업명        ③ 파일명 별칭
  ④ 괄호 제거형          ⑤ 조사 차이          ⑥ 공백 차이
  ⑦ 안전한 축약형(핵심 낱말만, 순서 유지)    ⑧ 기관+축약형
그리고 가짜 이름 3종(존재하지 않는 사업·다른 기관 혼합·일반어만)을 함께 넣는다.
공식 자료는 읽기만 한다.

사용: python tools/evalset/sweep_doc_resolution.py --out <결과.json> [--repo-root R] [--data-root D]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import add_path_args, data_root, official_paths, repo_root  # noqa: E402

GENERIC = {"사업", "용역", "구축", "고도화", "개발", "운영", "시스템", "공고", "및", "등",
           "위한", "관련", "지원", "관리", "정보화", "서비스", "사업의", "기타"}


def _strip_brackets(text: str) -> str:
    return re.sub(r"[\[(【][^\])】]*[\])】]", " ", text)


def _core_words(title: str, limit: int = 3) -> list[str]:
    words = [w for w in re.findall(r"[가-힣A-Za-z0-9]+", _strip_brackets(title))
             if len(w) >= 2 and w not in GENERIC]
    return words[:limit]


def _questions(rec) -> list[tuple[str, str]]:
    org, title = rec.buyer_org, rec.project_name
    alias = getattr(rec, "filename_project_alias", "") or ""
    nb = _strip_brackets(title).strip()
    nospace = re.sub(r"\s+", "", nb)
    core = _core_words(title)
    qs = [
        ("기관명", f"{org} 사업 예산 알려줘"),
        ("정식 사업명", f"{title} 예산 알려줘"),
        ("괄호 제거형", f"{nb} 예산 알려줘"),
        ("조사 차이", f"{nb}의 예산은 얼마야?"),
        ("공백 차이", f"{nospace} 예산 알려줘"),
    ]
    if alias:
        qs.append(("파일명 별칭", f"{alias} 예산 알려줘"))
    if len(core) >= 2:
        qs.append(("축약형", f"{' '.join(core)} 사업 예산 알려줘"))
        qs.append(("기관+축약형", f"{org} {' '.join(core)} 사업 예산 알려줘"))
    return qs


def _fake_questions(rec) -> list[tuple[str, str]]:
    org = rec.buyer_org
    return [
        ("가짜 사업명", f"{org} 화성기지 관제시스템 구축 사업 예산 알려줘"),
        ("일반어만", "시스템 구축 사업 예산 알려줘"),
    ]


def main(out: Path, repo: Path, srv: Path) -> int:
    sys.path.insert(0, str(repo / "src" / "rag"))
    from doc_resolver import resolve_document          # noqa: E402
    from identity_metadata import load_identity        # noqa: E402
    index = load_identity(official_paths(srv)["identity"])
    rows = []
    for doc_id in sorted(index.document_ids()):
        rec = index.get(doc_id)
        for kind, q in _questions(rec):
            r = resolve_document(q, index)
            rows.append({"document": doc_id, "kind": kind, "question": q,
                         "resolved": r.document_id, "method": r.method,
                         "n_candidates": len(r.candidates),
                         "outcome": ("정확" if r.document_id == doc_id else
                                     "오확정" if r.document_id else
                                     "되묻기" if len(r.candidates) > 1 else "미확정")})
        for kind, q in _fake_questions(rec):
            r = resolve_document(q, index)
            rows.append({"document": doc_id, "kind": kind, "question": q,
                         "resolved": r.document_id, "method": r.method,
                         "n_candidates": len(r.candidates),
                         "outcome": ("가짜 오확정" if r.document_id else "정확히 거부")})
    summary = {"n_documents": len(index.document_ids()), "n_questions": len(rows)}
    for key in ("정확", "오확정", "되묻기", "미확정", "가짜 오확정", "정확히 거부"):
        summary[key] = sum(1 for r in rows if r["outcome"] == key)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="공식 100문서 문서 특정 전수 점검")
    ap.add_argument("--out", required=True)
    add_path_args(ap)
    a = ap.parse_args()
    sys.exit(main(Path(a.out), repo_root(a.repo_root), data_root(a.data_root)))
