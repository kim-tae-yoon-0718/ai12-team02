"""
인덱스 생성 진입점 (팀 규약 "공통 실행 진입점" 중 하나 — "인덱스 만들기").
    청크 읽기 → 임베딩 → 저장소 생성

박예진님 C단계(청킹) 실제 산출물 필드명은 애초 이 파일이 기대하던 이름과
다르다 — 실제 chunks.jsonl 기준으로 매핑한다 (JSONL 자체는 맞음, 한 줄=청크 1개):
{
  "chunk_id": "...", "document_id": "RFP-000001", "document_version": "...",
  "processed_sha256": "...", "sidecar_sha256": "...", "corpus_version": "v2",
  "search_text": "검색용 본문 (문서명·장절이 이미 앞에 붙어 있음)",  ← text
  "source_document_title": "...",                                  ← document_name
  "section_path": ["1. 추진배경 및 방향"],                          ← chapter(join)
  "block_type": "text" | "table",                                  ← chunk_type
  "table_idx": null, "row_start": null, "row_end": null,
  "part": null, "of": null,
  "table_degraded": false, "oversize": false,
  "retrieval_eligible": true   ← 청크 단위. 문서 단위(레지스트리)와 별개로 적용
}

사용 예:
  export RAG_ROOT=/srv/rfp
  export OPENAI_API_KEY=...
  python build_index.py --chunks /path/to/chunks.jsonl --registry /path/to/document_registry_v2.json
  python build_index.py --chunks /path/to/chunks.jsonl --registry /path/to/document_registry_v2.json \
      --only RFP-000001,RFP-000037   # 문서 단위 부분 갱신 — 지정 문서만 재색인
"""
from __future__ import annotations
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# scripts/build_index.py 기준 ../rag를 sys.path에 추가 — 라이브러리 모듈은
# src/rag/에, 진입점은 src/scripts/에 나뉘어 있는 구조 대응.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "rag"))

from config import load_config, index_dir, chunks_version_file
from git_info import get_git_info, warn_if_dirty
from embedding_client import EmbeddingClient
from vector_store import (VectorStore, ChunkMetadata, IndexTag, config_mismatch_check,
                          ConfigMismatchError, validate_index_tag, IndexTagError)


def load_chunks(chunks_path: Path) -> list[dict]:
    """⚠️ 정정(리뷰 반영): 예전엔 파일 끝만 잘린 경우 경고 후 스킵하고
    계속 진행했는데, 이건 조용한 데이터 손실이었다(에러 없이 문서가
    빠짐). baseline 단계라 문제를 숨기지 않고 바로 드러내는 게 맞다 —
    잘린 파일은 무조건 에러로 막는다. 원본을 다시 받아야 한다."""
    chunks = []
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                chunks.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"{chunks_path}:{line_no} JSON 파싱 실패 — 파일이 중간에 잘렸을 "
                    f"수 있습니다(업로드/생성 도중 truncate). 원본을 다시 확인하세요. "
                    f"원인: {e}"
                ) from e
    return chunks


def map_chunk(c: dict) -> dict:
    """실제 청크 필드명 → ChunkMetadata가 기대하는 이름으로 매핑.
    필수 키가 없으면 KeyError로 바로 죽는다(조용히 빈 문자열로 메꾸지 않음).
    ⚠️ 정정(리뷰 반영): document_version/processed_sha256/sidecar_sha256도
    필수로 바꿨다 — 예전엔 없으면 빈 문자열로 조용히 채웠는데, 이건 인덱스
    재현성·버전 추적을 깨는 조용한 데이터 손실이었다."""
    required = (
        "chunk_id", "document_id", "search_text", "block_type",
        "document_version", "processed_sha256", "sidecar_sha256",
    )
    for field in required:
        if field not in c or c[field] in (None, ""):
            raise KeyError(
                f"청크에 필수 필드 '{field}'가 없습니다: {c.get('chunk_id', '(chunk_id도 없음)')}"
            )
    section_path = c.get("section_path") or []
    return {
        "chunk_id": c["chunk_id"],
        "document_id": c["document_id"],
        "document_version": c["document_version"],
        "processed_sha256": c["processed_sha256"],
        "sidecar_sha256": c["sidecar_sha256"],
        "corpus_version": c.get("corpus_version", ""),
        "chunking_version": c.get("chunking_version", ""),
        "text": c["search_text"],
        "document_name": c.get("source_document_title", ""),
        "chapter": " > ".join(section_path) if section_path else "",
        "chunk_type": c["block_type"],
        "table_idx": c.get("table_idx"),
        "row_start": c.get("row_start"),
        "row_end": c.get("row_end"),
        "part": c.get("part"),
        "of": c.get("of"),
        "table_degraded": bool(c.get("table_degraded", False)),
        "oversize": bool(c.get("oversize", False)),
        "chunk_retrieval_eligible": bool(c.get("retrieval_eligible", True)),
        "location_label": c.get("location_label", ""),
        "section_path": section_path,
        "section_paths": c.get("section_paths") or [],
        "md_line_start": c.get("md_line_start"),
        "md_line_end": c.get("md_line_end"),
        "preprocess_version": c.get("preprocess_version", ""),
    }


def validate_chunk_consistency(chunks: list[dict]) -> None:
    """⚠️ 버그 수정(리뷰 반영, 문제2): base.yaml 확정 규칙 "같은 문서 안에
    서로 다른 문서 버전이나 파일 지문이 섞이면 인덱싱을 중단한다"가 실제로는
    "값이 있는지"만 확인하고 "문서 내에서 서로 같은 값인지"는 확인 안 됐다
    — 예: RFP-000001의 청크 절반이 document_version=1, 나머지가 =2여도
    통과할 수 있었다. 같은 document_id 안에서 아래 값이 전부 같은지 검사."""
    check_keys = [
        "document_version", "processed_sha256", "sidecar_sha256",
        "corpus_version", "chunking_version",
    ]
    by_doc: dict[str, list[dict]] = {}
    for c in chunks:
        by_doc.setdefault(c["document_id"], []).append(c)

    for doc_id, doc_chunks in by_doc.items():
        for key in check_keys:
            values = {c.get(key) for c in doc_chunks}
            if len(values) > 1:
                raise ValueError(
                    f"{doc_id}: 청크마다 '{key}' 값이 다릅니다({values}) — "
                    f"같은 문서 안에 서로 다른 버전이 섞여 있습니다. 인덱싱을 중단합니다."
                )


class ChunkContractError(ValueError):
    pass


_VERSION_TXT_KEYS = {
    "chunking version": "chunking_version",
    "corpus version": "corpus_version",
    "preprocess version": "preprocess_version",
    "table version": "extraction_version",
    "문서 수": "document_count",
    "청크 수": "chunk_count",
}


def parse_chunks_version_file(path: Path) -> dict[str, str]:
    """chunks_vN/VERSION.txt를 읽어 공식 선언 버전을 뽑는다(교차 확인용)."""
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip().lstrip("#").strip()
        mapped = _VERSION_TXT_KEYS.get(key)
        if mapped:
            out[mapped] = value.strip()
    return out


def validate_chunk_versions(
    chunks: list[dict], cfg: dict, version_file: Path | None = None,
) -> dict[str, str]:
    """청크 버전 계약 검사 (4-2 확정).

    - corpus_version / chunking_version이 없으면 실패(빈 값도 실패)
    - 설정(base.yaml)이 요구하는 버전과 다르면 실패
    - 청크끼리 값이 같다는 이유만으로 통과시키지 않는다 — 공식 메타데이터
      (chunks_vN/VERSION.txt)와도 교차 확인한다
    """
    required = {
        "corpus_version": cfg.get("corpus"),
        "chunking_version": cfg.get("chunking_version"),
    }
    optional = {"preprocess_version": cfg.get("preprocess")}

    for key, expected in list(required.items()):
        missing = [c["chunk_id"] for c in chunks if not c.get(key)]
        if missing:
            raise ChunkContractError(
                f"청크에 필수 버전 필드 '{key}'가 없습니다({len(missing)}개, "
                f"예: {missing[:3]}). 인덱싱을 중단합니다."
            )
        actual = {c[key] for c in chunks}
        if len(actual) > 1:
            raise ChunkContractError(
                f"청크 파일 안에 '{key}' 값이 섞여 있습니다: {sorted(actual)}"
            )
        got = next(iter(actual))
        if expected is not None and got != expected:
            raise ChunkContractError(
                f"청크의 {key}={got!r}가 설정 요구값({expected!r})과 다릅니다. "
                f"폴더 이름이 아니라 파일 안의 값으로 판정합니다 — 인덱싱을 중단합니다."
            )

    for key, expected in optional.items():
        actual = {c.get(key) for c in chunks if c.get(key)}
        if len(actual) > 1:
            raise ChunkContractError(
                f"청크 파일 안에 '{key}' 값이 섞여 있습니다: {sorted(actual)}"
            )
        if actual and expected is not None and next(iter(actual)) != expected:
            raise ChunkContractError(
                f"청크의 {key}={next(iter(actual))!r}가 설정 요구값({expected!r})과 다릅니다."
            )

    declared: dict[str, str] = {}
    if version_file is not None:
        declared = parse_chunks_version_file(version_file)
        if not declared:
            raise ChunkContractError(
                f"공식 청크 메타데이터를 찾지 못했습니다: {version_file}. "
                f"청크 내부 값끼리 같다는 것만으로는 최신 자료임을 인정하지 않습니다."
            )
        for key in ("corpus_version", "chunking_version", "preprocess_version"):
            expected = cfg.get({"corpus_version": "corpus",
                                "chunking_version": "chunking_version",
                                "preprocess_version": "preprocess"}[key])
            if key in declared and expected is not None and declared[key] != expected:
                raise ChunkContractError(
                    f"{version_file}: {key}={declared[key]!r}가 설정 요구값"
                    f"({expected!r})과 다릅니다."
                )
        if "chunk_count" in declared and declared["chunk_count"].isdigit():
            if int(declared["chunk_count"]) != len(chunks):
                raise ChunkContractError(
                    f"{version_file}: 선언 청크 수({declared['chunk_count']})와 실제"
                    f"({len(chunks)})가 다릅니다."
                )
        if "document_count" in declared and declared["document_count"].isdigit():
            actual_docs = len({c["document_id"] for c in chunks})
            if int(declared["document_count"]) != actual_docs:
                raise ChunkContractError(
                    f"{version_file}: 선언 문서 수({declared['document_count']})와 실제"
                    f"({actual_docs})가 다릅니다."
                )
    return declared


def load_registry_documents(registry_path: Path) -> list[dict]:
    with open(registry_path, "r", encoding="utf-8") as f:
        doc = json.load(f)
    if "documents" not in doc:
        raise ValueError(
            f"{registry_path}: 최상위 객체에 'documents' 키가 없습니다. "
            "존재하지 않는 registry.jsonl 형식을 가정하지 마세요."
        )
    return doc["documents"]


def cross_check_registry(chunks: list[dict], registry_docs: list[dict]) -> dict:
    """청크 ↔ 공식 등록부 대조 (4-2 확정).

    문서마다 document_id · document_version · processed_sha256 · sidecar_sha256 ·
    검색 대상 여부 · 중복 제외 관계를 비교한다. 하나라도 누락·추가·불일치하면
    공식 인덱스 생성을 중단한다.
    """
    reg_by_id = {r["document_id"]: r for r in registry_docs}
    chunk_ids = {c["document_id"] for c in chunks}

    missing = sorted(set(reg_by_id) - chunk_ids)      # 등록부엔 있는데 청크에 없음
    extra = sorted(chunk_ids - set(reg_by_id))        # 청크엔 있는데 등록부에 없음
    problems: list[str] = []
    if missing:
        problems.append(f"등록부에는 있으나 청크에 없는 문서 {len(missing)}건: {missing[:5]}")
    if extra:
        problems.append(f"등록부에 없는 문서가 청크에 있음 {len(extra)}건: {extra[:5]}")

    by_doc: dict[str, dict] = {}
    for c in chunks:
        by_doc.setdefault(c["document_id"], c)

    for doc_id, c in sorted(by_doc.items()):
        reg = reg_by_id.get(doc_id)
        if reg is None:
            continue
        for chunk_key, reg_key in (
            ("document_version", "document_version"),
            ("processed_sha256", "processed_sha256"),
            ("sidecar_sha256", "sidecar_sha256"),
        ):
            got, want = str(c.get(chunk_key)), str(reg.get(reg_key))
            if got != want:
                problems.append(
                    f"{doc_id}: {chunk_key} 불일치 — 청크={got!r} / 등록부={want!r}"
                )

    if problems:
        raise ChunkContractError(
            "청크와 공식 등록부가 일치하지 않습니다. 공식 인덱스 생성을 중단합니다.\n"
            + "\n".join(f"  - {p}" for p in problems[:20])
        )

    eligible = {r["document_id"] for r in registry_docs if r.get("retrieval_eligible", True)}
    excluded = [
        {"document_id": r["document_id"],
         "duplicate_of_document_id": r.get("duplicate_of_document_id") or None,
         "relation_status": r.get("relation_status")}
        for r in registry_docs if not r.get("retrieval_eligible", True)
    ]
    return {
        "registry_document_count": len(registry_docs),
        "chunk_document_count": len(chunk_ids),
        "retrieval_eligible_count": len(eligible),
        "excluded": excluded,
        "hash_checked_documents": len(by_doc),
    }


def load_retrieval_eligible_ids(registry_path: Path | None) -> set[str] | None:
    """공식 document_registry_v2.json(단일 JSON, 'documents' 키)에서
    retrieval_eligible=true인 document_id만 추출한다.
    4-9-1 확정: 활성 100건 중 검색 대상 98건(콘텐츠 중복 2건 제외).
    ⚠️ registry.jsonl 형식이 아니다 — JSONL로 읽지 않는다."""
    if registry_path is None:
        print("⚠️  --registry 미지정 — 전체 문서를 검색 대상으로 취급합니다. "
              "실제로는 document_registry에서 retrieval_eligible 필터링 필요.")
        return None
    with open(registry_path, "r", encoding="utf-8") as f:
        doc = json.load(f)
    if "documents" not in doc:
        raise ValueError(
            f"{registry_path}: 최상위 객체에 'documents' 키가 없습니다. "
            "존재하지 않는 registry.jsonl 형식을 가정하지 마세요."
        )
    eligible = {
        row["document_id"] for row in doc["documents"]
        if row.get("retrieval_eligible", True)
    }
    excluded = [
        (row["document_id"], row.get("duplicate_of_document_id"))
        for row in doc["documents"] if not row.get("retrieval_eligible", True)
    ]
    if excluded:
        print(f"   문서 등록부 기준 검색 제외 {len(excluded)}건: "
              + ", ".join(f"{d}->{dup}" for d, dup in excluded))
    return eligible


def load_registry_document_ids(registry_path: Path) -> set[str]:
    """등록부에 있는 전체 document_id(검색 대상 여부 무관). --only에
    오타 문서ID(예: RFP-00001)를 넣었을 때 "등록부에 아예 없는 ID"와
    "등록부엔 있지만 이번엔 제외 대상"을 구분하는 데 씀(리뷰 문제4)."""
    with open(registry_path, "r", encoding="utf-8") as f:
        doc = json.load(f)
    if "documents" not in doc:
        raise ValueError(f"{registry_path}: 최상위 객체에 'documents' 키가 없습니다.")
    return {row["document_id"] for row in doc["documents"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", required=True, help="C단계 청크 JSONL 경로")
    parser.add_argument("--registry", required=False, help="document_registry_v2 JSON 경로")
    parser.add_argument("--experiment-config", required=False, help="실험별 config.yaml (선택)")
    parser.add_argument(
        "--only", required=False,
        help="쉼표로 구분한 document_id 목록 — 지정하면 그 문서들만 기존 인덱스에서 "
             "교체(문서 단위 증분 갱신). 생략하면 전체 재생성."
    )
    parser.add_argument(
        "--chunks-version-file", required=False,
        help="chunks_vN/VERSION.txt 경로. 생략하면 청크 파일 옆 또는 base.yaml 기준으로 찾는다.")
    parser.add_argument(
        "--allow-no-chunks-version-file", action="store_true",
        help="공식 청크 메타데이터 없이 진행(테스트 목적). 기본은 필수.")
    parser.add_argument(
        "--index-out", required=False,
        help="인덱스를 쓸 폴더. 생략하면 base.yaml의 index 버전 경로. "
             "공식 인덱스를 덮어쓰지 않고 임시 검증용으로 만들 때 쓴다.")
    parser.add_argument(
        "--allow-no-registry", action="store_true",
        help="문서 등록부 없이 인덱싱을 허용한다(테스트 목적). 기본값은 필수 — "
             "등록부 없이 돌리면 검색 제외 대상(콘텐츠 중복 등)이 그대로 섞여 들어간다."
    )
    args = parser.parse_args()
    only_ids = set(args.only.split(",")) if args.only else None

    cfg = load_config(args.experiment_config)
    warn_if_dirty(purpose="인덱싱")

    if not cfg.get("embedding_provider"):
        print("❌ embedding_provider가 비어 있습니다 — 오픈AI 트랙 확정이 철회되고 "
              "후보 비교 단계로 되돌아갔습니다(4-8). base.yaml에 embedding_provider를 "
              "명시적으로 채우기 전까진 실행하지 않습니다.")
        sys.exit(1)
    if cfg.get("embedding_provider") != "openai":
        print(f"이 스크립트는 현재 오픈AI 구현만 있습니다(embedding_provider="
              f"{cfg.get('embedding_provider')!r}). 로컬 트랙 스크립트는 별도입니다.")
        sys.exit(1)

    # ⚠️ 버그 수정(리뷰 반영, 문제3): 예전엔 --registry가 완전히 선택사항이라
    # 안 주면 경고만 하고 100건 전부(콘텐츠 중복 2건 포함) 검색 대상으로
    # 삼았다. 공식 인덱스는 등록부가 필수 — 명시적으로 우회해야만 생략 가능.
    if not args.registry and not args.allow_no_registry:
        print("❌ --registry가 없습니다. 공식 인덱스는 문서 등록부가 필수입니다 "
              "(콘텐츠 중복 문서 제외를 위해). 테스트 목적으로 등록부 없이 돌리려면 "
              "--allow-no-registry를 명시하세요.")
        sys.exit(1)

    out_dir = Path(args.index_out) if args.index_out else index_dir(cfg)
    try:
        config_mismatch_check(out_dir, cfg)
    except ConfigMismatchError as e:
        print(f"❌ {e}")
        sys.exit(1)

    raw_chunks = load_chunks(Path(args.chunks))
    print(f"청크 {len(raw_chunks)}개 로드 완료(원본 필드명 그대로)")
    chunks = [map_chunk(c) for c in raw_chunks]

    try:
        validate_chunk_consistency(chunks)
    except ValueError as e:
        print(f"❌ {e}")
        sys.exit(1)

    # ⭐ 청크 버전 계약 — 폴더 이름이 아니라 파일 안의 값 + 공식 메타데이터로 판정
    version_file = Path(args.chunks_version_file) if args.chunks_version_file else None
    if version_file is None and not args.allow_no_chunks_version_file:
        candidate = Path(args.chunks).parent / "VERSION.txt"
        if candidate.exists():
            version_file = candidate
        else:
            try:
                version_file = chunks_version_file(cfg)
            except Exception:
                version_file = None
        if version_file is None or not version_file.exists():
            print("❌ 공식 청크 메타데이터(VERSION.txt)를 찾지 못했습니다. "
                  "--chunks-version-file로 지정하거나, 테스트 목적이면 "
                  "--allow-no-chunks-version-file을 명시하세요.")
            sys.exit(1)
    try:
        declared_versions = validate_chunk_versions(chunks, cfg, version_file)
    except ChunkContractError as e:
        print(f"❌ {e}")
        sys.exit(1)
    if declared_versions:
        print(f"   청크 메타데이터 교차 확인 OK: {declared_versions}")

    registry_path = Path(args.registry) if args.registry else None
    registry_report = None
    if registry_path is not None:
        try:
            registry_docs = load_registry_documents(registry_path)
            registry_report = cross_check_registry(chunks, registry_docs)
        except (ChunkContractError, ValueError) as e:
            print(f"❌ {e}")
            sys.exit(1)
        print(f"   등록부 대조 OK: 문서 {registry_report['registry_document_count']}건, "
              f"해시 대조 {registry_report['hash_checked_documents']}건, "
              f"검색 제외 {len(registry_report['excluded'])}건")

    eligible_ids = load_retrieval_eligible_ids(registry_path)
    all_registry_ids = load_registry_document_ids(registry_path) if registry_path else None

    before = len(chunks)
    if eligible_ids is not None:
        chunks = [c for c in chunks if c["document_id"] in eligible_ids]
    # 문서 단위 제외와 별개로 청크 단위 retrieval_eligible도 적용(예: degraded
    # 표 등 청크 개별 제외 — message.txt 5번, 실제 chunks.jsonl 스키마 대조 결과)
    chunks = [c for c in chunks if c["chunk_retrieval_eligible"]]
    print(f"retrieval_eligible 필터링(문서+청크 단위): {before} → {len(chunks)}개 청크")

    is_incremental = only_ids is not None
    if is_incremental:
        # ⚠️ 버그 수정(리뷰 반영, 문제4): 예전엔 "청크가 없는 문서"를 전부
        # 같은 경고 하나로 뭉뚱그렸다 — 그러면 다음 세 가지를 구분 못 한다:
        # (a) 사용자가 문서 ID를 오타(RFP-00001)로 입력한 경우 — 에러로 중단
        # (b) 등록부엔 있지만 이번엔 검색 제외로 바뀐 경우 — 기존 벡터만 비활성화
        # (c) 등록부엔 검색 대상인데 청크 파일에서 실수로 빠진 경우 — 에러로 중단
        missing = only_ids - {c["document_id"] for c in chunks}
        if missing and all_registry_ids is not None:
            not_in_registry = missing - all_registry_ids
            if not_in_registry:
                print(f"❌ --only에 지정한 문서 ID가 등록부에 아예 없습니다(오타 "
                      f"가능성): {sorted(not_in_registry)}")
                sys.exit(1)
            still_missing = missing - not_in_registry
            excluded_now = still_missing & (all_registry_ids - (eligible_ids or set()))
            data_gap = still_missing - excluded_now
            if data_gap:
                print(f"❌ --only에 지정한 문서가 등록부엔 검색 대상인데 청크 "
                      f"파일에 없습니다(청크 생성 누락 가능성): {sorted(data_gap)}")
                sys.exit(1)
            if excluded_now:
                print(f"   이번에 검색 제외로 바뀐 문서 — 기존 벡터만 비활성화: "
                      f"{sorted(excluded_now)}")
        elif missing:
            print(f"⚠️  --only로 지정했지만 색인할 청크가 없는 문서: {sorted(missing)} "
                  f"— 등록부가 없어 오타/제외/누락을 구분 못 합니다.")
        chunks = [c for c in chunks if c["document_id"] in only_ids]
        print(f"--only 필터링: {only_ids} 대상 청크 {len(chunks)}개")

    client = EmbeddingClient(cfg)
    texts = [c["text"] for c in chunks]
    if texts:
        print(f"임베딩 시작 ({client.model})...")
        vectors = client.embed_batch(texts)
        print(f"임베딩 완료: {len(vectors)}개 벡터")
    else:
        vectors = []
        print("임베딩할 청크가 없습니다(모두 --only 대상에서 제외됨).")

    # 실제 사용량·비용 기록 (인덱싱은 문항 단위 비용과 별개로 따로 남긴다)
    from pricing import compute_cost
    index_cost, index_cost_detail = compute_cost(cfg, client.usage)
    print(f"   임베딩 API 요청 {client.usage.embedding_requests}건 / "
          f"토큰 {client.usage.embedding_tokens:,} / 비용 ${index_cost}")

    if is_incremental:
        store = VectorStore.load(out_dir)
        if store.vectors is None:
            print(f"❌ {out_dir}에 기존 인덱스가 없습니다 — --only(문서 단위 증분 갱신)는 "
                  f"기존 인덱스가 있어야 합니다. 먼저 --only 없이 전체 생성하세요.")
            sys.exit(1)
        print(f"기존 인덱스 로드 OK: 청크 {len(store.metadata)}개 "
              f"(그중 비활성 {sum(1 for m in store.metadata if not m.active)}개)")
        # 청크가 하나도 안 남았지만 기존 인덱스에 있던 문서 = 새로 제외된 문서 →
        # soft delete만 수행
        for doc_id in missing:
            if any(m.document_id == doc_id for m in store.metadata):
                store.soft_delete_document(doc_id)
                print(f"   {doc_id}: 기존 청크 비활성화(soft delete) — retrieval_eligible=false로 전환")
    else:
        store = VectorStore()

    by_doc: dict[str, list[int]] = {}
    for i, c in enumerate(chunks):
        by_doc.setdefault(c["document_id"], []).append(i)

    for doc_id, idxs in by_doc.items():
        pairs = [(texts[i], vectors[i]) for i in idxs]
        metas = [
            ChunkMetadata(
                chunk_id=chunks[i]["chunk_id"],
                document_id=chunks[i]["document_id"],
                document_version=chunks[i]["document_version"],
                processed_sha256=chunks[i]["processed_sha256"],
                sidecar_sha256=chunks[i]["sidecar_sha256"],
                corpus_version=chunks[i].get("corpus_version") or cfg["corpus"],
                text=chunks[i]["text"],
                document_name=chunks[i]["document_name"],
                chapter=chunks[i]["chapter"],
                chunk_type=chunks[i]["chunk_type"],
                table_idx=chunks[i]["table_idx"],
                row_start=chunks[i]["row_start"],
                row_end=chunks[i]["row_end"],
                part=chunks[i]["part"],
                of=chunks[i]["of"],
                table_degraded=chunks[i]["table_degraded"],
                oversize=chunks[i]["oversize"],
                location_label=chunks[i]["location_label"],
                section_path=chunks[i]["section_path"],
                section_paths=chunks[i]["section_paths"],
                md_line_start=chunks[i]["md_line_start"],
                md_line_end=chunks[i]["md_line_end"],
                preprocess_version=chunks[i].get("preprocess_version", ""),
                chunking_version=chunks[i].get("chunking_version", ""),
            )
            for i in idxs
        ]
        store.upsert_document(doc_id, pairs, metas)

    active_doc_count = len({m.document_id for m in store.metadata if m.active})
    active_chunk_count = sum(1 for m in store.metadata if m.active)

    git = get_git_info()
    tag = IndexTag(
        chunk_size=cfg.get("chunk_size") or -1,
        chunk_overlap=cfg.get("chunk_overlap") or -1,
        embedding_model=client.model,
        embedding_provider="openai",
        preprocess_version=cfg.get("preprocess", ""),
        corpus_version=cfg["corpus"],
        registry_version=cfg.get("document_registry_version", cfg["corpus"]),
        extraction_version=cfg.get("extraction_version", ""),
        chunking_version=cfg.get("chunking_version", ""),
        vector_dimension=(store.dimension or 0),
        document_count=active_doc_count,
        chunk_count=active_chunk_count,
        build_timestamp=datetime.now(timezone.utc).isoformat(),
        extra={**git, "embedding_usage": client.usage.as_dict(),
               "embedding_cost_usd": index_cost,
               "pricing_unit": cfg.get("pricing_unit"),
               "registry_cross_check": registry_report},
    )
    store.save(out_dir, tag)

    # ⭐ 만든 인덱스가 실행 경로의 검증(결함 1-1)을 그대로 통과하는지 여기서 확인한다.
    #    빌드와 실행이 같은 함수를 쓰므로 기준이 둘로 갈리지 않는다.
    try:
        validate_index_tag(out_dir, cfg)
    except IndexTagError as e:
        print(f"❌ 방금 만든 인덱스가 실행 검증을 통과하지 못했습니다:\n{e}")
        sys.exit(1)
    print("   인덱스 꼬리표 검증 통과(실행 경로와 동일 기준)")

    mode = f"증분 갱신({sorted(only_ids)})" if is_incremental else "전체 재생성"
    print(f"✅ 인덱스 저장 완료: {out_dir} [{mode}]")
    print(f"   활성 문서 {active_doc_count}건, 활성 청크 {active_chunk_count}개, "
          f"git_dirty={git['git_dirty']}")


if __name__ == "__main__":
    main()
