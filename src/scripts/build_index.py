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

from config import load_config, index_dir
from git_info import get_git_info, warn_if_dirty
from embedding_client import EmbeddingClient
from vector_store import VectorStore, ChunkMetadata, IndexTag, config_mismatch_check, ConfigMismatchError


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

    out_dir = index_dir(cfg)
    try:
        config_mismatch_check(out_dir, cfg)
    except ConfigMismatchError as e:
        print(f"❌ {e}")
        sys.exit(1)

    raw_chunks = load_chunks(Path(args.chunks))
    print(f"청크 {len(raw_chunks)}개 로드 완료(원본 필드명 그대로)")
    chunks = [map_chunk(c) for c in raw_chunks]

    eligible_ids = load_retrieval_eligible_ids(
        Path(args.registry) if args.registry else None
    )
    before = len(chunks)
    if eligible_ids is not None:
        chunks = [c for c in chunks if c["document_id"] in eligible_ids]
    # 문서 단위 제외와 별개로 청크 단위 retrieval_eligible도 적용(예: degraded
    # 표 등 청크 개별 제외 — message.txt 5번, 실제 chunks.jsonl 스키마 대조 결과)
    chunks = [c for c in chunks if c["chunk_retrieval_eligible"]]
    print(f"retrieval_eligible 필터링(문서+청크 단위): {before} → {len(chunks)}개 청크")

    is_incremental = only_ids is not None
    if is_incremental:
        missing = only_ids - {c["document_id"] for c in chunks}
        # --only로 지정했는데 청크가 하나도 안 남은 문서 = 청크 파일에 아예 없거나
        # retrieval_eligible=false로 이번에 새로 제외된 경우. 후자는 기존 인덱스에서
        # soft delete만 하고 새로 넣지는 않는다 — 조용히 넘어가지 않고 알린다.
        if missing:
            print(f"⚠️  --only로 지정했지만 색인할 청크가 없는 문서: {sorted(missing)} "
                  f"— 청크 파일에 없거나 retrieval_eligible=false일 수 있습니다. "
                  f"기존 인덱스에 있었다면 비활성화(soft delete)만 적용합니다.")
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
        extra=git,
    )
    store.save(out_dir, tag)
    mode = f"증분 갱신({sorted(only_ids)})" if is_incremental else "전체 재생성"
    print(f"✅ 인덱스 저장 완료: {out_dir} [{mode}]")
    print(f"   활성 문서 {active_doc_count}건, 활성 청크 {active_chunk_count}개, "
          f"git_dirty={git['git_dirty']}")


if __name__ == "__main__":
    main()
