"""
경량 벡터 저장소 (4-9-1 확정 방식) — numpy 배열 기반.
근거: 조각 수 약 1만 개 추정 규모라 전용 벡터DB 없이도 시작 가능(4-9-1).
갱신 범위는 추가+수정까지, 물리 삭제 없이 상태 플래그로 흡수(4-9-1).
인덱싱 전 config_mismatch_check로 설정 불일치를 사전 차단한다(4-9-2).

[2026-08-31 message.txt 9번 반영]
사업명 기반 콘텐츠 중복 탐지(refresh_duplicate_name_flags)는 제거했다.
공식 콘텐츠 중복 관계는 document_registry_v2.json의 retrieval_eligible·
duplicate_of_document_id가 유일한 근거이고, 그 판단은 build_index.py가
문서를 색인에 넣기 전에 이미 적용한다(제외된 문서는 벡터스토어에 아예
들어오지 않음) — 벡터스토어가 사업명만으로 다시 판정할 이유가 없다.
"""
from __future__ import annotations
import json
import shutil
import tempfile
import numpy as np
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


@dataclass
class ChunkMetadata:
    chunk_id: str
    document_id: str
    document_version: str
    processed_sha256: str
    sidecar_sha256: str
    corpus_version: str
    text: str = ""  # 검색용 본문 — 생성 단계(J)에서 컨텍스트로 그대로 사용
    active: bool = True                      # soft delete 플래그
    # 출처 표기용 위치 정보 (4-13)
    document_name: str = ""
    chapter: str = ""
    chunk_type: str = "text"  # "text" | "table"
    table_idx: int | None = None
    row_start: int | None = None
    row_end: int | None = None
    part: int | None = None
    of: int | None = None
    # 4-7-1 확정 — degraded 표는 값 생성 대신 원문 위치만 안내해야 함
    table_degraded: bool = False
    # 임베딩 모델 최대 길이 초과 플래그(청킹 산출물에서 옴) — 조용히 잘리지 않았는지 확인용
    oversize: bool = False


@dataclass
class IndexTag:
    """인덱스 생성 설정 꼬리표 (4-9-2 확정 항목 + message.txt 9번 확장)."""
    chunk_size: int
    chunk_overlap: int
    embedding_model: str
    embedding_provider: str
    preprocess_version: str
    corpus_version: str
    build_timestamp: str
    registry_version: str = ""
    extraction_version: str = ""
    chunking_version: str = ""
    vector_dimension: int = 0
    document_count: int = 0
    chunk_count: int = 0
    extra: dict[str, Any] = field(default_factory=dict)  # git_commit·git_dirty 등


class ConfigMismatchError(RuntimeError):
    pass


def config_mismatch_check(index_dir: Path, cfg: dict[str, Any]) -> None:
    """인덱싱 시작 전 기존 인덱스 꼬리표와 현재 config를 비교.
    다르면 실행을 중단하고 새 버전을 만들라고 안내한다(4-9-2 확정).
    """
    tag_path = index_dir / "index_tag.json"
    if not tag_path.exists():
        return  # 최초 생성이면 비교 대상 없음
    with open(tag_path, "r", encoding="utf-8") as f:
        existing = json.load(f)

    check_keys = ["chunk_size", "chunk_overlap", "embedding_model", "embedding_provider"]
    mismatches = {
        k: (existing.get(k), cfg.get(k))
        for k in check_keys
        if existing.get(k) != cfg.get(k)
    }
    if mismatches:
        detail = "\n".join(f"  - {k}: 기존={old} / 현재={new}" for k, (old, new) in mismatches.items())
        raise ConfigMismatchError(
            f"인덱스 설정 불일치 발견 — {index_dir}\n{detail}\n"
            f"기존 인덱스({cfg.get('index')})에 다른 설정을 밀어넣지 마세요. "
            f"index_v{int(cfg['index'].lstrip('v')) + 1}을 새로 만드세요."
        )


class VectorStoreError(RuntimeError):
    pass


class VectorStore:
    """numpy 기반 경량 벡터 저장소. upsert·soft delete·doc_id 필터 지원."""

    def __init__(self):
        self.vectors: np.ndarray | None = None  # shape (N, dim)
        self.metadata: list[ChunkMetadata] = []

    @property
    def dimension(self) -> int | None:
        return None if self.vectors is None or len(self.vectors) == 0 else self.vectors.shape[1]

    # ---- 적재 ----
    def upsert_document(
        self,
        document_id: str,
        chunk_texts_and_vectors: list[tuple[str, list[float]]],
        chunk_metadata: list[ChunkMetadata],
    ) -> None:
        """문서 하나 단위로 upsert. 같은 document_id의 기존 청크는 전부
        비활성화(soft delete)한 뒤 새 청크를 추가한다."""
        if len(chunk_texts_and_vectors) != len(chunk_metadata):
            raise VectorStoreError(
                f"{document_id}: 벡터 수({len(chunk_texts_and_vectors)})와 "
                f"청크 메타데이터 수({len(chunk_metadata)})가 다릅니다."
            )
        bad_doc = [m.document_id for m in chunk_metadata if m.document_id != document_id]
        if bad_doc:
            raise VectorStoreError(
                f"upsert_document(document_id={document_id!r})에 다른 document_id를 "
                f"가진 메타데이터가 섞여 있습니다: {set(bad_doc)}"
            )
        chunk_ids = [m.chunk_id for m in chunk_metadata]
        if len(chunk_ids) != len(set(chunk_ids)):
            dupes = {c for c in chunk_ids if chunk_ids.count(c) > 1}
            raise VectorStoreError(f"{document_id}: 입력 배치 안에 chunk_id 중복: {dupes}")
        existing_ids = {m.chunk_id for m in self.metadata if m.document_id != document_id}
        clash = existing_ids & set(chunk_ids)
        if clash:
            raise VectorStoreError(f"{document_id}: 다른 문서와 chunk_id가 겹칩니다: {clash}")

        new_vecs = np.array([v for _, v in chunk_texts_and_vectors], dtype=np.float32)
        if self.vectors is not None and len(self.vectors) > 0 and new_vecs.shape[1] != self.dimension:
            raise VectorStoreError(
                f"{document_id}: 벡터 차원({new_vecs.shape[1]})이 기존 인덱스 차원"
                f"({self.dimension})과 다릅니다 — 임베딩 모델이 섞였을 수 있습니다."
            )

        self.soft_delete_document(document_id)

        if self.vectors is None:
            self.vectors = new_vecs
        else:
            self.vectors = np.vstack([self.vectors, new_vecs])
        self.metadata.extend(chunk_metadata)

    def soft_delete_document(self, document_id: str) -> None:
        for m in self.metadata:
            if m.document_id == document_id:
                m.active = False

    # ---- 검색 ----
    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        active_only: bool = True,
        document_id: str | None = None,
    ) -> list[tuple[ChunkMetadata, float]]:
        """document_id를 주면 그 문서로 검색 범위를 좁힌다(4-14 후속질문 —
        활성 문서 ID로 범위 제한, 쿼리 재작성은 하지 않음)."""
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
            raise VectorStoreError(f"top_k는 양의 정수여야 합니다: {top_k!r}")
        if self.vectors is None or len(self.metadata) == 0:
            return []
        if len(self.vectors) != len(self.metadata):
            raise VectorStoreError(
                f"벡터 수({len(self.vectors)})와 메타데이터 수({len(self.metadata)})가 "
                "일치하지 않습니다 — 인덱스가 손상됐을 수 있습니다."
            )
        q = np.array(query_vector, dtype=np.float32)
        if self.dimension is not None and q.shape[-1] != self.dimension:
            raise VectorStoreError(
                f"질의 벡터 차원({q.shape[-1]})이 인덱스 차원({self.dimension})과 다릅니다."
            )
        q = q / (np.linalg.norm(q) + 1e-10)
        vecs = self.vectors / (np.linalg.norm(self.vectors, axis=1, keepdims=True) + 1e-10)
        scores = vecs @ q  # cosine similarity

        indices = np.argsort(-scores)
        results: list[tuple[ChunkMetadata, float]] = []
        for i in indices:
            m = self.metadata[i]
            if active_only and not m.active:
                continue
            if document_id is not None and m.document_id != document_id:
                continue
            results.append((m, float(scores[i])))
            if len(results) >= top_k:
                break
        return results

    # ---- 저장/로드 ----
    def save(self, index_dir: Path, tag: IndexTag) -> None:
        """임시 디렉터리에 전부 쓴 뒤 최종 위치로 교체한다(원자적 저장) —
        저장 도중 실패해도 기존 인덱스가 반쪽짜리로 덮이지 않는다."""
        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)
        tmp_dir = Path(tempfile.mkdtemp(prefix=".tmp_index_", dir=index_dir.parent))
        try:
            if self.vectors is not None:
                np.save(tmp_dir / "vectors.npy", self.vectors)
            with open(tmp_dir / "metadata.jsonl", "w", encoding="utf-8") as f:
                for m in self.metadata:
                    f.write(json.dumps(asdict(m), ensure_ascii=False) + "\n")
            with open(tmp_dir / "index_tag.json", "w", encoding="utf-8") as f:
                json.dump(asdict(tag), f, ensure_ascii=False, indent=2)

            for name in ("vectors.npy", "metadata.jsonl", "index_tag.json"):
                src = tmp_dir / name
                if src.exists():
                    shutil.move(str(src), str(index_dir / name))
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    @classmethod
    def load(cls, index_dir: Path) -> "VectorStore":
        store = cls()
        vec_path = Path(index_dir) / "vectors.npy"
        if vec_path.exists():
            store.vectors = np.load(vec_path)
        meta_path = Path(index_dir) / "metadata.jsonl"
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                store.metadata = [ChunkMetadata(**json.loads(line)) for line in f]
        if store.vectors is not None and len(store.vectors) != len(store.metadata):
            raise VectorStoreError(
                f"{index_dir}: 불러온 벡터 행 수({len(store.vectors)})와 메타데이터 행 수"
                f"({len(store.metadata)})가 다릅니다 — 인덱스가 손상됐을 수 있습니다."
            )
        return store
