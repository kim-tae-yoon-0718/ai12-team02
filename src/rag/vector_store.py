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
import os
import shutil
import tempfile
import numpy as np
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


def chunk_index_from_id(chunk_id: str) -> int | None:
    """chunk_id("RFP-000001-0007")의 접미 순번을 정수로. 형식이 다르면 None."""
    if not chunk_id:
        return None
    tail = str(chunk_id).rsplit("-", 1)[-1]
    return int(tail) if tail.isdigit() else None


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
    # ⚠️ 버그 수정(리뷰 반영, 문제8): 청크에 실려오는 location_label(예:
    # "제18조 평가배점 · 문단 1~4")을 예전엔 인덱스에 저장 안 해서, 일반
    # 문단 출처는 문서명+장절까지만 남고 정확한 문단 위치가 사라졌다.
    location_label: str = ""
    # 2026-09-02 추가 — 하루님 채점기가 요구하는 청크 형식(contexts/retrieved)에
    # 필요. chapter는 " > "로 이미 합쳐진 문자열이라 원래 리스트 형태를 잃는데,
    # section_path는 원본 리스트를 그대로 보존한다.
    section_path: list = None  # type: ignore[assignment]
    # 2026-09-02 추가 — 청크가 여러 절에 걸쳐 있을 때의 원본 목록(chunks_v3
    # 'section_paths'). 있으면 그대로 보존한다(합쳐서 버리지 않는다).
    section_paths: list = None  # type: ignore[assignment]
    md_line_start: int | None = None
    md_line_end: int | None = None
    # 2026-09-02 추가 — 텍스트 청크의 block_index를 None으로 지우지 않기 위한
    # 문서 내 청크 순번(chunk_id 접미 4자리). 표 청크는 table_idx를 쓴다.
    chunk_index: int | None = None
    preprocess_version: str = ""
    chunking_version: str = ""

    def __post_init__(self):
        if self.section_path is None:
            self.section_path = []
        if self.section_paths is None:
            self.section_paths = []
        if self.chunk_index is None:
            self.chunk_index = chunk_index_from_id(self.chunk_id)

    @property
    def block_index(self) -> int | None:
        """근거 위치용 블록 번호. 표는 표 번호, 그 밖에는 문서 내 청크 순번.
        ⚠️ 텍스트 청크에서 None으로 비우지 않는다(4-3 확정)."""
        if self.chunk_type == "table" and self.table_idx is not None:
            return self.table_idx
        return self.chunk_index


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


class IndexTagError(RuntimeError):
    """인덱스 꼬리표(index_tag.json)가 없거나 깨졌거나 설정과 다를 때."""


# 인덱스 꼬리표에 반드시 있어야 하는 항목. 하나라도 없으면 그 인덱스는 신뢰하지 않는다.
INDEX_TAG_REQUIRED_KEYS = (
    "chunk_size", "chunk_overlap", "embedding_model", "embedding_provider",
    "corpus_version", "preprocess_version", "chunking_version",
    "registry_version", "extraction_version", "vector_dimension",
    "build_timestamp",
)

# 설정(cfg) 키 → 꼬리표(tag) 키. 빌드와 실행이 **같은 표**를 본다(기준 이중화 금지).
INDEX_TAG_CFG_MAP = (
    ("chunk_size", "chunk_size"),
    ("chunk_overlap", "chunk_overlap"),
    ("embedding_model", "embedding_model"),
    ("embedding_provider", "embedding_provider"),
    ("corpus", "corpus_version"),
    ("preprocess", "preprocess_version"),
    ("chunking_version", "chunking_version"),
    ("__registry__", "registry_version"),
    ("extraction_version", "extraction_version"),
)


def _cfg_value(cfg: dict[str, Any], cfg_key: str) -> Any:
    if cfg_key == "__registry__":
        return cfg.get("document_registry_version", cfg.get("corpus"))
    return cfg.get(cfg_key)


def read_index_tag(index_dir: Path) -> dict[str, Any]:
    """index_tag.json 을 읽는다. 없거나 JSON 이 깨졌으면 IndexTagError."""
    tag_path = Path(index_dir) / "index_tag.json"
    if not tag_path.exists():
        raise IndexTagError(
            f"인덱스 꼬리표가 없습니다: {tag_path}\n"
            f"버전을 확인할 수 없는 인덱스는 사용하지 않습니다 — "
            f"build_index.py 로 다시 만드세요."
        )
    try:
        with open(tag_path, "r", encoding="utf-8") as f:
            tag = json.load(f)
    except json.JSONDecodeError as e:
        raise IndexTagError(f"{tag_path}: 꼬리표 JSON 을 읽지 못했습니다 — {e}") from e
    if not isinstance(tag, dict):
        raise IndexTagError(f"{tag_path}: 꼬리표가 객체가 아닙니다({type(tag).__name__}).")
    return tag


def validate_index_tag(
    index_dir: Path, cfg: dict[str, Any], *, check_vectors: bool = True,
) -> dict[str, Any]:
    """실행 전 인덱스 검증 (2026-09-02 확정, 결함 1-1).

    ⚠️ 예전엔 인덱싱할 때만 버전을 봤고, 답변 실행 경로는 `VectorStore.load()` 를
    그냥 불렀다. 그래서 chunks_v3 설정으로 옛 index_v1(chunking v1 / extraction v2)을
    실수로 불러도 아무도 막지 않았다.

    검사 항목
      ① 꼬리표 존재·JSON 파싱          ② 필수 항목 누락
      ③ 코퍼스·전처리·청킹·등록부·추출표 버전, 임베딩 모델/제공자, 청크 크기/겹침
      ④ 벡터 차원 (꼬리표 vs 실제 vectors.npy)

    ★ 이 함수는 build_index.py 와 answer_pipeline.build_runtime 이 **같이** 쓴다.
    ★ 반드시 임베딩·생성 클라이언트를 만들기 **전에** 부른다(실패 시 API 호출 0회).
    """
    index_dir = Path(index_dir)
    tag = read_index_tag(index_dir)

    missing = [k for k in INDEX_TAG_REQUIRED_KEYS if k not in tag]
    if missing:
        raise IndexTagError(
            f"{index_dir}: 인덱스 꼬리표에 필수 항목이 없습니다: {missing}\n"
            f"꼬리표가 오래된 형식일 수 있습니다 — 인덱스를 다시 만드세요."
        )

    mismatches: list[str] = []
    for cfg_key, tag_key in INDEX_TAG_CFG_MAP:
        expected = _cfg_value(cfg, cfg_key)
        actual = tag.get(tag_key)
        if expected is None:
            continue
        if actual != expected:
            label = "document_registry_version" if cfg_key == "__registry__" else cfg_key
            mismatches.append(
                f"  - {tag_key}: 인덱스={actual!r} / 설정({label})={expected!r}"
            )

    dim_note = None
    if check_vectors:
        vec_path = index_dir / "vectors.npy"
        if not vec_path.exists():
            raise IndexTagError(f"{index_dir}: vectors.npy 가 없습니다 — 인덱스가 불완전합니다.")
        actual_dim = int(np.load(vec_path, mmap_mode="r").shape[1])
        tag_dim = tag.get("vector_dimension")
        if tag_dim != actual_dim:
            mismatches.append(
                f"  - vector_dimension: 꼬리표={tag_dim!r} / 실제 vectors.npy={actual_dim}"
            )
        dim_note = actual_dim
        expected_dim = cfg.get("embedding_dimension")
        if expected_dim is not None and actual_dim != expected_dim:
            mismatches.append(
                f"  - vector_dimension: 실제={actual_dim} / 설정(embedding_dimension)={expected_dim}"
            )

    if mismatches:
        raise IndexTagError(
            f"인덱스가 현재 설정과 맞지 않습니다 — {index_dir}\n"
            + "\n".join(mismatches)
            + "\n낡은 인덱스로 답변하면 잘못된 근거를 내놓습니다. "
              "설정에 맞는 인덱스를 만들거나(build_index.py --index-out) "
              "올바른 인덱스 경로를 --index 로 지정하세요."
        )
    return {"tag": tag, "vector_dimension": dim_note, "index_dir": str(index_dir)}


def config_mismatch_check(index_dir: Path, cfg: dict[str, Any]) -> None:
    """인덱싱 시작 전 기존 인덱스 꼬리표와 현재 config를 비교.
    다르면 실행을 중단하고 새 버전을 만들라고 안내한다(4-9-2 확정).

    ⚠️ 범위 확장(리뷰 반영): 예전엔 chunk_size·chunk_overlap·embedding_model·
    embedding_provider 4개만 봤다. 코퍼스·전처리·청킹·등록부·추출표 버전이
    바뀌어도 안 걸렸다 — "옛 버전 벡터가 섞이면 중단" 조건을 충족 못 했다.
    cfg 쪽 키 이름(corpus/preprocess/...)과 IndexTag 쪽 키 이름
    (corpus_version/preprocess_version/...)이 다르므로 매핑해서 비교한다.
    ⚠️ 아직 못 하는 것: 문서 단위 해시(processed_sha256 등) 불일치는 이
    함수(인덱스 전체 꼬리표) 수준이 아니라 문서 단위라 별도 검사가 필요 —
    현재 코드엔 없음(추후 과제로 남김).
    """
    tag_path = index_dir / "index_tag.json"
    if not tag_path.exists():
        return  # 최초 생성이면 비교 대상 없음
    with open(tag_path, "r", encoding="utf-8") as f:
        existing = json.load(f)

    check_pairs: list[tuple[str, Any]] = [
        ("chunk_size", cfg.get("chunk_size")),
        ("chunk_overlap", cfg.get("chunk_overlap")),
        ("embedding_model", cfg.get("embedding_model")),
        ("embedding_provider", cfg.get("embedding_provider")),
        ("corpus_version", cfg.get("corpus")),
        ("preprocess_version", cfg.get("preprocess")),
        ("chunking_version", cfg.get("chunking_version")),
        ("registry_version", cfg.get("document_registry_version", cfg.get("corpus"))),
        ("extraction_version", cfg.get("extraction_version")),
    ]
    mismatches = {
        tag_key: (existing.get(tag_key), new_val)
        for tag_key, new_val in check_pairs
        if existing.get(tag_key) != new_val
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
        """임시 디렉터리에 세 파일을 전부 완성한 뒤, 그 폴더 자체를 최종
        위치로 통째로 교체한다(원자적 저장) — 저장 도중 실패해도 기존
        인덱스가 반쪽짜리로 섞이지 않는다.

        ⚠️ 버그 수정(리뷰 반영): 예전엔 tmp_dir 안에 세 파일을 완성해두고도
        마지막에 파일을 하나씩 옮겨서(vectors.npy → metadata.jsonl →
        index_tag.json 순), 두 번째 파일을 옮기다 실패하면 "벡터는 새 버전,
        메타데이터는 옛 버전" 같은 섞임이 생길 수 있었다. 이제 tmp_dir을
        os.rename으로 통째로 index_dir 자리에 밀어넣는다 — 같은 파일시스템
        안에서 디렉터리 rename은 단일 원자적 연산이라 중간 상태가 없다."""
        index_dir = Path(index_dir)
        index_dir.parent.mkdir(parents=True, exist_ok=True)
        tmp_dir = Path(tempfile.mkdtemp(prefix=".tmp_index_", dir=index_dir.parent))
        try:
            if self.vectors is not None:
                np.save(tmp_dir / "vectors.npy", self.vectors)
            with open(tmp_dir / "metadata.jsonl", "w", encoding="utf-8") as f:
                for m in self.metadata:
                    f.write(json.dumps(asdict(m), ensure_ascii=False) + "\n")
            with open(tmp_dir / "index_tag.json", "w", encoding="utf-8") as f:
                json.dump(asdict(tag), f, ensure_ascii=False, indent=2)

            backup_dir = None
            if index_dir.exists():
                backup_dir = index_dir.with_name(index_dir.name + f".old_{os.getpid()}")
                os.rename(index_dir, backup_dir)
            try:
                os.rename(tmp_dir, index_dir)
            except Exception:
                if backup_dir is not None:
                    os.rename(backup_dir, index_dir)  # 실패하면 원래 상태로 복구
                raise
            if backup_dir is not None:
                shutil.rmtree(backup_dir, ignore_errors=True)
        except Exception:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise

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
