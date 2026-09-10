"""
config 로더 — base.yaml을 전부 읽고, 실험별 config.yaml의 '바뀐 줄만'을 덮어씀.
팀 규약(실험 인프라 규약 2-1) 그대로 구현.
"""
from __future__ import annotations
import os
import yaml
from pathlib import Path
from typing import Any


def _find_base_yaml() -> Path:
    """config/base.yaml을 찾는다. 폴더 깊이(parents[1] 등)를 가정하지 않고
    config.py 기준 상위로 올라가며 config/base.yaml을 탐색한다 — 저장소
    구조가 flat이든 src/rag/ 형태든 동일하게 동작한다.
    RAG_CONFIG_PATH 환경변수가 있으면 그 경로를 그대로 우선 사용한다."""
    override = os.environ.get("RAG_CONFIG_PATH")
    if override:
        p = Path(override)
        if not p.exists():
            raise RuntimeError(f"RAG_CONFIG_PATH가 가리키는 파일이 없습니다: {p}")
        return p

    here = Path(__file__).resolve().parent
    for candidate_root in [here, *here.parents]:
        candidate = candidate_root / "config" / "base.yaml"
        if candidate.exists():
            return candidate
    raise RuntimeError(
        "config/base.yaml을 찾지 못했습니다. config.py 기준 상위 경로 어디에도 "
        "config/base.yaml이 없습니다. RAG_CONFIG_PATH 환경변수로 직접 지정하거나 "
        "config/base.yaml을 만들어 두세요."
    )


_REQUIRED_NON_NULL = ["top_k"]  # null이면 실행 전 즉시 중단해야 하는 필수값
# ⚠️ prompt_generate는 여기 넣지 않았다. 기존 테스트가 쓰는 축약 config에는 이 키가
#    없어서 config 단계에서 막으면 기존 테스트의 기대값을 바꿔야 한다. 대신 실제로
#    프롬프트를 싣는 지점(generation_client._load_system_prompt_template)에서 비어 있으면
#    즉시 중단하므로, 조용히 옛 프롬프트로 되돌아가는 경로는 이미 닫혀 있다.


def _validate_config(cfg: dict[str, Any]) -> None:
    missing = [k for k in _REQUIRED_NON_NULL if cfg.get(k) in (None, "")]
    if missing:
        raise RuntimeError(
            f"필수 설정값이 비어 있습니다: {missing}. base.yaml 또는 실험 config에서 "
            "채운 뒤 다시 실행하세요. 코드가 조용히 기본값을 넣지 않습니다."
        )
    top_k = cfg.get("top_k")
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
        raise RuntimeError(f"top_k는 양의 정수여야 합니다. 현재 값: {top_k!r}")


def load_config(experiment_config_path: str | None = None) -> dict[str, Any]:
    """base.yaml을 읽고, experiment_config_path가 있으면 그 위에 덮어쓴다.

    실험별 config는 '바뀐 줄만' 적는 게 규칙이다. 두 줄 이상 바뀐 파일이면
    "한 번에 하나만 바꾼다"(4-2) 원칙 위반이므로 경고만 남기고 막지는 않는다
    (강제 차단은 오탐 위험이 있어 team 합의 시 추가).
    """
    base_path = _find_base_yaml()
    with open(base_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    if experiment_config_path:
        with open(experiment_config_path, "r", encoding="utf-8") as f:
            override = yaml.safe_load(f) or {}
        changed_keys = [k for k in override if k not in ("name", "weight", "changed")]
        if len(changed_keys) > 1:
            print(
                f"⚠️  실험 config에 설정값이 {len(changed_keys)}개 바뀜: {changed_keys} "
                "— '한 번에 하나만 바꾼다'(4-2) 원칙 확인 필요"
            )
        cfg.update(override)

    _validate_config(cfg)
    return cfg


def resolve_path(cfg: dict[str, Any], *parts: str) -> Path:
    """RAG_ROOT 환경변수 기준으로 절대경로 조립. config엔 버전만 적고
    경로는 여기서 코드가 조립한다 (2-3 원칙 — 절대경로 하드코딩 금지)."""
    root = os.environ.get("RAG_ROOT")
    if not root:
        raise RuntimeError(
            "RAG_ROOT 환경변수가 없습니다. ~/.bashrc에 "
            "'export RAG_ROOT=/srv/rfp' 추가 후 source 하세요."
        )
    return Path(root).joinpath(*parts)


def corpus_dir(cfg: dict[str, Any]) -> Path:
    return resolve_path(cfg, "shared_data", "processed", f"corpus_{cfg['corpus']}")


class ExtractionTablePathError(RuntimeError):
    """공식 추출표를 찾지 못했거나, 두 곳의 같은 버전이 서로 다를 때.

    ★일반 RuntimeError 와 구분하는 이유: 호출부(answer_pipeline._try_derive)가
      경로 조립 실패를 '경고 후 계속'으로 넘긴다. 이 오류만은 그대로 터뜨려서
      "왜 못 찾았는지"가 '공식 추출표를 찾지 못했습니다'라는 뭉뚱그린 문구로
      덮이지 않게 한다.
    """


def repo_root() -> Path:
    """저장소 루트 — config/base.yaml 이 있는 폴더의 부모."""
    return _find_base_yaml().resolve().parent.parent


def _sha256(path: Path) -> str:
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extraction_table_candidates(cfg: dict[str, Any]) -> list[Path]:
    """추출표 폴더 후보를 **우선순위 순서대로** 돌려준다.

      ② 저장소 공식 자료 (data/preprocessed/rfp_extraction_table_vN)
      ③ 서버 공용 경로 ($RAG_ROOT/shared_data/processed/rfp_extraction_table_vN)

    (①'사용자가 명시한 경로'는 CLI 인자라 이 함수 위에서 처리된다.)

    ⚠️ 버전 폴더 이름에 cfg['table'] 을 그대로 쓴다. 못 찾았다고 다른 버전으로
      되돌아가지 않는다 — 조용한 버전 강등은 채점 결과를 통째로 무의미하게 만든다.
    """
    version = cfg["table"]
    out: list[Path] = []
    try:
        out.append(repo_root() / "data" / "preprocessed" / f"rfp_extraction_table_{version}")
    except Exception:
        pass
    try:
        out.append(resolve_path(cfg, "shared_data", "processed",
                                f"rfp_extraction_table_{version}"))
    except Exception:
        pass   # RAG_ROOT 없음 — 저장소 경로만으로 진행
    return out


def extraction_table_dir(cfg: dict[str, Any]) -> Path:
    """실제로 쓸 공식 추출표 폴더. 우선순위는 extraction_table_candidates 참조.

    두 곳에 **같은 버전**이 다 있는데 파일 지문이 다르면, 어느 쪽이 진짜인지
    코드가 판단할 수 없으므로 하나를 고르지 않고 중단한다.
    """
    version = cfg["table"]
    filename = f"extraction_table_{version}.json"
    candidates = extraction_table_candidates(cfg)
    found = [d for d in candidates if (d / filename).exists()]

    if not found:
        raise ExtractionTablePathError(
            f"공식 추출표 {version} 을 어디에서도 찾지 못했습니다. 찾아본 곳:\n"
            + "\n".join(f"  - {d / filename}" for d in candidates)
            + "\n--extraction-table 로 직접 지정하거나 공식 자료를 배치하세요. "
              "다른 버전으로 자동 대체하지 않습니다."
        )

    if len(found) > 1:
        digests = {str(d): _sha256(d / filename) for d in found}
        if len(set(digests.values())) > 1:
            detail = "\n".join(f"  - {k}/{filename}: {v}" for k, v in digests.items())
            raise ExtractionTablePathError(
                f"같은 버전({version})의 추출표가 두 곳에 있는데 내용이 다릅니다 — "
                f"어느 쪽이 공식인지 코드가 정할 수 없어 중단합니다.\n{detail}\n"
                f"--extraction-table 로 쓸 파일을 명시하세요."
            )
    return found[0]


def extraction_table_path(cfg: dict[str, Any]) -> Path:
    """공식 추출표 JSON 파일 — extraction_table_vN.json (rows 안에 1,200행).
    ⚠️ table.jsonl 같은 파일은 존재하지 않는다. 폴더 안 실제 파일명을
    가정할 수 없으면 --extraction-table로 파일 경로를 직접 넘긴다."""
    return extraction_table_dir(cfg) / f"extraction_table_{cfg['table']}.json"


def extraction_metadata_path(cfg: dict[str, Any]) -> Path:
    """공식 이름표 — extraction_metadata.json (VERSION.txt 아님)."""
    return extraction_table_dir(cfg) / "extraction_metadata.json"


def document_registry_dir(cfg: dict[str, Any]) -> Path:
    # base.yaml의 document_registry_version이 정본. 없으면 corpus 버전을 쓴다.
    version = cfg.get("document_registry_version", cfg["corpus"])
    return resolve_path(
        cfg, "shared_data", "processed", f"document_registry_{version}"
    )


def document_registry_path(cfg: dict[str, Any]) -> Path:
    """공식 문서 등록부 JSON 파일 — document_registry_v2.json (documents 안에
    100건, retrieval_eligible·duplicate_of_document_id 포함).
    ⚠️ registry.jsonl 같은 파일은 존재하지 않는다."""
    version = cfg.get("document_registry_version", cfg["corpus"])
    return document_registry_dir(cfg) / f"document_registry_{version}.json"


def index_dir(cfg: dict[str, Any]) -> Path:
    return resolve_path(cfg, "shared_data", "processed", f"index_{cfg['index']}")


def identity_path(cfg: dict[str, Any]) -> Path:
    """공식 identity_v2 — document_identity_v2.csv.

    ⚠️ 2026-09-02 (4-4 확정): 마감일·발주기관·사업명의 유일한 공식 출처.
    예전 코드가 쓰던 shared_data/raw/ 아래 원본 수집 CSV 경로는 폐기했다.
    (그 원본은 identity_v2를 만든 재료일 뿐, 실행 경로가 직접 읽지 않는다.)"""
    version = cfg.get("document_registry_version", cfg["corpus"])
    filename = cfg.get("identity_csv_filename", f"document_identity_{version}.csv")
    return document_registry_dir(cfg) / filename


def chunks_dir(cfg: dict[str, Any]) -> Path:
    """공식 청크 폴더 — chunks_v3. 청킹 버전으로 폴더가 갈린다."""
    return resolve_path(
        cfg, "shared_data", "processed", f"chunks_{cfg['chunking_version']}"
    )


def chunks_path(cfg: dict[str, Any]) -> Path:
    return chunks_dir(cfg) / "chunks.jsonl"


def chunks_version_file(cfg: dict[str, Any]) -> Path:
    """chunks_vN/VERSION.txt — 청크 파일 안의 값과 교차 확인할 공식 메타데이터."""
    return chunks_dir(cfg) / "VERSION.txt"


def models_home() -> Path:
    root = os.environ.get("HF_HOME")
    if not root:
        raise RuntimeError("HF_HOME 환경변수가 없습니다.")
    return Path(root)
