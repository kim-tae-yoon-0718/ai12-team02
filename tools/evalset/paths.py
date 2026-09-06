"""평가셋·추출표 후보 생성 도구의 공통 경로 규칙.

원칙(2026-09-04 §13)
  · 저장소 루트는 **이 파일 위치에서 계산**하거나 `--repo-root` 로 명시한다.
  · 공식 자료 루트(/srv/rfp)는 기본값이지만 `--data-root` 로 바꿀 수 있다.
  · 후보 평가셋·후보 추출표·결정 CSV·출력 폴더는 전부 인자로 받는다.
  · 다른 사람의 작업 폴더를 조용히 읽지 않는다 — 마커가 없으면 즉시 실패한다.
  · 필수 파일이 없으면 **어느 파일이 없는지** 이름을 찍고 멈춘다.
"""
from __future__ import annotations

import os
from pathlib import Path

# 저장소 루트임을 확인하는 마커 — 둘 다 있어야 한다(우연히 상위 폴더를 잡지 않게)
REPO_MARKERS = ("pyproject.toml", "src/checks/selection_policy.py")
DEFAULT_DATA_ROOT = Path("/srv/rfp")


class MissingInput(SystemExit):
    """필수 입력이 없을 때 — 어느 파일인지 이름을 담아 종료한다."""

    def __init__(self, label: str, path: Path | str, hint: str = "") -> None:
        super().__init__(f"❌ 필수 입력이 없습니다 — {label}: {path}"
                         + (f"\n   {hint}" if hint else ""))


def _has_markers(root: Path) -> bool:
    return all((root / m).exists() for m in REPO_MARKERS)


def repo_root(explicit: str | Path | None = None) -> Path:
    """저장소 루트. 우선순위: --repo-root → RFP_REPO_ROOT → 이 파일에서 위로 탐색."""
    if explicit:
        root = Path(explicit).expanduser().resolve()
        if not _has_markers(root):
            raise MissingInput("--repo-root 가 저장소 루트가 아님", root,
                               f"다음 파일이 모두 있어야 합니다: {', '.join(REPO_MARKERS)}")
        return root
    env = os.environ.get("RFP_REPO_ROOT")
    if env:
        return repo_root(env)
    here = Path(__file__).resolve()
    for cand in here.parents:
        if _has_markers(cand):
            return cand
    raise MissingInput(
        "저장소 루트를 찾지 못함", here,
        "`--repo-root <저장소>` 로 직접 지정하거나 RFP_REPO_ROOT 환경변수를 설정하세요. "
        f"마커: {', '.join(REPO_MARKERS)}")


def data_root(explicit: str | Path | None = None) -> Path:
    """공식 자료 루트(기본 /srv/rfp). 존재하지 않으면 즉시 실패한다."""
    root = Path(explicit).expanduser().resolve() if explicit else DEFAULT_DATA_ROOT
    if not root.exists():
        raise MissingInput("공식 자료 루트(--data-root)", root)
    return root


def official_paths(data_root_path: Path) -> dict[str, Path]:
    p = data_root_path / "shared_data/processed"
    return {
        "registry": p / "document_registry_v2/document_registry_v2.json",
        "identity": p / "document_registry_v2/document_identity_v2.csv",
        "table": p / "rfp_extraction_table_v3/extraction_table_v3.json",
        "chunks": p / "chunks_v3/chunks.jsonl",
        "corpus": p / "corpus_v2",
        "index": p / "index_v2",
        "official_items": data_root_path / "evalset/v1/items.jsonl",
        "practice_items": data_root_path / "evalset/practice_items.jsonl",
        "data_list_csv": data_root_path / "shared_data/raw/data_list.csv",
    }


def require(label: str, path: Path | str | None, hint: str = "") -> Path:
    """필수 입력 확인 — 없으면 어느 파일인지 이름을 찍고 멈춘다."""
    if path is None:
        raise MissingInput(label, "(인자 없음)", hint)
    p = Path(path).expanduser()
    if not p.exists():
        raise MissingInput(label, p, hint)
    return p.resolve()


def add_path_args(parser, *, need_table: bool = False) -> None:
    """모든 생성 도구가 같은 인자 이름을 쓰게 한다."""
    parser.add_argument("--repo-root", default=None,
                        help="저장소 루트(기본: 이 파일 위치에서 자동 탐색)")
    parser.add_argument("--data-root", default=None,
                        help=f"공식 자료 루트(기본 {DEFAULT_DATA_ROOT})")
    parser.add_argument("--table", default=None,
                        required=need_table,
                        help="후보 추출표 JSON(없으면 공식 표)")
