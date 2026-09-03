"""grader.diagnostics — 평가 결과 진단 (점수 하나로는 뭘 고칠지 모른다, 부품별로 나눈다).

  statistics.py    집계 프리미티브 (cell_report / severity_* / abstention / format / integrated)
  report.py        3-12 최종 집계 리포트 + 3-1-0 진단표 (full_report / diagnose / render_table)
  retrieval.py     3-2 검색 결과 집계
  citation.py      3-4-3 출처 좌표 집계
  extraction.py    3-2-1/3-2-2 추출·문서특정 집계 + D7 순환 경보
  regression.py    3-13 변동 폭 / 3-13-1 회귀 판정 / 3-17 원인 분리
  contamination.py 4-9 실험 오염 방지
"""

from __future__ import annotations

from .citation import aggregate_citation
from .contamination import ExperimentRecord, check_contamination
from .extraction import aggregate_doc_selection, circularity_flag
from .regression import (
    RegressionPolicy,
    attribute_change,
    cell_sizes_of,
    flatten_report,
    judge_regression,
    measure_variance,
    pp_per_item,
)
from .report import (
    DEFAULT_THRESHOLDS,
    STAGE_MAP,
    classify,
    diagnose,
    full_report,
    main,
    main_metrics,
    render_table,
)
from .retrieval import aggregate_retrieval
from .statistics import (
    MIN_CELL,
    abstention_report,
    by_axis,
    cell_report,
    format_report,
    integrated_score,
    severity_report,
    severity_weighted_score,
)

__all__ = [
    # statistics
    "MIN_CELL", "by_axis", "cell_report", "severity_report", "severity_weighted_score",
    "abstention_report", "format_report", "integrated_score",
    # report / 진단표
    "full_report", "main_metrics", "render_table", "classify", "diagnose",
    "DEFAULT_THRESHOLDS", "STAGE_MAP", "main",
    # 집계
    "aggregate_retrieval", "aggregate_citation", "aggregate_doc_selection", "circularity_flag",
    # regression / contamination
    "RegressionPolicy", "judge_regression", "measure_variance", "attribute_change",
    "pp_per_item", "flatten_report", "cell_sizes_of",
    "ExperimentRecord", "check_contamination",
]
