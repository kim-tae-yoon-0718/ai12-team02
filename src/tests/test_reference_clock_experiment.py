"""reference_clock_experiment — now() 실험 진입점이 공식 경로와 분리돼 있는지 확인."""
from datetime import datetime, timedelta

import pytest


class TestReferenceClockExperiment:
    def test_now_reference_datetime_is_close_to_actual_now(self):
        from reference_clock_experiment import now_reference_datetime
        before = datetime.now()
        got = now_reference_datetime()
        after = datetime.now()
        assert before <= got <= after

    def test_resolve_defers_to_official_path_when_not_now(self, base_cfg):
        from reference_clock_experiment import resolve_reference_datetime
        assert resolve_reference_datetime(base_cfg) == datetime(2024, 6, 1)

    def test_resolve_uses_now_only_when_experimental_flag_set(self, base_cfg):
        from reference_clock_experiment import resolve_reference_datetime
        cfg = dict(base_cfg)
        cfg["experimental_reference_datetime_source"] = "now"
        before = datetime.now()
        got = resolve_reference_datetime(cfg)
        after = datetime.now()
        assert before <= got <= after
        # 고정값(2024-06-01)과는 확실히 다르다
        assert got - datetime(2024, 6, 1) > timedelta(days=1)

    def test_official_function_still_rejects_now(self, base_cfg):
        """4-10-2 확정: 공식 경로는 여전히 external/None만 받는다 — 이 실험 모듈이
        존재해도 공식 함수의 계약은 바뀌지 않는다."""
        from identity_metadata import reference_datetime_from_config
        bad = dict(base_cfg)
        bad["reference_datetime_source"] = "now"
        with pytest.raises(RuntimeError):
            reference_datetime_from_config(bad)
