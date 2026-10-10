"""Phase 5 export: model outputs appear only where they were scored."""
import math

import pytest

pd = pytest.importorskip("pandas")

from filing_signals import export  # noqa: E402


def test_unscored_reports_omit_model_fields_but_unknown_answers_stay():
    df = pd.DataFrame([
        # outside the test years: no Phase 3 outputs at all
        {"accession": "a", "risk_pct": math.nan, "event_next_12m": None,
         "answerable": None, "qlora_usd": math.nan, "zero_shot_usd": math.nan},
        # a test report where the model answered "unknown"
        {"accession": "b", "risk_pct": 0.42, "event_next_12m": False,
         "answerable": True, "qlora_usd": math.nan, "zero_shot_usd": 1.5e9},
    ])
    a, b = export._records(df)
    assert "qlora_usd" not in a and "risk_pct" not in a
    assert b["qlora_usd"] is None and b["zero_shot_usd"] == 1.5e9 and b["risk_pct"] == 0.42


def test_clean_rounds_small_values_and_keeps_large_ones_whole():
    assert export._clean(0.987654321) == 0.9877
    assert export._clean(68_636_146_000.0) == 68_636_146_000
    assert export._clean(math.nan) is None
