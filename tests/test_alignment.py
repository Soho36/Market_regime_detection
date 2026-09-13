"""Lookahead and alignment tests. Run from the project root: python -m pytest tests"""
import numpy as np
import pandas as pd
import pytest

import config as C
from cotlib.features import build_features
from cotlib.outcomes import forward_window
from cotlib.release import DELAYED, nominal_friday

PANEL = C.PANEL / "cot_panel.parquet"


@pytest.fixture(scope="module")
def panel():
    if not PANEL.exists():
        pytest.skip("run 3_build_panel.py first")
    return pd.read_parquet(PANEL)


@pytest.fixture(scope="module")
def cot():
    return pd.read_csv(C.RAW_CFTC / "tff_nasdaq.csv", dtype={"code": str}, parse_dates=["report_date"])


def test_entry_strictly_after_nominal_friday_and_publication(panel):
    p = panel[panel["entry_date"].notna()]
    assert (p["entry_date"] > nominal_friday(p["report_date"])).all()
    assert (p["entry_date"] >= p["publication_date"]).all()
    assert (p["publication_date"] >= nominal_friday(p["report_date"])).all()


def test_entry_close_to_release(panel):
    """Guards against snapping a report to a distant session (e.g. 2006 reports onto 2010 prices)."""
    p = panel[panel["entry_date"].notna()]
    earliest = np.maximum(nominal_friday(p["report_date"]) + pd.Timedelta(days=1),
                          pd.DatetimeIndex(p["publication_date"]))
    assert ((p["entry_date"] - earliest) <= pd.Timedelta(days=7)).all()


def test_no_prices_for_reports_outside_price_history(panel):
    """No as-of price before the first session; no entry for reports usable > 7 days before it.

    A report released just before the first session (2010-06-01, out 06-04) may still enter on it.
    """
    first_session = panel["asof_session"].min()
    assert panel.loc[panel["report_date"] < first_session, "asof_px"].isna().all()
    earliest = np.maximum(nominal_friday(panel["report_date"]) + pd.Timedelta(days=1),
                          pd.DatetimeIndex(panel["publication_date"]))
    before = panel[earliest < first_session - pd.Timedelta(days=7)]
    assert len(before) > 0
    assert before["entry_date"].isna().all() and before["fwd_ret_1w"].isna().all()


def test_asof_session_recent(panel):
    p = panel[panel["asof_session"].notna()]
    assert ((p["report_date"] - p["asof_session"]) <= pd.Timedelta(days=7)).all()


def test_exits_after_entry_and_ordered(panel):
    p = panel[panel["exit_13w"].notna()]
    assert (p["exit_1w"] > p["entry_date"]).all()
    assert (p["exit_4w"] > p["exit_1w"]).all()
    assert (p["exit_13w"] > p["exit_4w"]).all()


def test_asof_price_not_after_report_date(panel):
    p = panel[panel["asof_session"].notna()]
    assert (p["asof_session"] <= p["report_date"]).all()


def test_delayed_reports_wait_for_publication(panel):
    p = panel[panel["primary"]].set_index("report_date")
    for as_of, (pub, _) in DELAYED.items():
        row = p.loc[pd.Timestamp(as_of)]
        if pd.notna(row["entry_date"]):
            assert row["entry_date"] >= pd.Timestamp(pub), as_of


@pytest.mark.parametrize("cut", [200, 500, 800, 1000])
def test_features_ignore_future_reports(cot, cut):
    g = cot[cot["code"] == C.COT_PRIMARY].sort_values("report_date").reset_index(drop=True)
    full = build_features(g)
    truncated = build_features(g.iloc[:cut])
    pd.testing.assert_frame_equal(full.iloc[:cut].reset_index(drop=True), truncated, check_exact=False)


def _sessions(n=6):
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    base = pd.DataFrame({"ref_adj": 100.0, "low_adj": 99.0, "high_adj": 101.0,
                         "low_pre_adj": 99.0, "high_pre_adj": 101.0,
                         "low_post_adj": 99.5, "high_post_adj": 100.5}, index=idx)
    base["ref_adj"] = [100, 101, 102, 101, 103, 104]
    return base


def test_forward_window_excludes_bars_before_entry_and_after_exit():
    s = _sessions()
    clean = forward_window(s, 1, 4)
    s.loc[s.index[1], "low_pre_adj"] = 1.0      # crash before the entry close
    s.loc[s.index[4], "high_post_adj"] = 1e6    # spike after the exit close
    s.loc[s.index[0], "low_adj"] = 1.0          # crash on an earlier session
    assert forward_window(s, 1, 4) == clean


def test_forward_window_values():
    s = _sessions()
    out = forward_window(s, 1, 4)
    assert out["ret"] == pytest.approx(103 / 101 - 1)
    assert out["mdd"] == pytest.approx(99.0 / 101 - 1)
    assert out["mru"] == pytest.approx(101.0 / 101 - 1)
    r = np.diff(np.log([101, 102, 101, 103]))
    assert out["vol"] == pytest.approx(r.std(ddof=1) * np.sqrt(252))
