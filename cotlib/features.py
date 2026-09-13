"""Positioning features from one COT series (one contract code), in report order.

All position features are % of open interest, so they are unit-free (the consolidated series
switched from $100 to $20 units on 2023-05-02). Weekly changes are left empty where reports
are missing or where the window crosses a definition break.
"""
import numpy as np
import pandas as pd

CATEGORIES = {
    "dealer": ("dealer_positions_long_all", "dealer_positions_short_all"),
    "asset_mgr": ("asset_mgr_positions_long", "asset_mgr_positions_short"),
    "lev_money": ("lev_money_positions_long", "lev_money_positions_short"),
    "other_rept": ("other_rept_positions_long", "other_rept_positions_short"),
    "nonrept": ("nonrept_positions_long_all", "nonrept_positions_short_all"),
}
PCT_WINDOW = 156  # reports ≈ 3 years


def rolling_percentile(x, window=PCT_WINDOW):
    """Share of the last `window` values (current included) that are <= the current value."""
    return x.rolling(window, min_periods=window).apply(lambda w: (w <= w[-1]).mean(), raw=True)


def build_features(cot, breaks=()):
    """cot: rows of one code sorted by report_date. breaks: dates where the series definition changes."""
    cot = cot.sort_values("report_date").reset_index(drop=True)
    date = cot["report_date"]
    oi = cot["open_interest_all"].astype(float)

    # a k-report change is valid only if those k reports span ~k weeks and no definition break
    era = np.searchsorted(pd.DatetimeIndex(pd.to_datetime(list(breaks))).sort_values(), date, side="right")
    era = pd.Series(era, index=cot.index)

    def valid(k):
        span_ok = (date - date.shift(k)).dt.days <= 7 * k + 1
        return span_ok & era.eq(era.shift(k))

    ok1, ok4 = valid(1), valid(4)
    out = pd.DataFrame({"report_date": date, "oi": oi})
    for cat, (long_col, short_col) in CATEGORIES.items():
        long_pct = cot[long_col] / oi * 100
        short_pct = cot[short_col] / oi * 100
        net = long_pct - short_pct
        out[f"{cat}_long_pct"] = long_pct
        out[f"{cat}_short_pct"] = short_pct
        out[f"{cat}_net_pct"] = net                                   # F1
        out[f"{cat}_d1"] = net.diff(1).where(ok1)                    # F2
        out[f"{cat}_d4"] = net.diff(4).where(ok4)                    # F3
        out[f"{cat}_pct3y"] = rolling_percentile(net)                 # F4
    out["oi_chg4"] = (oi / oi.shift(4) - 1).where(ok4)               # O1
    out["weeks_since_prev"] = (date - date.shift(1)).dt.days / 7
    return out
