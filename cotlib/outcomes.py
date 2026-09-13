"""Price side: sessions, entry/exit selection, forward and trailing outcomes.

A "session" is a CME trade date with >= 30 one-minute bars and a reference close (drops the
two one-bar Sunday stubs). A "full" session has its reference bar at 14:59 CT. Entries and
exits are placed on full sessions only; holiday stubs and early closes still count for
daily returns and for highs/lows.
"""
import numpy as np
import pandas as pd

from cotlib.release import nominal_friday

HORIZONS_W = (1, 4, 13)
ANNUAL = np.sqrt(252)
TRAIL_SESSIONS = (20, 65)
# A session further than this from the date it stands for means the price data doesn't cover
# that date (e.g. E-mini reports from 2006-2010, before the price history starts).
MAX_STALE = pd.Timedelta(days=7)


def load_sessions(path):
    px = pd.read_parquet(path)
    px = px[(px["bars"] >= 30) & px["ref"].notna()].copy()
    px["full"] = px["ref_bar_ct"].dt.strftime("%H:%M").eq("14:59")
    return px.set_index("trade_date").sort_index()


def forward_window(s, i, j):
    """From the reference close of session i to the reference close of session j (positions).

    Highs/lows use only bars after the entry close (post segment of i), whole sessions strictly
    between, and bars before the exit close (pre segment of j).
    """
    p0 = s["ref_adj"].iat[i]
    lows = np.r_[s["low_post_adj"].iat[i], s["low_adj"].iloc[i + 1:j].to_numpy(), s["low_pre_adj"].iat[j]]
    highs = np.r_[s["high_post_adj"].iat[i], s["high_adj"].iloc[i + 1:j].to_numpy(), s["high_pre_adj"].iat[j]]
    r = np.diff(np.log(s["ref_adj"].iloc[i:j + 1].to_numpy()))
    return {
        "ret": s["ref_adj"].iat[j] / p0 - 1,
        "vol": r.std(ddof=1) * ANNUAL if len(r) >= 3 else np.nan,
        "mdd": np.nanmin(lows) / p0 - 1,
        "mru": np.nanmax(highs) / p0 - 1,
    }


def trailing_window(s, i, n):
    if i < n:
        return np.nan, np.nan
    ref = s["ref_adj"].iloc[i - n:i + 1].to_numpy()
    r = np.diff(np.log(ref))
    return r.std(ddof=1) * ANNUAL, ref[-1] / ref[0] - 1


def align(reports, s, rolls):
    """reports: report_date, publication_date (one code, sorted). Returns price columns per report."""
    idx, full_idx = s.index, s.index[s["full"]]
    last_day = idx[-1]
    rep = reports.reset_index(drop=True)
    n = len(rep)

    # Q-A: price at the as-of date = last session on or before it
    asof_pos = idx.searchsorted(rep["report_date"], side="right") - 1
    in_span = (asof_pos >= 0) & (rep["report_date"] <= last_day).to_numpy()
    in_span[in_span] = (rep["report_date"].to_numpy()[in_span] - idx[asof_pos[in_span]]) <= MAX_STALE
    out = pd.DataFrame({"asof_session": pd.NaT, "asof_px": np.nan}, index=rep.index)
    out.loc[in_span, "asof_session"] = idx[asof_pos[in_span]]
    out.loc[in_span, "asof_px"] = s["ref_adj"].to_numpy()[asof_pos[in_span]]

    cum_vol = s["volume_all"].cumsum().to_numpy()
    vol_week = np.full(n, np.nan)
    for k in range(1, n):
        if in_span[k] and in_span[k - 1]:
            vol_week[k] = cum_vol[asof_pos[k]] - cum_vol[asof_pos[k - 1]]
    out["volume_week"] = vol_week

    roll_dates = pd.DatetimeIndex(rolls["roll_date"])
    prev = rep["report_date"].shift(1)
    out["roll_in_week"] = [bool(((roll_dates > p) & (roll_dates <= d)).any()) if pd.notna(p) else False
                           for p, d in zip(prev, rep["report_date"])]

    # Q-B: entry = first full session after the nominal Friday and on/after publication
    friday = nominal_friday(rep["report_date"])
    earliest = np.maximum(friday + pd.Timedelta(days=1), pd.DatetimeIndex(rep["publication_date"]))
    k_entry = full_idx.searchsorted(earliest, side="left")
    entry = pd.Series([full_idx[k] if k < len(full_idx) and full_idx[k] - day <= MAX_STALE else pd.NaT
                       for k, day in zip(k_entry, earliest)])
    out["nominal_friday"] = friday
    out["entry_date"] = entry

    pos = {d: p for p, d in enumerate(idx)}
    records = []
    for r, e in enumerate(entry):
        rec = {}
        if pd.isna(e):
            records.append(rec)
            continue
        i = pos[e]
        rec["entry_px"] = s["ref_adj"].iat[i]
        for m in TRAIL_SESSIONS:
            rec[f"trail_vol_{m}d"], rec[f"trail_ret_{m}d"] = trailing_window(s, i, m)
        for h in HORIZONS_W:
            k = full_idx.searchsorted(e + pd.Timedelta(weeks=h), side="left")
            if k >= len(full_idx):
                continue
            x = full_idx[k]
            rec[f"exit_{h}w"] = x
            for key, val in forward_window(s, i, pos[x]).items():
                rec[f"fwd_{key}_{h}w"] = val
        records.append(rec)
    return pd.concat([out, pd.DataFrame(records, index=rep.index)], axis=1)
