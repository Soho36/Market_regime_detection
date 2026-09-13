"""Step 3 — one row per COT report per contract code: features, release timing, aligned NQ outcomes.

in : data/raw/cftc/tff_nasdaq.csv, data/prices/nq_daily.parquet, data/prices/nq_rolls.csv
out: data/panel/cot_panel.parquet

Columns
- report_date, publication_date, release_quality       (cotlib.release)
- <category>_{long_pct,short_pct,net_pct,d1,d4,pct3y}, oi, oi_chg4   (cotlib.features)
- asof_session, asof_px, volume_week, roll_in_week       price at the as-of date (Q-A)
- entry_date, entry_px, exit_{h}w, fwd_{ret,vol,mdd,mru}_{h}w, trail_*   (Q-B, cotlib.outcomes)
Prices are back-adjusted reference closes (15:00 CT).
"""
import pandas as pd

import config as C
from cotlib.features import build_features
from cotlib.outcomes import HORIZONS_W, align, load_sessions
from cotlib.release import DELAYED, release_table


def main():
    C.PANEL.mkdir(parents=True, exist_ok=True)
    cot = pd.read_csv(C.RAW_CFTC / "tff_nasdaq.csv", dtype={"code": str}, parse_dates=["report_date"])
    sessions = load_sessions(C.PRICES / "nq_daily.parquet")
    rolls = pd.read_csv(C.PRICES / "nq_rolls.csv", parse_dates=["roll_date"])

    primary_dates = set(cot.loc[cot["code"] == C.COT_PRIMARY, "report_date"])
    missing = [d for d in DELAYED if pd.Timestamp(d) not in primary_dates]
    assert not missing, f"DELAYED as-of dates not found in {C.COT_PRIMARY}: {missing}"

    parts = []
    for code, g in cot.groupby("code"):
        g = g.sort_values("report_date").reset_index(drop=True)
        feats = build_features(g, breaks=C.COT_BREAKS.get(code, ()))
        rel = release_table(g["report_date"])
        px = align(rel, sessions, rolls)
        part = pd.concat([feats, rel.drop(columns="report_date"), px], axis=1)
        part.insert(0, "code", code)
        part.insert(1, "series", C.COT_CODES[code])
        part.insert(2, "primary", code == C.COT_PRIMARY)
        parts.append(part)
    panel = pd.concat(parts, ignore_index=True)
    panel.to_parquet(C.PANEL / "cot_panel.parquet", index=False)
    qa(panel, sessions)


def qa(panel, sessions):
    print(f"panel rows {len(panel)}; price sessions {sessions.index[0].date()} -> {sessions.index[-1].date()}")
    for code, g in panel.groupby("code"):
        print(f"\n{code} ({g['series'].iat[0]}{', PRIMARY' if g['primary'].iat[0] else ''}): {len(g)} reports; "
              f"with as-of price {g['asof_px'].notna().sum()}; with entry {g['entry_date'].notna().sum()}; "
              + "; ".join(f"fwd {h}w {g[f'fwd_ret_{h}w'].notna().sum()}" for h in HORIZONS_W))

    p = panel[panel["primary"] & panel["entry_date"].notna()].copy()
    full_idx = sessions.index[sessions["full"]]
    lag = full_idx.searchsorted(p["entry_date"]) - full_idx.searchsorted(p["nominal_friday"], side="right")
    print("\nprimary: entry weekday", p["entry_date"].dt.day_name().value_counts().to_dict())
    print("primary: full sessions between nominal Friday and entry", pd.Series(lag).value_counts().sort_index().to_dict())
    print("primary: release quality", p["release_quality"].value_counts().to_dict())
    print("\ndelayed reports (primary):")
    print(p.loc[p["release_quality"] != "normal", ["report_date", "publication_date", "release_quality", "entry_date"]]
          .to_string(index=False))

    span = p[p["fwd_ret_13w"].notna()]
    feat_cols = [c for c in p.columns if c.endswith(("_net_pct", "_d1", "_d4", "_pct3y"))]
    print(f"\nprimary rows with 13w outcome: {len(span)} ({span['report_date'].min().date()} -> "
          f"{span['report_date'].max().date()}); missing feature values in those rows:",
          {c: int(n) for c, n in span[feat_cols].isna().sum().items() if n})
    cols = [f"fwd_{k}_{h}w" for h in HORIZONS_W for k in ("ret", "vol", "mdd", "mru")]
    print("\nprimary forward outcomes (mean / median):")
    print(span[cols].agg(["mean", "median", "min", "max"]).T.round(4).to_string())
    print("share of rows where the forward low never went below entry (mdd > 0):",
          {h: round((span[f"fwd_mdd_{h}w"] > 0).mean(), 3) for h in HORIZONS_W})
    print("\nprimary latest 3 rows:")
    print(p[["report_date", "publication_date", "entry_date", "exit_4w", "asset_mgr_net_pct",
             "lev_money_net_pct", "dealer_net_pct", "asset_mgr_pct3y", "fwd_ret_4w"]].tail(3).to_string(index=False))


if __name__ == "__main__":
    main()
