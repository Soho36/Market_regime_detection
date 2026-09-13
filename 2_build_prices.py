"""Step 2 — raw Databento 1-minute NQ bars -> roll-adjusted daily series.

in : data/raw/databento/databento-ohlcv-1m.csv  (GLBX.MDP3 ohlcv-1m, parent NQ.FUT, UTC)
out: data/prices/nq_contract_daily.parquet  one row per contract x CME trade date
     data/prices/nq_daily.parquet           front contract per trade date, raw and back-adjusted
     data/prices/nq_rolls.csv               every roll: date, from, to, ratio

Conventions
- Only outright quarterly contracts (NQH/M/U/Z + year digit). Calendar spreads are dropped.
- The symbol's year digit repeats every decade (NQU0 = Sep 2010 and Sep 2020): the expiry is
  the first third-Friday of that month, in a year ending in that digit, on or after the bar's trade date.
- Trade date: CME session starts 17:00 CT, so trade_date = (time in CT + 7h).date().
- Reference close ("ref"): close of the last bar starting before 15:00 CT (normally the 14:59
  bar) = 16:00 ET cash close. This is the daily price COT positions are compared against.
- Each day's range is split at 15:00 CT ("pre" / "post") so a window that starts or ends on
  a reference close can exclude bars on the wrong side of it.
- Front contract: highest daily volume, never stepping back to an earlier expiry.
- Back-adjustment is multiplicative. On the first day of a new front, all earlier prices are
  scaled by ref_new / ref_old on the previous trade date. The latest prices stay real.
"""
import re

import numpy as np
import pandas as pd

import config as C

MONTHS = {"H": 3, "M": 6, "U": 9, "Z": 12}
OUTRIGHT = re.compile(r"^NQ([HMUZ])(\d)$")
PRICE_COLS = ["open", "high", "low", "close", "ref", "high_pre", "low_pre", "high_post", "low_post"]


def third_friday(year, month):
    first = pd.Timestamp(year=year, month=month, day=1)
    return first + pd.Timedelta(days=(4 - first.dayofweek) % 7 + 14)


def expiry_for(symbol, trade_date):
    month_code, digit = OUTRIGHT.match(symbol).groups()
    month = MONTHS[month_code]
    year = trade_date.year - trade_date.year % 10 + int(digit)
    while third_friday(year, month) < trade_date:
        year += 10
    return third_friday(year, month)


def load_bars():
    df = pd.read_csv(
        C.NQ_1M_RAW,
        usecols=["ts_event", "instrument_id", "open", "high", "low", "close", "volume", "symbol"],
        dtype={"instrument_id": "int64", "volume": "int64", "symbol": "category"},
    )
    cats = pd.Series(df["symbol"].cat.categories)
    outright = cats[cats.str.match(OUTRIGHT.pattern)]
    spreads = cats[cats.str.contains("-")]
    other = cats[~cats.isin(outright) & ~cats.isin(spreads)]
    keep = df["symbol"].isin(outright)
    print(f"raw rows {len(df):,}; spread rows dropped {df['symbol'].isin(spreads).sum():,}; "
          f"other symbols dropped {sorted(other)}")
    df = df[keep].copy()
    df["symbol"] = df["symbol"].astype(str)

    ts_ct = pd.to_datetime(df["ts_event"], utc=True, format="ISO8601").dt.tz_convert(C.EXCHANGE_TZ)
    local = ts_ct.dt.tz_localize(None)
    df["ts_ct"] = local
    df["trade_date"] = (local + pd.Timedelta(hours=7)).dt.normalize()
    # Session runs 17:00 CT -> next day; bars from 17:00 are before the ref close, 15:00 onward after it
    hour = local.dt.hour
    df["pre"] = (hour >= 17) | (hour < C.REF_CLOSE_HOUR_CT)
    df.drop(columns="ts_event", inplace=True)

    keys = df[["symbol", "trade_date"]].drop_duplicates()
    keys["expiry"] = [expiry_for(s, d) for s, d in zip(keys["symbol"], keys["trade_date"])]
    df = df.merge(keys, on=["symbol", "trade_date"], how="left")
    far = (df["expiry"] - df["trade_date"]).dt.days > 3 * 365
    print(f"outright rows {len(df):,}; rows with expiry > 3y ahead (ambiguous decode): {far.sum()}")
    ids = df.groupby("instrument_id")["expiry"].nunique()
    print(f"instrument_ids {len(ids)}; ids mapping to more than one expiry: {(ids > 1).sum()}")
    return df.sort_values(["expiry", "ts_ct"]).reset_index(drop=True)


def contract_daily(df):
    keys = ["expiry", "trade_date"]
    g = df.groupby(keys, sort=True)
    daily = g.agg(symbol=("symbol", "first"), open=("open", "first"), high=("high", "max"),
                  low=("low", "min"), close=("close", "last"), volume=("volume", "sum"),
                  bars=("close", "size"), last_bar_ct=("ts_ct", "last"))
    pre = df[df["pre"]].groupby(keys).agg(ref=("close", "last"), ref_bar_ct=("ts_ct", "last"),
                                          high_pre=("high", "max"), low_pre=("low", "min"))
    post = df[~df["pre"]].groupby(keys).agg(high_post=("high", "max"), low_post=("low", "min"))
    daily = daily.join(pre).join(post).reset_index()
    daily["contract"] = daily["symbol"].str[:3] + daily["expiry"].dt.year.astype(str)
    return daily


def front_series(daily):
    vol = daily.pivot(index="trade_date", columns="expiry", values="volume")
    front = vol.idxmax(axis=1).cummax().rename("expiry")
    assert (front.values >= front.index.values).all(), "front contract expired before its trade date"

    by_key = daily.set_index(["expiry", "trade_date"])
    fr = by_key.loc[list(zip(front.values, front.index))].reset_index()
    fr["volume_all"] = vol.sum(axis=1).values
    fr["roll"] = fr["expiry"].ne(fr["expiry"].shift()) & fr.index.to_series().gt(0)

    rolls, ratio = [], pd.Series(1.0, index=fr.index)
    for i in fr.index[fr["roll"]]:
        prev_date, old, new = fr.at[i - 1, "trade_date"], fr.at[i - 1, "expiry"], fr.at[i, "expiry"]
        old_ref = by_key.at[(old, prev_date), "ref"]
        new_ref = by_key.at[(new, prev_date), "ref"] if (new, prev_date) in by_key.index else np.nan
        ratio[i] = new_ref / old_ref
        rolls.append({"roll_date": fr.at[i, "trade_date"], "from": fr.at[i - 1, "contract"],
                      "to": fr.at[i, "contract"], "prev_date": prev_date,
                      "old_ref": old_ref, "new_ref": new_ref, "ratio": ratio[i],
                      "trading_days_to_old_expiry": int(((fr["trade_date"] > prev_date)
                                                        & (fr["trade_date"] <= old)).sum())})
    rolls = pd.DataFrame(rolls)
    if rolls["ratio"].isna().any():
        raise ValueError(f"missing ref price for roll(s):\n{rolls[rolls['ratio'].isna()]}")

    # factor[d] = product of ratios of all rolls strictly after d
    fr["adj_factor"] = ratio[::-1].cumprod()[::-1].shift(-1).fillna(1.0).values
    for col in PRICE_COLS:
        fr[f"{col}_adj"] = fr[col] * fr["adj_factor"]
    return fr, rolls


def qa(fr, rolls):
    print(f"\ntrade dates {len(fr)}: {fr['trade_date'].min().date()} -> {fr['trade_date'].max().date()}")
    print("weekday:", fr["trade_date"].dt.day_name().value_counts().to_dict())
    print("per year:", fr.groupby(fr["trade_date"].dt.year).size().to_dict())
    ref_t = fr["ref_bar_ct"].dt.strftime("%H:%M").value_counts()
    print("ref bar start time (CT), top:", ref_t.head(6).to_dict())

    print(f"\nrolls {len(rolls)}; per year:", rolls.groupby(rolls["roll_date"].dt.year).size().to_dict())
    print("trading days from roll to old expiry:",
          rolls["trading_days_to_old_expiry"].describe()[["min", "50%", "max"]].to_dict())
    print(rolls.assign(ratio_pct=((rolls["ratio"] - 1) * 100).round(3))
          [["roll_date", "from", "to", "trading_days_to_old_expiry", "ratio_pct"]]
          .to_string(index=False))

    gap_raw = fr["open"] / fr["close"].shift() - 1
    gap_adj = fr["open_adj"] / fr["close_adj"].shift() - 1
    on_roll = fr["roll"]
    print("\nopen gap vs previous close, % (median |gap| / max |gap|)")
    print(f"  roll days  raw {gap_raw[on_roll].abs().median() * 100:.3f} / {gap_raw[on_roll].abs().max() * 100:.3f}"
          f"   adjusted {gap_adj[on_roll].abs().median() * 100:.3f} / {gap_adj[on_roll].abs().max() * 100:.3f}")
    print(f"  other days raw {gap_raw[~on_roll].abs().median() * 100:.3f} / {gap_raw[~on_roll].abs().max() * 100:.3f}")

    r_raw = fr["ref"] / fr["ref"].shift() - 1
    r_adj = fr["ref_adj"] / fr["ref_adj"].shift() - 1
    diff = (r_raw - r_adj)[~on_roll].abs().max()
    print(f"max |raw - adjusted| daily ref return on non-roll days: {diff:.2e}")

    yearly = fr.groupby(fr["trade_date"].dt.year)["ref_adj"].agg(["first", "min", "max", "last"])
    yearly["ret_pct"] = (yearly["last"] / yearly["first"] - 1) * 100
    print("\nref_adj by year\n", yearly.round(2).to_string())


def main():
    C.PRICES.mkdir(parents=True, exist_ok=True)
    bars = load_bars()
    daily = contract_daily(bars)
    del bars
    daily.to_parquet(C.PRICES / "nq_contract_daily.parquet", index=False)
    fr, rolls = front_series(daily)
    fr.to_parquet(C.PRICES / "nq_daily.parquet", index=False)
    rolls.to_csv(C.PRICES / "nq_rolls.csv", index=False)
    qa(fr, rolls)


if __name__ == "__main__":
    main()
