"""Step 1 — download CFTC TFF (futures-only) history for the Nasdaq-100 contracts.

out: data/raw/cftc/tff_nasdaq.json   API response as received
     data/raw/cftc/tff_nasdaq.csv    same rows, numeric columns typed, sorted
     data/raw/cftc/fetch_log.json    urls, fetch time, rows per code, sha256 of both files

Checks printed at the end (nothing is dropped or fixed here):
- report dates per code: span, count, weekday, duplicates, gaps other than 7 days
- consolidated vs E-mini (+ Micro/10) — how CFTC builds the consolidated series
"""
import hashlib
import json
from datetime import datetime, timezone

import pandas as pd
import requests
import truststore

import config as C

# Verify HTTPS against the Windows certificate store (certifi's bundle lacks this machine's root)
truststore.inject_into_ssl()

PAGE = 50_000
# Identifier columns that look numeric but must stay text
ID_COLS = {"id", "cftc_contract_market_code", "cftc_market_code", "cftc_region_code",
           "cftc_commodity_code", "cftc_subgroup_code"}


def fetch():
    codes = ", ".join(f"'{c}'" for c in C.COT_CODES)
    params = {
        "$where": f"cftc_contract_market_code in ({codes})",
        "$order": "report_date_as_yyyy_mm_dd, cftc_contract_market_code",
        "$limit": PAGE,
        "$offset": 0,
    }
    rows, urls = [], []
    while True:
        r = requests.get(C.CFTC_TFF_URL, params=params, timeout=120)
        r.raise_for_status()
        batch = r.json()
        rows.extend(batch)
        urls.append(r.url)
        if len(batch) < PAGE:
            return rows, urls
        params["$offset"] += PAGE


def to_frame(rows):
    df = pd.DataFrame(rows)
    for col in df.columns.difference(ID_COLS):
        num = pd.to_numeric(df[col], errors="coerce")
        if num.notna().sum() == df[col].notna().sum():  # convert only fully numeric columns
            df[col] = num
    df["code"] = df["cftc_contract_market_code"].str.strip()
    df["series"] = df["code"].map(C.COT_CODES)
    df["report_date"] = pd.to_datetime(df["report_date_as_yyyy_mm_dd"]).dt.normalize()
    return df.sort_values(["code", "report_date"]).reset_index(drop=True)


def check_dates(df):
    for code, g in df.groupby("code"):
        d = g["report_date"].reset_index(drop=True)
        step = d.diff().dt.days
        print(f"\n{code} ({C.COT_CODES[code]}): {len(d)} reports, "
              f"{d.iloc[0].date()} -> {d.iloc[-1].date()}, duplicates {d.duplicated().sum()}")
        print("  report-date weekday:", d.dt.day_name().value_counts().to_dict())
        odd = step.index[step.notna() & step.ne(7)]
        print(f"  gaps other than 7 days: {len(odd)}")
        for i in odd:
            print(f"    {d[i - 1].date()} -> {d[i].date()}  ({int(step[i])} days)")


def check_consolidation(df):
    print("\nConsolidated vs components, ratio by year (median / min / max)")
    for field in ("open_interest_all", "asset_mgr_positions_long", "lev_money_positions_short"):
        p = df.pivot_table(index="report_date", columns="code", values=field)
        both = p.dropna(subset=["20974+", "209742"])
        micro = p.get("209747", pd.Series(dtype=float)).reindex(both.index)
        ratios = pd.DataFrame({
            "cons/emini": both["20974+"] / both["209742"],
            "cons/(emini+micro/10)": both["20974+"] / (both["209742"] + micro.fillna(0) / 10),
            "cons/(emini+micro)": both["20974+"] / (both["209742"] + micro.fillna(0)),
        })
        summary = ratios.groupby(ratios.index.year).agg(["median", "min", "max"]).round(3)
        print(f"\n  {field}")
        print(summary.to_string())


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    C.RAW_CFTC.mkdir(parents=True, exist_ok=True)
    fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    rows, urls = fetch()

    json_path = C.RAW_CFTC / "tff_nasdaq.json"
    json_path.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    df = to_frame(rows)
    csv_path = C.RAW_CFTC / "tff_nasdaq.csv"
    df.to_csv(csv_path, index=False)

    log = {
        "fetched_at_utc": fetched_at,
        "urls": urls,
        "rows": len(df),
        "rows_per_code": df["code"].value_counts().to_dict(),
        "report_date_range": [str(df["report_date"].min().date()), str(df["report_date"].max().date())],
        "sha256": {json_path.name: sha256(json_path), csv_path.name: sha256(csv_path)},
    }
    (C.RAW_CFTC / "fetch_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    print(f"{len(df)} rows, {df.shape[1]} columns -> {csv_path}")

    check_dates(df)
    check_consolidation(df)


if __name__ == "__main__":
    main()
