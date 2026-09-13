# COT Positioning vs NQ Price — Study Plan

*Drafted 2026-09-14, updated the same day after steps 1–3 (Gate A).
Scope deliberately narrow: one data source (CFTC TFF) against one market (NQ).*

## 1. The question

**Are there measurable, stable regularities between what the CFTC trader categories hold in
Nasdaq-100 futures and what NQ price does?**

We ask it two ways:

- **Q-A: does positioning *follow* price?** Which categories add when price rises, and which
  take the other side? This is a descriptive check. If the data shows nothing here, suspect a
  bug before believing a result.
- **Q-B: does positioning *lead* price?** Does a report, once published, say anything about the
  next 1–13 weeks: return, volatility, drawdown, range?

We move on only if Q-B shows something stable.

**Out of scope for now:** RR/GG strategies, other data sources, ML, live trading.

## 2. Data (built and checked)

### COT: CFTC Traders in Financial Futures (TFF), futures-only

- API: `https://publicreporting.cftc.gov/resource/gpe5-46if.json` (Socrata), fetched by `1_fetch_cot.py`
- **Primary: E-mini `209742`.** One definition from 2006-06-13 to 2026-09-08, 1,057 reports.
  It was renamed "NASDAQ MINI" in 2022-02, with no change in content.
- Cross-checks:
  - `20974+` consolidated
  - `209747` Micro, from 2020-08, with missing weeks in 2020–21
- **Why not consolidated** (it was primary in the first draft):
  - **Until 2023-04-25** it is in big-NQ units ($100 per point). It holds big NQ + E-mini/5 and
    **no Micro**.
  - **From 2023-05-02** it is in E-mini units and equals E-mini + Micro/10 exactly.
  - CFTC nets each trader's positions across contracts, so category positions are not simple sums.
  - Micro/10 is only about 4% of E-mini open interest, so E-mini-only loses little.
  - Its 3-year percentile would not start until 2013.

### Price: Databento GLBX.MDP3 `ohlcv-1m`, parent `NQ.FUT` → `2_build_prices.py`

- `data/raw/databento/databento-ohlcv-1m.csv`: 2010-06-06 → 2026-02-27, UTC, per contract
- Cleaning:
  - dropped 502,852 calendar-spread rows
  - decoded the repeating year digit by date; 69 instrument ids, no conflicts
  - dropped 2 one-bar Sunday stubs
- **4,051 CME trade dates.** The reference close is the 14:59 CT bar on 3,915 of them; the rest
  are early closes and holiday sessions.
- **63 rolls**, 4 per year, at the volume crossover 4–6 trading days before expiry.
  - The roll ratio follows carry: −0.2% while dividends exceeded rates, about +1.0–1.4% in 2023–25.
  - Median roll-day open gap: 0.25% raw, 0.04% after back-adjustment.
  - On non-roll days, raw and adjusted returns match exactly.
- The MT5-converted file is not used. It has unadjusted roll jumps, a broken clock before 2016,
  and negative prices.

### Panel → `3_build_panel.py` → `data/panel/cot_panel.parquet`

- **E-mini: 821 reports with prices, 803 with a 13-week outcome** (as-of 2010-06-01 → 2025-10-14)
- Plausibility checks:
  - mean forward return 0.36% / 1.41% / 4.42% at 1 / 4 / 13 weeks (consistent across horizons)
  - annualized volatility about 17–19%
  - calendar-year returns: 2022 −34%, 2023 +48%

## 3. Alignment (implemented in `cotlib/`)

| Item | Rule |
|---|---|
| Clock | UTC → `America/Chicago`. Trade date = (CT + 7h).date(), so a session opening at 17:00 CT belongs to the next day. |
| Reference close | Close of the 14:59 CT bar = 15:00 CT = 16:00 ET cash close |
| Q-A price | Reference close on the last session on or before the as-of date |
| Contemporaneous week | As-of session → next as-of session, the same window as the COT weekly change |
| Q-B entry | **First full session after the nominal Friday, and on or after publication.** This is at most one trading day later than the real release and never earlier, with no holiday calendar needed. Entries are Monday for 706 reports and Tuesday for 89 (Monday holidays); the rest are delayed releases. |
| Delayed releases | `cotlib/release.py`, 35 E-mini reports. 2023 ION and 2025 shutdown: **exact** CFTC schedules. 2018–19: **derived** from CFTC's Tuesday/Friday pattern, anchored on stated dates. 2013 (and the tail of 2019): **upper bound** = end of the week CFTC gave. |
| Staleness guard | A session more than 7 days from the date it stands for is treated as "no price" |
| Forward highs/lows | Bars after the entry close, whole sessions in between, bars before the exit close |
| Rolls | Multiplicative back-adjustment; `roll_in_week` flag kept for robustness |

**Tests** (`python -m pytest tests`) check that:
- entry comes after release, and within 7 days of it
- the as-of price is never after the as-of date
- delayed reports wait for publication
- features are unchanged when future reports are removed
- forward windows ignore bars before entry and after exit

## 4. What we measure (kept small on purpose)

**Positioning features.** Main categories: Dealer, Asset Manager, Leveraged Funds. Other and
Non-reportables are secondary.

| Code | Column | Feature |
|---|---|---|
| F1 | `<cat>_net_pct` | net % OI = (long − short) / OI |
| F2 | `<cat>_d1` | 1-week change in F1 (empty across missing weeks or a definition break) |
| F3 | `<cat>_d4` | 4-week change in F1 |
| F4 | `<cat>_pct3y` | percentile of F1 within the last 156 reports, current included |
| O1 | `oi_chg4` | 4-week change in total OI |
| — | `<cat>_long_pct`, `<cat>_short_pct` | exploratory |

**Price outcomes** (back-adjusted):

| Code | Column | Outcome |
|---|---|---|
| P1 | `fwd_ret_{1,4,13}w` | reference-close return |
| P2 | `fwd_vol_{h}w` | annualized volatility of daily reference-close log returns |
| P3 | `fwd_mdd_{h}w` | worst low vs entry close |
| P4 | `fwd_mru_{h}w` | best high vs entry close |
| V1 | `volume_week` | contract volume between as-of sessions (compare with its 13-week average) |
| control | `trail_vol_20d`, `trail_vol_65d`, `trail_ret_*` | trailing, up to and including the entry session |

## 5. Tests

**Stage 1: descriptive (sanity, not findings)**
- Chart NQ price against each category's F1 and F4, 2010–2026.
- Contemporaneous correlation of F2 with the same week's return (`asof_px` to the next as-of), per category.

**Stage 2: lead–lag**
- Cross-correlation of F2(t) with weekly return(t+k), k = −8 … +8.
  - Significant at k < 0 means positioning follows price.
  - Significant at k > 0 means it leads.

**Stage 3: conditional forward outcomes**
- F3 and F4 in terciles (extremes <10 / >90 as secondary) → P1–P4 at 4w and 13w, compared with
  the unconditional average. NQ rose about 12× over the sample, so every bucket has positive
  returns. A regularity is a *difference* from the unconditional average.
- Regression on the standardized feature. For P2/P3, add `trail_vol_20d` as a control.

**Stage 4: stability**
- Sub-periods: 2010–2015 / 2016–2020 / 2021–2026.
- Drop each year in turn.
- Exclude `roll_in_week` rows.
- Exclude `derived` / `upper_bound` release rows.
- Swap the E-mini series for consolidated.

**Statistics**

| Problem | Remedy |
|---|---|
| Overlapping forward windows | Newey–West (HAC) standard errors |
| Persistent features inflate naive significance | Circular-shift permutation: rotate the feature series against price by random offsets ≥ 26 weeks, 2,000 times |
| Many tests | Small, fixed primary set, adjusted with Benjamini–Hochberg. Everything else is labeled exploratory. |

## 6. What counts as a regularity

**Primary tests** (to be locked in `HYPOTHESES.md` at Gate B):
3 categories (Dealer, AM, LF) × 2 features (F3, F4) × 2 horizons (4w, 13w) × 3 outcomes
(P1, P2, P3) = **36 tests**.

A primary test counts as **FOUND** only if all four hold:
1. Benjamini–Hochberg q < 0.10, using circular-shift p-values
2. Same sign in all three sub-periods
3. Survives dropping any single year and excluding roll weeks
4. For P2/P3: survives the trailing-vol control

Verdicts:
- **FOUND:** all four criteria met.
- **WEAK:** criterion 1 met, but not all of 2–4.
- **NONE:** criterion 1 not met. This is a legitimate answer; we stop and pick the next source.

**Honest prior.** Much of equity-index futures open interest is hedging and basis or
relative-value trading, so "net" is not a clean directional bet. Direction (P1) is the least
likely place to find something. Volatility, drawdown and extremes are more plausible.

## 7. Steps and gates

| Step | Status | Work |
|---|---|---|
| 1 | ✅ | `1_fetch_cot.py`: CFTC API → `data/raw/cftc/` (+ `fetch_log.json` with sha256) |
| 2 | ✅ | `2_build_prices.py`: raw 1m → per-contract daily → front, back-adjusted → `data/prices/` |
| 3 | ✅ | `3_build_panel.py`: features + release timing + outcomes → `data/panel/cot_panel.parquet`, with tests |
| **Gate A** | see below | data checks |
| 4 | next | `4_explore.py`: Stages 1–2 → `reports/01_explore.html` |
| **Gate B** | | review together; lock primary tests and pass bar in `HYPOTHESES.md` |
| 5 | | `5_test.py`: Stages 3–4 → `reports/02_tests.html`, `results/tests.csv`, `FINDINGS.md` |
| **Gate C** | | FOUND → decide next phase. NONE → document and choose the next source. |

**Gate A checklist**
- [x] No missing COT weeks. The only irregular gaps are holiday weeks with a Monday as-of date.
- [x] Consolidated vs components explained (unit and Micro change on 2023-05-02)
- [x] Spot-check against CFTC's published reports: all 15 position/OI fields match exactly for E-mini and consolidated on 2015-06-16 and 2026-09-08
- [x] Quarterly roll jumps removed; adjusted = raw returns on non-roll days
- [x] Calendar-year returns and volatility plausible
- [x] Lookahead and alignment tests pass

## 8. Layout

```
Market_regime_detection/
  PLAN.md  requirements.txt  config.py  .gitignore
  1_fetch_cot.py  2_build_prices.py  3_build_panel.py   (4_explore.py, 5_test.py next)
  cotlib/
    release.py     publication dates, delayed-release table
    features.py    F1–F4, O1
    outcomes.py    sessions, entry/exit, forward and trailing outcomes
  tests/test_alignment.py
  data/raw/cftc/  data/raw/databento/  data/prices/  data/panel/
  reports/  results/
  venv/
```

## 9. Later, not now

- ES COT as a cross-check (same code, different market)
- Extend prices past 2026-02-27 (COT already runs to 2026-09-08)
- Linking any finding to RR/GG trade outcomes
- Other sources: volatility term structure, rates, credit, breadth
