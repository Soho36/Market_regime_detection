"""Paths, CFTC codes and conventions shared by the numbered scripts."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RAW_CFTC = DATA / "raw" / "cftc"
RAW_DATABENTO = DATA / "raw" / "databento"
PRICES = DATA / "prices"
PANEL = DATA / "panel"
REPORTS = ROOT / "reports"
RESULTS = ROOT / "results"

# GLBX.MDP3 ohlcv-1m, parent NQ.FUT, UTC, one row per contract per minute (spreads included)
NQ_1M_RAW = RAW_DATABENTO / "databento-ohlcv-1m.csv"

# CFTC Traders in Financial Futures, futures-only (Socrata dataset)
CFTC_TFF_URL = "https://publicreporting.cftc.gov/resource/gpe5-46if.json"
COT_CODES = {
    "20974+": "consolidated",  # big NQ + E-mini; from 2023-05-02 E-mini + Micro, in $20 units
    "209742": "emini",         # one definition 2006 -> today — primary
    "209747": "micro",
}
# E-mini is primary: the consolidated series changed units and composition on 2023-05-02,
# and Micro adds only ~4% of E-mini-equivalent open interest.
COT_PRIMARY = "209742"
# Dates where a series' definition changes; weekly changes are not computed across them
COT_BREAKS = {"20974+": ["2023-05-02"]}

EXCHANGE_TZ = "America/Chicago"
# Daily reference close: the 14:59 CT bar's close = 15:00 CT = 16:00 ET cash close.
REF_CLOSE_HOUR_CT = 15
