"""When each COT report became public.

Normal schedule: positions as of Tuesday, published Friday 15:30 ET (14:30 CT). Holiday weeks
move the as-of date to Monday and/or the release to the next business day. Rather than model
every holiday, outcomes.py uses a report only from the first full session *after* its nominal
Friday: at most one trading day later than necessary, never earlier.

Delayed periods come from CFTC's published schedules. Quality labels:
  exact        date stated by CFTC
  derived      follows a stated pattern that is anchored by an exact date
  upper_bound  CFTC gave only a week; the end of that week is used (never earlier than the truth)
"""
import pandas as pd

DELAYED = {
    # 2013 lapse in appropriations. CFTC PR 6745-13 (2013-10-23): first delayed report 10/25,
    # two more in the week of 10/28, more in the week of 11/04, normal schedule by 11/08.
    "2013-10-01": ("2013-10-25", "exact"),
    "2013-10-08": ("2013-11-01", "upper_bound"),
    "2013-10-15": ("2013-11-01", "upper_bound"),
    "2013-10-22": ("2013-11-08", "upper_bound"),
    "2013-10-29": ("2013-11-08", "upper_bound"),
    # 2018-19 shutdown. CFTC PR 7864-19 (2019-01-29): 12/24 report on 02/01, then one report each
    # Tuesday and Friday. The report due 02/08 (as of 02/05) is stated as published 02/22, which
    # matches that pattern. Normal schedule from March.
    "2018-12-24": ("2019-02-01", "exact"),
    "2018-12-31": ("2019-02-05", "derived"),
    "2019-01-08": ("2019-02-08", "derived"),
    "2019-01-15": ("2019-02-12", "derived"),
    "2019-01-22": ("2019-02-15", "derived"),
    "2019-01-29": ("2019-02-19", "derived"),
    "2019-02-05": ("2019-02-22", "exact"),
    "2019-02-12": ("2019-02-26", "derived"),
    "2019-02-19": ("2019-03-05", "upper_bound"),
    "2019-02-26": ("2019-03-05", "upper_bound"),
    # 2023 ION cyber incident. CFTC COT Historical Special Announcements, Feb-Mar 2023.
    "2023-01-31": ("2023-02-24", "exact"),
    "2023-02-07": ("2023-03-03", "exact"),
    "2023-02-14": ("2023-03-08", "exact"),
    "2023-02-21": ("2023-03-10", "exact"),
    "2023-02-28": ("2023-03-14", "exact"),
    "2023-03-07": ("2023-03-16", "exact"),
    "2023-03-14": ("2023-03-21", "exact"),
    # 2025 lapse in appropriations. CFTC PR 9147-25 (2025-12-09), accelerated schedule.
    "2025-09-30": ("2025-11-19", "exact"),
    "2025-10-07": ("2025-11-21", "exact"),
    "2025-10-14": ("2025-11-25", "exact"),
    "2025-10-21": ("2025-12-02", "exact"),
    "2025-10-28": ("2025-12-05", "exact"),
    "2025-11-04": ("2025-12-09", "exact"),
    "2025-11-10": ("2025-12-10", "exact"),
    "2025-11-18": ("2025-12-12", "exact"),
    "2025-11-25": ("2025-12-15", "exact"),
    "2025-12-02": ("2025-12-17", "exact"),
    "2025-12-09": ("2025-12-19", "exact"),
    "2025-12-16": ("2025-12-23", "exact"),
    "2025-12-23": ("2025-12-29", "exact"),
}


def nominal_friday(report_dates):
    d = pd.DatetimeIndex(report_dates)
    return d + pd.to_timedelta((4 - d.dayofweek) % 7, unit="D")


def release_table(report_dates):
    """publication_date (nominal Friday unless delayed) and release_quality per report date."""
    d = pd.DatetimeIndex(report_dates)
    delayed = {pd.Timestamp(k): (pd.Timestamp(v), q) for k, (v, q) in DELAYED.items()}
    pub = nominal_friday(d)
    out = pd.DataFrame({"report_date": d, "publication_date": pub, "release_quality": "normal"})
    for i, day in enumerate(d):
        if day in delayed:
            out.at[i, "publication_date"], out.at[i, "release_quality"] = delayed[day]
    return out
