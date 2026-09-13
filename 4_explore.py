"""Step 4 — Stages 1–2: how positioning moves with NQ, and whether it leads or follows price.

in : data/panel/cot_panel.parquet
out: reports/01_explore.html
     results/explore_contemporaneous.csv   Q-A: weekly change in net % OI vs the same week's return
     results/explore_leadlag.csv           corr(change at t, return at t+k), k = -8..+8
     results/explore_leadlag_abs.csv       same with |change| and |return| (activity, exploratory)

Descriptive only — nothing here is a finding. The pre-registered tests come after Gate B.

Weekly return = reference close at one as-of session to the next, the same window as the COT
weekly change. Significance: circular shifts (positioning rotated against returns by >= 26
weeks) so each series keeps its own autocorrelation. Intervals: moving-block bootstrap.
k > 0 windows start at the as-of date, three days before publication, so they describe
the data rather than a tradable signal (that is Q-B, Stage 3).
"""
import hashlib
from datetime import datetime
from html import escape

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import spearmanr

import config as C
from cotlib import stats as S

# Category -> (label, categorical slot). Colour follows the category in every chart.
CATS = {
    "dealer": ("Dealers", "#2a78d6"),
    "asset_mgr": ("Asset managers", "#eb6834"),
    "lev_money": ("Leveraged funds", "#1baf7a"),
    "other_rept": ("Other reportables", "#eda100"),
    "nonrept": ("Non-reportables", "#e87ba4"),
}
MAIN = ["dealer", "asset_mgr", "lev_money"]
SECONDARY = ["other_rept", "nonrept"]
# Sub-periods on an ordinal grey ramp (early = light), so they never read as a category
PERIODS = [("2010–15", 2010, 2015, "#b5b3ab"), ("2016–20", 2016, 2020, "#7d7b74"), ("2021–26", 2021, 2026, "#3f3e3b")]
LAGS = np.arange(-8, 9)
DRAWS, MIN_SHIFT, BLOCK, SEED = 2000, 26, 8, 20260914
X_RANGE = ["2010-05-01", "2026-04-01"]

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'


# ---------------------------------------------------------------- data

def weekly(panel, code):
    p = panel[(panel["code"] == code) & panel["asof_px"].notna()].sort_values("report_date").reset_index(drop=True)
    gap1 = (p["report_date"] - p["report_date"].shift(1)).dt.days
    gap4 = (p["report_date"] - p["report_date"].shift(4)).dt.days
    p["ret_w"] = (p["asof_px"] / p["asof_px"].shift(1) - 1).where(gap1 <= 8)
    p["ret_4w"] = (p["asof_px"] / p["asof_px"].shift(4) - 1).where(gap4 <= 29)
    return p


def contemporaneous(p, cons):
    years = p["report_date"].dt.year
    rows = []
    for cat in CATS:
        x, y = p[f"{cat}_d1"].to_numpy(), p["ret_w"].to_numpy()
        r = S.corr(x, y)
        null = S.circular_shift_null(x, y, [0], DRAWS, MIN_SHIFT, SEED)[:, 0]
        boot = S.block_bootstrap(x, y, S.corr, BLOCK, DRAWS, SEED)
        x4, y4 = p[f"{cat}_d4"].to_numpy(), p["ret_4w"].to_numpy()
        r4 = S.corr(x4, y4)
        null4 = S.circular_shift_null(x4, y4, [0], DRAWS, MIN_SHIFT, SEED)[:, 0]
        row = {"category": cat, "label": CATS[cat][0], "n": S.pairs(x, y), "r": r,
               "ci_lo": np.nanpercentile(boot, 2.5), "ci_hi": np.nanpercentile(boot, 97.5),
               "p_shift": S.p_two_sided(r, null), "spearman": spearmanr(x, y, nan_policy="omit")[0],
               "r_4w": r4, "p_shift_4w": S.p_two_sided(r4, null4)}
        for name, a, b, _ in PERIODS:
            m = years.between(a, b).to_numpy()
            row[f"r_{name}"], row[f"n_{name}"] = S.corr(x[m], y[m]), S.pairs(x[m], y[m])
        row["r_consolidated"] = S.corr(cons[f"{cat}_d1"].to_numpy(), cons["ret_w"].to_numpy())
        rows.append(row)
    return pd.DataFrame(rows)


def leadlag(p, absolute=False):
    rows = []
    for cat in CATS:
        x, y = p[f"{cat}_d1"].to_numpy(), p["ret_w"].to_numpy()
        if absolute:
            x, y = np.abs(x), np.abs(y)
        obs = S.cross_corr(x, y, LAGS)
        null = np.abs(S.circular_shift_null(x, y, LAGS, DRAWS, MIN_SHIFT, SEED))
        band = np.nanpercentile(null, 95, axis=0)
        any_lag = np.nanpercentile(np.nanmax(null[:, LAGS != 0], axis=1), 95)
        for k, r, b, col in zip(LAGS, obs, band, null.T):
            rows.append({"category": cat, "k": int(k), "r": r, "band95": b, "any_lag_band95": any_lag,
                         "p_shift": (1 + np.sum(col >= abs(r))) / (1 + DRAWS), "outside": abs(r) > b})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- charts

def style(fig, height, legend=False):
    fig.update_layout(template="none", height=height, paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
                      font=dict(family=FONT, size=12, color=INK2), showlegend=legend,
                      margin=dict(l=64, r=120, t=40 if legend else 30, b=48),
                      hoverlabel=dict(bgcolor="#ffffff", bordercolor=GRID, font=dict(family=FONT, color=INK)),
                      legend=dict(orientation="h", x=0, y=1.0, yanchor="bottom", font=dict(color=INK2)),
                      barcornerradius=4)
    fig.update_xaxes(showgrid=False, showline=True, linecolor=AXIS, ticks="outside", tickcolor=AXIS,
                     tickfont=dict(color=MUTED), title_font=dict(color=INK2, size=12), zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor=GRID, gridwidth=1, showline=False, zeroline=False,
                     tickfont=dict(color=MUTED), title_font=dict(color=INK2, size=12))
    fig.update_annotations(font=dict(family=FONT, color=INK2, size=12))
    return fig


def fig_price(p):
    fig = go.Figure(go.Scatter(x=p["asof_session"], y=p["asof_px"], mode="lines", line=dict(color=INK2, width=2),
                               hovertemplate="%{x|%Y-%m-%d}<br>NQ %{y:,.0f}<extra></extra>"))
    fig.update_yaxes(type="log", title_text="NQ, back-adjusted", tickvals=[2500, 5000, 10000, 20000],
                     ticktext=["2,500", "5,000", "10,000", "20,000"])
    fig.update_xaxes(range=X_RANGE)
    return style(fig, 240)


def fig_lines(p, cats, col, ytitle, height, hover="%{y:.1f}", zero=True, end_labels=True):
    """end_labels=False where lines finish too close together; the legend and tooltip carry identity."""
    fig = go.Figure()
    for cat in cats:
        label, color = CATS[cat]
        fig.add_trace(go.Scatter(x=p["report_date"], y=p[f"{cat}{col}"], mode="lines", name=label,
                                 line=dict(color=color, width=2), hovertemplate=f"{hover}<extra>{label}</extra>"))
        if not end_labels:
            continue
        last = p[["report_date", f"{cat}{col}"]].dropna().iloc[-1]
        fig.add_trace(go.Scatter(x=[last.iloc[0]], y=[last.iloc[1]], mode="markers", showlegend=False, hoverinfo="skip",
                                 marker=dict(size=9, color=color, line=dict(color=SURFACE, width=2))))
        fig.add_annotation(x=last.iloc[0], y=last.iloc[1], text=label, showarrow=False, xanchor="left", xshift=9)
    if zero:
        fig.add_hline(y=0, line=dict(color=AXIS, width=1))
    fig.update_layout(hovermode="x unified")
    fig.update_xaxes(range=X_RANGE, hoverformat="%Y-%m-%d")
    fig.update_yaxes(title_text=ytitle)
    return style(fig, height, legend=True)


def fig_percentiles(p):
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=[CATS[c][0] for c in MAIN])
    for i, cat in enumerate(MAIN, start=1):
        label, color = CATS[cat]
        fig.add_trace(go.Scatter(x=p["report_date"], y=p[f"{cat}_pct3y"] * 100, mode="lines",
                                 line=dict(color=color, width=2),
                                 hovertemplate="%{x|%Y-%m-%d}<br>%{y:.0f}th percentile<extra>" + label + "</extra>"),
                      row=i, col=1)
        fig.update_yaxes(range=[-4, 104], tickvals=[0, 10, 50, 90, 100], row=i, col=1)
    fig.update_yaxes(title_text="percentile of net % OI, last 3 years", row=2, col=1)
    fig.update_xaxes(range=X_RANGE)
    style(fig, 540)
    fig.update_annotations(x=0, xanchor="left")
    return fig


def fig_oi(p):
    fig = go.Figure(go.Scatter(x=p["report_date"], y=p["oi"], mode="lines", line=dict(color=INK2, width=2),
                               hovertemplate="%{x|%Y-%m-%d}<br>%{y:,.0f} contracts<extra></extra>"))
    fig.update_yaxes(title_text="open interest, contracts", tickformat=",")
    fig.update_xaxes(range=X_RANGE)
    return style(fig, 220)


def fig_scatter(p):
    fig = make_subplots(rows=1, cols=3, shared_yaxes=True, horizontal_spacing=0.04,
                        subplot_titles=[CATS[c][0] for c in MAIN])
    y = p["ret_w"] * 100
    dates = p["report_date"].dt.strftime("%Y-%m-%d")
    for j, cat in enumerate(MAIN, start=1):
        label, color = CATS[cat]
        x = p[f"{cat}_d1"]
        m = x.notna() & y.notna()
        fig.add_trace(go.Scatter(x=x[m], y=y[m], mode="markers", customdata=dates[m],
                                 marker=dict(size=8, color=color, opacity=0.5, line=dict(color=SURFACE, width=1)),
                                 hovertemplate="%{customdata}<br>change %{x:+.2f} pp<br>NQ %{y:+.2f}%<extra>" + label + "</extra>"),
                      row=1, col=j)
        slope, icpt = np.polyfit(x[m], y[m], 1)
        xs = np.array([x[m].min(), x[m].max()])
        fig.add_trace(go.Scatter(x=xs, y=icpt + slope * xs, mode="lines", line=dict(color=INK2, width=2), hoverinfo="skip"),
                      row=1, col=j)
        fig.add_hline(y=0, line=dict(color=AXIS, width=1), row=1, col=j)
        fig.add_vline(x=0, line=dict(color=AXIS, width=1), row=1, col=j)
        fig.update_xaxes(title_text="weekly change, pp of OI", row=1, col=j)
    fig.update_yaxes(title_text="NQ return, same week (%)", row=1, col=1)
    fig.update_layout(hovermode="closest")
    style(fig, 400)
    return fig


def fig_contemporaneous(ct):
    order = list(CATS)
    ypos = {cat: len(order) - 1 - i for i, cat in enumerate(order)}
    fig = go.Figure()
    for off, (name, _, _, color) in zip((0.09, -0.09, -0.27), PERIODS):
        fig.add_trace(go.Scatter(
            x=ct[f"r_{name}"], y=[ypos[c] + off for c in ct["category"]], mode="markers", name=name,
            marker=dict(size=9, color=color, line=dict(color=SURFACE, width=2)),
            customdata=np.c_[ct["label"], ct[f"n_{name}"]],
            hovertemplate="%{customdata[0]}, " + name + "<br>r = %{x:+.2f} (n = %{customdata[1]})<extra></extra>"))
    for _, row in ct.iterrows():
        label, color = CATS[row["category"]]
        fig.add_trace(go.Scatter(
            x=[row["r"]], y=[ypos[row["category"]] + 0.27], mode="markers", showlegend=False,
            marker=dict(size=12, color=color, line=dict(color=SURFACE, width=2)),
            error_x=dict(type="data", symmetric=False, array=[row["ci_hi"] - row["r"]],
                         arrayminus=[row["r"] - row["ci_lo"]], color=INK2, thickness=1.5, width=0),
            hovertemplate=(f"{label}, 2010–26<br>r = {row['r']:+.2f}  [{row['ci_lo']:+.2f}, {row['ci_hi']:+.2f}]"
                           f"<br>shift p = {row['p_shift']:.3f}<extra></extra>")))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name="2010–26, with 95% interval",
                             marker=dict(size=12, color=INK2)))
    fig.add_vline(x=0, line=dict(color=AXIS, width=1))
    fig.update_yaxes(tickvals=list(ypos.values()), ticktext=[CATS[c][0] for c in ypos], showgrid=False,
                     range=[-0.6, len(order) - 0.4], tickfont=dict(color=INK2))
    fig.update_xaxes(title_text="correlation of weekly change in net % OI with the same week's NQ return",
                     showgrid=True, gridcolor=GRID)
    fig.update_layout(hovermode="closest")
    style(fig, 440, legend=True)
    fig.update_layout(margin=dict(l=150))  # room for the category names on the y-axis
    return fig


def fig_leadlag(ll, cats, ytitle):
    fig = make_subplots(rows=1, cols=len(cats), shared_yaxes=True, horizontal_spacing=0.035,
                        subplot_titles=[CATS[c][0] for c in cats])
    lim = 0.0
    for j, cat in enumerate(cats, start=1):
        label, color = CATS[cat]
        d = ll[ll["category"] == cat]
        k, r, band, pv = d["k"].to_numpy(), d["r"].to_numpy(), d["band95"].to_numpy(), d["p_shift"].to_numpy()
        lim = max(lim, np.nanmax(np.abs(r)), np.nanmax(band))
        fig.add_trace(go.Scatter(x=np.r_[k, k[::-1]], y=np.r_[band, -band[::-1]], fill="toself", mode="lines",
                                 line=dict(width=0), fillcolor="rgba(137,135,129,0.16)", hoverinfo="skip"),
                      row=1, col=j)
        outside = np.abs(r) > band
        for mask, opacity in ((outside, 1.0), (~outside, 0.35)):
            fig.add_trace(go.Bar(x=k[mask], y=r[mask], marker=dict(color=color, opacity=opacity, line=dict(width=0)),
                                 customdata=np.c_[band[mask], pv[mask]],
                                 hovertemplate=("k = %{x:+d} weeks<br>r = %{y:+.3f}<br>95% band ±%{customdata[0]:.3f}"
                                                "<br>shift p = %{customdata[1]:.3f}<extra>" + label + "</extra>")),
                          row=1, col=j)
        fig.add_hline(y=0, line=dict(color=AXIS, width=1), row=1, col=j)
        fig.update_xaxes(tickvals=[-8, -4, 0, 4, 8], title_text="k, weeks", row=1, col=j)
    fig.update_yaxes(range=[-lim * 1.15, lim * 1.15], title_text=ytitle, row=1, col=1)
    fig.update_layout(barmode="overlay", bargap=0.3, hovermode="closest")
    return style(fig, 330)


# ---------------------------------------------------------------- html

def num(v, spec="+.2f"):
    return "–" if pd.isna(v) else format(v, spec)


def key(cat):
    label, color = CATS[cat]
    return f'<span class="key" style="background:{color}"></span>{escape(label)}'


def pcell(pv):
    return f'<td class="{"sig" if pv < 0.05 else "ns"}">{pv:.3f}</td>'


def table_contemporaneous(ct):
    head = ("<tr><th>Category</th><th>n</th><th>r</th><th>95% interval</th><th>shift p</th><th>Spearman</th>"
            + "".join(f"<th>r {n}</th>" for n, *_ in PERIODS)
            + "<th>r consolidated</th><th>r, 4-week</th><th>shift p, 4-week</th></tr>")
    body = ""
    for _, r in ct.iterrows():
        body += (f"<tr><td>{key(r['category'])}</td><td>{r['n']}</td><td>{num(r['r'])}</td>"
                 f"<td>{num(r['ci_lo'])} … {num(r['ci_hi'])}</td>{pcell(r['p_shift'])}<td>{num(r['spearman'])}</td>"
                 + "".join(f"<td>{num(r[f'r_{n}'])}</td>" for n, *_ in PERIODS)
                 + f"<td>{num(r['r_consolidated'])}</td><td>{num(r['r_4w'])}</td>{pcell(r['p_shift_4w'])}</tr>")
    return f'<div class="tablewrap"><table>{head}{body}</table></div>'


def table_leadlag(ll):
    head = "<tr><th>Category</th>" + "".join(f"<th>{k:+d}</th>" for k in LAGS) + "<th>any-lag band</th></tr>"
    body = ""
    for cat in CATS:
        d = ll[ll["category"] == cat]
        cells = "".join(f'<td class="{"sig" if o else "ns"}">{r:+.2f}</td>' for r, o in zip(d["r"], d["outside"]))
        body += f"<tr><td>{key(cat)}</td>{cells}<td>±{d['any_lag_band95'].iat[0]:.2f}</td></tr>"
    return f'<div class="tablewrap"><table>{head}{body}</table></div>'


def table_yearly(p):
    cols = [f"{c}_net_pct" for c in CATS]
    y = p.groupby(p["report_date"].dt.year)[cols].mean()
    head = "<tr><th>Year</th>" + "".join(f"<th>{key(c)}</th>" for c in CATS) + "</tr>"
    body = "".join(f"<tr><td>{yr}</td>" + "".join(f"<td>{v:+.1f}</td>" for v in row) + "</tr>"
                   for yr, row in zip(y.index, y.to_numpy()))
    return f'<div class="tablewrap"><table>{head}{body}</table></div>'


def lags_outside(ll):
    out = {}
    for cat in CATS:
        d = ll[(ll["category"] == cat) & ll["outside"] & (ll["k"] != 0)]
        out[cat] = [f"{k:+d} ({r:+.2f})" for k, r in zip(d["k"], d["r"])]
    return out


CSS = """
:root{color-scheme:light;--surface:#fcfcfb;--page:#f9f9f7;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;
--grid:#e1e0d9;--border:rgba(11,11,11,.10)}
*{box-sizing:border-box}
body{margin:0;background:var(--page);color:var(--ink);font:14px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1180px;margin:0 auto;padding:36px 24px 72px}
h1{font-size:26px;font-weight:600;margin:0 0 6px;letter-spacing:-.01em}
.sub{color:var(--ink2);margin:0 0 8px;max-width:820px}
.meta{color:var(--muted);font-size:12px;margin:0 0 28px}
section{margin:44px 0 0}
h2{font-size:19px;font-weight:600;margin:0 0 6px}
.lead{color:var(--ink2);max-width:820px;margin:0 0 14px}
.card{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:18px 18px 10px;margin:14px 0}
.card h3{font-size:14px;font-weight:600;margin:0 0 2px}
.note{font-size:12.5px;color:var(--ink2);margin:2px 0 8px;max-width:900px}
.tablewrap{overflow-x:auto}
table{border-collapse:collapse;font-size:13px;font-variant-numeric:tabular-nums;width:100%}
th,td{padding:6px 10px;text-align:right;border-bottom:1px solid var(--grid);white-space:nowrap}
th:first-child,td:first-child{text-align:left}
th{color:var(--ink2);font-weight:600}
.key{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:7px}
.sig{font-weight:600;color:var(--ink)}.ns{color:var(--muted)}
details{margin:8px 0 6px}summary{cursor:pointer;color:var(--ink2);font-size:13px}
.callout{border-left:3px solid var(--grid);padding:4px 0 4px 14px;color:var(--ink2);max-width:900px;margin:12px 0}
ul.facts{margin:6px 0 0;padding-left:18px;color:var(--ink2)}ul.facts li{margin:3px 0}
"""


def card(title, note, body):
    return f'<div class="card"><h3>{title}</h3><p class="note">{note}</p>{body}</div>'


def main():
    C.REPORTS.mkdir(parents=True, exist_ok=True)
    C.RESULTS.mkdir(parents=True, exist_ok=True)
    panel_path = C.PANEL / "cot_panel.parquet"
    panel = pd.read_parquet(panel_path)
    p = weekly(panel, C.COT_PRIMARY)
    cons = weekly(panel, "20974+")

    ct = contemporaneous(p, cons)
    ll = leadlag(p)
    lla = leadlag(p, absolute=True)
    ct.to_csv(C.RESULTS / "explore_contemporaneous.csv", index=False)
    ll.to_csv(C.RESULTS / "explore_leadlag.csv", index=False)
    lla.to_csv(C.RESULTS / "explore_leadlag_abs.csv", index=False)

    pd.set_option("display.width", 200)
    print("contemporaneous (primary):")
    print(ct.drop(columns="label").round(3).to_string(index=False))
    print("\nlead-lag, lags outside the pointwise 95% band (k != 0):", lags_outside(ll))
    print("any-lag bands:", ll.groupby("category")["any_lag_band95"].first().round(3).to_dict())
    print("\n|change| vs |return|, lags outside band:", lags_outside(lla))
    print("|r| at k=0:", lla[lla["k"] == 0].set_index("category")["r"].round(3).to_dict())

    figs = {
        "price": fig_price(p),
        "net": fig_lines(p, MAIN, "_net_pct", "net position, % of open interest", 380, "%{y:+.1f}%"),
        "net2": fig_lines(p, SECONDARY, "_net_pct", "net position, % of open interest", 300, "%{y:+.1f}%",
                          end_labels=False),
        "pct": fig_percentiles(p),
        "oi": fig_oi(p),
        "scatter": fig_scatter(p),
        "ct": fig_contemporaneous(ct),
        "ll": fig_leadlag(ll, MAIN, "correlation"),
        "ll2": fig_leadlag(ll, SECONDARY, "correlation"),
        "lla": fig_leadlag(lla, MAIN, "correlation of magnitudes"),
    }
    html_figs, first = {}, True
    for name, fig in figs.items():
        html_figs[name] = fig.to_html(full_html=False, include_plotlyjs=first, default_width="100%",
                                      config={"displaylogo": False, "responsive": True})
        first = False

    sha = hashlib.sha256(panel_path.read_bytes()).hexdigest()[:12]
    span = f"{p['report_date'].min():%Y-%m-%d} → {p['report_date'].max():%Y-%m-%d}"
    n_w = int(p["ret_w"].notna().sum())
    outside = lags_outside(ll)
    facts_ct = "".join(
        f"<li>{key(r['category'])}: r = {r['r']:+.2f} (shift p = {r['p_shift']:.3f}); sub-periods "
        + ", ".join(f"{r[f'r_{n}']:+.2f}" for n, *_ in PERIODS) + "</li>"
        for _, r in ct.iterrows())
    facts_ll = "".join(
        f"<li>{key(c)}: {', '.join(outside[c]) if outside[c] else 'none'}</li>" for c in CATS)

    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>COT vs NQ — Exploration</title>
<style>{CSS}</style></head><body><main>
<h1>COT positioning vs NQ — exploration</h1>
<p class="sub">Stages 1–2 of the plan: how each CFTC trader category's position moves with NQ, and whether
the changes come before or after price moves. Descriptive only. Nothing on this page is a finding; the
pre-registered tests come after Gate B.</p>
<p class="meta">E-mini Nasdaq-100 (CFTC 209742), futures-only TFF · {len(p)} weekly reports, {span} · {n_w} weekly returns ·
back-adjusted NQ reference closes (15:00 CT) · built {datetime.now():%Y-%m-%d %H:%M} · panel sha256 {sha}</p>

<section><h2>1 · Who holds what, 2010–2026</h2>
<p class="lead">Net position = long minus short, as % of open interest. Positive means the category is net long.</p>
{card("NQ", "Back-adjusted reference close on each report's as-of date, log scale.", html_figs["price"])}
{card("Net position — main categories", "Dealers, asset managers and leveraged funds.", html_figs["net"])}
{card("Net position — other categories", "Other reportables and non-reportable (small) traders.", html_figs["net2"])}
{card("Where each category sits within its own last 3 years",
      "Percentile of net % OI among the latest 156 reports (current included). Gridlines at 10 and 90 mark the extremes used as a secondary test.",
      html_figs["pct"])}
{card("Open interest", "E-mini contracts. The saw-tooth is the quarterly expiry cycle.", html_figs["oi"])}
<details><summary>Table view: average net % OI by year</summary>{table_yearly(p)}</details>
</section>

<section><h2>2 · Same week: does positioning move with price? (Q-A)</h2>
<p class="lead">Change in net % OI between two reports against the NQ return over exactly the same Tuesday-to-Tuesday
window. This is the sanity check: if nothing shows here, suspect the alignment before believing anything later.</p>
{card("Weekly change vs same-week return", "Each dot is one week. Grey line: least-squares fit.", html_figs["scatter"])}
{card("Correlation by category and sub-period",
      f"Large dot: 2010–26 with a moving-block bootstrap 95% interval ({BLOCK}-week blocks, {DRAWS:,} resamples). Small grey dots: sub-periods, light to dark.",
      html_figs["ct"])}
<div class="card"><h3>Table view</h3><p class="note">Bold p-values are below 0.05. Shift p: share of {DRAWS:,} circular shifts
(≥ {MIN_SHIFT} weeks) with |r| at least as large. 4-week: change over 4 reports vs 4-week return (overlapping windows; the shift test accounts for that).</p>
{table_contemporaneous(ct)}<ul class="facts">{facts_ct}</ul></div>
</section>

<section><h2>3 · Lead–lag: does positioning follow price or come before it?</h2>
<p class="lead">Correlation of the weekly change at report <i>t</i> with the NQ return in week <i>t + k</i>.
<b>k &lt; 0</b>: the return came first, so positioning <i>follows</i> price. <b>k &gt; 0</b>: the change came first.
Weeks at k = +1 start at the as-of date, three days before the report is published, so a k &gt; 0 bar is not yet a
tradable signal. That question is Stage 3.</p>
{card("Main categories",
      "Grey band: 95% of |r| under circular shifts, per lag. Full-colour bars are outside it. With 16 non-zero lags, about one bar per panel lands outside by chance; the any-lag band in the table is the stricter bar.",
      html_figs["ll"])}
{card("Other categories", "Same construction.", html_figs["ll2"])}
<div class="card"><h3>Table view</h3><p class="note">Bold: outside the per-lag 95% band. Any-lag band: 95th percentile of the
largest |r| across all non-zero lags under the shift null.</p>{table_leadlag(ll)}
<p class="note">Non-zero lags outside the per-lag band:</p><ul class="facts">{facts_ll}</ul></div>
{card("Exploratory: size of the change vs size of the move",
      "Correlation of |weekly change| with |return| at lag k. Asks whether big repositioning weeks cluster with big price weeks, regardless of direction.",
      html_figs["lla"])}
</section>
</main></body></html>"""
    out = C.REPORTS / "01_explore.html"
    out.write_text(html, encoding="utf-8")
    print(f"\nwrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
