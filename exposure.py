#!/usr/bin/env python3
"""Hidden-bet unbundler: how many separate bets is this portfolio, really?

Fifteen tickers can be one bet. This measures it from a year of daily returns:

  * how much of the portfolio's day-to-day movement is explained by the
    average stock, by the tech giants, by AI chips, by consumer spending and
    by interest rates (each measured after removing the ones before it)
  * the "effective number of bets": 8 positions that move together count as
    far fewer
  * groups of holdings that move together beyond what the market explains
  * each stock's sensitivity to each driver, and its share of portfolio risk

It reports on the paper account's open positions (state/portfolio.json) and,
for comparison, on the whole universe held in equal amounts.

Honest limits
  * Correlation over one year. Drivers are proxied by ETFs; relationships
    shift, and a stock can share a business driver that prices have not yet
    shown. Treat the groups as questions to ask, not as facts.
  * Not financial advice.

Usage
    python exposure.py               # writes state/exposure.json
    python exposure.py --days 504    # two years instead of one
"""
import argparse
import datetime as dt
import json
import os
import sys

import numpy as np
import pandas as pd

from scan import DEFAULT_TICKERS

# name -> (ETF, ETF to subtract first or None, plain-language label)
FACTORS = [
    ("Market", "RSP", None, "the average US large company (S&P 500, equal weight)"),
    ("Big tech", "QQQ", "RSP", "the tech giants vs the average company"),
    ("AI chips", "SMH", "QQQ", "chip makers vs the rest of big tech"),
    ("Consumer spending", "XLY", "RSP", "consumer discretionary vs the average company"),
    ("Interest rates", "TLT", None, "long Treasury bonds (up when yields fall)"),
]
GROUP_CORR = 0.30      # residual correlation that puts two stocks in a group


def fetch_prices(symbols, days):
    import yfinance as yf
    out = {}
    for s in symbols:
        try:
            h = yf.Ticker(s).history(period="3y", auto_adjust=True)["Close"]
            h.index = h.index.tz_localize(None).normalize()
            if len(h) > 60:
                out[s] = h
        except Exception as e:
            print(f"  ! {s}: {type(e).__name__}", file=sys.stderr)
    return pd.DataFrame(out).sort_index().ffill().iloc[-(days + 1):]


def build_factors(rets):
    """Factor return table. Each spread factor is the ETF minus its base, and
    every factor is then cleaned of the ones before it, so shares add up."""
    cols = {}
    for name, etf, base, _ in FACTORS:
        if etf not in rets or (base and base not in rets):
            continue
        f = rets[etf] - (rets[base] if base else 0)
        if cols:
            X = np.column_stack(list(cols.values()))
            f = f - X @ np.linalg.lstsq(X, f.values, rcond=None)[0]
        cols[name] = f.values
    return pd.DataFrame(cols, index=rets.index)


def regress(y, F):
    """-> (betas, share of variance per factor, residual series)."""
    X = np.column_stack([np.ones(len(F)), F.values])
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ b
    var = y.var()
    # factors are orthogonal by construction, so the shares are additive
    share = {c: float(b[k + 1] ** 2 * F[c].var() / var) if var else 0.0
             for k, c in enumerate(F.columns)}
    return dict(zip(F.columns, map(float, b[1:]))), share, resid


def groups_of(resid_corr):
    """Connected groups of stocks whose residual correlation exceeds the bar."""
    names, seen, out = list(resid_corr.index), set(), []
    for n in names:
        if n in seen:
            continue
        stack, grp = [n], []
        while stack:
            x = stack.pop()
            if x in seen:
                continue
            seen.add(x)
            grp.append(x)
            stack += [m for m in names if m not in seen
                      and resid_corr.loc[x, m] >= GROUP_CORR]
        if len(grp) > 1:
            sub = resid_corr.loc[grp, grp].values
            avg = float(sub[np.triu_indices(len(grp), 1)].mean())
            out.append(dict(members=sorted(grp), avg_corr=avg))
    return sorted(out, key=lambda g: -len(g["members"]))


def analyse(weights, rets, F, label):
    """weights: {ticker: weight of the invested part, summing to 1}."""
    tks = [t for t in weights if t in rets]
    w = np.array([weights[t] for t in tks])
    w = w / w.sum()
    R = rets[tks]
    port = R.values @ w
    betas, share, _ = regress(port, F)
    cov = R.cov().values
    pvar = float(w @ cov @ w)
    risk = w * (cov @ w) / pvar                      # risk share per stock
    # effective number of bets: 1 / sum of squared risk shares by principal
    # component (all in one component -> 1; spread evenly -> many)
    vals, vecs = np.linalg.eigh(cov)
    comp = vals * (vecs.T @ w) ** 2
    comp = comp / comp.sum()
    enb = float(1 / (comp ** 2).sum())
    corr = R.corr().values
    avg_corr = float(corr[np.triu_indices(len(tks), 1)].mean()) if len(tks) > 1 else 1.0
    stocks, resid = [], {}
    for k, t in enumerate(tks):
        b, s, r = regress(R[t].values, F)
        resid[t] = r
        stocks.append(dict(ticker=t, weight=float(w[k]), risk=float(risk[k]),
                           vol=float(R[t].std() * np.sqrt(252)), betas=b,
                           explained=float(sum(s.values())),
                           top=max(s, key=s.get)))
    rc = pd.DataFrame(resid).corr()
    explained = float(sum(share.values()))
    top = max(share, key=share.get)
    return dict(
        label=label, n=len(tks), vol=float(port.std() * np.sqrt(252)),
        betas=betas, share=share, explained=explained, own=1 - explained,
        top_factor=top, eff_bets=enb, avg_corr=avg_corr,
        groups=groups_of(rc),
        stocks=sorted(stocks, key=lambda s: -s["risk"]))


def run(tickers, sdir, days=252):
    px = fetch_prices(list(tickers) + sorted({f[1] for f in FACTORS}), days)
    missing = [t for t in tickers if t not in px]
    rets = px.pct_change().dropna(how="any")
    if len(rets) < 60:
        raise SystemExit("Not enough price history (network blocked?).")
    F = build_factors(rets)
    have = [t for t in tickers if t in rets]
    views = []
    ppath = os.path.join(sdir, "portfolio.json")
    if os.path.exists(ppath):
        pos = json.load(open(ppath))["engine"]["positions"]
        val = {t: p["shares"] * p["last"] for t, p in pos.items() if t in rets}
        if val:
            tot = sum(val.values())
            views.append(analyse({t: v / tot for t, v in val.items()}, rets,
                                 F, "Paper account, open positions"))
    views.append(analyse({t: 1 / len(have) for t in have}, rets, F,
                         f"All {len(have)} stocks, equal amounts"))
    rep = dict(as_of=str(rets.index[-1].date()), days=len(rets),
               start=str(rets.index[0].date()), missing=missing,
               factors=[dict(name=n, etf=e, base=b, what=w)
                        for n, e, b, w in FACTORS if n in F.columns],
               group_corr=GROUP_CORR, views=views,
               generated=str(dt.date.today()))
    with open(os.path.join(sdir, "exposure.json"), "w") as f:
        json.dump(rep, f, indent=1)
    return rep


def print_report(rep):
    print(f"\nHIDDEN BETS  {rep['start']} -> {rep['as_of']} "
          f"({rep['days']} trading days)")
    for v in rep["views"]:
        print(f"\n{v['label']}: {v['n']} stocks behave like "
              f"{v['eff_bets']:.1f} independent bets "
              f"(average correlation {v['avg_corr']:.2f}, "
              f"volatility {v['vol']:.0%} a year)")
        print("  What moves it, as a share of daily ups and downs:")
        for name, s in sorted(v["share"].items(), key=lambda x: -x[1]):
            print(f"    {name:<19}{s:>5.0%}   sensitivity {v['betas'][name]:+.2f}")
        print(f"    {'Stock-specific':<19}{v['own']:>5.0%}")
        for g in v["groups"]:
            print(f"  Move together beyond those drivers: "
                  f"{', '.join(g['members'])} (correlation {g['avg_corr']:.2f})")
        print(f"  {'Stock':<7}{'Weight':>7}{'Risk':>7}  Biggest driver")
        for s in v["stocks"]:
            b = s["betas"][s["top"]]
            print(f"  {s['ticker']:<7}{s['weight']:>7.0%}{s['risk']:>7.0%}  "
                  f"{'with' if b > 0 else 'against'} {s['top']} ({b:+.2f})")
    if rep["missing"]:
        print(f"\nNo prices for: {', '.join(rep['missing'])}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    p.add_argument("--state", default="state")
    p.add_argument("--days", type=int, default=252)
    a = p.parse_args()
    os.makedirs(a.state, exist_ok=True)
    print_report(run(a.tickers, a.state, a.days))


if __name__ == "__main__":
    main()
