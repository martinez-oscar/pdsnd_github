#!/usr/bin/env python3
"""Multi-stock research scan: run every layer + rule on a basket of big stocks.

For each ticker it (1) builds walk-forward conviction signals from past prices
only, (2) simulates several PAPER trading rules, and (3) pools everything so
one lucky stock can't fool you. It also prints today's ranked watchlist using
all layers (including news / earnings / fundamentals, which can't be
backtested).

Rules compared (all long-only, same costs / stops / sizing as trading.py):
  Sure (ensemble)        P(up) >= 55% AND >= 60% of layers bullish
  Ensemble P(up) only    P(up) >= 55%, ignore agreement
  Trend only (50/200d)   price > 200d average and 50d > 200d
  Sure + trend           both of the above must hold
  (Buy-and-hold shown as the baseline everything must beat.)

NO REAL TRADING happens anywhere. Nothing connects to a broker.

READ THIS BEFORE TRUSTING ANY RESULT
  * Survivorship bias: the default basket is "big companies today". They are
    big BECAUSE they went up, so backtests on them flatter any long strategy
    and buy-and-hold in particular. Treat all results as optimistic.
  * Multiple testing: with N stocks x 4 rules, some will look great by luck.
    The report prints how many "wins" chance alone would produce.
  * Choose your rule BEFORE seeing results, or confirm it on the second half
    of the period (printed) / on new data later. Picking the best-looking row
    afterwards is how people fool themselves.

Usage:
    pip install numpy pandas yfinance
    python scan.py --save results/scan.txt            # default 15 stocks
    python scan.py --tickers NVDA MSFT AAPL --years 10
    python scan.py --demo --tickers A B C D           # offline synthetic test
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

import stock_strategies as ss
import trading as tr
from stock_strategies import TD

# Approximate "largest US-listed companies" from the assistant's knowledge --
# NOT a live market-cap ranking. Edit freely or pass --tickers.
DEFAULT_TICKERS = ["NVDA", "MSFT", "AAPL", "GOOGL", "AMZN", "META", "AVGO",
                   "TSLA", "BRK-B", "LLY", "JPM", "WMT", "V", "ORCL", "NFLX"]
NSIM = 300


class Tee:
    """Write everything printed to the screen and to a file."""

    def __init__(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.f, self.out = open(path, "w"), sys.stdout

    def write(self, s):
        self.out.write(s)
        self.f.write(s)

    def flush(self):
        self.out.flush()
        self.f.flush()


def make_variants(prices, sig):
    """Rule variants expressed as modified signal frames for tr.simulate()."""
    p = prices.to_numpy()
    t = sig.index.to_numpy()
    sma200 = pd.Series(p).rolling(200).mean().to_numpy()
    sma50 = pd.Series(p).rolling(50).mean().to_numpy()
    trend = (p[t] > sma200[t]) & (sma50[t] > sma200[t])
    v = {"Sure (ensemble)": sig}
    s = sig.copy()
    s["agree"] = 1.0
    v["Ensemble P(up) only"] = s
    s = sig.copy()
    s["p_up"], s["agree"] = np.where(trend, 1.0, 0.0), 1.0
    v["Trend only (50/200d)"] = s
    s = sig.copy()
    s["p_up"] = np.where(trend, sig.p_up, 0.0)
    v["Sure + trend"] = s
    return v


def random_sums(px, start, days, rng):
    """For each of NSIM runs: sum of returns of trades with the same lengths
    but random entry dates."""
    n = len(px)
    tot = np.zeros(NSIM)
    for d in days:
        i = rng.integers(start, max(start + 1, n - d - 1), NSIM)
        tot += px[np.minimum(i + d, n - 1)] / px[i] * (1 - tr.COST) ** 2 - 1
    return tot


def load_bundle(tk, k, a, shared_bench):
    if a.demo:
        b = ss.synthetic(a.years, seed=k)
        tr.attach_benchmark(b, a)
        return b
    b = ss.fetch_live(tk, a.years)
    if shared_bench is not None:
        b.benchmark = shared_bench
    return b


def analyse(tk, b, a):
    h = a.horizon
    prices = b.prices
    if len(prices) < 504 + h + 100:
        print(f"[skip] {tk}: only {len(prices)} days of history")
        return None
    bench = getattr(b, "benchmark", None)
    sig = tr.build_signals(prices, bench, h, a.step)
    px = prices.to_numpy()
    rng = np.random.default_rng(sum(ord(c) * 31 ** i for i, c in enumerate(tk)) % 2 ** 32)
    out = {"ticker": tk, "bundle": b, "sig": sig, "var": {}}
    for name, s in make_variants(prices, sig).items():
        eq, trd, start = tr.simulate(prices, s, h)
        rs = random_sums(px, start, trd.days.tolist(), rng) if len(trd) else None
        out["var"][name] = dict(eq=pd.Series(eq, index=prices.index),
                                trades=trd, start=start, rand=rs)
    out["start"] = start
    bh = pd.Series(px / px[start], index=prices.index)
    bh.iloc[:start] = np.nan
    out["bh"] = bh
    out["today"] = tr.conviction(b, tr.ALL_LAYERS, h)
    return out


# --------------------------------------------------------------------------
def report(results, a):
    h = a.horizon
    real = not a.demo
    print(f"\nSource: {'Yahoo Finance' if real else 'SYNTHETIC DEMO DATA'}   "
          f"stocks analysed: {len(results)}   horizon: {h} trading days   "
          f"history: {a.years}y")

    # ---- today's watchlist -------------------------------------------------
    print("\n1. TODAY'S WATCHLIST (all layers; ranked by P(up); paper only)")
    print(f"{'Ticker':<8}{'Last':>10}{'P(up)':>7}{'Agree':>7}{'Median':>10}"
          f"{'5%':>10}{'95%':>10}  Verdict")
    rows = []
    for r in results:
        p_up, agree, fc, _ = r["today"]
        rows.append((p_up, agree, fc, r["ticker"],
                     float(r["bundle"].prices.iloc[-1])))
    for p_up, agree, fc, tk, last in sorted(rows, key=lambda x: -x[0]):
        sure = p_up >= tr.P_ENTRY and agree >= tr.AGREE_MIN
        print(f"{tk:<8}{last:>10,.2f}{p_up:>7.0%}{agree:>7.0%}"
              f"{fc['median']:>10,.2f}{fc['lo']:>10,.2f}{fc['hi']:>10,.2f}  "
              f"{'SURE' if sure else '-'}")

    # ---- pooled conviction study -------------------------------------------
    frames = []
    for k, r in enumerate(results):
        s = r["sig"].copy()
        s.index = s.index / len(r["bundle"].prices) + k * 1e-9
        frames.append(s)
    pooled = pd.concat(frames).sort_index()
    print("\n2. POOLED CONVICTION STUDY (all stocks together; walk-forward)")
    print("   Stocks move together, so the 'independent' count below is "
          "optimistic.")
    tr.conviction_study(pooled, h, a.step)

    # ---- rule comparison ---------------------------------------------------
    print("\n3. RULE COMPARISON (pooled paper trades across all stocks)")
    print(f"{'Rule':<24}{'Trades':>7}{'Win%':>6}{'Avg trade':>10}"
          f"{'1st half':>10}{'2nd half':>10}{'Beats random':>14}")
    names = list(results[0]["var"])
    for nm in names:
        allt, rsum = [], None
        for r in results:
            v = r["var"][nm]
            t = v["trades"].copy()
            if len(t):
                n = len(r["bundle"].prices)
                t["late"] = t.entry >= v["start"] + (n - v["start"]) // 2
                allt.append(t)
                rsum = v["rand"] if rsum is None else rsum + v["rand"]
        if not allt:
            print(f"{nm:<24}{0:>7}   (never traded)")
            continue
        t = pd.concat(allt)
        pct = float(np.mean(rsum / len(t) <= t.ret.mean()))
        e, l = t[~t.late], t[t.late]
        print(f"{nm:<24}{len(t):>7}{(t.ret > 0).mean():>6.0%}"
              f"{t.ret.mean():>10.1%}"
              f"{(e.ret.mean() if len(e) else float('nan')):>10.1%}"
              f"{(l.ret.mean() if len(l) else float('nan')):>10.1%}"
              f"{pct:>14.0%}")
    print("   'Beats random' = share of 300 random-timing runs (same trade "
          "lengths) the rule's average\n   trade beat. ~50% = no skill. A rule "
          "is only interesting if it holds in BOTH halves.")

    # ---- per stock ----------------------------------------------------------
    print("\n4. PER STOCK  (total return over the test period; MaxDD in brackets)")
    print(f"{'Ticker':<8}{'Buy&hold':>16}" +
          "".join(f"{nm[:18]:>22}" for nm in names))
    wins = {nm: 0 for nm in names}
    lucky = {nm: 0 for nm in names}
    for r in results:
        st = r["start"]
        bh = r["bh"].iloc[st:].to_numpy()
        bdd = (bh / np.maximum.accumulate(bh) - 1).min()
        line = f"{r['ticker']:<8}{bh[-1] - 1:>9.0%} ({bdd:>4.0%})"
        for nm in names:
            v = r["var"][nm]
            e = v["eq"].iloc[st:].to_numpy()
            dd = (e / np.maximum.accumulate(e) - 1).min()
            line += f"{e[-1] / e[0] - 1:>14.0%} ({dd:>4.0%})"
            wins[nm] += (e[-1] / e[0]) > bh[-1]
            if v["rand"] is not None and len(v["trades"]):
                lucky[nm] += np.mean(v["rand"] / len(v["trades"])
                                     <= v["trades"].ret.mean()) >= 0.95
        print(line)
    print(f"\n   Rule beat buy-and-hold on:  " +
          ",  ".join(f"{nm[:14]} {wins[nm]}/{len(results)}" for nm in names))
    print(f"   Passed the 95% random-timing test on:  " +
          ",  ".join(f"{nm[:14]} {lucky[nm]}/{len(results)}" for nm in names))
    print(f"   Pure luck would pass ~{0.05 * len(results):.1f} of "
          f"{len(results)} per rule. Counts near that are noise.")

    # ---- equal-weight portfolios --------------------------------------------
    print("\n5. EQUAL-WEIGHT PORTFOLIOS (each stock gets 1/N, daily rebalanced; "
          "common period)")
    st = max(r["start"] for r in results)
    idx0 = max(r["bundle"].prices.index[st] for r in results)

    def port(getter):
        cols = []
        for r in results:
            s = getter(r)
            cols.append(s[s.index >= idx0].pct_change())
        ret = pd.concat(cols, axis=1).mean(axis=1).dropna()
        return (1 + ret).cumprod().to_numpy()

    print(f"   Period: {idx0.date()} -> {results[0]['bundle'].prices.index[-1].date()}")
    print(f"{'':<26}{'Total':>9}{'CAGR':>9}{'MaxDD':>9}{'Sharpe':>8}")
    print(tr.stats(port(lambda r: r["bh"]), "Buy & hold (all stocks)"))
    for nm in names:
        print(tr.stats(port(lambda r, nm=nm: r["var"][nm]["eq"]), nm))
    bm = getattr(results[0]["bundle"], "benchmark", None)
    if bm is not None:
        bm = bm[bm.index >= idx0].dropna()
        if len(bm) > TD:
            print(tr.stats((bm / bm.iloc[0]).to_numpy(), "SPY buy & hold"))

    print("\nCAVEATS: survivorship bias (today's giants), overlapping samples, "
          "multiple testing, and\nno taxes/slippage beyond 10 bps. Paper "
          "simulation only; not financial advice.")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    p.add_argument("--years", type=int, default=10)
    p.add_argument("--horizon", type=int, default=63)
    p.add_argument("--step", type=int, default=5)
    p.add_argument("--demo", action="store_true")
    p.add_argument("--csv", default=None, help=argparse.SUPPRESS)
    p.add_argument("--save", help="also write the report to this file")
    a = p.parse_args()
    if a.save:
        sys.stdout = Tee(a.save)

    shared_bench = None
    if not a.demo:
        try:
            import yfinance as yf
            spy = yf.Ticker("SPY").history(period=f"{a.years}y",
                                           auto_adjust=True)["Close"]
            spy.index = spy.index.tz_localize(None).normalize()
            shared_bench = spy if len(spy) else None
        except Exception as e:
            ss._warn(f"SPY benchmark unavailable: {e}")

    results = []
    for k, tk in enumerate(a.tickers):
        try:
            print(f"... {tk} ({k + 1}/{len(a.tickers)})", file=sys.stderr)
            b = load_bundle(tk, k, a, shared_bench)
            r = analyse(tk, b, a)
            if r:
                results.append(r)
        except SystemExit as e:
            print(f"[skip] {tk}: {e}")
        except Exception as e:
            print(f"[skip] {tk}: {type(e).__name__}: {e}")
        if not a.demo:
            time.sleep(0.5)                        # be polite to Yahoo
    if not results:
        raise SystemExit("No stocks could be analysed.")
    report(results, a)


if __name__ == "__main__":
    main()
