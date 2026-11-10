#!/usr/bin/env python3
"""Extra forecast layers + a PAPER-trading simulator, built on stock_strategies.

NO REAL TRADING: nothing here connects to a broker or spends money. "Trading"
means (a) a historical simulation and (b) an optional paper ledger in a local
JSON file.

New layers (all use only past prices, so they CAN be backtested honestly):
  6. TrendFollowing   price vs 50/200-day averages + 12-1 month momentum
  7. MeanReversion    stretch vs 20-day average / RSI (matters at short horizons)
  8. VolatilityRegime widens the forecast range when volatility is spiking
  9. MarketRegime     benchmark (SPY) trend + the stock's relative strength

"Sure" means BOTH:
  * P(up)  = model probability that the price is higher after the horizon
  * agree  = share of layers (with a view) that lean bullish

The question asked: if we bought only when "sure", would it have won?
This script answers it two ways, using only information available at each
past date (walk-forward, no look-ahead):
  A. Conviction study: hit-rate / average return of "sure" signals vs. all days
  B. Trade simulation: buy when sure, exit on lost conviction / stop / time,
     with costs, vs. buy-and-hold and vs. 500 random-timing runs

Only the price-based layers are in the historical test. Fundamentals, news and
earnings layers exist in `today's signal` but have no point-in-time history.

Usage:
    python trading.py --demo                 # offline synthetic data
    python trading.py --ticker TSLA --years 10
    python trading.py --paper                # today's paper decision
    python trading.py --paper --paper-execute  # also record it in the ledger
"""
import argparse
import json
import math
import os

import numpy as np
import pandas as pd

import stock_strategies as ss
from stock_strategies import Signal, TD, _vol

# --------------------------------------------------------------------------
# CONFIG (assumptions -- not fitted to data; set BEFORE looking at results)
# --------------------------------------------------------------------------
TREND_K, REVERSION_K, MARKET_K = 0.20, 0.30, 0.15   # tilt in sigma*sqrt(h) units
ss.OVERLAY_WEIGHT.update({"TrendFollowing": 0.5, "MeanReversion": 0.5,
                          "MarketRegime": 0.5, "VolatilityRegime": 1.0})
P_ENTRY, AGREE_MIN, P_EXIT = 0.55, 0.60, 0.50
STOP_LOSS, COST, TARGET_VOL = 0.15, 0.001, 0.30     # 10 bps per side
LEDGER = "paper_ledger.json"


def norm_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


# --------------------------------------------------------------------------
# Extra layers
# --------------------------------------------------------------------------
class TrendFollowing:
    name, kind = "TrendFollowing", "overlay"

    def predict(self, b, horizons):
        p = b.prices.to_numpy()
        if len(p) < 253:
            return None
        sma50, sma200 = p[-50:].mean(), p[-200:].mean()
        mom = np.log(p[-22] / p[-253])               # 12-1 month momentum
        score = np.mean([np.sign(p[-1] - sma200), np.sign(sma50 - sma200),
                         np.sign(mom)])
        sd = _vol(b.prices)
        return Signal({h: TREND_K * score * sd * np.sqrt(h) for h in horizons},
                      f"trend score {score:+.2f} (price/200d, 50d/200d, 12-1m "
                      f"momentum)")


class MeanReversion:
    name, kind = "MeanReversion", "overlay"

    def predict(self, b, horizons):
        lp = np.log(b.prices.to_numpy())
        if len(lp) < 30:
            return None
        w = lp[-20:]
        z = (lp[-1] - w.mean()) / (w.std(ddof=1) + 1e-9)
        d = np.diff(lp[-15:])
        up, dn = d[d > 0].sum(), -d[d < 0].sum()
        rsi = 0.5 if up + dn == 0 else up / (up + dn)
        score = -np.clip(0.5 * z / 2 + 0.5 * (rsi - 0.5) * 2, -1, 1)
        sd = _vol(b.prices)
        return Signal({h: REVERSION_K * score * sd * np.sqrt(h) *
                       math.exp(-h / 30) for h in horizons},
                      f"20d z {z:+.1f}, RSI {rsi:.0%}; fades by ~1 month")


class VolatilityRegime:
    name, kind = "VolatilityRegime", "overlay"

    def predict(self, b, horizons):
        r = ss._daily_log_returns(b.prices)
        if len(r) < 126:
            return None
        ratio = r[-21:].std(ddof=1) / (r.std(ddof=1) + 1e-12)
        sd = _vol(b.prices)
        extra = {h: max(0.0, ratio ** 2 - 1) * 0.5 * sd ** 2 * h
                 for h in horizons}
        return Signal({h: 0.0 for h in horizons},
                      f"1-month vol is {ratio:.2f}x long-run "
                      f"({'widening range' if ratio > 1 else 'calm'})", extra)


class MarketRegime:
    name, kind = "MarketRegime", "overlay"

    def predict(self, b, horizons):
        bm = getattr(b, "benchmark", None)
        if bm is None:
            return None
        bm = bm.reindex(b.prices.index).ffill().dropna()
        if len(bm) < 253:
            return None
        p = b.prices.reindex(bm.index)
        trend = np.sign(bm.iloc[-1] - bm.iloc[-200:].mean())
        rel = np.log(p.iloc[-1] / p.iloc[-127]) - np.log(bm.iloc[-1] / bm.iloc[-127])
        score = 0.5 * trend + 0.5 * np.clip(rel / 0.3, -1, 1)
        sd = _vol(b.prices)
        return Signal({h: MARKET_K * score * sd * np.sqrt(h) for h in horizons},
                      f"market {'up' if trend > 0 else 'down'}trend, "
                      f"6m relative strength {rel:+.0%}")


def _by_name(n):
    return next(s for s in ss.STRATEGIES if s.name == n)


PRICE_LAYERS = [_by_name("HistoricalDrift"), _by_name("TechnicalML"),
                TrendFollowing(), MeanReversion(), VolatilityRegime(),
                MarketRegime()]
ALL_LAYERS = [_by_name("HistoricalDrift"), _by_name("Fundamentals"),
              _by_name("TechnicalML"), _by_name("NewsSentiment"),
              _by_name("Earnings"), TrendFollowing(), MeanReversion(),
              VolatilityRegime(), MarketRegime()]


# --------------------------------------------------------------------------
# Benchmark data
# --------------------------------------------------------------------------
def attach_benchmark(b, args):
    if args.demo:
        rng = np.random.default_rng(1)
        r = np.diff(np.log(b.prices.to_numpy()), prepend=np.log(b.prices.iloc[0]))
        rb = 0.0003 + 0.12 * (r - r.mean()) + rng.normal(0, 0.008, len(r))
        b.benchmark = pd.Series(400 * np.exp(np.cumsum(rb)), index=b.prices.index)
    elif not args.csv:
        try:
            import yfinance as yf
            h = yf.Ticker("SPY").history(period=f"{args.years}y",
                                         auto_adjust=True)["Close"]
            h.index = h.index.tz_localize(None).normalize()
            b.benchmark = h
        except Exception as e:
            ss._warn(f"benchmark unavailable, MarketRegime skipped: {e}")
    return b


# --------------------------------------------------------------------------
# Conviction
# --------------------------------------------------------------------------
def conviction(b, layers, h):
    """-> (p_up, agree, forecast, results) using only data inside bundle b."""
    res = ss.run_strategies(b, layers, [h])
    fc = ss.ensemble(b, res, [h])[h]
    p_up = norm_cdf(fc["mu"] / fc["sd"])
    votes = []
    for s, sig in res.values():
        if sig is None or h not in sig.value:
            continue
        v = sig.value[h] - (ss.MARKET_DRIFT * h / TD if s.kind == "base" else 0)
        if abs(v) > 1e-4:
            votes.append(v > 0)
    agree = float(np.mean(votes)) if votes else 0.5
    return p_up, agree, fc, res


def build_signals(prices, bench, h, step, start=504):
    """Walk-forward: conviction at every `step`-th day, past data only."""
    rows = {}
    bench_al = None if bench is None else bench.reindex(prices.index).ffill()
    for t in range(start, len(prices) - 1, step):
        b = ss.Bundle(prices=prices.iloc[: t + 1])
        if bench_al is not None:
            b.benchmark = bench_al.iloc[: t + 1]
        p_up, agree, fc, _ = conviction(b, PRICE_LAYERS, h)
        fwd = (math.log(prices.iloc[t + h] / prices.iloc[t])
               if t + h < len(prices) else np.nan)
        rows[t] = dict(p_up=p_up, agree=agree, mu=fc["mu"],
                       sigma_ann=_vol(b.prices) * math.sqrt(TD), fwd=fwd)
    return pd.DataFrame.from_dict(rows, orient="index")


# --------------------------------------------------------------------------
# A. Conviction study
# --------------------------------------------------------------------------
def conviction_study(sig, h, step):
    d = sig.dropna(subset=["fwd"])
    base = (d.fwd > 0).mean()
    neff_factor = min(1.0, step / h)          # overlapping windows
    print(f"\nA. CONVICTION STUDY  ({len(d)} walk-forward signals, "
          f"horizon {h} trading days)")
    print(f"   Overlap: signals are {step} days apart but outcomes span {h}, "
          f"so ~{max(1, int(len(d) * neff_factor))} are independent.")
    print(f"{'Group':<34}{'N':>5}{'Up-rate':>9}{'Avg ret':>9}{'Median':>9}"
          f"{'z vs all':>10}")

    def row(label, g):
        if len(g) == 0:
            print(f"{label:<34}{0:>5}   (no signals)")
            return
        n_eff = max(1.0, len(g) * neff_factor)
        z = (g.fwd.gt(0).mean() - base) / math.sqrt(base * (1 - base) / n_eff)
        print(f"{label:<34}{len(g):>5}{g.fwd.gt(0).mean():>9.0%}"
              f"{np.expm1(g.fwd.mean()):>9.1%}{np.expm1(g.fwd.median()):>9.1%}"
              f"{z:>10.1f}")

    row("All days (base rate)", d)
    for q in (0.5, 0.75, 0.9):
        cut = d.p_up.quantile(q)
        row(f"Top {round((1 - q) * 100)}% by P(up) (>= {cut:.2f})", d[d.p_up >= cut])
    sure = d[(d.p_up >= P_ENTRY) & (d.agree >= AGREE_MIN)]
    row(f"SURE: P(up)>={P_ENTRY}, agree>={AGREE_MIN:.0%}", sure)
    row("NOT sure (everything else)", d.drop(sure.index))
    if len(sure) >= 10:
        mid = d.index[len(d) // 2]
        row("  SURE, first half of period", sure[sure.index < mid])
        row("  SURE, second half of period", sure[sure.index >= mid])
    print("   z > ~2 would be statistically notable; below that, differences "
          "are within noise.")


# --------------------------------------------------------------------------
# B. Trade simulation
# --------------------------------------------------------------------------
def simulate(prices, sig, h):
    px = prices.to_numpy()
    n = len(px)
    smap = sig.to_dict("index")
    start = int(sig.index.min()) + 1
    cash, shares, entry_px, entry_i, eq_in = 1.0, 0.0, 0.0, 0, 0.0
    equity = np.full(n, np.nan)
    trades = []

    def close(i, why):
        nonlocal cash, shares
        proceeds = shares * px[i] * (1 - COST)
        trades.append(dict(entry=entry_i, exit=i, why=why,
                           ret=proceeds / eq_in - 1, days=i - entry_i))
        cash += proceeds
        shares = 0.0

    for i in range(start, n):
        if shares > 0:
            if px[i] / entry_px - 1 <= -STOP_LOSS:
                close(i, "stop")
            elif i - entry_i >= 2 * h:
                close(i, "time")
        row = smap.get(i - 1)                 # signal known at prior close
        if row is not None:
            if shares > 0 and row["p_up"] < P_EXIT:
                close(i, "signal")
            elif (shares == 0 and row["p_up"] >= P_ENTRY
                  and row["agree"] >= AGREE_MIN):
                size = float(np.clip(TARGET_VOL / row["sigma_ann"], 0, 1))
                eq_in = cash * size
                shares = eq_in * (1 - COST) / px[i]
                cash -= eq_in
                entry_px, entry_i = px[i], i
        equity[i] = cash + shares * px[i]
    if shares > 0:
        close(n - 1, "end")
        equity[n - 1] = cash
    return equity, pd.DataFrame(trades), start


def stats(eq, label):
    e = eq[~np.isnan(eq)]
    r = np.diff(e) / e[:-1]
    yrs = len(e) / TD
    dd = (e / np.maximum.accumulate(e) - 1).min()
    sharpe = r.mean() / r.std() * math.sqrt(TD) if r.std() > 0 else 0.0
    return (f"{label:<26}{e[-1] / e[0] - 1:>9.0%}{(e[-1] / e[0]) ** (1 / yrs) - 1:>9.1%}"
            f"{dd:>9.0%}{sharpe:>8.2f}")


def trade_report(prices, sig, h):
    equity, tr, start = simulate(prices, sig, h)
    px = prices.to_numpy()
    n = len(px)
    bh = px / px[start] * (1 - COST)
    bh[:start] = np.nan
    print(f"\nB. TRADE SIMULATION  (long-only, paper; cost {COST * 1e4:.0f} bps "
          f"per side, stop {STOP_LOSS:.0%}, vol-target {TARGET_VOL:.0%})")
    print(f"   Period: {prices.index[start].date()} -> {prices.index[-1].date()}")
    print(f"{'':<26}{'Total':>9}{'CAGR':>9}{'MaxDD':>9}{'Sharpe':>8}")
    print(stats(equity[start:], "Strategy ('buy when sure')"))
    print(stats(bh[start:], "Buy & hold"))
    if tr.empty:
        print("   No trades: the 'sure' rule never fired. Loosen thresholds "
              "(--p-entry / --agree).")
        return
    expo = np.nanmean([t["days"] for t in tr.to_dict("records")]) * len(tr) \
        / (n - start)
    wins = tr.ret > 0
    pf = tr.ret[wins].sum() / max(1e-9, -tr.ret[~wins].sum())
    print(f"\n   Trades: {len(tr)}   Win rate: {wins.mean():.0%}   "
          f"Avg trade: {tr.ret.mean():+.1%}   Median: {tr.ret.median():+.1%}   "
          f"Profit factor: {pf:.2f}")
    print(f"   Time invested: {expo:.0%}   Exits: "
          + ", ".join(f"{k} {v}" for k, v in tr.why.value_counts().items()))
    # skill test: same trades (same lengths), random entry dates
    rng = np.random.default_rng(7)
    sims = []
    for _ in range(500):
        rets = []
        for d in tr.days:
            i = int(rng.integers(start, max(start + 1, n - d - 1)))
            rets.append(px[min(i + d, n - 1)] / px[i] * (1 - COST) ** 2 - 1)
        sims.append(np.mean(rets))
    pct = float(np.mean(np.array(sims) <= tr.ret.mean()))
    print(f"   Skill check: avg trade beats {pct:.0%} of 500 random-timing runs "
          f"(same trade lengths; ~50% = no skill, >95% = notable)")


# --------------------------------------------------------------------------
# Paper trading (no broker, no money)
# --------------------------------------------------------------------------
def paper_decision(b, h):
    p_up, agree, fc, res = conviction(b, ALL_LAYERS, h)
    sure = p_up >= P_ENTRY and agree >= AGREE_MIN
    return p_up, agree, fc, res, sure


def paper_step(b, sure, p_up, execute):
    price = float(b.prices.iloc[-1])
    led = {"cash": 10000.0, "shares": 0.0, "log": []}
    if os.path.exists(LEDGER):
        with open(LEDGER) as f:
            led = json.load(f)
    action = "HOLD"
    if led["shares"] == 0 and sure:
        action = "BUY"
    elif led["shares"] > 0 and p_up < P_EXIT:
        action = "SELL"
    print(f"\nPAPER DECISION (hypothetical only): {action}   "
          f"[P(up) {p_up:.0%}; ledger: ${led['cash']:,.0f} cash, "
          f"{led['shares']:.2f} shares]")
    if execute and action != "HOLD":
        if action == "BUY":
            led["shares"] = led["cash"] * (1 - COST) / price
            led["cash"] = 0.0
        else:
            led["cash"] = led["shares"] * price * (1 - COST)
            led["shares"] = 0.0
        led["log"].append({"date": str(b.prices.index[-1].date()),
                           "action": action, "price": price})
        with open(LEDGER, "w") as f:
            json.dump(led, f, indent=2)
        print(f"   Recorded {action} at ${price:,.2f} in {LEDGER}")
    elif execute:
        print("   Nothing to record.")


# --------------------------------------------------------------------------
def main():
    global P_ENTRY, AGREE_MIN
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ticker", default="TSLA")
    p.add_argument("--years", type=int, default=10)
    p.add_argument("--csv")
    p.add_argument("--demo", action="store_true")
    p.add_argument("--horizon", type=int, default=63, help="trading days")
    p.add_argument("--step", type=int, default=5)
    p.add_argument("--p-entry", type=float, default=P_ENTRY)
    p.add_argument("--agree", type=float, default=AGREE_MIN)
    p.add_argument("--paper", action="store_true")
    p.add_argument("--paper-execute", action="store_true")
    a = p.parse_args()
    P_ENTRY, AGREE_MIN = a.p_entry, a.agree

    b = ss.synthetic(a.years) if a.demo else ss.from_csv(a.csv) if a.csv \
        else ss.fetch_live(a.ticker, a.years)
    attach_benchmark(b, a)
    h = a.horizon
    print(f"Source: {b.source}\nHistory: {b.prices.index[0].date()} -> "
          f"{b.prices.index[-1].date()}   Last close: "
          f"${float(b.prices.iloc[-1]):,.2f}")

    p_up, agree, fc, res, sure = paper_decision(b, h)
    print(f"\nTODAY ({h}-day view, all layers)")
    for name, (s, sg) in res.items():
        v = "-- skipped (no data) --" if sg is None else \
            f"{sg.value[h]:+.1%}  {sg.note}"
        print(f"  {name:<17}{s.kind:<9}{v}")
    f = fc
    print(f"  Ensemble median ${f['median']:,.2f}  (5-95%: ${f['lo']:,.2f} - "
          f"${f['hi']:,.2f})   P(up) {p_up:.0%}   agreement {agree:.0%}   "
          f"-> {'SURE' if sure else 'not sure'}")

    if a.paper:
        paper_step(b, sure, p_up, a.paper_execute)
        return

    bench = getattr(b, "benchmark", None)
    if len(b.prices) < 504 + h + 100:
        raise SystemExit("Not enough history for the backtest; use --years 8+.")
    sig = build_signals(b.prices, bench, h, a.step)
    print(f"\nBacktest layers (price-only, point-in-time): "
          + ", ".join(s.name for s in PRICE_LAYERS
                      if bench is not None or s.name != "MarketRegime"))
    conviction_study(sig, h, a.step)
    trade_report(b.prices, sig, h)
    print("\nPast results do not predict future returns. Simulation only; not "
          "financial advice.")


if __name__ == "__main__":
    main()
