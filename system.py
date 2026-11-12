#!/usr/bin/env python3
"""Paper-trading SYSTEM: one portfolio, many stocks, risk rules, a dashboard.

NO REAL MONEY. There is no broker connection anywhere in this code. Orders are
filled inside a local JSON file (the "paper account").

What it does
  backtest   Walk-forward portfolio test on past prices (price-only layers, no
             look-ahead): the chosen rule vs. the other rules, vs. equal-weight
             buy-and-hold, vs. SPY, and vs. a "shifted-signal" skill test.
  run        Daily cycle for the paper account: fetch data, score every stock
             with ALL layers, apply exits / risk rules / entries, save state.
  dashboard  Build a self-contained dashboard.html from the saved state.
  all        backtest + run + dashboard.

Risk rules (CONFIG below; assumptions chosen up front, not tuned to results)
  * at most MAX_POSITIONS stocks, each at most MAX_WEIGHT of the account
  * volatile stocks get smaller positions (VOL_BUDGET / annual volatility)
  * trailing stop: sell if a stock falls TRAIL_STOP below its high since entry
  * time stop: sell after MAX_HOLD trading days
  * circuit breaker: after a DD_HALT account drawdown, no new buys for
    HALT_DAYS trading days
  * cash buffer and a 10 bps cost on every fill

Honest limits
  * `run` fills at the latest close. Real fills differ.
  * The backtest uses only price-based layers; news / earnings / fundamentals
    have no point-in-time history. Today's signals use all layers, so live
    behaviour will not match the backtest exactly.
  * The default universe is today's largest companies (survivorship bias).
  * Not financial advice.

Usage
    python system.py all --demo            # offline, synthetic data
    python system.py all                   # real data (needs yfinance)
    python system.py run                   # once per trading day, after close
    python system.py dashboard --open
"""
import argparse
import datetime as dt
import json
import math
import os
import sys
import types

import numpy as np
import pandas as pd

import stock_strategies as ss
import trading as tr
from scan import DEFAULT_TICKERS
from stock_strategies import TD

# --------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------
CFG = dict(
    start_cash=100_000.0,
    rule="sure",            # sure | trend | sure+trend
    horizon=63,             # trading days the forecast looks ahead
    max_positions=8,
    max_weight=0.15,
    vol_budget=0.04,        # weight = vol_budget / annual vol (capped)
    trail_stop=0.20,
    max_hold=126,
    dd_halt=0.15,
    halt_days=21,
    cash_buffer=0.02,
    cost=0.001,
)
RULES = ["sure", "trend", "sure+trend"]
RULE_LABEL = {"sure": "Sure (ensemble)", "trend": "Trend only (50/200d)",
              "sure+trend": "Sure + trend"}
WARMUP = 504                # days of history needed before the first signal


def decide(s, rule):
    """-> (enter, exit) for one stock's signal row under a rule."""
    sure = s["p_up"] >= tr.P_ENTRY and s["agree"] >= tr.AGREE_MIN
    if rule == "sure":
        return sure, s["p_up"] < tr.P_EXIT
    if rule == "trend":
        return bool(s["trend"]), not s["trend"]
    if rule == "sure+trend":
        return sure and bool(s["trend"]), s["p_up"] < tr.P_EXIT or not s["trend"]
    raise ValueError(rule)


# --------------------------------------------------------------------------
# Portfolio engine (shared by the backtest and the live paper account)
# --------------------------------------------------------------------------
class Engine:
    def __init__(self, cfg, state=None):
        s = state or {}
        self.cfg = cfg
        self.cash = s.get("cash", cfg["start_cash"])
        self.pos = s.get("positions", {})
        self.peak = s.get("peak", self.cash)
        self.halt_left = s.get("halt_left", 0)
        self.hist = s.get("hist", [])
        self.trades = s.get("trades", [])

    def state(self):
        return dict(cash=self.cash, positions=self.pos, peak=self.peak,
                    halt_left=self.halt_left, hist=self.hist,
                    trades=self.trades)

    def equity(self):
        return self.cash + sum(p["shares"] * p["last"] for p in self.pos.values())

    def _sell(self, tk, date, why):
        p = self.pos.pop(tk)
        proceeds = p["shares"] * p["last"] * (1 - self.cfg["cost"])
        self.cash += proceeds
        self.trades.append(dict(
            ticker=tk, entry_date=p["entry_date"], exit_date=date,
            entry_px=p["entry_px"], exit_px=p["last"], days=p["days"],
            ret=proceeds / p["basis"] - 1, pnl=proceeds - p["basis"], why=why))
        return dict(date=date, action="SELL", ticker=tk, price=p["last"],
                    shares=p["shares"], value=proceeds, why=why)

    def step(self, date, px, sig):
        """Advance one day. px: {ticker: close}. sig: {ticker: row} or None,
        where row has enter / exit / rank / sigma_ann."""
        c, orders = self.cfg, []
        for tk, p in self.pos.items():
            if tk in px:
                p["last"] = px[tk]
                p["high"] = max(p["high"], px[tk])
            p["days"] = int(np.busday_count(p["entry_date"], date))
        for tk in list(self.pos):
            p, why = self.pos[tk], None
            if p["last"] <= p["high"] * (1 - c["trail_stop"]):
                why = "stop"
            elif p["days"] >= c["max_hold"]:
                why = "time"
            elif sig and tk in sig and sig[tk]["exit"]:
                why = "signal"
            if why:
                orders.append(self._sell(tk, date, why))

        eq = self.equity()
        self.peak = max(self.peak, eq)
        if self.halt_left > 0:
            self.halt_left -= 1
            if self.halt_left == 0:
                self.peak = eq                      # restart the drawdown clock
        elif eq / self.peak - 1 <= -c["dd_halt"]:
            self.halt_left = c["halt_days"]

        if sig and self.halt_left == 0:
            cands = sorted((tk for tk, s in sig.items()
                            if s["enter"] and tk not in self.pos and tk in px),
                           key=lambda tk: -sig[tk]["rank"])
            for tk in cands:
                if len(self.pos) >= c["max_positions"]:
                    break
                w = min(c["max_weight"],
                        c["vol_budget"] / max(sig[tk]["sigma_ann"], 1e-6))
                dollars = min(w * eq, self.cash - c["cash_buffer"] * eq)
                if dollars < 0.01 * eq:
                    continue
                shares = dollars * (1 - c["cost"]) / px[tk]
                self.cash -= dollars
                self.pos[tk] = dict(shares=shares, entry_px=px[tk],
                                    entry_date=date, high=px[tk], last=px[tk],
                                    basis=dollars, days=0)
                orders.append(dict(date=date, action="BUY", ticker=tk,
                                   price=px[tk], shares=shares, value=dollars,
                                   why="signal"))
        eq = self.equity()
        self.hist.append([date, eq, 1 - self.cash / eq if eq else 0.0])
        return orders


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def load_universe(a):
    """-> ({ticker: Bundle}, benchmark Series or None)"""
    bundles, bench = {}, None
    if a.demo:
        ns = types.SimpleNamespace(demo=True, csv=None, years=a.years)
        for k, tk in enumerate(a.tickers):
            b = ss.synthetic(a.years, seed=k)
            tr.attach_benchmark(b, ns)
            bundles[tk] = b
        bench = bundles[a.tickers[0]].benchmark
        return bundles, bench
    try:
        import yfinance as yf
        spy = yf.Ticker("SPY").history(period=f"{a.years}y",
                                       auto_adjust=True)["Close"]
        spy.index = spy.index.tz_localize(None).normalize()
        bench = spy if len(spy) else None
    except Exception as e:
        ss._warn(f"SPY benchmark unavailable: {e}")
    for k, tk in enumerate(a.tickers):
        print(f"... loading {tk} ({k + 1}/{len(a.tickers)})", file=sys.stderr)
        try:
            b = ss.fetch_live(tk, a.years)
        except (SystemExit, Exception) as e:
            ss._warn(f"skip {tk}: {e}")
            continue
        if bench is not None:
            b.benchmark = bench
        bundles[tk] = b
    if not bundles:
        raise SystemExit("No data could be loaded (network blocked? try --demo).")
    return bundles, bench


def price_frame(bundles):
    return pd.concat({tk: b.prices for tk, b in bundles.items()},
                     axis=1).sort_index().ffill()


def _trend_flags(prices):
    s200, s50 = prices.rolling(200).mean(), prices.rolling(50).mean()
    return ((prices > s200) & (s50 > s200)).to_numpy()


def signal_grid(bundles, grid_dates, h, layers, label=""):
    """Point-in-time signals: {date_str: {ticker: raw row}}; each row only
    uses prices up to and including that date."""
    out = {}
    for k, (tk, b) in enumerate(bundles.items()):
        if label:
            print(f"... {label} {tk} ({k + 1}/{len(bundles)})", file=sys.stderr)
        pr = b.prices
        bench = getattr(b, "benchmark", None)
        if bench is not None:
            bench = bench.reindex(pr.index).ffill()
        trend = _trend_flags(pr)
        pos = pr.index.searchsorted(grid_dates, side="right") - 1
        for d, t in zip(grid_dates, pos):
            if t < WARMUP or (d - pr.index[t]).days > 5:
                continue
            bb = ss.Bundle(prices=pr.iloc[: t + 1])
            if bench is not None:
                bb.benchmark = bench.iloc[: t + 1]
            p_up, agree, _, _ = tr.conviction(bb, layers, h)
            out.setdefault(str(d.date()), {})[tk] = dict(
                p_up=p_up, agree=agree, trend=bool(trend[t]),
                sigma_ann=ss._vol(bb.prices) * math.sqrt(TD))
    return out


def apply_rule(grid, rule):
    out = {}
    for d, rows in grid.items():
        out[d] = {}
        for tk, s in rows.items():
            enter, exit_ = decide(s, rule)
            out[d][tk] = dict(enter=enter, exit=exit_, rank=s["p_up"],
                              sigma_ann=s["sigma_ann"])
    return out


def shifted(sigmap, rng):
    """Skill-test null: every stock keeps its own sequence of decisions, but
    slid in time by a random amount, so it no longer lines up with prices."""
    dates = sorted(sigmap)
    tks = sorted({tk for d in dates for tk in sigmap[d]})
    out = {d: {} for d in dates}
    for tk in tks:
        ds = [d for d in dates if tk in sigmap[d]]
        if len(ds) < 4:
            continue
        k = int(rng.integers(len(ds) // 4, 3 * len(ds) // 4 + 1))
        for i, d in enumerate(ds):
            src = sigmap[ds[(i + k) % len(ds)]][tk]
            out[d][tk] = dict(src, sigma_ann=sigmap[d][tk]["sigma_ann"])
    return out


# --------------------------------------------------------------------------
# Backtest
# --------------------------------------------------------------------------
def run_engine(cfg, dates, px_rows, sigmap):
    eng = Engine(cfg)
    for i in range(1, len(dates)):
        eng.step(dates[i], px_rows[i], sigmap.get(dates[i - 1]))
    return eng


def curve_stats(e):
    e = np.asarray(e, float)
    r = np.diff(e) / e[:-1]
    yrs = max(len(e) / TD, 1e-9)
    return dict(total=e[-1] / e[0] - 1, cagr=(e[-1] / e[0]) ** (1 / yrs) - 1,
                maxdd=float((e / np.maximum.accumulate(e) - 1).min()),
                vol=float(r.std() * math.sqrt(TD)),
                sharpe=float(r.mean() / r.std() * math.sqrt(TD))
                if r.std() > 0 else 0.0)


def full_stats(e, eng=None):
    s = curve_stats(e)
    m = len(e) // 2
    s["h1"], s["h2"] = e[m] / e[0] - 1, e[-1] / e[m] - 1
    if eng is not None:
        t = eng.trades
        s["trades"] = len(t)
        s["win"] = float(np.mean([x["ret"] > 0 for x in t])) if t else None
        s["avg_trade"] = float(np.mean([x["ret"] for x in t])) if t else None
        s["exposure"] = float(np.mean([h[2] for h in eng.hist]))
    return s


def backtest(bundles, bench, cfg, a):
    px = price_frame(bundles)
    cal = px.index
    if len(cal) < WARMUP + 2 * cfg["horizon"] + 50:
        raise SystemExit("Not enough history for a backtest; use --years 8+.")
    grid = signal_grid(bundles, cal[WARMUP::a.step], cfg["horizon"],
                       tr.PRICE_LAYERS, "signals")
    sub = px.iloc[WARMUP:]
    dates = [str(d.date()) for d in sub.index]
    px_rows = [{tk: float(v) for tk, v in zip(sub.columns, row) if v == v}
               for row in sub.to_numpy()]

    rules, curves, engines = [], {}, {}
    for rule in RULES:
        eng = run_engine(dict(cfg, rule=rule), dates, px_rows,
                         apply_rule(grid, rule))
        e = np.array([cfg["start_cash"]] + [h[1] for h in eng.hist])
        curves[rule], engines[rule] = e, eng
        rules.append(dict(full_stats(e, eng), name=RULE_LABEL[rule], key=rule))

    rets = sub.pct_change().iloc[1:]
    bh = np.concatenate([[1.0], (1 + rets.mean(axis=1)).cumprod().to_numpy()])
    rules.append(dict(full_stats(bh), name="Buy & hold, equal weight",
                      key="bh"))
    spy = None
    if bench is not None:
        s = bench.reindex(sub.index).ffill().bfill().to_numpy()
        if not np.isnan(s).any():
            spy = s / s[0]
            rules.append(dict(full_stats(spy), name="SPY buy & hold",
                              key="spy"))

    rng = np.random.default_rng(11)
    chosen = apply_rule(grid, cfg["rule"])
    sims = []
    for i in range(a.nsim):
        eng = run_engine(cfg, dates, px_rows, shifted(chosen, rng))
        sims.append(curve_stats([cfg["start_cash"]] + [h[1] for h in eng.hist]))
    me = next(r for r in rules if r["key"] == cfg["rule"])
    skill = None
    if sims:
        skill = dict(
            n=len(sims),
            pct_total=float(np.mean([s["total"] <= me["total"] for s in sims])),
            pct_sharpe=float(np.mean([s["sharpe"] <= me["sharpe"] for s in sims])),
            median_total=float(np.median([s["total"] for s in sims])))

    eng = engines[cfg["rule"]]
    by_tk = {}
    for t in eng.trades:
        d = by_tk.setdefault(t["ticker"], dict(ticker=t["ticker"], trades=0,
                                               wins=0, pnl=0.0))
        d["trades"] += 1
        d["wins"] += t["ret"] > 0
        d["pnl"] += t["pnl"]
    c0 = curves[cfg["rule"]][0]
    return dict(
        period=[dates[0], dates[-1]], dates=dates, rule=cfg["rule"],
        strat=(curves[cfg["rule"]] / c0 * 100).round(2).tolist(),
        bh=(bh * 100).round(2).tolist(),
        spy=None if spy is None else (spy * 100).round(2).tolist(),
        exposure=[0.0] + [round(h[2], 3) for h in eng.hist],
        rules=rules, skill=skill,
        by_ticker=sorted(by_tk.values(), key=lambda d: -d["pnl"]),
        n_stocks=len(bundles), step=a.step)


# --------------------------------------------------------------------------
# Live paper cycle
# --------------------------------------------------------------------------
def todays_signals(bundles, cfg):
    h, rows, sig = cfg["horizon"], [], {}
    for tk, b in bundles.items():
        p_up, agree, fc, res = tr.conviction(b, tr.ALL_LAYERS, h)
        raw = dict(p_up=p_up, agree=agree,
                   trend=bool(_trend_flags(b.prices)[-1]),
                   sigma_ann=ss._vol(b.prices) * math.sqrt(TD))
        enter, exit_ = decide(raw, cfg["rule"])
        sig[tk] = dict(enter=enter, exit=exit_, rank=p_up,
                       sigma_ann=raw["sigma_ann"])
        rows.append(dict(
            ticker=tk, last=float(b.prices.iloc[-1]), p_up=p_up, agree=agree,
            trend=raw["trend"], vol=raw["sigma_ann"], median=fc["median"],
            lo=fc["lo"], hi=fc["hi"], enter=enter, exit=exit_,
            layers={n: (None if sg is None or h not in sg.value
                        else float(sg.value[h]) -
                        (ss.MARKET_DRIFT * h / TD if s.kind == "base" else 0))
                    for n, (s, sg) in res.items()}))
    return sig, sorted(rows, key=lambda r: -r["p_up"])


def cycle(bundles, cfg, a, sdir):
    path = os.path.join(sdir, "portfolio.json")
    state = json.load(open(path)) if os.path.exists(path) else None
    px = price_frame(bundles)
    today = str(px.index[-1].date())
    orders = []
    if state is None:
        eng = Engine(cfg)
        n = min(a.replay, len(px) - WARMUP - 2)
        if n > 0:      # warm-start: replay recent days with price-only layers
            days = px.index[-(n + 2):-1]
            grid = apply_rule(signal_grid(bundles, days, cfg["horizon"],
                                          tr.PRICE_LAYERS, "replay"),
                              cfg["rule"])
            ds = [str(d.date()) for d in days]
            for i in range(1, len(ds)):
                row = px.loc[days[i]].dropna().to_dict()
                eng.step(ds[i], row, grid.get(ds[i - 1]))
        state = dict(replay_days=max(n, 0), created=today)
    else:
        eng = Engine(cfg, state["engine"])

    sig, rows = todays_signals(bundles, cfg)
    if state.get("last_date") == today and not a.force:
        print(f"Already ran for {today}; refreshing signals only "
              f"(use --force to re-run).")
        orders = state.get("orders_today", [])
    else:
        orders = eng.step(today, px.iloc[-1].dropna().to_dict(), sig)

    hd = pd.to_datetime([h[0] for h in eng.hist])
    base = px.reindex(hd, method="ffill")
    bench = (base / base.iloc[0]).mean(axis=1) * eng.hist[0][1]
    state.update(engine=eng.state(), last_date=today, orders_today=orders,
                 signals=rows, cfg=cfg, tickers=list(bundles),
                 bench=[round(float(v), 2) for v in bench],
                 mode="demo" if a.demo else "paper")
    with open(path, "w") as f:
        json.dump(state, f, default=float)
    eq = eng.equity()
    print(f"\nPAPER ACCOUNT {today}: equity ${eq:,.0f} "
          f"({eq / cfg['start_cash'] - 1:+.1%}), cash ${eng.cash:,.0f}, "
          f"{len(eng.pos)} positions"
          + (f", HALTED {eng.halt_left}d" if eng.halt_left else ""))
    for o in orders:
        print(f"  {o['action']:<4} {o['ticker']:<6} {o['shares']:.2f} @ "
              f"${o['price']:,.2f}  (${o['value']:,.0f}, {o['why']})")
    if not orders:
        print("  No orders today.")


# --------------------------------------------------------------------------
# Dashboard
# --------------------------------------------------------------------------
def build_dashboard(sdir, open_it=False):
    here = os.path.dirname(os.path.abspath(__file__))
    data = {"generated": dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
            "layer_order": [s.name for s in tr.ALL_LAYERS],
            "thresholds": dict(p_entry=tr.P_ENTRY, agree=tr.AGREE_MIN,
                               p_exit=tr.P_EXIT),
            "rule_label": RULE_LABEL}
    for key in ("portfolio", "backtest"):
        p = os.path.join(sdir, key + ".json")
        data[key] = json.load(open(p)) if os.path.exists(p) else None
    if not data["portfolio"] and not data["backtest"]:
        raise SystemExit(f"Nothing in {sdir}/ yet. Run: python system.py all")
    html = open(os.path.join(here, "dashboard_template.html")).read()
    blob = json.dumps(data, default=float).replace("</", "<\\/")
    out = os.path.join(sdir, "dashboard.html")
    with open(out, "w") as f:
        f.write(html.replace("/*__DATA__*/null", blob))
    print(f"Dashboard written to {out}")
    if open_it:
        import webbrowser
        webbrowser.open("file://" + os.path.abspath(out))


# --------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["backtest", "run", "dashboard", "all"])
    p.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    p.add_argument("--years", type=int, default=10)
    p.add_argument("--rule", choices=RULES, default=CFG["rule"])
    p.add_argument("--horizon", type=int, default=CFG["horizon"])
    p.add_argument("--step", type=int, default=5,
                   help="days between backtest signals")
    p.add_argument("--nsim", type=int, default=100,
                   help="shifted-signal runs for the skill test")
    p.add_argument("--replay", type=int, default=0,
                   help="when creating the paper account, replay this many "
                        "recent days (price-only layers) to give it a history")
    p.add_argument("--cash", type=float, default=CFG["start_cash"])
    p.add_argument("--demo", action="store_true", help="synthetic data")
    p.add_argument("--force", action="store_true")
    p.add_argument("--open", action="store_true", help="open the dashboard")
    a = p.parse_args()

    cfg = dict(CFG, rule=a.rule, horizon=a.horizon, start_cash=a.cash)
    sdir = "state_demo" if a.demo else "state"
    os.makedirs(sdir, exist_ok=True)
    if a.demo and a.replay == 0:
        a.replay = 120

    if a.command in ("backtest", "run", "all"):
        bundles, bench = load_universe(a)
    if a.command in ("backtest", "all"):
        bt = backtest(bundles, bench, cfg, a)
        bt["mode"] = "demo" if a.demo else "paper"
        with open(os.path.join(sdir, "backtest.json"), "w") as f:
            json.dump(bt, f, default=float)
        print(f"\nBACKTEST {bt['period'][0]} -> {bt['period'][1]}  "
              f"({bt['n_stocks']} stocks, rule: {RULE_LABEL[cfg['rule']]})")
        print(f"{'':<28}{'Total':>8}{'CAGR':>8}{'MaxDD':>8}{'Sharpe':>8}"
              f"{'1st half':>10}{'2nd half':>10}{'Trades':>8}{'Win%':>6}")
        for r in bt["rules"]:
            tail = (f"{r['trades']:>8}{r['win']:>6.0%}"
                    if r.get("trades") else "")
            print(f"{r['name']:<28}{r['total']:>8.0%}{r['cagr']:>8.1%}"
                  f"{r['maxdd']:>8.0%}{r['sharpe']:>8.2f}{r['h1']:>10.0%}"
                  f"{r['h2']:>10.0%}{tail}")
        if bt["skill"]:
            k = bt["skill"]
            print(f"Skill test: the rule's return beats {k['pct_total']:.0%} "
                  f"and its Sharpe beats {k['pct_sharpe']:.0%} of {k['n']} "
                  f"shifted-signal runs (~50% = no timing skill).")
    if a.command in ("run", "all"):
        cycle(bundles, cfg, a, sdir)
    if a.command in ("dashboard", "all"):
        build_dashboard(sdir, a.open)


if __name__ == "__main__":
    main()
