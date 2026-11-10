#!/usr/bin/env python3
"""Five stock-forecasting strategies plus an ensemble, for TSLA (or any ticker).

Strategies (each returns an expected log-return for 6m / 12m / 2y):

  BASE strategies -- produce a full expected return, averaged together:
    1. HistoricalDrift  price history; drift shrunk toward a market prior
    2. Fundamentals     valuation gap (analyst target + growth-adjusted P/E)
                        closing slowly toward fair value

  OVERLAY strategies -- produce a tilt added on top of the base return:
    3. TechnicalML      ridge regression on momentum / volatility / RSI
    4. NewsSentiment    lexicon score of recent headlines (decaying effect)
    5. Earnings         post-earnings drift vs. EPS surprise + earnings-day risk

The ensemble combines them and builds a 5th/50th/95th percentile price range
whose width includes (a) normal volatility, (b) uncertainty in the drift
estimate itself, and (c) known earnings-day jump risk.

IMPORTANT HONESTY NOTES
  * Several coefficients (market prior, sentiment strength, fair-value
    P/E-to-growth ratio, overlay weights) are documented ASSUMPTIONS, not
    values fitted to data. Change them in the CONFIG block.
  * Only HistoricalDrift and TechnicalML can be backtested honestly. Yahoo
    only gives *current* news / fundamentals, so testing those on the past
    would leak the future. --backtest therefore excludes them.
  * Nothing here can predict price to "90% accuracy". This is not
    financial advice.

Usage:
    pip install numpy pandas yfinance
    python stock_strategies.py                    # live data for TSLA
    python stock_strategies.py --ticker NVDA --years 10
    python stock_strategies.py --csv prices.csv   # prices only (Date,Close)
    python stock_strategies.py --demo --backtest  # offline synthetic data
"""
import argparse
import re
import sys
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# CONFIG (assumptions -- edit freely)
# --------------------------------------------------------------------------
TD = 252                                   # trading days per year
HORIZONS = {"6 months": 126, "12 months": 252, "2 years": 504}
MARKET_DRIFT = 0.07                        # prior annual log drift (~8% arith.)
PRIOR_DRIFT_SD = 0.15                      # how far a stock's true drift may
                                           # plausibly differ from the market
SENTIMENT_MAX_TILT = 0.03                  # max sentiment effect (log return)
SENTIMENT_DECAY_DAYS = 21                  # effect builds/decays on this scale
HEADLINE_HALF_LIFE = 7                     # days; older headlines count less
PEG_FAIR = 1.5                             # "fair" P/E = growth% * PEG_FAIR
VALUATION_HALF_LIFE = 252                  # days to close half the value gap
ANALYST_TARGET_HAIRCUT = 0.5               # analyst targets run optimistic
OVERLAY_WEIGHT = {"TechnicalML": 0.5, "NewsSentiment": 1.0, "Earnings": 1.0}
Z90 = 1.6448536269514722                   # z for the 5th / 95th percentile


# --------------------------------------------------------------------------
# Data containers and loaders
# --------------------------------------------------------------------------
@dataclass
class Bundle:
    """Everything the strategies may look at. Missing parts are just empty."""
    prices: pd.Series
    headlines: list = field(default_factory=list)   # [(Timestamp, text)]
    earnings: pd.DataFrame = None                   # index=date, cols below
    next_earnings: pd.Timestamp = None
    fundamentals: dict = field(default_factory=dict)
    source: str = ""


def _warn(msg):
    print(f"[warn] {msg}", file=sys.stderr)


def fetch_live(ticker, years):
    """Download prices, news, earnings and fundamentals from Yahoo Finance.
    Each component is fetched independently so one failure doesn't kill all."""
    try:
        import yfinance as yf
    except ImportError:
        raise SystemExit("pip install yfinance (or use --csv / --demo)")
    t = yf.Ticker(ticker)
    hist = t.history(period=f"{years}y", auto_adjust=True)
    if hist.empty:
        raise SystemExit("No price data downloaded (network blocked? "
                         "use --csv or --demo).")
    prices = hist["Close"].dropna()
    prices.index = prices.index.tz_localize(None).normalize()
    b = Bundle(prices=prices, source=f"Yahoo Finance ({ticker})")

    try:
        for item in t.news or []:
            c = item.get("content", item)
            title = c.get("title") or ""
            when = c.get("pubDate") or c.get("displayTime")
            if when is None and "providerPublishTime" in item:
                when = pd.to_datetime(item["providerPublishTime"], unit="s")
            if title and when is not None:
                ts = pd.to_datetime(when).tz_localize(None) \
                    if pd.to_datetime(when).tzinfo else pd.to_datetime(when)
                b.headlines.append((ts, title))
    except Exception as e:
        _warn(f"news unavailable: {e}")

    try:
        ed = t.earnings_dates
        if ed is not None and not ed.empty:
            ed = ed.copy()
            ed.index = ed.index.tz_localize(None).normalize()
            future = ed[ed["Reported EPS"].isna()]
            if len(future):
                b.next_earnings = future.index.min()
            b.earnings = ed.dropna(subset=["Reported EPS"]).sort_index()
    except Exception as e:
        _warn(f"earnings unavailable: {e}")

    try:
        info = t.info or {}
        keys = ["currentPrice", "targetMeanPrice", "forwardPE", "trailingPE",
                "earningsGrowth", "revenueGrowth", "profitMargins"]
        b.fundamentals = {k: info[k] for k in keys if info.get(k) is not None}
    except Exception as e:
        _warn(f"fundamentals unavailable: {e}")
    return b


def from_csv(path):
    df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
    return Bundle(prices=df["Close"].dropna().sort_index(), source=path)


def synthetic(years, seed=0):
    """Fake but plausible data so the whole pipeline can run offline."""
    rng = np.random.default_rng(seed)
    n = TD * years
    rets = rng.normal(0.0006, 0.03, n)
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n)
    prices = pd.Series(200 * np.exp(np.cumsum(rets)), index=idx, name="Close")
    today = idx[-1]
    heads = [("Record deliveries beat expectations, shares surge", 2),
             ("Analysts upgrade stock after strong growth", 4),
             ("Company faces recall and investigation over safety", 6),
             ("Margins miss as demand weakens, downgrade follows", 9),
             ("New factory expansion announced", 12)]
    news = [(today - pd.Timedelta(days=d), h) for h, d in heads]
    ev = pd.DatetimeIndex([idx[i] for i in range(len(idx) - 20, 40, -63)])
    sur = rng.normal(0, 12, len(ev))
    earn = pd.DataFrame({"Reported EPS": 1.0, "EPS Estimate": 1.0,
                         "Surprise(%)": sur}, index=ev).sort_index()
    fund = {"currentPrice": float(prices.iloc[-1]),
            "targetMeanPrice": float(prices.iloc[-1]) * 1.10,
            "forwardPE": 80.0, "earningsGrowth": 0.25,
            "revenueGrowth": 0.12, "profitMargins": 0.08}
    return Bundle(prices, news, earn, today + pd.Timedelta(days=40), fund,
                  source="SYNTHETIC DEMO DATA")


# --------------------------------------------------------------------------
# Signal
# --------------------------------------------------------------------------
@dataclass
class Signal:
    """Per-horizon result of one strategy. `value` is a total expected log
    return for BASE strategies and an additive tilt for OVERLAYS."""
    value: dict            # {horizon_days: log return}
    note: str = ""
    extra_var: dict = field(default_factory=dict)   # {horizon_days: variance}


def _daily_log_returns(prices):
    return np.diff(np.log(prices.to_numpy()))


def _vol(prices):
    """Daily vol: blend of full-sample and last-3-month volatility."""
    r = _daily_log_returns(prices)
    full = r.std(ddof=1)
    recent = r[-63:].std(ddof=1) if len(r) >= 63 else full
    return float(np.sqrt(0.6 * full ** 2 + 0.4 * recent ** 2))


# --------------------------------------------------------------------------
# 1. Historical drift (BASE)
# --------------------------------------------------------------------------
class HistoricalDrift:
    name, kind = "HistoricalDrift", "base"

    def predict(self, b, horizons):
        r = _daily_log_returns(b.prices)
        n = len(r)
        if n < 120:
            return None
        mu_hist = r.mean() * TD                      # annual
        sigma = r.std(ddof=1) * np.sqrt(TD)
        se = sigma / np.sqrt(n / TD)                 # std error of the drift
        w = PRIOR_DRIFT_SD ** 2 / (PRIOR_DRIFT_SD ** 2 + se ** 2)
        mu = w * mu_hist + (1 - w) * MARKET_DRIFT    # empirical-Bayes shrink
        note = (f"hist drift {mu_hist:+.0%}/yr, shrunk (weight {w:.2f}) "
                f"toward {MARKET_DRIFT:.0%} -> {mu:+.1%}/yr")
        return Signal({h: mu * h / TD for h in horizons}, note)


# --------------------------------------------------------------------------
# 2. Fundamentals / valuation (BASE)
# --------------------------------------------------------------------------
class Fundamentals:
    name, kind = "Fundamentals", "base"

    def predict(self, b, horizons):
        f = b.fundamentals
        price = f.get("currentPrice") or float(b.prices.iloc[-1])
        gaps, parts = [], []
        tgt = f.get("targetMeanPrice")
        if tgt and tgt > 0:
            g = ANALYST_TARGET_HAIRCUT * np.log(tgt / price)
            gaps.append(g)
            parts.append(f"analyst target ${tgt:,.0f} (gap {g:+.0%} after haircut)")
        fpe, gr = f.get("forwardPE"), f.get("earningsGrowth")
        if fpe and fpe > 0 and gr and gr > 0:
            fair_pe = float(np.clip(gr * 100 * PEG_FAIR, 10, 60))
            g = np.log(fair_pe / fpe)
            gaps.append(g)
            parts.append(f"fwd P/E {fpe:.0f} vs growth-fair {fair_pe:.0f} "
                         f"(gap {g:+.0%})")
        if not gaps:
            return None
        gap = float(np.clip(np.mean(gaps), -0.5, 0.5))
        out = {}
        for h in horizons:
            closed = 1 - 0.5 ** (h / VALUATION_HALF_LIFE)
            out[h] = MARKET_DRIFT * h / TD + closed * gap
        return Signal(out, "; ".join(parts))


# --------------------------------------------------------------------------
# 3. Technical ML (OVERLAY)
# --------------------------------------------------------------------------
def _features(logp, t):
    """Feature vector using data up to index t (inclusive) only."""
    r = np.diff(logp[: t + 1])
    ma200 = logp[t - 199: t + 1].mean()
    d = r[-14:]
    up, dn = d[d > 0].sum(), -d[d < 0].sum()
    rsi = 0.5 if up + dn == 0 else up / (up + dn)
    return np.array([logp[t] - logp[t - 21], logp[t] - logp[t - 63],
                     logp[t] - logp[t - 126], logp[t] - logp[t - 252],
                     r[-63:].std(ddof=1) * np.sqrt(TD),
                     logp[t] - ma200, rsi])


class TechnicalML:
    name, kind = "TechnicalML", "overlay"
    MIN_SAMPLES = 20

    def predict(self, b, horizons):
        logp = np.log(b.prices.to_numpy())
        T = len(logp) - 1
        if T < 252 + 60:
            return None
        sigma_d = _vol(b.prices)
        out, used = {}, []
        for h in horizons:
            ts = list(range(252, T - h + 1, 5))      # targets fully realised
            if len(ts) < self.MIN_SAMPLES:
                continue
            X = np.array([_features(logp, t) for t in ts])
            y = np.array([logp[t + h] - logp[t] for t in ts])
            mu, sd = X.mean(0), X.std(0) + 1e-9
            Z = (X - mu) / sd
            alpha = 5.0 * len(ts)                    # strong shrinkage
            yc = y - y.mean()
            beta = np.linalg.solve(Z.T @ Z + alpha * np.eye(Z.shape[1]),
                                   Z.T @ yc)
            tilt = float(((_features(logp, T) - mu) / sd) @ beta)
            cap = 0.5 * sigma_d * np.sqrt(h)
            out[h] = float(np.clip(tilt, -cap, cap))
            used.append(h)
        if not out:
            return None
        return Signal(out, f"ridge on 7 technical features; "
                           f"horizons trained: {len(used)}/{len(horizons)}")


# --------------------------------------------------------------------------
# 4. News sentiment (OVERLAY)
# --------------------------------------------------------------------------
POS = {"beat", "beats", "surge", "surges", "soar", "soars", "jump", "jumps",
       "rally", "record", "strong", "growth", "upgrade", "upgrades",
       "outperform", "profit", "profitable", "gain", "gains", "rise", "rises",
       "boost", "boosts", "approval", "approved", "expansion", "bullish",
       "exceeds", "wins", "breakthrough", "optimistic", "buy"}
NEG = {"miss", "misses", "plunge", "plunges", "fall", "falls", "drop", "drops",
       "slump", "weak", "weakens", "downgrade", "downgrades", "recall",
       "recalls", "lawsuit", "probe", "investigation", "loss", "losses",
       "cut", "cuts", "decline", "declines", "fraud", "crash", "bearish",
       "warns", "warning", "delay", "delays", "layoffs", "sell", "concern",
       "concerns", "slows", "fear", "fears"}
NEGATORS = {"not", "no", "never", "without", "fails", "despite"}


def headline_score(text):
    """Score in [-1, 1]: (pos - neg) / (pos + neg + 2) with simple negation."""
    toks = re.findall(r"[a-z']+", text.lower())
    pos = neg = 0
    for i, w in enumerate(toks):
        flip = any(t in NEGATORS for t in toks[max(0, i - 2): i])
        if w in POS:
            pos, neg = (pos, neg + 1) if flip else (pos + 1, neg)
        elif w in NEG:
            pos, neg = (pos + 1, neg) if flip else (pos, neg + 1)
    return (pos - neg) / (pos + neg + 2)


class NewsSentiment:
    name, kind = "NewsSentiment", "overlay"

    def predict(self, b, horizons):
        if not b.headlines:
            return None
        now = b.prices.index[-1]
        ws, ss = [], []
        for ts, text in b.headlines:
            age = max((now - ts).days, 0)
            ws.append(0.5 ** (age / HEADLINE_HALF_LIFE))
            ss.append(headline_score(text))
        ws, ss = np.array(ws), np.array(ss)
        if ws.sum() < 1e-6:
            return None
        agg = float((ws * ss).sum() / ws.sum())
        agg *= ws.sum() / (ws.sum() + 3.0)           # few headlines -> shrink
        out = {h: SENTIMENT_MAX_TILT * agg *
               (1 - np.exp(-h / SENTIMENT_DECAY_DAYS)) for h in horizons}
        return Signal(out, f"{len(ss)} headlines, weighted score {agg:+.2f}; "
                           f"max effect {SENTIMENT_MAX_TILT:.0%} (assumed)")


# --------------------------------------------------------------------------
# 5. Earnings (OVERLAY)
# --------------------------------------------------------------------------
class Earnings:
    name, kind = "Earnings", "overlay"
    WINDOW = 21                                       # trading days after

    def predict(self, b, horizons):
        e = b.earnings
        if e is None or len(e) < 6 or "Surprise(%)" not in e:
            return None
        idx = b.prices.index
        sur, post, react = [], [], []
        for d, s in e["Surprise(%)"].dropna().items():
            pos = idx.searchsorted(d, side="right") - 1
            if pos < 0 or pos + self.WINDOW >= len(idx):
                continue
            sur.append(float(np.clip(s, -50, 50)))
            post.append(np.log(b.prices.iloc[pos + self.WINDOW] /
                               b.prices.iloc[pos]))
            react.append(abs(np.log(b.prices.iloc[pos + 1] /
                                    b.prices.iloc[pos])))
        n = len(sur)
        if n < 6:
            return None
        sur, post = np.array(sur), np.array(post)
        var_s = sur.var()
        slope = 0.0 if var_s < 1e-9 else \
            np.cov(sur, post, ddof=0)[0, 1] / var_s
        slope *= n / (n + 8.0)                         # shrink small samples
        # tilt from the latest report's surprise, only while drift is pending
        latest = e["Surprise(%)"].dropna()
        tilt = 0.0
        if len(latest):
            d, s = latest.index[-1], float(np.clip(latest.iloc[-1], -50, 50))
            days_since = (idx[-1] - d).days
            remaining = max(0.0, 1 - days_since / 30)
            tilt = slope * s * remaining
        # earnings-day jump risk if a report falls inside the horizon
        move = float(np.mean(react))
        out, ev = {}, {}
        for h in horizons:
            out[h] = tilt
            if b.next_earnings is not None:
                days_ahead = (b.next_earnings - idx[-1]).days
                if 0 <= days_ahead <= h * 365 / TD:
                    # roughly one report per quarter inside the horizon
                    ev[h] = (h / 63.0) * move ** 2
            else:
                ev[h] = (h / 63.0) * move ** 2
        return Signal(out, f"{n} past reports; slope {slope:+.4f} per 1% "
                           f"surprise; avg 1-day move {move:.1%}", ev)


# --------------------------------------------------------------------------
# Ensemble
# --------------------------------------------------------------------------
STRATEGIES = [HistoricalDrift(), Fundamentals(), TechnicalML(),
              NewsSentiment(), Earnings()]


def run_strategies(bundle, strategies=None, horizons=None):
    horizons = horizons or list(HORIZONS.values())
    results = {}
    for s in strategies or STRATEGIES:
        try:
            sig = s.predict(bundle, horizons)
        except Exception as e:                       # one bad strategy != crash
            _warn(f"{s.name} failed: {e}")
            sig = None
        results[s.name] = (s, sig)
    return results


def ensemble(bundle, results, horizons=None):
    """Combine signals -> {h: dict(median, lo, hi, mu)} price forecast."""
    horizons = horizons or list(HORIZONS.values())
    last = float(bundle.prices.iloc[-1])
    sigma_d = _vol(bundle.prices)
    n = len(bundle.prices) - 1
    out = {}
    for h in horizons:
        base = [sig.value[h] for s, sig in results.values()
                if sig and s.kind == "base" and h in sig.value]
        mu = float(np.mean(base)) if base else MARKET_DRIFT * h / TD
        for s, sig in results.values():
            if sig and s.kind == "overlay" and h in sig.value:
                mu += OVERLAY_WEIGHT.get(s.name, 1.0) * sig.value[h]
        var = sigma_d ** 2 * h                           # ordinary volatility
        var += (h * sigma_d / np.sqrt(n)) ** 2           # drift uncertainty
        var += sum(sig.extra_var.get(h, 0.0) for _, sig in results.values()
                   if sig)                               # earnings jumps
        sd = np.sqrt(var)
        out[h] = {"mu": mu, "sd": sd, "median": last * np.exp(mu),
                  "lo": last * np.exp(mu - Z90 * sd),
                  "hi": last * np.exp(mu + Z90 * sd)}
    return out


# --------------------------------------------------------------------------
# Walk-forward backtest (history-only strategies; no look-ahead)
# --------------------------------------------------------------------------
def backtest(prices, window=504, step=21):
    hist_only = [HistoricalDrift(), TechnicalML()]
    horizons = list(HORIZONS.values())
    names = {v: k for k, v in HORIZONS.items()}
    print("\nBACKTEST: HistoricalDrift + TechnicalML only (news, earnings and "
          "fundamentals have no\npoint-in-time data, so testing them on the "
          "past would leak the future).")
    print(f"{'Horizon':<11}{'Tests':>6}{'In 90%':>8}{'Dir.hit':>9}"
          f"{'MAE model':>11}{'MAE flat':>10}{'Within10%':>11}")
    for h in horizons:
        hits = dirs = within = tests = 0
        e_model = e_flat = 0.0
        for t in range(window, len(prices) - h, step):
            b = Bundle(prices=prices.iloc[: t + 1])
            res = run_strategies(b, hist_only, [h])
            fc = ensemble(b, res, [h])[h]
            actual = float(prices.iloc[t + h])
            now = float(prices.iloc[t])
            hits += fc["lo"] <= actual <= fc["hi"]
            dirs += np.sign(fc["median"] - now) == np.sign(actual - now)
            e_model += abs(fc["median"] / actual - 1)
            e_flat += abs(now / actual - 1)
            within += abs(fc["median"] / actual - 1) <= 0.10
            tests += 1
        if not tests:
            print(f"{names[h]:<11}  not enough history (use more --years)")
            continue
        print(f"{names[h]:<11}{tests:>6}{hits / tests:>8.0%}{dirs / tests:>9.0%}"
              f"{e_model / tests:>11.0%}{e_flat / tests:>10.0%}"
              f"{within / tests:>11.0%}")
    print("In 90% = actual inside the 5-95% range (target ~90%). Dir.hit = "
          "predicted direction right.\nMAE flat = error of 'price stays the "
          "same'; the model is only useful if MAE model < MAE flat.\n"
          "Windows overlap, so treat results as a rough guide.")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ticker", default="TSLA")
    p.add_argument("--years", type=int, default=5)
    p.add_argument("--csv", help="prices only: CSV with Date,Close")
    p.add_argument("--demo", action="store_true", help="synthetic offline data")
    p.add_argument("--backtest", action="store_true")
    a = p.parse_args()

    b = synthetic(a.years) if a.demo else from_csv(a.csv) if a.csv \
        else fetch_live(a.ticker, a.years)
    last = float(b.prices.iloc[-1])
    print(f"Source: {b.source}\nHistory: {b.prices.index[0].date()} -> "
          f"{b.prices.index[-1].date()}   Last close: ${last:,.2f}\n")

    results = run_strategies(b)
    horizons = list(HORIZONS.values())
    print("STRATEGY SIGNALS (expected log return; overlays are tilts)")
    print(f"{'Strategy':<16}{'Type':<9}" +
          "".join(f"{k:>11}" for k in HORIZONS) + "  Notes")
    for name, (s, sig) in results.items():
        if sig is None:
            print(f"{name:<16}{s.kind:<9}{'-- skipped: not enough data --'}")
            continue
        vals = "".join(f"{sig.value[h]:>+11.1%}" if h in sig.value
                       else f"{'n/a':>11}" for h in horizons)
        print(f"{name:<16}{s.kind:<9}{vals}  {sig.note}")

    fc = ensemble(b, results)
    print(f"\nENSEMBLE FORECAST (from ${last:,.2f})")
    print(f"{'Horizon':<11}{'5th pct':>11}{'Median':>11}{'95th pct':>11}"
          f"{'Median chg':>12}")
    for k, h in HORIZONS.items():
        f = fc[h]
        print(f"{k:<11}${f['lo']:>9,.2f}${f['median']:>9,.2f}${f['hi']:>9,.2f}"
              f"{f['median'] / last - 1:>11.1%}")
    print("\nRange = 90% interval under the model's assumptions. Not financial "
          "advice.")

    if a.backtest:
        backtest(b.prices)


if __name__ == "__main__":
    main()
