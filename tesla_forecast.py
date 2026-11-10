#!/usr/bin/env python3
"""Forecast Tesla (TSLA) stock price at 6 months, 12 months and 2 years.

Method: Monte Carlo simulation of geometric Brownian motion (GBM), with the
drift and volatility estimated from historical daily log returns. The output
is a probability distribution (5th / 50th / 95th percentile), not a single
"true" prediction -- stock prices are very noisy and this is NOT financial
advice.

Usage:
    pip install numpy pandas yfinance matplotlib   # matplotlib optional
    python tesla_forecast.py                       # download data via yfinance
    python tesla_forecast.py --csv tsla.csv        # use a CSV with Date,Close
    python tesla_forecast.py --years 5 --sims 20000 --plot
    python tesla_forecast.py --demo                # synthetic data (offline test)
    python tesla_forecast.py --backtest --years 10 # check how reliable it was
"""
import argparse

import numpy as np
import pandas as pd

TRADING_DAYS = 252
HORIZONS = {"6 months": 126, "12 months": 252, "2 years": 504}


def load_prices(args):
    """Return a Series of daily closing prices indexed by date."""
    if args.demo:
        rng = np.random.default_rng(0)
        rets = rng.normal(0.0008, 0.035, TRADING_DAYS * args.years)
        idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=len(rets))
        return pd.Series(200 * np.exp(np.cumsum(rets)), index=idx, name="Close")

    if args.csv:
        df = pd.read_csv(args.csv, parse_dates=["Date"], index_col="Date")
        return df["Close"].dropna().sort_index()

    try:
        import yfinance as yf
    except ImportError:
        raise SystemExit("yfinance not installed. Run: pip install yfinance "
                         "(or use --csv / --demo).")
    df = yf.download("TSLA", period=f"{args.years}y", auto_adjust=True,
                     progress=False)
    if df.empty:
        raise SystemExit("No data downloaded. Check your connection or use --csv.")
    close = df["Close"]
    if isinstance(close, pd.DataFrame):  # newer yfinance returns a 1-col frame
        close = close.iloc[:, 0]
    return close.dropna()


def simulate(prices, n_sims, max_days, seed=42):
    """Simulate GBM paths; returns array of shape (max_days, n_sims)."""
    log_ret = np.log(prices / prices.shift(1)).dropna().to_numpy()
    mu = log_ret.mean()          # daily mean log return
    sigma = log_ret.std(ddof=1)  # daily volatility
    rng = np.random.default_rng(seed)
    shocks = rng.normal(mu, sigma, size=(max_days, n_sims))
    paths = prices.iloc[-1] * np.exp(np.cumsum(shocks, axis=0))
    return paths, mu, sigma


Z90 = 1.6448536269514722  # z-score for the 5th / 95th percentile


def backtest(prices, window=504, step=21):
    """Walk-forward test of the forecast on past data.

    At each origin date, estimate drift/volatility from the previous `window`
    trading days only (no look-ahead), build the closed-form GBM 90% interval
    for each horizon, and compare it with the price that actually occurred.
    Origins are `step` days apart, so results overlap and are correlated:
    treat them as a rough guide, not a precise statistic.
    """
    log_p = np.log(prices.to_numpy())
    rets = np.diff(log_p)
    print(f"\nBacktest (estimation window {window} days, origin every {step} days)")
    print(f"{'Horizon':<11}{'Tests':>7}{'In 90% range':>14}{'Median |err|':>14}"
          f"{'Within 10%':>12}")
    for name, h in HORIZONS.items():
        hits, errs, tests = 0, [], 0
        for t in range(window, len(prices) - h, step):
            r = rets[t - window:t]
            mu, sigma = r.mean(), r.std(ddof=1)
            centre = h * mu
            spread = Z90 * sigma * np.sqrt(h)
            actual = log_p[t + h] - log_p[t]            # realised log return
            hits += centre - spread <= actual <= centre + spread
            errs.append(abs(np.exp(actual - centre) - 1))  # |actual/median - 1|
            tests += 1
        if tests == 0:
            print(f"{name:<11}{'not enough history (use more --years)':>46}")
            continue
        errs = np.array(errs)
        print(f"{name:<11}{tests:>7}{hits / tests:>14.0%}{np.median(errs):>14.0%}"
              f"{(errs <= 0.10).mean():>12.0%}")
    print("'In 90% range' should be near 90% if the interval is well calibrated.\n"
          "'Within 10%' = how often the median forecast was within 10% of the "
          "actual price.")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", help="CSV file with Date and Close columns")
    p.add_argument("--years", type=int, default=5, help="years of history (default 5)")
    p.add_argument("--sims", type=int, default=10000, help="Monte Carlo paths")
    p.add_argument("--plot", action="store_true", help="save forecast.png")
    p.add_argument("--backtest", action="store_true",
                   help="walk-forward test of past forecast accuracy")
    p.add_argument("--demo", action="store_true", help="use synthetic data")
    args = p.parse_args()

    prices = load_prices(args)
    last = float(prices.iloc[-1])
    max_days = max(HORIZONS.values())
    paths, mu, sigma = simulate(prices, args.sims, max_days)

    print(f"History: {prices.index[0].date()} -> {prices.index[-1].date()} "
          f"({len(prices)} days)")
    print(f"Last close: ${last:,.2f}")
    print(f"Annualised drift: {mu * TRADING_DAYS:.1%}   "
          f"Annualised volatility: {sigma * np.sqrt(TRADING_DAYS):.1%}\n")

    print(f"{'Horizon':<11}{'Median':>11}{'5th pct':>11}{'95th pct':>11}"
          f"{'Median chg':>12}")
    for name, days in HORIZONS.items():
        end = paths[days - 1]
        lo, med, hi = np.percentile(end, [5, 50, 95])
        print(f"{name:<11}${med:>9,.2f}${lo:>9,.2f}${hi:>9,.2f}"
              f"{med / last - 1:>11.1%}")
    print("\nThe 5th-95th range covers 90% of simulated outcomes. "
          "Not financial advice.")

    if args.backtest:
        backtest(prices)

    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            raise SystemExit("matplotlib not installed. Run: pip install matplotlib")
        days = np.arange(1, max_days + 1)
        q5, q50, q95 = np.percentile(paths, [5, 50, 95], axis=1)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(range(-len(prices) + 1, 1), prices.to_numpy(), label="History")
        ax.plot(days, q50, label="Median forecast")
        ax.fill_between(days, q5, q95, alpha=0.25, label="90% range")
        for name, d in HORIZONS.items():
            ax.axvline(d, ls=":", color="gray")
            ax.text(d, ax.get_ylim()[1], name, rotation=90, va="top", ha="right")
        ax.set_xlabel("Trading days from today")
        ax.set_ylabel("Price (USD)")
        ax.set_title("TSLA Monte Carlo forecast")
        ax.legend()
        fig.tight_layout()
        fig.savefig("forecast.png", dpi=150)
        print("Saved forecast.png")


if __name__ == "__main__":
    main()
