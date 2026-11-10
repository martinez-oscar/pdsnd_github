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


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", help="CSV file with Date and Close columns")
    p.add_argument("--years", type=int, default=5, help="years of history (default 5)")
    p.add_argument("--sims", type=int, default=10000, help="Monte Carlo paths")
    p.add_argument("--plot", action="store_true", help="save forecast.png")
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
