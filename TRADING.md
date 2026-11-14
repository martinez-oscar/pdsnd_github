# Paper trading system (retired 2026-11-14)

**Retired.** This system is no longer developed or run. The thesis monitor and the hidden-bet analysis
moved to the private trading-bot project; nothing else was carried over. The code and the paper
account below are kept as a record only.

Research code. **No real money, no broker connection.** Not financial advice.

## Setup
```
python3 -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install numpy pandas yfinance
```

## Commands
| Command | What it does |
|---|---|
| `python system.py all --demo --open` | Everything on synthetic data, then opens the dashboard |
| `python system.py all --open` | Backtest + today's paper cycle + dashboard on real data |
| `python system.py run` | Daily paper cycle (run once per trading day, after the close) |
| `python system.py dashboard --open` | Rebuild and open the dashboard |
| `python thesis.py init` | Draft three written assumptions per stock (only for stocks without one) |
| `python thesis.py check` | Test every assumption against the latest reported quarter |
| `python exposure.py` | Hidden bets: what really drives the portfolio |
| `python news.py scan` | Fetch, archive and score news on its own |
| `python news.py selftest` | Check the news scorer and its evaluation (offline) |
| `python scan.py --save results/scan.txt` | Per-stock research report |
| `python trading.py --ticker TSLA` | Single-stock conviction study |

Options: `--tickers A B C`, `--rule sure|trend|sure+trend`, `--cash 50000`,
`--replay 120` (give a new paper account a simulated recent history).

## Files
- `stock_strategies.py` – five forecast layers and the ensemble
- `trading.py` – four more price layers, the "sure" rule, single-stock simulator
- `news.py` – news scanner, headline scoring, story archive, evidence test
- `thesis.py` – thesis ledger, kill-switch monitor, accounting red flags
- `exposure.py` – hidden-bet analysis (drivers, effective number of bets)
- `scan.py` – multi-stock research scan
- `system.py` – portfolio engine, risk rules, backtest, daily paper cycle
- `dashboard_template.html` – dashboard page; `system.py` fills in the data
- `state/` – real paper account, backtest results, `dashboard.html`
- `state/theses.json` – the written assumptions and their kill lines; **edit this**
- `state/fundamentals/` – quarterly numbers per stock; grows with every check
- `state/news/` – the story archive, one file per stock; it grows with every run
- `state_demo/` – the same for `--demo` (git-ignored)

To start the paper account over, delete `state/portfolio.json`.

## Sharing results
`git add state && git commit -m "Update paper account" && git push`

## News scanner
Every `system.py run` (or `news.py scan`) pulls headlines from Yahoo Finance
and Google News, merges duplicates, and adds them to `state/news/`. Each
headline is classified by event type and scored; opinion pieces, price recaps
and law-firm notices are discounted. A stock's news score is its tone relative
to the other stocks that day.

The archive is the point: after about 20 trading days of daily runs the system
starts measuring whether the score was followed by returns, and sizes the news
layer from that measurement. Before then it uses a small assumed skill (0.02).
Missing a day loses that day's headlines for good, so run it daily.

## Thesis monitor
`state/theses.json` holds three falsifiable assumptions per stock, each tied to
one reported number with a kill line and an early-warning line. `thesis.py
check` (also run by `system.py run`) marks each INTACT, WATCH or BROKEN, adds
an accounting red-flag screen and lists recent negative headlines. A broken
assumption on a held stock prints `SELL THESIS BROKEN`. It is an alert only;
the system never sells on it.

`init` writes machine drafts: the metric per stock was chosen by hand, the
lines come from the last year of numbers. Rewrite them so they say what you
believe, and set `"author"` to your name. Existing theses are never
overwritten unless you pass `--force`.

Data is Yahoo Finance quarterly statements: company totals only. Segment
numbers, footnotes, debt maturities and earnings-call text need SEC filings.

## Hidden bets
`exposure.py` (also run by `system.py run`) regresses a year of daily returns
on five drivers and reports the share of the portfolio's movement each one
explains, the effective number of independent bets, and which stock carries
how much of the risk.
