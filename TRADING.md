# Paper trading system

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
- `scan.py` – multi-stock research scan
- `system.py` – portfolio engine, risk rules, backtest, daily paper cycle
- `dashboard_template.html` – dashboard page; `system.py` fills in the data
- `state/` – real paper account, backtest results, `dashboard.html`
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
