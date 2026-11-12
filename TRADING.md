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
| `python scan.py --save results/scan.txt` | Per-stock research report |
| `python trading.py --ticker TSLA` | Single-stock conviction study |

Options: `--tickers A B C`, `--rule sure|trend|sure+trend`, `--cash 50000`,
`--replay 120` (give a new paper account a simulated recent history).

## Files
- `stock_strategies.py` – five forecast layers and the ensemble
- `trading.py` – four more price layers, the "sure" rule, single-stock simulator
- `scan.py` – multi-stock research scan
- `system.py` – portfolio engine, risk rules, backtest, daily paper cycle
- `dashboard_template.html` – dashboard page; `system.py` fills in the data
- `state/` – real paper account, backtest results, `dashboard.html`
- `state_demo/` – the same for `--demo` (git-ignored)

To start the paper account over, delete `state/portfolio.json`.

## Sharing results
`git add state && git commit -m "Update paper account" && git push`
