#!/usr/bin/env python3
"""Thesis ledger and kill-switch monitor.

For every stock in the universe this keeps three written, falsifiable
assumptions ("revenue keeps growing at least 8% a year", "operating margin
stays above 40%") in state/theses.json, then checks each one against the
company's latest reported quarter. An assumption is

    INTACT   the number is comfortably on the right side of the line
    WATCH    it has crossed the early-warning line
    BROKEN   it has crossed the kill line  ->  "SELL THESIS BROKEN" alert

It also runs a short accounting red-flag screen (receivables outrunning sales,
profit not turning into cash, capex eating the cash flow, dilution, leverage)
and lists recent negative headlines from the news archive next to each stock.

Honest limits
  * `init` DRAFTS the assumptions: the metric per stock was picked by hand,
    the lines are set mechanically from the last year of reported numbers.
    They are a starting point. Edit state/theses.json so they say what YOU
    believe; the monitor never rewrites an existing thesis.
  * Data is Yahoo Finance quarterly statements (about five quarters). It has
    no segment detail (cloud revenue, data-center revenue), no footnotes and
    no earnings-call text. Those need SEC filings.
  * This is an alert, not an order. system.py does not sell on it.
  * Not financial advice.

Usage
    python thesis.py init            # draft theses for stocks that have none
    python thesis.py check           # refresh data, test every assumption
    python thesis.py check --offline # use cached fundamentals only
"""
import argparse
import datetime as dt
import json
import os
import sys

from scan import DEFAULT_TICKERS

FINANCIALS = {"JPM", "BRK-B"}     # cash-flow and margin screens don't apply
CACHE_DAYS = 3
ORDER = {"BROKEN": 0, "WATCH": 1, "NO DATA": 2, "INTACT": 3}

# metric -> (label, format, better direction)
METRICS = {
    "rev_yoy":         ("Revenue growth, year over year", "pct", ">="),
    "gross_margin":    ("Gross margin, last 4 quarters", "pct", ">="),
    "op_margin":       ("Operating margin, last 4 quarters", "pct", ">="),
    "net_margin":      ("Net margin, last 4 quarters", "pct", ">="),
    "fcf_margin":      ("Free-cash-flow margin, last 4 quarters", "pct", ">="),
    "cfo_ni":          ("Operating cash flow / net income, 4 quarters", "x", ">="),
    "equity_yoy":      ("Book value growth, year over year", "pct", ">="),
    "shares_yoy":      ("Share count change, year over year", "pct", "<="),
    "net_debt_ebitda": ("Net debt / EBITDA", "x", "<="),
}

# The three assumptions per stock: (metric, the belief in plain words).
TEMPLATES = {
    "NVDA": [("rev_yoy", "AI chip demand keeps revenue growing fast"),
             ("gross_margin", "Pricing power holds: no margin squeeze from competition or custom chips"),
             ("fcf_margin", "Growth keeps converting into free cash flow")],
    "MSFT": [("rev_yoy", "Cloud and AI keep company revenue growing at a double-digit pace"),
             ("op_margin", "AI spending does not erode operating margin"),
             ("fcf_margin", "Data-center capex does not swallow free cash flow")],
    "AAPL": [("rev_yoy", "Revenue keeps growing (iPhone base plus services)"),
             ("gross_margin", "Services mix keeps gross margin high"),
             ("shares_yoy", "Buybacks keep shrinking the share count")],
    "GOOGL": [("rev_yoy", "Search and cloud keep growing despite AI competitors"),
              ("op_margin", "AI costs do not erode operating margin"),
              ("fcf_margin", "Capex does not swallow free cash flow")],
    "AMZN": [("rev_yoy", "AWS and retail keep revenue growing"),
             ("op_margin", "Operating margin gains are kept"),
             ("fcf_margin", "Free cash flow stays positive through the capex build")],
    "META": [("rev_yoy", "Ad revenue keeps growing at a double-digit pace"),
             ("op_margin", "AI and Reality Labs spending does not erode margin"),
             ("fcf_margin", "Capex does not swallow free cash flow")],
    "AVGO": [("rev_yoy", "AI networking and custom chips keep revenue growing"),
             ("gross_margin", "Mix keeps gross margin high"),
             ("net_debt_ebitda", "Acquisition debt keeps getting paid down")],
    "TSLA": [("rev_yoy", "Revenue returns to and stays in growth"),
             ("gross_margin", "Price cuts stop eroding gross margin"),
             ("fcf_margin", "The business funds itself: free cash flow stays positive")],
    "BRK-B": [("equity_yoy", "Book value keeps compounding"),
              ("rev_yoy", "Operating businesses keep growing"),
              ("net_margin", "Underwriting and operating earnings stay healthy")],
    "LLY": [("rev_yoy", "GLP-1 drugs keep revenue growing fast"),
            ("gross_margin", "Pricing pressure does not erode gross margin"),
            ("op_margin", "Growth keeps dropping through to operating profit")],
    "JPM": [("rev_yoy", "Revenue holds up as interest rates move"),
            ("net_margin", "Credit costs stay contained: net margin holds"),
            ("equity_yoy", "Book value keeps growing")],
    "WMT": [("rev_yoy", "Sales keep growing steadily"),
            ("op_margin", "Ads and e-commerce keep operating margin from slipping"),
            ("cfo_ni", "Profit keeps turning into cash")],
    "V": [("rev_yoy", "Payment volume keeps revenue growing"),
          ("op_margin", "The network keeps its very high operating margin"),
          ("fcf_margin", "Nearly all profit stays free cash flow")],
    "ORCL": [("rev_yoy", "Cloud contracts turn into reported revenue growth"),
             ("op_margin", "The cloud build does not erode operating margin"),
             ("net_debt_ebitda", "Debt taken on for data centers stays manageable")],
    "NFLX": [("rev_yoy", "Subscribers, pricing and ads keep revenue growing"),
             ("op_margin", "Operating margin keeps its gains"),
             ("fcf_margin", "Content spending leaves solid free cash flow")],
}
GENERIC = [("rev_yoy", "Revenue keeps growing"),
           ("op_margin", "Operating margin holds"),
           ("cfo_ni", "Profit keeps turning into cash")]


# --------------------------------------------------------------------------
# Fundamentals: fetch, cache, merge (the cache grows past Yahoo's 5 quarters)
# --------------------------------------------------------------------------
ROWS = {
    "revenue": ("inc", ["Total Revenue", "Operating Revenue"]),
    "gross_profit": ("inc", ["Gross Profit"]),
    "op_income": ("inc", ["Operating Income", "Pretax Income"]),
    "net_income": ("inc", ["Net Income"]),
    "ebitda": ("inc", ["EBITDA", "Normalized EBITDA"]),
    "cfo": ("cf", ["Operating Cash Flow"]),
    "capex": ("cf", ["Capital Expenditure"]),
    "receivables": ("bs", ["Accounts Receivable", "Receivables"]),
    "inventory": ("bs", ["Inventory"]),
    "equity": ("bs", ["Stockholders Equity"]),
    "debt": ("bs", ["Total Debt"]),
    "cash": ("bs", ["Cash Cash Equivalents And Short Term Investments",
                    "Cash And Cash Equivalents"]),
    "shares": ("bs", ["Ordinary Shares Number", "Share Issued"]),
}


def _cache_path(sdir, tk):
    return os.path.join(sdir, "fundamentals", tk + ".json")


def _download(tk):
    import yfinance as yf
    t = yf.Ticker(tk)
    frames = {"inc": t.quarterly_income_stmt, "bs": t.quarterly_balance_sheet,
              "cf": t.quarterly_cashflow}
    out = {}
    for field, (src, names) in ROWS.items():
        df = frames[src]
        if df is None or df.empty:
            continue
        for name in names:
            if name not in df.index:
                continue
            for col, v in df.loc[name].items():
                if v == v and v is not None:
                    out.setdefault(str(col.date()), {})[field] = float(v)
            break
    # Yahoo's own trailing-12-month totals: they survive missing quarters
    for src, df in (("inc", t.ttm_income_stmt), ("cf", t.ttm_cashflow)):
        if df is None or df.empty:
            continue
        for field, (fsrc, names) in ROWS.items():
            name = next((n for n in names if n in df.index), None)
            if fsrc != src or name is None:
                continue
            for col, v in df.loc[name].items():
                if v == v and v is not None:
                    out.setdefault(str(col.date()), {})["ttm_" + field] = float(v)
    return out


def load_fundamentals(tk, sdir, offline=False):
    """-> list of quarters, oldest first: [{date, revenue, ...}]"""
    path = _cache_path(sdir, tk)
    cache = json.load(open(path)) if os.path.exists(path) else {}
    q = cache.get("quarters", {})
    age = (dt.date.today() - dt.date.fromisoformat(cache["fetched"])).days \
        if cache.get("fetched") else 999
    if not offline and age >= CACHE_DAYS:
        try:
            for d, row in _download(tk).items():
                q.setdefault(d, {}).update(row)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                json.dump({"fetched": str(dt.date.today()), "quarters": q}, f,
                          indent=1, sort_keys=True)
        except Exception as e:
            print(f"  ! {tk}: fundamentals fetch failed ({type(e).__name__}); "
                  f"using cache", file=sys.stderr)
    return [dict(date=d, **q[d]) for d in sorted(q)]


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def _back(qs, i, quarters):
    """Index of the row `quarters` quarters before row i, or None if Yahoo
    has a hole there. Matching is by date, so a missing quarter can never
    make a comparison silently span the wrong period."""
    want = dt.date.fromisoformat(qs[i]["date"]) - dt.timedelta(days=91.3 * quarters)
    for j in range(i - 1, -1, -1):
        gap = (dt.date.fromisoformat(qs[j]["date"]) - want).days
        if abs(gap) <= 20:
            return j
        if gap < -20:
            break
    return None


def _ttm(qs, i, field):
    """Last-4-quarter total at row i: Yahoo's own trailing figure when it
    has one, else the sum of four consecutive quarters, else None."""
    if qs[i].get("ttm_" + field) is not None:
        return qs[i]["ttm_" + field]
    idx = [i] + [_back(qs, i, k) for k in (1, 2, 3)]
    if any(j is None for j in idx):
        return None
    vals = [qs[j].get(field) for j in idx]
    return None if any(v is None for v in vals) else sum(vals)


def _ratio(a, b):
    return None if a is None or b is None or b <= 0 else a / b


def _yoy(qs, i, field):
    j = _back(qs, i, 4)
    if j is None:
        return None
    a, b = qs[i].get(field), qs[j].get(field)
    return None if a is None or b is None or b <= 0 else a / b - 1


def metric(qs, i, name):
    """Value of a metric at quarter index i (None when it can't be computed)."""
    if name in ("rev_yoy", "equity_yoy", "shares_yoy"):
        return _yoy(qs, i, {"rev_yoy": "revenue", "equity_yoy": "equity",
                            "shares_yoy": "shares"}[name])
    rev = _ttm(qs, i, "revenue")
    if name == "gross_margin":
        return _ratio(_ttm(qs, i, "gross_profit"), rev)
    if name == "op_margin":
        return _ratio(_ttm(qs, i, "op_income"), rev)
    if name == "net_margin":
        return _ratio(_ttm(qs, i, "net_income"), rev)
    if name == "fcf_margin":
        cfo, capex = _ttm(qs, i, "cfo"), _ttm(qs, i, "capex")
        return None if cfo is None or capex is None else _ratio(cfo + capex, rev)
    if name == "cfo_ni":
        return _ratio(_ttm(qs, i, "cfo"), _ttm(qs, i, "net_income"))
    if name == "net_debt_ebitda":
        d, c = qs[i].get("debt"), qs[i].get("cash")
        e = _ttm(qs, i, "ebitda")
        return None if d is None or c is None else _ratio(d - c, e)
    raise KeyError(name)


def latest(qs, name):
    """-> (value, date, previous value) using the newest quarter that has it."""
    for i in range(len(qs) - 1, -1, -1):
        v = metric(qs, i, name)
        if v is not None:
            j = _back(qs, i, 1)
            prev = metric(qs, j, name) if j is not None else None
            return v, qs[i]["date"], prev
    return None, None, None


def fmt(v, kind):
    if v is None:
        return "n/a"
    return f"{v:+.1%}" if kind == "pct" else f"{v:.2f}x"


# --------------------------------------------------------------------------
# Drafting the lines
# --------------------------------------------------------------------------
def draft_lines(name, base):
    """Kill line and early-warning line from today's value. Rules, not fits:
    growth may halve (or lose 5 points); margins may lose 4-5 points (less
    for thin-margin businesses: 15-20% of the margin); free
    cash flow may fall by half; buybacks may stall; leverage may rise 1 turn.
    """
    if name in ("rev_yoy", "equity_yoy"):
        kill = min(base * 0.5, base - 0.05)
    elif name == "gross_margin":
        kill = base - min(0.04, 0.15 * abs(base))
    elif name in ("op_margin", "net_margin"):
        kill = base - min(0.05, 0.20 * abs(base))
    elif name == "fcf_margin":
        kill = base * 0.5 if base > 0.04 else 0.0
    elif name == "cfo_ni":
        kill = min(0.8, base * 0.7)
    elif name == "shares_yoy":
        kill = max(base + 0.02, 0.005)
    elif name == "net_debt_ebitda":
        kill = max(base + 1.0, 1.0)
    else:
        raise KeyError(name)
    return round(kill, 4), round((base + kill) / 2, 4)


def cmd_init(a):
    path = os.path.join(a.state, "theses.json")
    book = json.load(open(path)) if os.path.exists(path) else {}
    made = 0
    for tk in a.tickers:
        if tk in book and not a.force:
            continue
        qs = load_fundamentals(tk, a.state, a.offline)
        rows = []
        for name, belief in TEMPLATES.get(tk, GENERIC):
            base, date, _ = latest(qs, name)
            if base is None:
                print(f"  ! {tk}: no data for {name}; assumption left out",
                      file=sys.stderr)
                continue
            kill, warn = draft_lines(name, base)
            rows.append(dict(belief=belief, metric=name,
                             op=METRICS[name][2], kill=kill, warn=warn,
                             baseline=round(base, 4), baseline_date=date))
        book[tk] = dict(written=str(dt.date.today()), author="draft",
                        assumptions=rows)
        made += 1
    os.makedirs(a.state, exist_ok=True)
    with open(path, "w") as f:
        json.dump(book, f, indent=1)
    print(f"{made} thesis draft(s) written to {path} "
          f"({len(book)} stocks in the ledger). Edit the file to make them "
          f"yours, then set \"author\" to your name.")


# --------------------------------------------------------------------------
# Checking
# --------------------------------------------------------------------------
def test(asm, qs):
    v, date, prev = latest(qs, asm["metric"])
    label, kind, _ = METRICS[asm["metric"]]
    bad = (lambda x, line: x < line) if asm["op"] == ">=" else \
          (lambda x, line: x > line)
    if v is None:
        status = "NO DATA"
    elif bad(v, asm["kill"]):
        status = "BROKEN"
    elif bad(v, asm["warn"]):
        status = "WATCH"
    else:
        status = "INTACT"
    worse = None if v is None or prev is None else bad(v, prev)
    return dict(belief=asm["belief"], metric=asm["metric"], label=label,
                kind=kind, op=asm["op"], value=v, prev=prev, as_of=date,
                kill=asm["kill"], warn=asm["warn"], baseline=asm["baseline"],
                status=status, worsening=worse)


def red_flags(tk, qs):
    """Accounting screen on the newest quarter. -> [text]"""
    if tk in FINANCIALS or not qs:
        return []
    i, out = len(qs) - 1, []
    rev = _yoy(qs, i, "revenue")
    ar = _yoy(qs, i, "receivables")
    if rev is not None and ar is not None and ar - rev > 0.10:
        out.append(f"Receivables up {ar:+.0%} vs revenue {rev:+.0%}: sales "
                   f"are being booked faster than cash is collected")
    inv = _yoy(qs, i, "inventory")
    r4, inv_now = _ttm(qs, i, "revenue"), qs[i].get("inventory")
    if (rev is not None and inv is not None and r4 and inv_now
            and inv_now / r4 > 0.02 and inv - rev > 0.15):
        out.append(f"Inventory up {inv:+.0%} vs revenue {rev:+.0%}: stock is "
                   f"building faster than it sells")
    c = metric(qs, i, "cfo_ni")
    if c is not None and c < 0.8:
        out.append(f"Operating cash flow is only {c:.2f}x net income over 4 "
                   f"quarters: reported profit is not turning into cash")
    cfo, capex = _ttm(qs, i, "cfo"), _ttm(qs, i, "capex")
    if cfo and capex is not None and cfo > 0 and -capex / cfo > 0.7:
        out.append(f"Capex takes {-capex / cfo:.0%} of operating cash flow: "
                   f"little is left for owners if returns on it disappoint")
    sh = _yoy(qs, i, "shares")
    if sh is not None and sh > 0.02:
        out.append(f"Share count up {sh:+.1%} in a year: owners are being "
                   f"diluted")
    lev = metric(qs, i, "net_debt_ebitda")
    if lev is not None and lev > 3:
        out.append(f"Net debt is {lev:.1f}x EBITDA: refinancing at higher "
                   f"rates would cut into earnings")
    return out


def bad_news(tk, sdir, days=14, top=3):
    """Recent negative, company-specific headlines from the news archive."""
    path = os.path.join(sdir, "news", tk + ".jsonl")
    if not os.path.exists(path):
        return []
    import news
    cut = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=days)).isoformat()
    hits = []
    for line in open(path):
        s = json.loads(line)
        if s.get("published", "") < cut:
            continue
        sc = news.score_headline(s["title"], tk, s.get("source"))
        if sc["s"] <= -0.25 and sc["rel"] >= 0.7 and sc["etype"] not in ("recap", "opinion",
                                                    "lawfirm", "other"):
            hits.append(dict(title=s["title"], source=s.get("source"),
                             date=s["published"][:10],
                             etype=news.EVENT_LABEL[sc["etype"]],
                             score=sc["s"] * sc["w"]))
    hits.sort(key=lambda h: h["score"])
    return hits[:top]


def check(tickers, sdir, offline=False):
    path = os.path.join(sdir, "theses.json")
    if not os.path.exists(path):
        raise SystemExit(f"No {path}. Run: python thesis.py init")
    book = json.load(open(path))
    rpath = os.path.join(sdir, "thesis_report.json")
    old = {}
    if os.path.exists(rpath):
        old = {s["ticker"]: s["status"]
               for s in json.load(open(rpath))["stocks"]}
    held = set()
    ppath = os.path.join(sdir, "portfolio.json")
    if os.path.exists(ppath):
        held = set(json.load(open(ppath))["engine"]["positions"])
    stocks = []
    for tk in tickers:
        if tk not in book:
            continue
        qs = load_fundamentals(tk, sdir, offline)
        rows = [test(asm, qs) for asm in book[tk]["assumptions"]]
        status = min((r["status"] for r in rows), key=ORDER.get,
                     default="NO DATA")
        stocks.append(dict(
            ticker=tk, status=status, was=old.get(tk), held=tk in held,
            written=book[tk]["written"], author=book[tk].get("author"),
            quarter=max((q["date"] for q in qs if "revenue" in q),
                        default=None), assumptions=rows,
            flags=red_flags(tk, qs), news=bad_news(tk, sdir)))
    stocks.sort(key=lambda s: (ORDER[s["status"]], -len(s["flags"]),
                               s["ticker"]))
    rep = dict(as_of=str(dt.date.today()), stocks=stocks)
    with open(rpath, "w") as f:
        json.dump(rep, f, indent=1)
    return rep


def print_report(rep):
    n = {k: sum(s["status"] == k for s in rep["stocks"]) for k in ORDER}
    print(f"\nTHESIS MONITOR {rep['as_of']}: {n['BROKEN']} broken, "
          f"{n['WATCH']} on watch, {n['INTACT']} intact"
          + (f", {n['NO DATA']} without data" if n["NO DATA"] else ""))
    for s in rep["stocks"]:
        head = f"\n{s['ticker']:<6} {s['status']}"
        if s["status"] == "BROKEN":
            head += "  ** SELL THESIS BROKEN **" if s["held"] else \
                    "  (not held: do not buy on this thesis)"
        elif s["held"]:
            head += "  (held)"
        if s["was"] and s["was"] != s["status"]:
            head += f"  [was {s['was']}]"
        print(head + f"   latest quarter {s['quarter']}")
        for r in s["assumptions"]:
            k = r["kind"]
            arrow = "" if r["worsening"] is None else \
                (" worsening" if r["worsening"] else " improving")
            stale = f" [as of {r['as_of']}]" if r["as_of"] and \
                r["as_of"] != s["quarter"] else ""
            print(f"  {r['status']:<7} {r['belief']}\n"
                  f"          {r['label']}: {fmt(r['value'], k)}{stale}"
                  + (f" (was {fmt(r['prev'], k)},{arrow})"
                     if r["prev"] is not None else "")
                  + f"; kill line {r['op']} {fmt(r['kill'], k)}")
        for f_ in s["flags"]:
            print(f"  FLAG    {f_}")
        for h in s["news"]:
            print(f"  NEWS    {h['date']} [{h['etype']}] {h['title'][:90]}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["init", "check"])
    p.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    p.add_argument("--state", default="state")
    p.add_argument("--offline", action="store_true")
    p.add_argument("--force", action="store_true",
                   help="init: overwrite existing theses")
    a = p.parse_args()
    if a.command == "init":
        cmd_init(a)
    else:
        print_report(check(a.tickers, a.state, a.offline))


if __name__ == "__main__":
    main()
