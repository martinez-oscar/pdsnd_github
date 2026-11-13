#!/usr/bin/env python3
"""News scanner + sentiment layer with a point-in-time archive.

Why this is built the way it is
  Most hobby "news sentiment" signals average a generic positivity score over
  whatever headlines an API returns today. That has four problems, each of
  which this module addresses:

  1. It can never be tested, because nobody kept yesterday's headlines.
     -> ARCHIVE: every scan appends timestamped stories to state/news/*.jsonl.
        After a few weeks you own a point-in-time dataset, and `evaluate`
        measures whether the signal was followed by returns (rank IC).
  2. Generic sentiment misreads finance. "Shares plunge" is a price move that
     already happened, not news; "class action reminder" is law-firm spam;
     "Is X a buy?" is opinion.
     -> EVENT TYPING: headlines are classified (earnings, guidance, analyst,
        legal, regulatory, M&A, management, capital return, operations,
        product, price recap, opinion, law-firm notice) and weighted by type.
  3. One story syndicated 30 times counts 30 times; roundups mention ten
     companies; content mills outnumber wires.
     -> DEDUP into story clusters, RELEVANCE to the specific company, and
        SOURCE tiers.
  4. The score-to-return mapping is invented ("+3% if positive").
     -> SIZING by the standard rule  alpha = IC x volatility x z-score,
        where z is sentiment relative to the other stocks today and IC starts
        at a cautious prior (0.02) and is replaced by the MEASURED IC, shrunk
        toward the prior, as the archive grows. A signal that does not work
        measures ~0 and switches itself off.

Honest limits
  * Headlines only (no article bodies), English only, rule-based scoring.
  * Free feeds are incomplete and delayed; professionals trade faster.
  * Published academic evidence for news-sentiment drift in large caps is
    weak and short-lived. Expect a small effect or none.
  * If you edit the lexicon after looking at results, re-test on NEW data.

Usage
    python news.py scan                # fetch, archive, print today's read
    python news.py scan --demo         # synthetic headlines, offline
    python news.py evaluate            # has the signal worked so far?
    python news.py selftest            # scorer checks + planted-signal test
"""
import argparse
import datetime as dt
import email.utils
import hashlib
import html
import json
import math
import os
import re
import sys
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import stock_strategies as ss

# --------------------------------------------------------------------------
# CONFIG (assumptions; see module docstring)
# --------------------------------------------------------------------------
IC_PRIOR = 0.02            # assumed skill before any evidence
IC_PRIOR_N = 2000          # prior counts as this many stock-days of evidence
IC_CAP = 0.10
MIN_EVAL_DAYS = 20         # archive days needed before measured IC is used
HALF_LIFE_DAYS = 3.0       # story weight halves every 3 days
WINDOW_DAYS = 14
EFFECT_DAYS = 21           # news effect assumed to play out within ~1 month
SHRINK_K = 2.0             # few stories -> score pulled toward 0
UA = "Mozilla/5.0 (compatible; research-news-scanner)"
NY, UTC = ZoneInfo("America/New_York"), dt.timezone.utc

COMPANY = {
    "NVDA": ["Nvidia"], "MSFT": ["Microsoft"], "AAPL": ["Apple"],
    "GOOGL": ["Alphabet", "Google"], "AMZN": ["Amazon"],
    "META": ["Meta Platforms", "Meta", "Facebook"], "AVGO": ["Broadcom"],
    "TSLA": ["Tesla"], "BRK-B": ["Berkshire Hathaway", "Berkshire"],
    "LLY": ["Eli Lilly", "Lilly"], "JPM": ["JPMorgan", "JP Morgan"],
    "WMT": ["Walmart"], "V": ["Visa"], "ORCL": ["Oracle"], "NFLX": ["Netflix"],
}
# Same word, different meaning -> not about the company.
FALSE_FRIENDS = {
    "V": r"\b(h-?1b|travel|immigration|student|tourist|work|golden) visas?\b|"
         r"\bvisas? (rules?|ban|policy|waiver|applicants?|fees?|program)\b",
    "META": r"\bmeta[- ](analysis|data)\b",
}
SOURCE_TIER = [
    (1.0, ["reuters", "bloomberg", "wall street journal", "wsj", "associated"
           " press", "ap news", "financial times", "cnbc", "barron",
           "marketwatch", "nytimes", "new york times", "the information",
           "business wire", "pr newswire", "globenewswire", "sec.gov"]),
    (0.4, ["motley fool", "zacks", "investorplace", "24/7 wall", "insider "
           "monkey", "tipranks", "simply wall", "stocktwits", "benzinga",
           "seeking alpha", "gurufocus", "marketbeat"]),
]
DEFAULT_TIER = 0.7

# event type -> importance weight
IMPORTANCE = {"guidance": 1.6, "earnings": 1.5, "mna": 1.3, "legal": 1.2,
              "regulatory": 1.2, "operations": 1.1, "analyst": 1.0,
              "management": 1.0, "capital": 0.9, "product": 0.8,
              "other": 0.6, "recap": 0.3, "opinion": 0.25, "lawfirm": 0.0}
EVENT_LABEL = {"guidance": "Guidance", "earnings": "Earnings", "mna": "M&A",
               "legal": "Legal", "regulatory": "Regulatory",
               "operations": "Operations", "analyst": "Analyst",
               "management": "Management", "capital": "Capital return",
               "product": "Product", "other": "Other",
               "recap": "Price recap", "opinion": "Opinion",
               "lawfirm": "Law-firm notice"}

_UP = r"(?:raises?|raised|lifts?|lifted|boosts?|boosted|hikes?|hiked|increases?|increased|ups)"
_DN = r"(?:cuts?|lowers?|lowered|slash(?:es|ed)?|trims?|trimmed|reduces?|reduced)"
_EST = r"(?:estimates?|expectations?|forecasts?|views?|consensus|street|targets?)"
_FIN = r"(?:earnings|profits?|revenues?|sales|eps|margins?|deliveries|bookings)"
# (event type, polarity, regex). Order = priority for the event label.
PATTERNS = [(t, p, re.compile(rx, re.I)) for t, p, rx in [
    ("lawfirm", 0, r"(?:shareholder|investor|stockholder)s? (?:alert|notice|reminder|deadline)|"
                   r"class action (?:reminder|deadline|filed|notice)|lead plaintiff|"
                   r"law (?:firm|offices?)|on behalf of (?:investors|shareholders)|"
                   r"rosen law|pomerantz|levi & korsinsky|bragar eagel|faruqi|"
                   r"glancy prongay|robbins geller|bernstein liebhard|encourages .{0,40}investors"),
    ("opinion", 0, r"^\d+ (?:top |best |magnificent |unstoppable |no-brainer )?(?:\w+ )?(?:stocks?|reasons?|things)|"
                   r"stocks? to (?:buy|watch|sell|avoid|own)|should you (?:buy|sell|invest)|"
                   r"is it (?:too late|time)|better buy|\bvs\.? |\bprediction\b|"
                   r"could (?:make|be worth|soar|double)|millionaire|"
                   r"what to (?:know|watch|expect)|\bif you(?:'d| had)? invested\b|"
                   r"here'?s (?:what|how)|\?\s*$"),
    ("recap", 0, r"(?:stock|shares?) .{0,25}(?:jump|surg|soar|rise|rose|climb|rall|gain|"
                 r"pop|fall|fell|drop|plung|sink|sank|slid|slip|tumbl|slump|dip|crash)|"
                 r"why .{0,40}(?:stock|shares)|(?:stock|shares) (?:is|are) (?:up|down|moving)|"
                 r"hits? (?:a |an |new )?(?:record|all-time|52-week) (?:high|low)|"
                 r"biggest (?:gainers|losers)|stock market today|pre-?market|movers|"
                 r"dow jones|s&p 500|nasdaq (?:rises|falls|ends|closes)"),
    ("guidance", +1.5, _UP + r" (?:its |the |full[- ]year |annual |fy\d* |\w+ )?(?:guidance|outlook|forecasts?)|"
                       r"guides? (?:above|higher)|upbeat (?:guidance|outlook|forecast)"),
    ("guidance", -1.5, r"(?:" + _DN + r"|withdraws?|suspends?|pulls?) (?:its |the |full[- ]year |annual |fy\d* |\w+ )?"
                       r"(?:guidance|outlook|forecasts?)|guides? (?:below|lower)|profit warning|"
                       r"warns? (?:on|of|about) .{0,20}(?:profits?|revenues?|sales|earnings|demand|margins?)|"
                       r"(?:weak|soft|disappointing|downbeat) (?:guidance|outlook|forecast)"),
    ("earnings", +1.3, r"(?:beats?|tops?|topped|exceeds?|exceeded|crush(?:es|ed)?|smash(?:es|ed)?|surpass(?:es|ed)?) "
                       r".{0,40}" + _EST + r"|" + _FIN + r" (?:beats?|jumps?|surges?|soars?|rises?|climbs?|grows?|doubles?)|"
                       r"record (?:quarter|quarterly )?" + _FIN + r"|record quarter|"
                       r"(?:strong|blowout|stellar) (?:quarter|results|earnings)|swings? to (?:a )?profit"),
    ("earnings", -1.3, r"(?:miss(?:es|ed)?|falls? short of|fell short of|trails?|lags?) .{0,40}" + _EST + r"|"
                       + _FIN + r" (?:miss(?:es)?|falls?|drops?|declines?|plunges?|slumps?|shrinks?|slides?)|"
                       r"(?:posts?|reports?|swings? to) (?:a |an )?(?:\w+ )?loss|"
                       r"(?:weak|disappointing|dismal) (?:quarter|results|earnings)"),
    ("analyst", +1.0, r"upgrad(?:e|es|ed)\b|" + _UP + r" .{0,30}price target|price target .{0,20}(?:raised|lifted|hiked|boosted|increased)|"
                      r"initiat\w+ .{0,40}(?:buy|overweight|outperform)|"
                      r"(?:reiterat|maintain)\w+ .{0,25}(?:buy|overweight|outperform)|top pick"),
    ("analyst", -1.0, r"downgrad(?:e|es|ed)\b|" + _DN + r" .{0,30}price target|price target .{0,20}(?:cut|lowered|slashed|trimmed|reduced)|"
                      r"initiat\w+ .{0,40}(?:sell|underweight|underperform)"),
    ("regulatory", +1.2, r"(?:fda|regulators?|eu|ftc|doj|commission|court|judge) .{0,30}(?:approv|clear|greenlight|sides? with)|"
                         r"(?:wins?|gets?|receives?|secures?|granted) .{0,25}(?:approval|clearance)|"
                         r"tariff (?:exemption|relief|pause)"),
    ("regulatory", -1.2, r"(?:fda|regulators?|eu|ftc|doj|sec|commission) .{0,30}(?:reject|block|ban|sues?|suing|fines?|probe|investigat|charges?)|"
                         r"complete response letter|antitrust (?:suit|case|probe|charges?|fine)|"
                         r"export (?:ban|curbs?|restrictions?|controls?)|new tariffs?|hit (?:by|with) tariffs?"),
    ("legal", +0.8, r"(?:wins?|won) .{0,25}(?:lawsuit|case|appeal|trial|ruling)|lawsuit (?:dismissed|tossed)|"
                    r"(?:cleared|acquitted) (?:of|in)"),
    ("legal", -1.0, r"\b(?:sued|indict\w*|fraud|subpoena\w*|whistleblower)\b|class[- ]action|"
                    r"(?:faces?|hit with|slapped with) .{0,25}(?:lawsuit|suit|fine|penalty|probe|investigation)|"
                    r"(?:fined|to pay) .{0,20}(?:\$|€|£)|(?:loses?|lost) .{0,25}(?:lawsuit|case|appeal|trial)|"
                    r"under investigation|accounting (?:probe|irregularit)"),
    ("management", -0.9, r"(?:ceo|cfo|coo|cto|chief \w+ officer|chairman|founder|president) .{0,30}"
                         r"(?:resign|steps? down|stepping down|depart|ousted|fired|to leave|exits?|quits?)|"
                         r"(?:resign|steps? down|ousted|fired|departure of).{0,30}(?:ceo|cfo|chief|chairman)"),
    ("management", 0, r"(?:names?|appoints?|hires?|taps?) .{0,40}(?:ceo|cfo|chief|chairman)"),
    ("capital", +0.9, r"buyback|repurchase|" + _UP + r" .{0,15}dividend|dividend (?:hike|increase|raise)|"
                      r"special dividend|stock split"),
    ("capital", -0.9, r"(?:" + _DN + r"|suspends?|scraps?) .{0,15}dividend|(?:share|stock|equity|convertible) (?:offering|sale)|"
                      r"\bdilut"),
    ("operations", -1.0, r"\blayoffs?\b|job cuts|cuts? .{0,15}(?:jobs|workforce|staff)|data breach|\bhack(?:ed)?\b|"
                         r"\boutage\b|\bstrike\b|walkout|halts? (?:production|sales|deliveries)|"
                         r"\brecalls?\b|recalling|bankrupt|shortage|(?:delays?|delayed|postpones?) .{0,30}"
                         r"(?:launch|production|rollout|release|deliveries)|plant (?:closure|shutdown)"),
    ("operations", +1.0, r"(?:wins?|won|secures?|lands?|signs?|inks?|awarded) .{0,35}(?:contract|deal|order|partnership|agreement)|"
                         r"record (?:orders|backlog|demand)|(?:strong|surging|booming) demand|"
                         r"(?:expands?|ramps? up|boosts?) .{0,20}(?:production|capacity)"),
    ("mna", 0, r"\b(?:to acquire|acquires|acquisition|merger|takeover|buyout)\b|"
               r"(?:to buy|buys|takes?) .{0,25}(?:stake|for \$)|spin-?off"),
    ("product", +0.6, r"\b(?:launch(?:es|ed)?|unveils?|unveiled|introduces?|rolls? out|debuts?)\b"),
    ("earnings", 0, r"\b(?:q[1-4]|quarter(?:ly)?|earnings|results)\b"),
]]
HEDGE = re.compile(r"\b(?:may|might|could|reportedly|rumou?rs?|said to|considering|"
                   r"weighs?|in talks|expected to|likely|plans? to|mulls?)\b", re.I)
NEGATE = re.compile(r"(?:\bnot\b|\bno\b|\bnever\b|n't\b|\bdenies\b|\bdeny\b|\bfails? to\b|"
                    r"\bwithout\b|\bunlikely\b|\bavoids?\b|\brules? out\b)[^,;:]{0,22}$", re.I)
LEX = {}
for _w, _words in [
    (+2, "breakthrough blockbuster"),
    (+1, "growth strong robust profitable upbeat optimistic bullish outperform "
         "accelerates accelerating milestone rebound recovery upside momentum "
         "wins innovative booming resilient tailwind tailwinds"),
    (-2, "scandal crisis collapse"),
    (-1, "weak weakness slowdown concern concerns risk risks bearish "
         "underperform downside pressure headwind headwinds struggles "
         "struggling fears trouble troubled probe setback uncertainty "
         "backlash boycott glut warning"),
]:
    for _x in _words.split():
        LEX[_x] = _w
LEX_NEG = {"not", "no", "never", "without", "despite", "eases", "easing",
           "fades", "fading"}
STOP = set("a an the of to in on for and or at by with from as is are be its "
           "it this that s after over into amid new says say".split())


# --------------------------------------------------------------------------
# Scoring one headline
# --------------------------------------------------------------------------
def _mentions(title, ticker, universe):
    """Does the title name this company? How many OTHER universe companies?"""
    def hit(tk):
        for name in universe.get(tk, []):
            if re.search(r"\b" + re.escape(name) + r"\b", title, re.I):
                return True
        if len(tk) >= 3:
            return re.search(r"(?<![A-Za-z])" + re.escape(tk) + r"(?![A-Za-z])",
                             title) is not None
        return re.search(r"[($:]\s?" + re.escape(tk) + r"\b", title) is not None
    me = hit(ticker)
    ff = FALSE_FRIENDS.get(ticker)
    if me and ff and re.search(ff, title, re.I):
        return False, 0, True
    return me, sum(hit(o) for o in universe if o != ticker), False


def source_tier(source):
    s = (source or "").lower()
    for tier, names in SOURCE_TIER:
        if any(n in s for n in names):
            return tier
    return DEFAULT_TIER


def score_headline(title, ticker="", source="", universe=None):
    """-> dict(s in [-1,1], etype, w weight, rel, hits)."""
    universe = COMPANY if universe is None else universe
    etype, raw, hits = None, 0.0, []
    for t, pol, rx in PATTERNS:
        m = rx.search(title)
        if not m:
            continue
        if etype is None:
            etype = t
        if etype in ("lawfirm", "opinion"):
            break
        if pol:
            if NEGATE.search(title[: m.start()]):
                pol = -pol
            raw += pol
            hits.append(f"{t}{pol:+.1f}")
    toks = re.findall(r"[a-z][a-z'-]+", title.lower())
    for i, w in enumerate(toks):
        if w in LEX:
            v = LEX[w] * (-1 if any(x in LEX_NEG for x in toks[max(0, i - 3): i])
                          else 1)
            raw += 0.5 * v
    etype = etype or "other"
    if etype in ("lawfirm", "opinion"):
        raw *= 0.3
    s = raw / (abs(raw) + 1.5)
    hedge = 0.6 if HEDGE.search(title) else 1.0
    if ticker:
        me, others, false_friend = _mentions(title, ticker, universe)
        rel = (0.0 if false_friend else
               (1.0 if others == 0 else 0.7 if others == 1 else 0.35) if me
               else 0.25)
    else:
        rel = 1.0
    w = IMPORTANCE[etype] * source_tier(source) * rel * hedge
    return dict(s=float(s), etype=etype, w=float(w), rel=rel, hits=hits)


# --------------------------------------------------------------------------
# Time handling
# --------------------------------------------------------------------------
def to_utc(x):
    if isinstance(x, (int, float)):
        return dt.datetime.fromtimestamp(x, UTC)
    if isinstance(x, dt.datetime):
        return x.astimezone(UTC) if x.tzinfo else x.replace(tzinfo=UTC)
    try:
        d = dt.datetime.fromisoformat(str(x).replace("Z", "+00:00"))
    except ValueError:
        d = email.utils.parsedate_to_datetime(str(x))
    return d.astimezone(UTC) if d.tzinfo else d.replace(tzinfo=UTC)


def trading_day(ts_utc):
    """The first trading day on which this story could be acted on at the
    close: news after 16:00 New York rolls to the next weekday."""
    loc = ts_utc.astimezone(NY)
    d = loc.date()
    if loc.hour >= 16:
        d += dt.timedelta(days=1)
    while d.weekday() >= 5:
        d += dt.timedelta(days=1)
    return d


# --------------------------------------------------------------------------
# Fetching (every source is optional and failure-tolerant)
# --------------------------------------------------------------------------
def _get(url, timeout=15):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read(2_000_000)


def _strip(s):
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", s or ""))).strip()


def parse_rss(data):
    out = []
    root = ET.fromstring(data)
    for it in root.iter("item"):
        title = _strip(it.findtext("title"))
        pub = it.findtext("pubDate")
        if not title or not pub:
            continue
        src = _strip(it.findtext("source"))
        if src and title.endswith(" - " + src):
            title = title[: -len(src) - 3]
        try:
            when = to_utc(pub)
        except Exception:
            continue
        out.append(dict(title=title, url=(it.findtext("link") or "").strip(),
                        published=when, source=src))
    return out


def src_yahoo_rss(tk):
    q = urllib.parse.quote(tk)
    items = parse_rss(_get("https://feeds.finance.yahoo.com/rss/2.0/headline"
                           f"?s={q}&region=US&lang=en-US"))
    for i in items:
        i["source"] = i["source"] or "Yahoo Finance"
    return items


def src_google_rss(tk):
    names = COMPANY.get(tk, [])
    q = f'"{names[0]}" stock' if names else f"{tk} stock"
    q = urllib.parse.quote(q + " when:7d")
    return parse_rss(_get("https://news.google.com/rss/search?q=" + q +
                          "&hl=en-US&gl=US&ceid=US:en"))


def src_yahoo_api(tk):
    import yfinance as yf
    out = []
    for item in yf.Ticker(tk).news or []:
        c = item.get("content") or item
        title = c.get("title")
        when = c.get("pubDate") or c.get("displayTime") or \
            item.get("providerPublishTime")
        if not title or when is None:
            continue
        prov = c.get("provider") or {}
        url = (c.get("canonicalUrl") or {}).get("url") or c.get("link") or \
            item.get("link") or ""
        out.append(dict(title=_strip(title), url=url, published=to_utc(when),
                        source=prov.get("displayName") or
                        item.get("publisher") or "Yahoo Finance"))
    return out


SOURCES = {"yahoo_api": src_yahoo_api, "yahoo_rss": src_yahoo_rss,
           "google_rss": src_google_rss}


def fetch(tk, status):
    items = []
    for name, fn in SOURCES.items():
        try:
            got = fn(tk)
            for g in got:
                g["feed"] = name
            items += got
            status[name]["ok"] += 1
            status[name]["items"] += len(got)
        except Exception as e:
            status[name]["fail"] += 1
            status[name]["error"] = f"{type(e).__name__}: {str(e)[:80]}"
    return items


# --------------------------------------------------------------------------
# Archive (append-only in spirit: stories are never dropped or re-dated)
# --------------------------------------------------------------------------
def _tokens(title):
    return frozenset(w for w in re.findall(r"[a-z0-9]+", title.lower())
                     if w not in STOP)


def story_id(title):
    return hashlib.sha1(" ".join(sorted(_tokens(title))).encode()).hexdigest()[:16]


def _path(sdir, tk):
    return os.path.join(sdir, "news", re.sub(r"[^A-Za-z0-9_-]", "_", tk) + ".jsonl")


def load_archive(sdir, tk):
    p = _path(sdir, tk)
    if not os.path.exists(p):
        return []
    with open(p) as f:
        return [json.loads(line) for line in f if line.strip()]


def save_archive(sdir, tk, stories):
    p = _path(sdir, tk)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    tmp = p + ".tmp"
    with open(tmp, "w") as f:
        for s in sorted(stories, key=lambda s: s["published"]):
            f.write(json.dumps(s) + "\n")
    os.replace(tmp, p)


def merge(stories, items, tk, now):
    """Add fetched items; fold syndicated near-duplicates into one story."""
    recent = [(s, _tokens(s["title"]), to_utc(s["published"])) for s in stories]
    added = 0
    for it in sorted(items, key=lambda i: i["published"]):
        if it["published"] > now + dt.timedelta(hours=1):
            continue                               # bad timestamp
        tok, twin = _tokens(it["title"]), None
        for s, t2, when in recent:                 # same story within 72 hours
            if abs((it["published"] - when).total_seconds()) > 72 * 3600:
                continue
            if tok == t2 or (len(tok) >= 4 and
                             len(tok & t2) / len(tok | t2) >= 0.7):
                twin = s
                break
        if twin is not None:
            seen = twin.setdefault("seen_in", [])
            key = (it.get("source") or "") + "|" + it.get("feed", "")
            if key not in seen:
                seen.append(key)
                twin["copies"] = len(seen)
                if source_tier(it.get("source")) > source_tier(twin.get("source")):
                    twin["source"], twin["url"] = it.get("source"), it.get("url")
            continue
        s = dict(id=story_id(it["title"]) + it["published"].strftime("-%Y%m%d"),
                 ticker=tk, title=it["title"][:300],
                 source=it.get("source") or "", url=it.get("url") or "",
                 published=it["published"].isoformat(),
                 first_seen=now.isoformat(), copies=1,
                 seen_in=[(it.get("source") or "") + "|" + it.get("feed", "")])
        stories.append(s)
        recent.append((s, tok, it["published"]))
        added += 1
    return added


def enrich(stories, universe=None):
    out = []
    for s in stories:
        sc = score_headline(s["title"], s["ticker"], s.get("source"), universe)
        pub = to_utc(s["published"])
        out.append(dict(s, **sc, pub=pub, tday=trading_day(pub)))
    return out


# --------------------------------------------------------------------------
# Aggregation: stories -> one reading per stock per day
# --------------------------------------------------------------------------
def aggregate(rich, day):
    """Reading for one stock as of trading day `day` (uses only stories whose
    trading day is <= day)."""
    num = den = 0.0
    n7 = n2 = copies = 0
    types = {}
    scores, weights = [], []
    for s in rich:
        age = (day - s["tday"]).days
        if age < 0 or age > WINDOW_DAYS or s["w"] <= 0:
            continue
        dw = s["w"] * 0.5 ** (age / HALF_LIFE_DAYS) * \
            (1 + 0.25 * math.log(max(1, s.get("copies", 1))))
        num += dw * s["s"]
        den += dw
        scores.append(s["s"])
        weights.append(dw)
        if age <= 7:
            n7 += 1
            copies += s.get("copies", 1)
            if s["etype"] not in ("recap", "opinion", "other"):
                types[s["etype"]] = types.get(s["etype"], 0) + 1
        if age <= 2:
            n2 += 1
    if den <= 0:
        return None
    level = num / (den + SHRINK_K)
    disp = float(np.sqrt(np.average((np.array(scores) - num / den) ** 2,
                                    weights=weights))) if len(scores) > 1 else 0.0
    return dict(level=level, n_eff=den, stories_7d=n7, stories_2d=n2,
                copies_7d=copies, dispersion=disp,
                events=sorted(types, key=lambda t: -types[t])[:3])


def cross_section(levels):
    """{ticker: level} -> {ticker: z}. Sentiment relative to the other stocks
    today, so an across-the-board bullish or bearish news day cancels out."""
    if len(levels) >= 5:
        v = np.array(list(levels.values()))
        med = np.median(v)
        sd = max(1.4826 * np.median(np.abs(v - med)), v.std(), 0.05)
        return {k: float(np.clip((x - med) / sd, -2.5, 2.5))
                for k, x in levels.items()}
    return {k: float(np.clip(x / 0.25, -2.5, 2.5)) for k, x in levels.items()}


def attention(rich, day):
    """Stories in the last 2 days vs. this stock's own 30-day norm."""
    days = [(day - s["tday"]).days for s in rich if s["w"] > 0]
    past = [d for d in days if 2 < d <= 32]
    if not past or max(days, default=0) < 15:
        return None
    base = len(past) / 15.0                       # per 2-day window
    return sum(0 <= d <= 2 for d in days) / base if base > 0 else None


# --------------------------------------------------------------------------
# Evidence: has the signal been followed by returns?
# --------------------------------------------------------------------------
def _spearman(a, b):
    return float(pd.Series(a).rank().corr(pd.Series(b).rank()))


def evaluate(rich_by_tk, px):
    """Rank IC of the cross-sectional news z-score vs forward returns, using
    the archive point-in-time. px: DataFrame of closes (index = dates)."""
    all_days = [s["tday"] for r in rich_by_tk.values() for s in r]
    ev = dict(stories=len(all_days), prior=IC_PRIOR, need_days=MIN_EVAL_DAYS,
              horizons={}, by_event=[], days=0, status="collecting",
              ic_used=IC_PRIOR, first_day=None)
    if not all_days:
        return ev
    first = min(all_days)
    ev["first_day"] = str(first)
    idx = px.index
    start = idx.searchsorted(pd.Timestamp(first) + pd.Timedelta(days=7))
    zs = {}
    for i in range(start, len(idx)):
        day = idx[i].date()
        lv = {}
        for tk, rich in rich_by_tk.items():
            if tk in px.columns and not np.isnan(px[tk].iloc[i]):
                a = aggregate(rich, day)
                if a and a["n_eff"] >= 0.3:
                    lv[tk] = a["level"]
        if len(lv) >= 5:
            zs[i] = cross_section(lv)
    ev["days"] = len(zs)
    logp = np.log(px)
    for h in (1, 5, 21):
        ics, n_obs = [], 0
        for i, z in zs.items():
            if i + h >= len(idx):
                continue
            tks = list(z)
            fwd = (logp[tks].iloc[i + h] - logp[tks].iloc[i]).to_numpy()
            zv = np.array([z[t] for t in tks])
            if np.std(zv) < 1e-9 or np.isnan(fwd).any():
                continue
            ics.append(_spearman(zv, fwd))
            n_obs += len(tks)
        if len(ics) >= 5:
            m, sd = float(np.mean(ics)), float(np.std(ics, ddof=1))
            n_ind = max(2.0, len(ics) / h)          # overlapping windows
            ev["horizons"][str(h)] = dict(
                ic=m, t=m / (sd / math.sqrt(n_ind)) if sd > 0 else 0.0,
                days=len(ics), n=n_obs)
    h5 = ev["horizons"].get("5")
    if h5 and h5["days"] >= MIN_EVAL_DAYS:
        n = h5["n"] / 5.0                           # de-overlap
        post = (n * h5["ic"] + IC_PRIOR_N * IC_PRIOR) / (n + IC_PRIOR_N)
        ev["ic_used"] = float(np.clip(post, 0.0, IC_CAP))
        ev["status"] = "measured"

    # which kinds of news were followed by moves (5-day, vs. universe average)
    pos = {d.date(): i for i, d in enumerate(idx)}
    uni = logp.diff(5).shift(-5).mean(axis=1).to_numpy()
    groups = {}
    for tk, rich in rich_by_tk.items():
        if tk not in px.columns:
            continue
        own = logp[tk].diff(5).shift(-5).to_numpy()
        for s in rich:
            i = pos.get(s["tday"])
            if i is None or abs(s["s"]) < 0.2 or s["w"] < 0.3 or np.isnan(own[i]):
                continue
            groups.setdefault((s["etype"], "good" if s["s"] > 0 else "bad"),
                              []).append(own[i] - uni[i])
    for (etype, sign), v in groups.items():
        if len(v) >= 8:
            v = np.array(v)
            ev["by_event"].append(dict(
                etype=EVENT_LABEL[etype], sign=sign, n=len(v),
                mean=float(v.mean()),
                t=float(v.mean() / (v.std(ddof=1) / math.sqrt(len(v))))
                if v.std() > 0 else 0.0))
    ev["by_event"].sort(key=lambda r: -abs(r["t"]))
    return ev


# --------------------------------------------------------------------------
# Demo data (random headlines; unrelated to prices by construction)
# --------------------------------------------------------------------------
_TEMPLATES = [
    "{c} beats quarterly profit estimates on strong demand",
    "{c} misses revenue expectations as sales slow",
    "{c} raises full-year guidance after record quarter",
    "{c} cuts annual outlook, warns of weaker demand",
    "Analyst upgrades {c} to buy, raises price target",
    "{c} downgraded to sell on valuation concerns",
    "{c} faces lawsuit over alleged accounting fraud",
    "Regulators approve {c} acquisition, clearing final hurdle",
    "EU regulators fine {c} in antitrust case",
    "{c} announces $10 billion buyback",
    "{c} CEO steps down unexpectedly",
    "{c} wins multi-year contract with major customer",
    "{c} announces layoffs affecting 5% of workforce",
    "{c} unveils new product line at annual event",
    "{c} recalls products over safety defect",
    "Why {c} stock is jumping today",
    "{c} shares fall as market slides",
    "3 reasons to buy {c} stock now",
    "Is {c} still a buy after its rally?",
    "SHAREHOLDER ALERT: law firm reminds {c} investors of class action deadline",
    "{c} to acquire startup in all-cash deal",
    "{c} may delay product launch, sources say",
    "{c} reports quarterly results",
    "{c} does not expect to miss delivery targets",
]
_DEMO_SOURCES = ["Reuters", "Bloomberg", "CNBC", "MarketWatch", "Motley Fool",
                 "Zacks", "Yahoo Finance", "Business Insider", "Benzinga"]


def demo_items(tk, k, days_index, plant=None, rate=1.1):
    """Random stories. plant = optional Series of forward returns: when given,
    headline tone leans toward the future return (used ONLY by selftest)."""
    rng = np.random.default_rng(1000 + k)
    name = COMPANY.get(tk, [tk])[0]
    good = [t for t in _TEMPLATES if score_headline(t.format(c=name))["s"] > 0.2]
    bad = [t for t in _TEMPLATES if score_headline(t.format(c=name))["s"] < -0.2]
    items = []
    for d in days_index:
        for _ in range(rng.poisson(rate)):
            if plant is not None and d in plant.index and plant[d] == plant[d] \
                    and rng.random() < 0.6:
                up = (plant[d] > 0) == (rng.random() < 0.72)
                tpl = good[rng.integers(len(good))] if up \
                    else bad[rng.integers(len(bad))]
            else:
                tpl = _TEMPLATES[rng.integers(len(_TEMPLATES))]
            when = dt.datetime.combine(d.date(), dt.time(
                int(rng.integers(12, 20)), int(rng.integers(60))), UTC)
            src = _DEMO_SOURCES[rng.integers(len(_DEMO_SOURCES))]
            items.append(dict(title=tpl.format(c=name), url="",
                              published=when, source=src, feed="demo"))
            if rng.random() < 0.15:                 # syndicated copy
                items.append(dict(items[-1], source="Yahoo Finance",
                                  title=items[-1]["title"] + " - report"))
    return items


# --------------------------------------------------------------------------
# Scan the universe
# --------------------------------------------------------------------------
def scan_universe(tickers, sdir, px, demo=False, quiet=False):
    """Fetch + archive + score. px: DataFrame of closes for the tickers.
    Returns dict(signals, stories, evidence, sources)."""
    now = dt.datetime.now(UTC)
    universe = {tk: COMPANY.get(tk, []) for tk in tickers}
    status = {n: dict(ok=0, fail=0, items=0, error=None) for n in SOURCES}
    rich_by_tk, added = {}, 0
    for k, tk in enumerate(tickers):
        stories = load_archive(sdir, tk)
        if demo:
            if not stories:
                items = demo_items(tk, k, px.index[-120:])
                added += merge(stories, items, tk, now + dt.timedelta(days=1))
        else:
            if not quiet:
                print(f"... news {tk} ({k + 1}/{len(tickers)})", file=sys.stderr)
            added += merge(stories, fetch(tk, status), tk, now)
        save_archive(sdir, tk, stories)
        rich_by_tk[tk] = enrich(stories, universe)

    evidence = evaluate(rich_by_tk, px)
    day = max(trading_day(now), px.index[-1].date())
    aggs = {tk: aggregate(r, day) for tk, r in rich_by_tk.items()}
    zs = cross_section({tk: a["level"] for tk, a in aggs.items()
                        if a and a["n_eff"] >= 0.3})
    signals = {}
    for tk in tickers:
        a = aggs.get(tk)
        col = px[tk].dropna() if tk in px.columns else pd.Series(dtype=float)
        signals[tk] = dict(
            a or dict(level=0.0, n_eff=0.0, stories_7d=0, stories_2d=0,
                      copies_7d=0, dispersion=0.0, events=[]),
            z=zs.get(tk), ic=evidence["ic_used"],
            attention=attention(rich_by_tk[tk], day),
            ret_5d=float(col.iloc[-1] / col.iloc[-6] - 1) if len(col) > 6 else None,
            events=[EVENT_LABEL[e] for e in (a["events"] if a else [])])
    feed = []
    for tk, rich in rich_by_tk.items():
        for s in rich:
            age = (day - s["tday"]).days
            if 0 <= age <= 7 and s["w"] > 0:
                feed.append((abs(s["s"]) * s["w"] * 0.5 ** (age / HALF_LIFE_DAYS),
                             dict(ticker=tk, title=s["title"], source=s["source"],
                                  url=s["url"] if re.match(r"https?://", s["url"] or "")
                                  else "", published=s["published"],
                                  s=s["s"], w=s["w"], etype=EVENT_LABEL[s["etype"]],
                                  copies=s.get("copies", 1))))
    feed.sort(key=lambda x: -x[0])
    evidence["added"] = added
    return dict(signals=signals, stories=[f[1] for f in feed[:30]],
                evidence=evidence, sources=None if demo else status,
                as_of=str(day))


# --------------------------------------------------------------------------
# Forecast layer
# --------------------------------------------------------------------------
class NewsLayer:
    """Overlay: tilt = IC x z x sigma x sqrt(min(h, EFFECT_DAYS))."""
    name, kind = "NewsSentiment", "overlay"

    def predict(self, b, horizons):
        n = getattr(b, "news", None)
        if n is None:                              # single-stock fallback
            heads = getattr(b, "headlines", None)
            if not heads:
                return None
            now = b.prices.index[-1]
            num = den = 0.0
            for ts, text in heads:
                sc = score_headline(text)
                w = sc["w"] * 0.5 ** (max((now - ts).days, 0) / HALF_LIFE_DAYS)
                num, den = num + w * sc["s"], den + w
            if den <= 0:
                return None
            n = dict(z=float(np.clip(num / (den + SHRINK_K) / 0.25, -2.5, 2.5)),
                     ic=IC_PRIOR, stories_7d=len(heads), n_eff=den)
        if n.get("z") is None or n.get("n_eff", 0) < 0.3:
            return None
        sd = ss._vol(b.prices)
        out = {h: n["ic"] * n["z"] * sd * math.sqrt(min(h, EFFECT_DAYS))
               for h in horizons}
        return ss.Signal(out, f"{n.get('stories_7d', 0)} stories/7d, z "
                              f"{n['z']:+.1f} vs peers, IC {n['ic']:.3f}")


# --------------------------------------------------------------------------
# Self-test
# --------------------------------------------------------------------------
CASES = [  # (headline, ticker, expected sign, expected event type)
    ("Tesla beats quarterly profit estimates on strong demand", "TSLA", +1, "earnings"),
    ("Nvidia misses revenue expectations as sales slow", "NVDA", -1, "earnings"),
    ("Apple raises full-year guidance after record quarter", "AAPL", +1, "guidance"),
    ("Microsoft cuts annual outlook, warns of weaker demand", "MSFT", -1, "guidance"),
    ("Morgan Stanley upgrades Amazon to overweight", "AMZN", +1, "analyst"),
    ("Analyst slashes Tesla price target", "TSLA", -1, "analyst"),
    ("Oracle downgraded to sell on valuation", "ORCL", -1, "analyst"),
    ("FDA approves Eli Lilly obesity drug", "LLY", +1, "regulatory"),
    ("EU regulators fine Meta in antitrust case", "META", -1, "regulatory"),
    ("Netflix announces $10 billion buyback", "NFLX", +1, "capital"),
    ("Broadcom CEO steps down unexpectedly", "AVGO", -1, "management"),
    ("Walmart announces layoffs affecting 5% of workforce", "WMT", -1, "operations"),
    ("Tesla recalls 200,000 vehicles over safety defect", "TSLA", -1, "operations"),
    ("JPMorgan wins lawsuit over trading losses", "JPM", +1, "legal"),
    ("Alphabet faces lawsuit over alleged fraud", "GOOGL", -1, "legal"),
    ("Apple does not expect to miss delivery targets", "AAPL", +1, "earnings"),
    ("Why Tesla stock is plunging today", "TSLA", 0, "recap"),
    ("Nvidia shares soar to record high", "NVDA", 0, "recap"),
    ("3 reasons to buy Amazon stock now", "AMZN", 0, "opinion"),
    ("Is Apple still a buy after its rally?", "AAPL", 0, "opinion"),
    ("SHAREHOLDER ALERT: Rosen Law Firm reminds Tesla investors of class action deadline", "TSLA", 0, "lawfirm"),
    ("Microsoft to acquire gaming studio", "MSFT", 0, "mna"),
    ("Visa reports quarterly results", "V", 0, "earnings"),
]


def selftest():
    bad = 0
    for title, tk, sign, etype in CASES:
        r = score_headline(title, tk)
        got = 0 if abs(r["s"]) < 0.12 else int(np.sign(r["s"]))
        ok = got == sign and r["etype"] == etype
        bad += not ok
        print(f"{'ok  ' if ok else 'FAIL'} {r['s']:+.2f} {r['etype']:<11} "
              f"w={r['w']:.2f}  {title}")
    # relevance + noise checks
    checks = [
        ("law-firm spam has zero weight",
         score_headline(CASES[20][0], "TSLA")["w"] == 0),
        ("travel-visa story is not about Visa Inc.",
         score_headline("US tightens H-1B visa rules", "V")["w"] == 0),
        ("roundup naming 3 companies is down-weighted",
         score_headline("Apple, Microsoft and Nvidia lead tech rally", "AAPL")["rel"] < 0.5),
        ("price recap weighs far less than earnings",
         score_headline(CASES[16][0], "TSLA")["w"] <
         0.3 * score_headline(CASES[0][0], "TSLA", "Reuters")["w"]),
        ("hedged headline is dampened",
         score_headline("Tesla may cut annual outlook", "TSLA")["w"] <
         score_headline("Tesla cuts annual outlook", "TSLA")["w"]),
        ("after-close news rolls to next trading day",
         trading_day(dt.datetime(2026, 11, 13, 22, 30, tzinfo=UTC)) ==
         dt.date(2026, 11, 16)),
    ]
    st = []
    n = merge(st, [dict(title="Tesla beats profit estimates on strong demand", url="",
                        published=dt.datetime(2026, 11, 10, 14, tzinfo=UTC), source="Reuters"),
                   dict(title="Tesla beats profit estimates on strong demand - report", url="",
                        published=dt.datetime(2026, 11, 10, 15, tzinfo=UTC), source="Yahoo")],
              "TSLA", dt.datetime(2026, 11, 11, tzinfo=UTC))
    checks.append(("syndicated copy folds into one story",
                   n == 1 and st[0]["copies"] == 2))
    for label, ok in checks:
        bad += not ok
        print(f"{'ok  ' if ok else 'FAIL'} {label}")

    # can the evaluation detect a real effect, and not invent one?
    tks = list(COMPANY)
    px = pd.concat({tk: ss.synthetic(3, seed=50 + k).prices
                    for k, tk in enumerate(tks)}, axis=1)
    fwd = np.log(px).diff(5).shift(-5)
    xs = fwd.sub(fwd.mean(axis=1), axis=0)
    res = {}
    for label, plant in (("planted", True), ("null", False)):
        rich = {}
        for k, tk in enumerate(tks):
            st = []
            merge(st, demo_items(tk, k, px.index[-400:],
                                 plant=xs[tk] if plant else None, rate=1.5),
                  tk, dt.datetime.now(UTC) + dt.timedelta(days=1))
            rich[tk] = enrich(st)
        res[label] = evaluate(rich, px)["horizons"]["5"]
        print(f"     {label:<8} 5-day rank IC {res[label]['ic']:+.3f}  "
              f"t = {res[label]['t']:+.1f}  ({res[label]['days']} days)")
    for label, ok in [("evaluation detects a planted effect (t > 3)",
                       res["planted"]["t"] > 3),
                      ("evaluation finds nothing in random news (|t| < 2.5)",
                       abs(res["null"]["t"]) < 2.5)]:
        bad += not ok
        print(f"{'ok  ' if ok else 'FAIL'} {label}")
    print(f"\n{'ALL PASSED' if not bad else str(bad) + ' FAILED'}")
    return bad


# --------------------------------------------------------------------------
def print_report(res):
    ev = res["evidence"]
    print(f"\nNEWS as of {res['as_of']}  (archive: {ev['stories']} stories "
          f"since {ev['first_day']}, {ev['added']} new this scan)")
    if res["sources"]:
        for n, s in res["sources"].items():
            print(f"  source {n:<11} ok {s['ok']:>2}  failed {s['fail']:>2}  "
                  f"items {s['items']:>4}" + (f"  last error: {s['error']}"
                                              if s["error"] else ""))
    print(f"{'Ticker':<8}{'Level':>7}{'z':>6}{'7d':>5}{'Attn':>6}{'5d px':>8}  Events")
    for tk, s in sorted(res["signals"].items(),
                        key=lambda kv: -(kv[1]["z"] if kv[1]["z"] is not None else -9)):
        z = f"{s['z']:+.1f}" if s["z"] is not None else "–"
        at = f"{s['attention']:.1f}x" if s["attention"] else "–"
        r5 = f"{s['ret_5d']:+.1%}" if s["ret_5d"] is not None else "–"
        print(f"{tk:<8}{s['level']:>+7.2f}{z:>6}{s['stories_7d']:>5}{at:>6}"
              f"{r5:>8}  {', '.join(s['events'])}")
    print("\nTop stories")
    for s in res["stories"][:10]:
        print(f"  {s['s']:+.2f} {s['ticker']:<6}{s['etype']:<13}"
              f"{s['title'][:80]}  [{s['source']}]")
    print("\nEVIDENCE")
    if ev["status"] == "collecting":
        print(f"  Collecting: {ev['days']} of {ev['need_days']} evaluable days. "
              f"Using cautious prior IC {ev['prior']}. Run the scan daily.")
    for h, r in ev["horizons"].items():
        print(f"  {h:>2}-day rank IC {r['ic']:+.3f}  t = {r['t']:+.1f}  "
              f"({r['days']} days)")
    if ev["status"] == "measured":
        print(f"  IC used for sizing (measured, shrunk to prior): "
              f"{ev['ic_used']:.3f}")
    for r in ev["by_event"][:8]:
        print(f"  {r['etype']:<15}{r['sign']:<5} n={r['n']:<4} next 5d vs peers "
              f"{r['mean']:+.2%}  t = {r['t']:+.1f}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["scan", "evaluate", "selftest"])
    p.add_argument("--tickers", nargs="+", default=list(COMPANY))
    p.add_argument("--years", type=int, default=3)
    p.add_argument("--demo", action="store_true")
    a = p.parse_args()
    if a.command == "selftest":
        raise SystemExit(selftest())
    sdir = "state_demo" if a.demo else "state"
    if a.demo:
        px = pd.concat({tk: ss.synthetic(a.years, seed=k).prices
                        for k, tk in enumerate(a.tickers)}, axis=1)
    else:
        cols = {}
        for tk in a.tickers:
            try:
                cols[tk] = ss.fetch_live(tk, a.years).prices
            except (SystemExit, Exception) as e:
                ss._warn(f"prices for {tk} unavailable: {e}")
        if not cols:
            raise SystemExit("No price data (network blocked? try --demo).")
        px = pd.concat(cols, axis=1).sort_index().ffill()
    print_report(scan_universe(a.tickers, sdir, px, demo=a.demo))


if __name__ == "__main__":
    main()
