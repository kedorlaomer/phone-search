"""Fetch result pages, extract their main text, and keep only the passages relevant to the question.

This is the 'sifting' step: the model never sees whole pages, only the top-ranked passages
(BM25 over ~600-char chunks), capped per source so one long page can't crowd out the rest.
"""
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import trafilatura
from rank_bm25 import BM25Okapi

from .browser import BrowserError, shared

WORD = re.compile(r"\w+", re.UNICODE)
# Reference lists and archive notes repeat the query terms, so BM25 likes them, but they hold no answers.
JUNK_LINE = re.compile(r"^\s*(-\s*)?(\u2191|\^|\d+(\s+\d+)*\s+\")|Retrieved \w+ \d|Archived from the original|"
                       r"Abgerufen am|ISBN|OCLC|doi:", re.I)


@dataclass
class Source:
    n: int
    title: str
    url: str
    passages: list = field(default_factory=list)


def tokens(text):
    return [w.lower() for w in WORD.findall(text)]


def split_passages(text, size):
    """Greedily pack paragraphs into ~size-char chunks, dropping reference-list lines."""
    out, cur = [], ""
    for para in (p.strip() for p in text.split("\n")):
        if not para or JUNK_LINE.search(para):
            continue
        if cur and len(cur) + len(para) > size:
            out.append(cur)
            cur = ""
        cur = f"{cur}\n{para}" if cur else para
        while len(cur) > 2 * size:
            out.append(cur[:size])
            cur = cur[size:]
    if cur:
        out.append(cur)
    return out


MIN_TEXT = 200  # shorter extractions are usually cookie walls, captchas or JS shells


def _extract(page):
    try:
        return trafilatura.extract(page, include_comments=False, include_tables=True) if page else None
    except Exception:
        return None


def fetch_text(url, timeout):
    """Plain HTTP first (fast, parallel); returns None if that didn't yield real text."""
    try:
        text = _extract(trafilatura.fetch_url(url, config=_traf_config(timeout)))
    except Exception:
        text = None
    return text if text and len(text) >= MIN_TEXT else None


def fetch_text_browser(url, timeout):
    """Slow path for pages that block plain HTTP or need JavaScript: render in headless Chromium."""
    try:
        page = shared().run(url, "document.readyState === 'complete'", "document.documentElement.outerHTML",
                            timeout=timeout)
    except BrowserError:
        return None
    text = _extract(page)
    return text if text and len(text) >= MIN_TEXT else None


_TRAF_CONFIGS = {}


def _traf_config(timeout):
    if timeout not in _TRAF_CONFIGS:
        from trafilatura.settings import use_config
        cfg = use_config()
        cfg.set("DEFAULT", "DOWNLOAD_TIMEOUT", str(int(timeout)))
        cfg.set("DEFAULT", "MAX_REDIRECTS", "3")
        _TRAF_CONFIGS[timeout] = cfg
    return _TRAF_CONFIGS[timeout]


def sift(question, query, hits, cfg, first_n=1):
    """Return numbered Sources holding the best passages. Numbering starts at first_n so that
    citations stay unique across several searches in one answer."""
    hits = hits[:cfg.pages_to_fetch]
    with ThreadPoolExecutor(max(1, len(hits))) as pool:
        texts = list(pool.map(lambda h: fetch_text(h.url, cfg.fetch_timeout), hits))
    # Retry failures in the browser (one tab, so sequential), at most cfg.browser_fetches pages.
    retry = [i for i, t in enumerate(texts) if t is None][:cfg.browser_fetches]
    for i in retry:
        texts[i] = fetch_text_browser(hits[i].url, cfg.fetch_timeout)

    candidates = []  # (hit index, passage)
    for i, (hit, text) in enumerate(zip(hits, texts)):
        chunks = split_passages(text, cfg.passage_chars) if text else []
        for chunk in chunks or ([hit.snippet] if hit.snippet else []):  # fall back to the search snippet
            candidates.append((i, chunk))
    if not candidates:
        return []

    bm25 = BM25Okapi([tokens(c) or ["_"] for _, c in candidates])
    scores = bm25.get_scores(tokens(f"{question} {query}"))
    picked, used = {}, 0
    for score, (i, chunk) in sorted(zip(scores, candidates), key=lambda x: -x[0]):
        if len(picked.get(i, [])) >= cfg.max_passages_per_source:
            continue
        picked.setdefault(i, []).append(chunk)
        used += 1
        if used >= cfg.max_passages:
            break

    # Keep search-engine order (a rough authority signal) rather than BM25 order.
    return [Source(first_n + k, hits[i].title, hits[i].url, picked[i]) for k, i in enumerate(sorted(picked))]


def format_sources(sources):
    if not sources:
        return "No usable results."
    return "\n\n".join(f"[{s.n}] {s.title[:120]}\n{s.url}\n" + "\n...\n".join(s.passages) for s in sources)
