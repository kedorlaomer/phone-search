"""Web search through headless Chromium: DuckDuckGo first, Bing as fallback.

No API keys. Plain HTTP scraping was blocked within minutes in testing; a real browser with the
phone's Chrome user agent was not (16/16 queries in en/de/he, 1-6 s each).
"""
import base64
import urllib.parse
from dataclasses import dataclass
from urllib.parse import parse_qs, urlparse

from .browser import BrowserError, ChallengeError, shared


class SearchError(Exception):
    pass


@dataclass
class Hit:
    title: str
    url: str
    snippet: str


DDG = {
    "url": "https://duckduckgo.com/?ia=web&kp=-2&q={q}",
    "ready": "document.querySelectorAll('article[data-testid=result]').length >= 3 || "
             "!!document.querySelector('[data-testid=no-results-message]')",
    "challenge": "!!document.querySelector('.anomaly-modal__modal, #challenge-form') || "
                 "/unusual traffic|anomaly/i.test(document.title)",
    "extract": """[...document.querySelectorAll('article[data-testid=result]')].map(a => ({
        title: (a.querySelector('[data-testid=result-title-a]') || {}).innerText || '',
        url: (a.querySelector('[data-testid=result-title-a]') || {}).href || '',
        snippet: (a.querySelector('[data-result=snippet]') || {}).innerText || ''}))""",
}

BING = {
    "url": "https://www.bing.com/search?q={q}",
    "ready": "document.querySelectorAll('li.b_algo').length >= 3 || !!document.querySelector('.b_no')",
    "challenge": "!!document.querySelector('#b_captcha, .captcha') || /verify|captcha/i.test(document.title)",
    "extract": """[...document.querySelectorAll('li.b_algo')].map(li => ({
        title: (li.querySelector('h2 a') || {}).innerText || '',
        url: (li.querySelector('h2 a') || {}).href || '',
        snippet: (li.querySelector('p, .b_caption p, .b_lineclamp2') || {}).innerText || ''}))""",
}


def _unwrap(url):
    """Bing wraps result links as bing.com/ck/a?...&u=a1<base64url>."""
    if "bing.com/ck/" in url:
        q = parse_qs(urlparse(url).query).get("u", [""])[0]
        if q.startswith("a1"):
            b64 = q[2:]
            try:
                return base64.urlsafe_b64decode(b64 + "=" * (-len(b64) % 4)).decode()
            except ValueError:
                pass
    return url


def _is_ad(url):
    return any(s in url for s in ("duckduckgo.com/y.js", "bing.com/aclick", "/aclk?"))


def _engine(spec, query, n):
    url = spec["url"].format(q=urllib.parse.quote_plus(query))
    rows = shared().run(url, spec["ready"], spec["extract"], spec["challenge"]) or []
    hits, seen = [], set()
    for r in rows:
        u = _unwrap(r["url"])
        if not u.startswith("http") or _is_ad(u) or u in seen:
            continue
        seen.add(u)
        hits.append(Hit(" ".join(r["title"].split()), u, " ".join(r["snippet"].split())))
    return hits[:n]


def web_search(query, n=6):
    errors, challenge = [], None
    for name, spec in (("duckduckgo", DDG), ("bing", BING)):
        try:
            hits = _engine(spec, query, n)
            if hits:
                return hits
            errors.append(f"{name}: no results")
        except ChallengeError as e:
            challenge = challenge or e
            errors.append(f"{name}: human check required")
        except BrowserError as e:
            errors.append(f"{name}: {e}")
    if challenge:
        raise challenge
    raise SearchError("; ".join(errors))
