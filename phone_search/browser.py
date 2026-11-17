"""A persistent headless Chromium driven over the DevTools protocol (CDP).

Why a real browser: plain HTTP scraping of DuckDuckGo/Bing gets bot pages within minutes, while
headless Chromium with the phone's own Chrome user agent passes. One browser process is started
lazily and reused; its profile (cookies, consent) persists in ~/.cache/phone-search/chromium.

Interactive fallback: when a search engine shows a challenge, `ChallengeError` carries the URL so
the UI can tell the user to open it in their normal browser. (Headless Chromium can't show a window
in Termux; see README for the trade-off.)
"""
import atexit
import json
import os
import shutil
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

from websockets.sync.client import connect

# The phone's real Chrome UA. With the default 'HeadlessChrome' UA, DuckDuckGo silently returns no results.
UA = ("Mozilla/5.0 (Linux; Android 16; A024) AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/157.0.0.0 Mobile Safari/537.36")
PROFILE = Path(os.environ.get("PS_BROWSER_PROFILE", Path.home() / ".cache" / "phone-search" / "chromium"))
PORT = int(os.environ.get("PS_BROWSER_PORT", "9333"))


class BrowserError(Exception):
    pass


class ChallengeError(BrowserError):
    def __init__(self, url, engine):
        super().__init__(f"{engine} is asking for a human check")
        self.url = url
        self.engine = engine


def _find_chromium():
    for name in ("chromium-browser", "chromium", "google-chrome", "chrome"):
        path = shutil.which(name)
        if path:
            return path
    mac = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
    if os.path.exists(mac):
        return mac
    raise BrowserError("no Chromium found (in Termux: pkg install x11-repo && pkg install chromium)")


class Browser:
    """One headless Chromium with a single reusable tab. Thread-safe (calls are serialized)."""

    def __init__(self):
        self._proc = None
        self._ws = None
        self._id = 0
        self._lock = threading.Lock()

    # -- lifecycle -------------------------------------------------------------------------
    def _start(self):
        PROFILE.mkdir(parents=True, exist_ok=True)
        # A previous run that was killed can leave Chromium behind holding the profile and the port.
        subprocess.run(["pkill", "-f", f"--user-data-dir={PROFILE}"], check=False,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        # Few processes: Android 12+ kills apps whose child-process count exceeds a limit (phantom
        # process killer), and Chromium normally spawns a zygote, GPU, utility and renderer processes.
        self._proc = subprocess.Popen(
            [_find_chromium(), "--headless=new", "--no-sandbox", "--no-zygote", "--renderer-process-limit=1",
             "--disable-gpu", "--in-process-gpu", "--disable-breakpad", "--disable-crash-reporter",
             "--enable-features=NetworkServiceInProcess2",  # fold network/storage services into the browser
             "--disable-features=Translate,OptimizationHints,MediaRouter,BackForwardCache",
             "--no-first-run",
             "--disable-extensions", "--mute-audio", "--blink-settings=imagesEnabled=false",
             f"--user-agent={UA}", "--lang=en-US", f"--user-data-dir={PROFILE}",
             f"--remote-debugging-port={PORT}", "--remote-allow-origins=*", "about:blank"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        deadline = time.time() + 20
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/json", timeout=1) as r:
                    pages = [t for t in json.load(r) if t["type"] == "page"]
                if pages:
                    self._ws = connect(pages[0]["webSocketDebuggerUrl"], max_size=64 * 2**20, open_timeout=10)
                    self._cmd("Page.enable")
                    return
            except OSError:
                pass
            if self._proc.poll() is not None:
                raise BrowserError(f"Chromium exited with code {self._proc.returncode}")
            time.sleep(0.25)
        self.close()
        raise BrowserError("Chromium did not start within 20 s")

    def _alive(self):
        return self._proc is not None and self._proc.poll() is None and self._ws is not None

    def close(self):
        if self._ws:
            try:
                self._ws.close()
            except Exception:
                pass
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        self._proc = self._ws = None

    # -- CDP -----------------------------------------------------------------------------
    def _cmd(self, method, timeout=30, **params):
        self._id += 1
        self._ws.send(json.dumps({"id": self._id, "method": method, "params": params}))
        deadline = time.time() + timeout
        while time.time() < deadline:
            msg = json.loads(self._ws.recv(timeout=max(0.1, deadline - time.time())))
            if msg.get("id") == self._id:
                if "error" in msg:
                    raise BrowserError(f"{method}: {msg['error'].get('message')}")
                return msg.get("result", {})
        raise BrowserError(f"{method} timed out")

    def _eval(self, expression):
        r = self._cmd("Runtime.evaluate", expression=expression, returnByValue=True, awaitPromise=True)
        if "exceptionDetails" in r:
            raise BrowserError(f"JS error: {r['exceptionDetails'].get('text')}")
        return r["result"].get("value")

    def run(self, url, ready_js, extract_js, challenge_js="false", timeout=12.0):
        """Navigate to url, wait until ready_js is truthy (or challenge_js), return extract_js's value."""
        with self._lock:
            for attempt in (1, 2):  # restart once if the browser died
                try:
                    if not self._alive():
                        self._start()
                    self._cmd("Page.navigate", url=url)
                    deadline = time.time() + timeout
                    while time.time() < deadline:
                        time.sleep(0.25)
                        state = self._eval(f"({ready_js}) ? 'ready' : (({challenge_js}) ? 'challenge' : '')")
                        if state == "challenge":
                            raise ChallengeError(url, url.split("/")[2])
                        if state == "ready":
                            return self._eval(extract_js)
                    return self._eval(extract_js)  # timed out: return whatever is there
                except ChallengeError:
                    raise
                except Exception as e:
                    self.close()
                    if attempt == 2:
                        raise BrowserError(str(e)) from e


_BROWSER = None


def shared():
    global _BROWSER
    if _BROWSER is None:
        _BROWSER = Browser()
        atexit.register(_BROWSER.close)
    return _BROWSER
