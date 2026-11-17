# phone-search

On-device search chatbot for the Nothing Phone (3), running in Termux. Qwen3.5-4B decides whether to search, the
app searches through headless Chromium, keeps the relevant passages, and the model answers with citations.

## Use (on the phone, in Termux)

    search          # terminal chat; /new resets, /think toggles answer-turn thinking, /quit exits
    search web      # web UI at http://127.0.0.1:8090 (open in the phone's browser)
    search stop     # stop llama-server and the web UI

`BACKEND=cpu|gpu|npu search` picks the inference backend (default `cpu`; see `scripts/start-llm.sh`).
Other settings are `PS_*` environment variables (see `phone_search/config.py`).

## Layout

- `phone_search/agent.py`: search loop: decide, search, sift, answer; search budget; history
- `phone_search/search.py`: DuckDuckGo, then Bing, via headless Chromium
- `phone_search/browser.py`: persistent headless Chromium over the DevTools protocol
- `phone_search/sift.py`: fetch pages, extract text (trafilatura), BM25 passage ranking
- `phone_search/llm.py`: streaming client for llama-server
- `phone_search/cli.py`, `web.py`, `static/index.html`: the two front ends
- `scripts/start-llm.sh`, `scripts/search`: launchers

## Assumptions (all open for iteration)

- **Backend = CPU.** The Hexagon NPU's matrix unit (HMX) gives wrong results on this v73 chip; with HMX off the NPU
  is correct but slower at reading prompts (30 tok/s) than the CPU (67 tok/s). The GPU reads prompts faster
  (141 tok/s) but writes slower (6 tok/s) and pushed free RAM down to ~1.7 GB.
- **Thinking** is on for the first turn (whether and what to search) and off for the answer.
- **At most 3 searches per question**; the final turn is sent without tools plus an "answer now" note.
- **Sifting:** top 5 pages per search, ~600-char passages, best 8 kept (max 2 per page) ≈ 4-5k chars.
- **Memory:** the last 6 question/answer pairs; search results are not carried over.
- **Bot checks:** the app links to the page so you can solve it in Chrome. Chrome doesn't share the headless
  browser's cookies, so this may not unblock it (untested; no block seen so far).
- **One question at a time**; everything listens on 127.0.0.1 only.

## Tests

    python -m unittest tests.test_core
