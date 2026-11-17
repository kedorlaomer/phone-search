# phone-search

A search chatbot that runs entirely on an Android phone (Nothing Phone (3), Snapdragon 8s Gen 4) in Termux.
A small local language model (Qwen3.5-4B) decides whether a question needs a web search, searches through a headless
Chromium, keeps the passages relevant to the question, and answers with citations. It handles English, German and
Hebrew questions. No cloud services or API keys are used.

```
> Was kostet das Deutschlandticket aktuell?
… searching: Deutschlandticket aktuelle Preise 2026
… reading 5 pages

Das Deutschlandticket kostet aktuell (2026) 63 € pro Monat im regulären Abo. […]
Wichtig: Ab dem 1. Januar 2027 steigt der reguläre Preis auf 66,80 € pro Monat.
(1 search(es), 2436 prompt + 197 generated tokens)
```

## How it works

1. **Decide.** The model gets the question plus a `web_search` tool. With thinking enabled, it either answers
   directly (arithmetic, rewriting, small talk) or calls the tool with a query.
2. **Search.** A persistent headless Chromium opens DuckDuckGo (falling back to Bing) and reads the result list.
   It uses the phone's real Chrome user agent: plain HTTP scraping got bot-check pages within minutes in testing,
   and Chromium's default "HeadlessChrome" user agent makes DuckDuckGo silently return nothing.
3. **Sift.** The top 5 result pages are fetched (plain HTTP first, the browser for pages that need it), their main
   text extracted with trafilatura, split into ~600-character passages and ranked with BM25 against the question.
   The best 8 passages (at most 2 per page, about 4-5k characters) go back to the model as numbered sources.
4. **Answer.** With thinking off, the model answers from the sources, or searches again. After at most 3 searches
   the final turn is sent without tools plus an "answer now" note; with tools still offered, the models we tested
   wrote the tool call into the answer as plain text instead.

The model runs in `llama-server` (llama.cpp), and the app talks to it over its OpenAI-compatible API on 127.0.0.1.

## Requirements

- Android phone with Termux (F-Droid build). Termux packages: `python`, `python-lxml`, `python-numpy`, `chromium`
  (from `x11-repo`). Python packages: `trafilatura`, `rank_bm25`, `websockets`. On the phone, install these as
  wheels or Termux packages; compiling them from source got Termux killed for lack of memory.
- A llama.cpp build for the phone with `llama-server`, and `Qwen3.5-4B-Q4_0.gguf` (unsloth) in `~/models`.
- 12 GB RAM phone: with the model loaded and Chromium running, free memory goes down to about 1.5 GB.

### Inference backend

`scripts/start-llm.sh` selects the backend with `BACKEND=npu|cpu|gpu`. Speeds measured on the Nothing Phone (3) for
Qwen3.5-4B Q4_0, all checked for correct output:

| Backend | Prompt reading | Generation | Build |
|---|---|---|---|
| **NPU (default)** | ~115-127 tok/s | ~12-14 tok/s | llama.cpp with [PR #29740](https://github.com/ggml-org/llama.cpp/pull/29740), in `~/npu-int` |
| CPU | 67 tok/s | 11 tok/s | upstream llama.cpp, in `~/npu/llama.cpp` |
| GPU (Adreno, OpenCL) | 141 tok/s | 6 tok/s | upstream llama.cpp; free RAM dropped to ~1.7 GB |

The phone's Hexagon v73 NPU has a matrix unit (HMX) without FP16 support. Stock llama.cpp uses FP16 HMX kernels,
so on this chip it produces garbage output ([issue #29473](https://github.com/ggml-org/llama.cpp/issues/29473)).
PR #29740 detects this and uses integer HMX kernels for Q4_0 models. Building the NPU backend needs the Qualcomm
Hexagon SDK; llama.cpp's `scripts/snapdragon/build.py` does it in Docker (`ghcr.io/snapdragon-toolchain/arm64-android`).

## Usage (in Termux)

    search          # terminal chat: /new resets, /think toggles answer-turn thinking, /quit exits
    search web      # web UI at http://127.0.0.1:8090 (open in the phone's browser)
    search stop     # stop llama-server and the web UI

`search` starts `llama-server` if it isn't running (and replaces it if the wrong build is running). Install by copying
the repository to `~/phone-search` and linking the launcher: `ln -s ~/phone-search/scripts/search $PREFIX/bin/search`.

### Settings

Environment variables, see `phone_search/config.py`:

| Variable | Default | Meaning |
|---|---|---|
| `PS_LLM_URL` | `http://127.0.0.1:8080` | llama-server endpoint |
| `PS_MAX_SEARCHES` | 3 | searches per question |
| `PS_PAGES_TO_FETCH` | 5 | result pages read per search |
| `PS_MAX_PASSAGES` | 8 | passages passed to the model per search |
| `PS_THINK_FOR_TOOLS` / `PS_THINK_FOR_ANSWER` | 1 / 0 | thinking for the search decision / for the answer |
| `PS_HISTORY_TURNS` | 6 | question/answer pairs kept as context |

## Design choices

- **Model:** Qwen3.5-4B was compared with LFM2.5-8B-A1B in a small blind evaluation on frozen search results
  (English, German, Hebrew). Both scored 20/30 on answer quality; Qwen with thinking made the right search decision in
  39/39 cases and is about 2-3x faster on the phone.
- **Thinking only for the decision:** without thinking, the model answered some follow-up questions from memory
  instead of searching. Thinking for the answer roughly doubles the time per answer for little gain.
- **Memory:** only final answers are kept in the conversation history, not the search results, to keep prompts short.
- **Local only:** `llama-server` and the web UI listen on 127.0.0.1.
- **Bot checks:** if a search engine shows a challenge, the app shows the link so you can open it in your browser.

## Development

    python -m unittest tests.test_core     # offline tests; LLM, search and page fetching are faked

| File | Purpose |
|---|---|
| `phone_search/agent.py` | the search loop: decide, search, sift, answer; search budget; history |
| `phone_search/search.py` | DuckDuckGo, then Bing, through the browser |
| `phone_search/browser.py` | persistent headless Chromium over the DevTools protocol |
| `phone_search/sift.py` | page fetching, text extraction, BM25 passage ranking |
| `phone_search/llm.py` | streaming client for llama-server |
| `phone_search/cli.py`, `web.py`, `static/index.html` | terminal and browser front ends |
| `scripts/start-llm.sh`, `scripts/search` | launchers |

See [TODO.md](TODO.md) for known problems.
