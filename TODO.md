# TODO

Known problems and open questions, roughly in priority order.

## Answers

- [ ] **No citation markers.** In on-phone runs the model never wrote `[1]`-style citations, so the source list under
  each answer stays empty. In the evaluation (same instruction, different tool-result wording) it did cite. Try:
  stronger wording in the system prompt, repeating the instruction in the tool result, or an example answer.
- [ ] **Unverified details.** Some answers add details that may not be in the sources (e.g. an hourly minimum wage of
  34.64 NIS and youth percentages in the Hebrew answer; in earlier runs, a Nokia/Meizu/Oppo career list for Carl Pei).
  Check them against the fetched passages and, if they're not there, tighten the grounding instruction.
- [ ] **Unanswerable questions.** In the evaluation, when the answer wasn't in the results, the model often stated a
  number or date from memory instead of saying it didn't know. Not yet tested in the app.
- [ ] **Search-result quality.** Measure how often the sifted passages actually contain the answer; tune
  `PAGES_TO_FETCH`, `MAX_PASSAGES` and the passage size.

## Stability on the phone

- [ ] **Termux gets killed** every 20-30 minutes when idle, taking `llama-server` with it. Suspected: Android's
  phantom-process killer (Termux runs sshd, llama-server and ~6 Chromium processes) and/or battery optimisation.
  Check with "Disable child process restrictions" in Developer options and battery optimisation off for Termux;
  restart `llama-server` automatically when it's gone.
- [ ] **Free memory is tight.** With the NPU server loaded, free memory sits at ~1.9 GB and dipped to ~1.5 GB during
  a run (Chromium included). Options: smaller context (`CTX`), close Chromium when idle, fewer browser fetches.
- [ ] **Heat.** Several questions in a row warmed the case to ~43 °C in earlier runs. At higher temperatures the
  phone's thermal policy turns off the Wi-Fi hotspot. Consider a cool-down pause or a lower thread count.
- [ ] **Bot checks are untested.** No search engine has blocked the headless browser yet. The plan (open the challenge
  link in Chrome) probably doesn't help, because Chrome doesn't share the headless browser's cookies.
  An interactive fallback would need a display for Chromium (e.g. Termux:X11).

## NPU backend

- [ ] **Depends on an unmerged llama.cpp PR** ([#29740](https://github.com/ggml-org/llama.cpp/pull/29740), built
  from commit `08c8373` of the author's branch). Rebuild from upstream once it's merged; until then `BACKEND=cpu`
  is the fallback.
- [ ] **Q4_0 only.** The integer-HMX path covers Q4_0 weights; other quantisations fall back to the slower HVX path.
- [ ] **Not bit-identical to CPU.** The PR author reports nearly the same perplexity (10.84 vs 10.85) but greedy output
  diverging after ~10 tokens. 8 of 2,985 `FLASH_ATTN_EXT` tests fail on this phone (the author sees similar misses
  with HMX disabled).

## App

- [ ] Restart `llama-server` from the app when it's unreachable, instead of failing.
- [ ] The web UI doesn't show the model's thinking or let you stop an answer.
- [ ] Re-run the blind evaluation against the app on the phone (NPU build), not just on the Mac.
