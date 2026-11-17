"""Terminal chat (works in Termux or over SSH).

Commands: /new (forget the conversation), /think (toggle thinking for the answer turn), /quit.
"""
import subprocess
import sys

from . import llm
from .agent import Agent, Events
from .config import Config

DIM, BOLD, RESET = "\033[2m", "\033[1m", "\033[0m"


def main():
    cfg = Config()
    if not llm.health(cfg.llm_url):
        print(f"llama-server is not reachable at {cfg.llm_url}. Start it with scripts/start-llm.sh.", file=sys.stderr)
        sys.exit(1)
    agent = Agent(cfg)
    show_reasoning = "--reasoning" in sys.argv
    interactive = sys.stdin.isatty()
    if interactive:
        print(f"{DIM}Ask anything. /new resets, /think toggles answer-turn thinking, /quit exits.{RESET}")
    while True:
        try:
            q = input(f"\n{BOLD}> {RESET}" if interactive else "").strip()
            if not interactive:
                print(f"\n> {q}")  # echo piped input so transcripts are readable
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not q:
            continue
        if q == "/quit":
            return
        if q == "/new":
            agent.reset()
            print(f"{DIM}(new conversation){RESET}")
            continue
        if q == "/think":
            cfg.think_for_answer = not cfg.think_for_answer
            print(f"{DIM}(answer-turn thinking {'on' if cfg.think_for_answer else 'off'}){RESET}")
            continue

        state = {"thinking": False}

        def on_reasoning(delta):
            if not state["thinking"]:
                state["thinking"] = True
                sys.stdout.write(f"{DIM}thinking…{RESET}" if not show_reasoning else DIM)
            if show_reasoning:
                sys.stdout.write(delta)
            sys.stdout.flush()

        def on_status(text):
            if state["thinking"]:
                sys.stdout.write(RESET + "\n")
                state["thinking"] = False
            print(f"{DIM}… {text}{RESET}")

        def on_answer_start():
            if state["thinking"]:
                sys.stdout.write(RESET + "\n")
                state["thinking"] = False
            sys.stdout.write("\n")

        def on_challenge(url):
            on_status(f"bot check — open in your browser, solve it, then ask again: {url}")
            try:  # Termux: hand the URL to Android so it opens in Chrome
                subprocess.run(["termux-open-url", url], timeout=5, check=False)
            except (OSError, subprocess.SubprocessError):
                pass

        ev = Events(on_status=on_status, on_reasoning=on_reasoning,
                    on_content=lambda d: (sys.stdout.write(d), sys.stdout.flush()),
                    on_answer_start=on_answer_start, on_challenge=on_challenge)
        try:
            ans = agent.ask(q, ev)
        except llm.LLMError as e:
            print(f"\nerror: {e}")
            continue
        except KeyboardInterrupt:
            print(f"\n{DIM}(interrupted){RESET}")
            continue
        print()
        for s in ans.sources:
            print(f"{DIM}[{s.n}] {s.title[:70]} — {s.url}{RESET}")
        u = ans.usage
        print(f"{DIM}({len(ans.searches)} search(es), {u.get('prompt_tokens', 0)} prompt + "
              f"{u.get('completion_tokens', 0)} generated tokens){RESET}")


if __name__ == "__main__":
    main()
