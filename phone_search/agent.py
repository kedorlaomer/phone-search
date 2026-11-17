"""The chat loop: decide whether to search, search + sift, answer with citations.

Turn structure for one user question:
  1. Decision call (thinking on by default): the model either answers directly or calls web_search.
  2. For each web_search call: search, fetch pages, sift passages, return them as the tool result.
  3. Follow-up calls (thinking off by default) may search again while the budget lasts.
  4. Once the budget is spent, a final call is made *without* tools and with an explicit
     "answer now" note: with tools still offered, both models we tested wrote the tool call into
     the answer as plain text instead.
"""
import datetime
import json
import re
from dataclasses import dataclass, field

from . import llm, search, sift
from .browser import ChallengeError

WEB_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web. Returns numbered sources with the most relevant passages from each page.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "The search query"}},
            "required": ["query"],
        },
    },
}

BUDGET_NOTE = ("Search budget exhausted: no more searches are possible. Answer the user now using only the "
               "results above. If they do not contain the answer, say so plainly.")

# Raw tool-call markup a model may emit as text when it shouldn't (Qwen and LFM formats).
RAW_TOOL_CALL = re.compile(r"<tool_call>.*?(</tool_call>|$)|<\|tool_call_start\|>.*?(<\|tool_call_end\|>|$)", re.S)
CITATION = re.compile(r"\[(\d+)\]")


def system_prompt(today):
    return f"""You are a helpful search assistant running on the user's phone. Today's date is {today}.
You have a web_search tool. Use it whenever the question depends on current events, specific facts, prices, dates, people, or anything you are not certain about. When in doubt, search. Do not search for things you can answer reliably yourself (arithmetic, rewriting, translation, general explanations, small talk).
When you answer from search results:
- Base your answer only on the results; cite the sources you used as [1], [2], ...
- If sources disagree, say so and prefer the most recent or most authoritative one.
- If the results do not contain the answer, say that clearly. Never fill the gap from memory.
Answer in the language of the user's question. Be concise."""


@dataclass
class Events:
    """UI callbacks; all optional."""
    on_status: callable = lambda text: None          # e.g. "searching: ..."
    on_reasoning: callable = lambda delta: None
    on_content: callable = lambda delta: None
    on_answer_start: callable = lambda: None          # final answer is about to stream
    on_challenge: callable = lambda url: None         # a search engine wants a human check


@dataclass
class Answer:
    text: str
    sources: list                                     # sift.Source objects that were cited
    searches: list = field(default_factory=list)      # queries run
    usage: dict = field(default_factory=dict)         # summed prompt/completion tokens


class Agent:
    def __init__(self, cfg):
        self.cfg = cfg
        self.history = []  # [{"role": "user"|"assistant", "content": ...}], final answers only

    def reset(self):
        self.history = []

    def ask(self, question, ev=None):
        ev = ev or Events()
        cfg = self.cfg
        today = datetime.date.today().isoformat()
        msgs = [{"role": "system", "content": system_prompt(today)},
                *self.history[-2 * cfg.history_turns:],
                {"role": "user", "content": question}]
        sources, queries, usage = [], [], {"prompt_tokens": 0, "completion_tokens": 0}

        def call(tools, think, stream_answer):
            # Stream content live only when this call is expected to produce the final answer.
            started = []

            def on_content(delta):
                if not started:
                    delta = delta.lstrip()  # templates often emit a leading newline
                    if not delta:
                        return
                    started.append(True)
                    ev.on_answer_start()
                ev.on_content(delta)

            out = llm.chat_stream(cfg.llm_url, msgs, tools=tools, think=think, max_tokens=cfg.max_tokens,
                                  on_reasoning=ev.on_reasoning, on_content=on_content if stream_answer else None)
            for k in usage:
                usage[k] += out["usage"].get(k, 0)
            return out

        think = cfg.think_for_tools
        while True:
            budget_left = cfg.max_searches - len(queries)
            tools = [WEB_SEARCH_TOOL] if budget_left > 0 else None
            # The decision turn may itself be the answer (no search needed), so always stream content.
            out = call(tools, think, stream_answer=True)
            calls = [c for c in out["tool_calls"] if c["name"] == "web_search"]
            if not calls:
                break
            ids = [c["id"] or f"call_{len(queries) + i}" for i, c in enumerate(calls)]
            msgs.append({"role": "assistant", "content": out["content"], "reasoning_content": out["reasoning"],
                         "tool_calls": [{"id": call_id, "type": "function",
                                         "function": {"name": c["name"], "arguments": c["arguments"]}}
                                        for call_id, c in zip(ids, calls)]})
            for i, (call_id, c) in enumerate(zip(ids, calls)):
                result = self._run_search(question, c["arguments"], sources, queries, budget_left - i, ev)
                msgs.append({"role": "tool", "tool_call_id": call_id, "content": result})
            if len(queries) >= cfg.max_searches:
                msgs[-1]["content"] += "\n\n" + BUDGET_NOTE
            think = cfg.think_for_answer  # after the first search, the next turn is usually the answer

        text = RAW_TOOL_CALL.sub("", out["content"]).strip()
        if not text:
            text = "(The model produced no answer. Try rephrasing the question.)"
        cited = sorted({int(n) for n in CITATION.findall(text)})
        used = [s for s in sources if s.n in cited]
        self.history += [{"role": "user", "content": question}, {"role": "assistant", "content": text}]
        return Answer(text, used, queries, usage)

    def _run_search(self, question, raw_args, sources, queries, budget_left, ev):
        try:
            query = json.loads(raw_args).get("query", "").strip()
        except (json.JSONDecodeError, AttributeError):
            query = raw_args.strip()
        if not query:
            return "Error: empty query."
        if budget_left <= 0:
            return "Search budget exhausted; this search was not run."
        queries.append(query)
        ev.on_status(f"searching: {query}")
        try:
            hits = search.web_search(query, n=self.cfg.results_per_search)
        except ChallengeError as e:
            ev.on_challenge(e.url)
            return ("Search blocked by a bot check. Tell the user the search engine wants a human check and "
                    "that they can open the link shown to solve it, then ask again.")
        except search.SearchError as e:
            ev.on_status(f"search failed: {e}")
            return f"Search failed ({e}). Tell the user the search did not work."
        ev.on_status(f"reading {min(len(hits), self.cfg.pages_to_fetch)} pages")
        new = sift.sift(question, query, hits, self.cfg, first_n=len(sources) + 1)
        sources.extend(new)
        return sift.format_sources(new)
