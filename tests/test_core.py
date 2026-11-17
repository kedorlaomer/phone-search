"""Offline tests: sifting, search-result cleanup and the agent loop (LLM, search and fetch are faked)."""
import json
import unittest
from unittest import mock

from phone_search import agent, search, sift
from phone_search.config import Config


class SiftTests(unittest.TestCase):
    def test_split_drops_reference_lines(self):
        text = "Everest is 8,848.86 m tall.\n- \u2191 Krakauer, Jon (1997). Retrieved 2 June 2023.\nMore text."
        out = sift.split_passages(text, 600)
        self.assertEqual(out, ["Everest is 8,848.86 m tall.\nMore text."])

    def test_split_packs_and_caps(self):
        paras = "\n".join(["x" * 250] * 5)
        out = sift.split_passages(paras, 600)
        self.assertTrue(all(len(p) <= 1200 for p in out))
        self.assertEqual(sum(p.count("x") for p in out), 1250)

    def test_sift_ranks_and_numbers(self):
        cfg = Config()
        hits = [search.Hit("A", "https://a", "a"), search.Hit("B", "https://b", "b")]
        pages = {"https://a": "Bananas are yellow.\n" * 30, "https://b": "Mount Everest is 8,848.86 metres high.\n" * 30}
        with mock.patch.object(sift, "fetch_text", side_effect=lambda u, t: pages[u]), \
             mock.patch.object(sift, "fetch_text_browser", return_value=None):
            out = sift.sift("How tall is Mount Everest?", "Everest height", hits, cfg, first_n=4)
        self.assertEqual([s.n for s in out], [4, 5][:len(out)])
        self.assertIn("Everest", "".join(p for s in out for p in s.passages))

    def test_sift_falls_back_to_snippet_then_browser(self):
        cfg = Config()
        hits = [search.Hit("A", "https://a", "snippet about everest height")]
        with mock.patch.object(sift, "fetch_text", return_value=None), \
             mock.patch.object(sift, "fetch_text_browser", return_value=None) as fb:
            out = sift.sift("everest height", "everest", hits, cfg)
        fb.assert_called_once()
        self.assertEqual(out[0].passages, ["snippet about everest height"])


class SearchTests(unittest.TestCase):
    def test_unwrap_bing(self):
        wrapped = "https://www.bing.com/ck/a?!&&p=x&u=a1aHR0cHM6Ly9leGFtcGxlLmNvbS9wYWdl&ntb=1"
        self.assertEqual(search._unwrap(wrapped), "https://example.com/page")

    def test_engine_filters_ads_and_dupes(self):
        rows = [{"title": "Ad", "url": "https://duckduckgo.com/y.js?ad", "snippet": ""},
                {"title": " Real \n title ", "url": "https://x.org/a", "snippet": "s"},
                {"title": "Dupe", "url": "https://x.org/a", "snippet": "s"}]
        with mock.patch.object(search, "shared") as sh:
            sh.return_value.run.return_value = rows
            hits = search._engine(search.DDG, "q", 5)
        self.assertEqual([(h.title, h.url) for h in hits], [("Real title", "https://x.org/a")])


def reply(content="", calls=(), reasoning=""):
    return {"content": content, "reasoning": reasoning, "finish": "tool_calls" if calls else "stop",
            "tool_calls": [{"id": f"c{i}", "name": "web_search", "arguments": json.dumps({"query": q})}
                           for i, q in enumerate(calls)],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5}}


class AgentTests(unittest.TestCase):
    def setUp(self):
        self.cfg = Config()
        self.cfg.max_searches = 2
        mock.patch.object(sift, "sift", side_effect=lambda q, query, hits, cfg, first_n=1:
                          [sift.Source(first_n, "T", f"https://t/{first_n}", ["passage"])]).start()
        mock.patch.object(search, "web_search", return_value=[search.Hit("T", "https://t", "s")]).start()
        self.addCleanup(mock.patch.stopall)

    def run_agent(self, replies):
        calls = []

        def fake_chat(url, msgs, tools=None, think=False, **kw):
            calls.append({"msgs": [dict(m) for m in msgs], "tools": tools, "think": think})
            out = replies.pop(0)
            if kw.get("on_content") and out["content"]:
                kw["on_content"](out["content"])
            return out

        with mock.patch("phone_search.llm.chat_stream", side_effect=fake_chat):
            ans = agent.Agent(self.cfg).ask("q?")
        return ans, calls

    def test_direct_answer_no_search(self):
        ans, calls = self.run_agent([reply("4")])
        self.assertEqual(ans.text, "4")
        self.assertEqual(ans.searches, [])
        self.assertTrue(calls[0]["think"])  # decision turn thinks

    def test_search_then_answer_with_citation(self):
        ans, calls = self.run_agent([reply(calls=["everest"]), reply("8,848.86 m [1].")])
        self.assertEqual(ans.searches, ["everest"])
        self.assertEqual([s.n for s in ans.sources], [1])
        self.assertFalse(calls[1]["think"])  # answer turn doesn't
        self.assertEqual(calls[1]["msgs"][-1]["role"], "tool")

    def test_budget_exhausted_forces_answer_without_tools(self):
        ans, calls = self.run_agent([reply(calls=["a"]), reply(calls=["b"]), reply("done [2]")])
        self.assertEqual(ans.searches, ["a", "b"])
        self.assertIsNone(calls[2]["tools"])
        self.assertIn("Search budget exhausted", calls[2]["msgs"][-1]["content"])
        self.assertEqual([s.n for s in ans.sources], [2])

    def test_raw_tool_markup_is_stripped(self):
        ans, _ = self.run_agent([reply("<tool_call>\n<function=web_search>x</function>\n</tool_call>")])
        self.assertIn("no answer", ans.text)

    def test_history_keeps_only_final_answers(self):
        a = agent.Agent(self.cfg)
        with mock.patch("phone_search.llm.chat_stream", side_effect=[reply(calls=["x"]), reply("one"), reply("two")]):
            a.ask("first")
            a.ask("second")
        self.assertEqual([m["role"] for m in a.history], ["user", "assistant", "user", "assistant"])
        self.assertEqual(a.history[1]["content"], "one")


if __name__ == "__main__":
    unittest.main()
