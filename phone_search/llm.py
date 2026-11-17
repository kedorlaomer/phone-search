"""Minimal client for llama-server's OpenAI-compatible chat API (stdlib only, streaming)."""
import json
import urllib.request


class LLMError(Exception):
    pass


# Vendor-recommended sampling for Qwen3.5 (model card), thinking vs. non-thinking.
SAMPLING_THINK = {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "presence_penalty": 1.5}
SAMPLING_NO_THINK = {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "presence_penalty": 1.5}


def chat_stream(url, messages, *, tools=None, think=False, max_tokens=2048, on_content=None, on_reasoning=None):
    """Stream a chat completion. Returns {'content', 'reasoning', 'tool_calls', 'finish', 'usage'}.

    on_content / on_reasoning are called with each text delta as it arrives.
    """
    body = {
        "messages": messages,
        "stream": True,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": think},
        "stream_options": {"include_usage": True},
        **(SAMPLING_THINK if think else SAMPLING_NO_THINK),
    }
    if tools:
        body["tools"] = tools
    req = urllib.request.Request(f"{url}/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    content, reasoning, calls, finish, usage = [], [], {}, None, {}
    try:
        resp = urllib.request.urlopen(req, timeout=600)
    except urllib.error.HTTPError as e:
        raise LLMError(f"HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:300]}") from e
    except OSError as e:
        raise LLMError(f"cannot reach llama-server at {url}: {e}") from e
    with resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            if chunk.get("usage"):
                usage = chunk["usage"]
            for choice in chunk.get("choices", []):
                delta = choice.get("delta", {})
                if delta.get("reasoning_content"):
                    reasoning.append(delta["reasoning_content"])
                    if on_reasoning:
                        on_reasoning(delta["reasoning_content"])
                if delta.get("content"):
                    content.append(delta["content"])
                    if on_content:
                        on_content(delta["content"])
                for tc in delta.get("tool_calls") or []:
                    slot = calls.setdefault(tc.get("index", 0), {"id": "", "name": "", "arguments": ""})
                    slot["id"] = tc.get("id") or slot["id"]
                    fn = tc.get("function") or {}
                    slot["name"] += fn.get("name") or ""
                    slot["arguments"] += fn.get("arguments") or ""
                finish = choice.get("finish_reason") or finish
    return {"content": "".join(content), "reasoning": "".join(reasoning),
            "tool_calls": [calls[k] for k in sorted(calls)], "finish": finish, "usage": usage}


def health(url, timeout=3):
    try:
        with urllib.request.urlopen(f"{url}/health", timeout=timeout) as r:
            return r.status == 200
    except OSError:
        return False
