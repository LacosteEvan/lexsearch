"""Answer a citizen's question by citing the articles that answer it. Three ways:

- closed book: the model cites articles from memory, no search. Measures what it already knows.
- one-shot RAG: a retriever returns the top articles, the model reads them all once and picks.
- agent: the model searches as many times as it wants (rephrasing in legal terms), reads what it chooses, then answers.

Every answer is scored on the articles it cites against the lawyers' labels. The articles the model was shown are
recorded too, which splits each miss in two: never found (search) or found but not cited (reading).
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict, dataclass, field

from . import data
from .llm import Chat, parse_json

ARTICLE_WORDS = 400  # 90% of articles are shorter than 310 words
SNIPPET_WORDS = 40
MAX_STEPS = 12

TASK = """You help people in Belgium find the law that applies to their situation. Given a question asked by a
citizen, find the statutory articles that answer it, from a corpus of 22,633 articles of Belgian codes (federal,
Brussels, Wallonia), in French.

Cite every article that answers part of the question, and only those: an article on the right topic that does not
answer this question is a wrong citation."""

ANSWER_FORMAT = """Answer with one JSON object and nothing else:
{"article_ids": ["<id>", ...], "answer": "<two or three sentences in French, citing the articles>"}"""

CLOSED_BOOK = TASK + """

You have no search tool: cite the articles from memory, by code and article number.
Answer with one JSON object and nothing else:
{"articles": [{"code": "<name of the code>", "article": "<article number>"}, ...], "answer": "<in French>"}"""

ONE_SHOT = TASK + """

A search engine returned the articles below, best first. Pick from them only.

""" + ANSWER_FORMAT

AGENT = TASK + """

Tools: `search` returns the 10 best articles for a query, with a short snippet; `read` returns full articles.
Citizens use everyday words and the codes use legal ones: if a search misses, rephrase it the way a statute would
say it, or search each part of the question separately. Read an article before citing it. When you are done, call
`answer`. You have at most """ + str(MAX_STEPS) + " tool calls."

TOOLS = [
    {"type": "function", "function": {
        "name": "search", "description": "Search the statutes. Returns the 10 best articles with a snippet.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "read", "description": "Full text of up to 5 articles, by id.",
        "parameters": {"type": "object", "properties": {"ids": {"type": "array", "items": {"type": "string"}}},
                       "required": ["ids"]}}},
    {"type": "function", "function": {
        "name": "answer", "description": "Final answer: the ids of the articles that answer the question.",
        "parameters": {"type": "object", "properties": {
            "article_ids": {"type": "array", "items": {"type": "string"}},
            "answer": {"type": "string", "description": "two or three sentences in French"}},
            "required": ["article_ids", "answer"]}}},
]


@dataclass
class Answer:
    id: str
    cited: list[str]
    answer: str
    shown: list[str]  # every article id the model saw, in a search result or in full
    cost_usd: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    latency_s: float = 0.0
    steps: int = 0
    queries: list[str] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def full(a: data.Article) -> str:
    words = a.text.split()
    text = " ".join(words[:ARTICLE_WORDS]) + (" […]" if len(words) > ARTICLE_WORDS else "")
    return f'<article id="{a.id}">\n{a.reference}\n{a.path}\n{text}\n</article>'


def snippet(a: data.Article) -> str:
    words = a.text.split()
    text = " ".join(words[:SNIPPET_WORDS]) + (" …" if len(words) > SNIPPET_WORDS else "")
    return f"[{a.id}] {a.reference} | {a.path}\n    {text}"


def _valid(ids, corpus: dict) -> list[str]:
    return list(dict.fromkeys(str(i).strip() for i in ids if str(i).strip() in corpus))


class Answerer:
    def __init__(self, chat: Chat, retriever=None, depth: int = 20) -> None:
        self.chat, self.retriever, self.depth = chat, retriever, depth
        self.corpus = data.articles()
        self._search_lock = threading.Lock()  # embedding models on one device: one query at a time

    def _search(self, query: str, k: int) -> list[str]:
        with self._search_lock:
            return self.retriever.search([query], k)[0]

    def one_shot(self, q: data.Question) -> Answer:
        start = time.monotonic()
        shown = self._search(q.text, self.depth)
        context = "\n\n".join(full(self.corpus[a]) for a in shown)
        messages = [{"role": "system", "content": ONE_SHOT},
                    {"role": "user", "content": f"Question: {q.text}\n\nArticles:\n\n{context}"}]
        msg, cost, i, o = self.chat(messages)
        try:
            out = parse_json(msg.content or "")
            return Answer(q.id, _valid(out.get("article_ids", []), set(shown)), out.get("answer", ""), shown, cost, i,
                          o, time.monotonic() - start, 1)
        except (ValueError, json.JSONDecodeError) as e:
            return Answer(q.id, [], "", shown, cost, i, o, time.monotonic() - start, 1, error=str(e)[:200])

    def agent(self, q: data.Question) -> Answer:
        start = time.monotonic()
        res = Answer(q.id, [], "", [])
        messages = [{"role": "system", "content": AGENT}, {"role": "user", "content": f"Question: {q.text}"}]
        nudged = False
        for step in range(MAX_STEPS + 1):
            last = step == MAX_STEPS
            if last:
                messages.append({"role": "user", "content": "No tool calls left: call `answer` now."})
            msg, cost, i, o = self.chat(messages, tools=TOOLS)
            res.cost_usd, res.tokens_in, res.tokens_out, res.steps = (res.cost_usd + cost, res.tokens_in + i,
                                                                      res.tokens_out + o, step + 1)
            calls = msg.tool_calls or []
            if not calls:
                try:  # some models answer in plain text instead of calling the tool
                    out = parse_json(msg.content or "")
                    res.cited, res.answer = _valid(out.get("article_ids", []), self.corpus), out.get("answer", "")
                except (ValueError, json.JSONDecodeError):
                    if not nudged:  # a prose answer: ask once for the tool call that carries the article ids
                        nudged = True
                        messages += [{"role": "assistant", "content": msg.content or ""},
                                     {"role": "user", "content": "Call `answer` with the ids of the articles."}]
                        continue
                    res.error = "no tool call and no JSON answer"
                break
            messages.append({"role": "assistant", "content": msg.content or "", "tool_calls": [
                {"id": c.id, "type": "function", "function": {"name": c.function.name,
                                                               "arguments": c.function.arguments}} for c in calls]})
            done = False
            for c in calls:
                try:
                    args = json.loads(c.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                name = c.function.name
                if name == "answer":
                    res.cited, res.answer, done = _valid(args.get("article_ids", []), self.corpus), \
                        args.get("answer", ""), True
                    result = "ok"
                elif name == "search" and not last:
                    query = str(args.get("query", ""))[:500]
                    hits = self._search(query, 10)
                    res.queries.append(query)
                    res.shown += hits
                    result = "\n".join(snippet(self.corpus[a]) for a in hits)
                elif name == "read" and not last:
                    ids = _valid(args.get("ids", []), self.corpus)[:5]
                    res.shown += ids
                    result = "\n\n".join(full(self.corpus[a]) for a in ids) or "No article with these ids."
                else:
                    result = "Not available."
                messages.append({"role": "tool", "tool_call_id": c.id, "content": result})
            if done:
                break
        res.shown = list(dict.fromkeys(res.shown))
        res.latency_s = time.monotonic() - start
        return res

    def closed_book(self, q: data.Question, resolve) -> Answer:
        start = time.monotonic()
        messages = [{"role": "system", "content": CLOSED_BOOK}, {"role": "user", "content": f"Question: {q.text}"}]
        msg, cost, i, o = self.chat(messages)
        try:
            out = parse_json(msg.content or "")
            cited = [a for c in out.get("articles", []) if (a := resolve(c.get("code", ""), c.get("article", "")))]
            return Answer(q.id, list(dict.fromkeys(cited)), out.get("answer", ""), [], cost, i, o,
                          time.monotonic() - start, 1)
        except (ValueError, json.JSONDecodeError, AttributeError) as e:
            return Answer(q.id, [], "", [], cost, i, o, time.monotonic() - start, 1, error=str(e)[:200])


def _norm_number(s: str) -> str:
    s = str(s).lower().replace("article", "").replace("art.", "").replace("art ", "")
    return "".join(s.split()).rstrip(".")


class Resolver:
    """Maps a citation given from memory ("Code Civil", "1180") to an article id: the closest code name, then the
    exact article number in that code."""

    def __init__(self, corpus: dict[str, data.Article]) -> None:
        import re

        self.index: dict[tuple[str, str], str] = {}
        for a in corpus.values():
            m = re.match(r"^Art\. ?([^,]+), (.+?)(?: \(|$)", a.reference)
            if m:
                self.index.setdefault((m.group(2).strip().lower(), _norm_number(m.group(1))), a.id)
        self.codes = sorted({code for code, _ in self.index})

    def __call__(self, code: str, number: str) -> str | None:
        import difflib

        match = difflib.get_close_matches(str(code).strip().lower(), self.codes, n=1, cutoff=0.6)
        return self.index.get((match[0], _norm_number(number))) if match else None
