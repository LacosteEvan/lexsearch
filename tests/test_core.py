import json
from types import SimpleNamespace

import numpy as np
import pytest

from lexsearch import answer, data
from lexsearch.llm import Chat, parse_json
from lexsearch.metrics import average_precision, ndcg_at, paired_diff, recall_at, set_scores
from lexsearch.retrievers import rrf

GOLD = frozenset({"a", "b"})


def test_ranking_metrics():
    ranked = ["a", "x", "b", "y"]
    assert recall_at(ranked, GOLD, 1) == 0.5
    assert recall_at(ranked, GOLD, 3) == 1.0
    assert average_precision(ranked, GOLD) == pytest.approx((1 + 2 / 3) / 2)
    assert ndcg_at(["a", "b"], GOLD) == pytest.approx(1.0)
    assert ndcg_at(["x", "y"], GOLD) == 0.0


def test_set_scores():
    assert set_scores(frozenset({"a", "z"}), GOLD) == {"precision": 0.5, "recall": 0.5, "f1": 0.5}
    assert set_scores(frozenset(), GOLD)["f1"] == 0.0


def test_paired_diff_sign():
    a, b = np.zeros(50), np.ones(50)
    mean, lo, hi = paired_diff(a, b)
    assert mean == 1.0 and lo == hi == 1.0


def test_rrf_rewards_agreement():
    assert rrf([["a", "b", "c"], ["b", "a", "d"]])[:2] in (["a", "b"], ["b", "a"])
    assert rrf([["a", "b"], ["c", "b"]])[0] == "b"


def test_dev_split_is_stable():
    shares = [data._hash_share(str(i)) for i in range(1000)]
    assert all(0 <= s <= 1 for s in shares)
    assert 0.15 < np.mean([s < data.DEV_SHARE for s in shares]) < 0.25
    assert data._hash_share("42") == data._hash_share("42")


def test_parse_json_tolerates_fences():
    assert parse_json('Voici:\n```json\n{"article_ids": ["1"]}\n```') == {"article_ids": ["1"]}


def test_parse_json_keeps_ids_around_bad_text():
    assert parse_json('{"article_ids": ["1"]}\nNote : voir aussi {art. 2}.') == {"article_ids": ["1"]}
    broken = '{"article_ids": ["1", "2"], "answer": "le "bail" est résilié"}'
    assert parse_json(broken) == {"article_ids": ["1", "2"], "answer": ""}


def test_paid_run_needs_a_cap():
    with pytest.raises(RuntimeError):
        Chat("haiku", max_cost=None)


ARTICLES = {
    "1": data.Article("1", "Art. 1180, Code Civil (Titre III)", "Titre III", "texte un"),
    "2": data.Article("2", "Art. VII.216/129, Code de Droit Economique (Livre VII)", "Livre VII", "texte deux"),
}


def test_resolver_matches_loose_citations():
    r = answer.Resolver(ARTICLES)
    assert r("Code civil", "art. 1180") == "1"
    assert r("code de droit économique", "VII.216/129") == "2"
    assert r("Code civil", "9") is None
    assert r("Code de la route", "1180") is None


class FakeRetriever:
    name = "fake"

    def search(self, queries, k):
        return [["1", "2"][:k] for _ in queries]


def _call(name, args, i):
    return SimpleNamespace(id=f"c{i}", function=SimpleNamespace(name=name, arguments=json.dumps(args)))


class ScriptedChat:
    """Plays a fixed list of assistant turns."""

    def __init__(self, turns):
        self.turns, self.seen = list(turns), []

    def __call__(self, messages, tools=None, max_tokens=4000):
        self.seen.append(messages)
        return self.turns.pop(0), 0.001, 100, 10


def test_agent_loop_records_what_it_saw(monkeypatch):
    monkeypatch.setattr(data, "articles", lambda: ARTICLES)
    chat = ScriptedChat([
        SimpleNamespace(content="", tool_calls=[_call("search", {"query": "dettes"}, 0)]),
        SimpleNamespace(content="", tool_calls=[_call("read", {"ids": ["2", "999"]}, 1)]),
        SimpleNamespace(content="", tool_calls=[_call("answer", {"article_ids": ["2", "999"], "answer": "oui"}, 2)]),
    ])
    res = answer.Answerer(chat, FakeRetriever()).agent(data.Question("q", "Dois-je payer ?", "", frozenset({"2"})))
    assert res.cited == ["2"]  # unknown ids are dropped
    assert res.shown == ["1", "2"]
    assert res.queries == ["dettes"] and res.steps == 3
    assert res.cost_usd == pytest.approx(0.003)
    tool_msgs = [m for m in chat.seen[-1] if m["role"] == "tool"]
    assert "texte deux" in tool_msgs[1]["content"]


def test_agent_answer_in_plain_json(monkeypatch):
    monkeypatch.setattr(data, "articles", lambda: ARTICLES)
    chat = ScriptedChat([SimpleNamespace(content='{"article_ids": ["1"], "answer": "x"}', tool_calls=None)])
    res = answer.Answerer(chat, FakeRetriever()).agent(data.Question("q", "?", "", frozenset({"1"})))
    assert res.cited == ["1"] and res.error is None


def test_one_shot_only_cites_shown(monkeypatch):
    monkeypatch.setattr(data, "articles", lambda: ARTICLES)
    chat = ScriptedChat([SimpleNamespace(content='{"article_ids": ["1", "3"], "answer": "x"}', tool_calls=None)])
    res = answer.Answerer(chat, FakeRetriever(), depth=1).one_shot(data.Question("q", "?", "", frozenset({"1"})))
    assert res.cited == ["1"] and res.shown == ["1"]


def test_agent_prose_answer_is_nudged_once(monkeypatch):
    monkeypatch.setattr(data, "articles", lambda: ARTICLES)
    chat = ScriptedChat([
        SimpleNamespace(content="Oui, selon l'article 1180.", tool_calls=None),
        SimpleNamespace(content="", tool_calls=[_call("answer", {"article_ids": ["1"], "answer": "oui"}, 0)]),
    ])
    res = answer.Answerer(chat, FakeRetriever()).agent(data.Question("q", "?", "", frozenset({"1"})))
    assert res.cited == ["1"] and res.error is None and res.steps == 2
