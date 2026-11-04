"""Answer-run tables and figures for the README: python scripts/report.py [split]

Writes results/answers/<split>/table.md and docs/{f1_vs_cost,gold_breakdown}-{light,dark}.png.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from lexsearch import data  # noqa: E402
from lexsearch.metrics import paired_diff, set_scores  # noqa: E402
from lexsearch.runs import summarize  # noqa: E402

split = sys.argv[1] if len(sys.argv) > 1 else "test"
RUNS = Path("results/answers", split)
FIGURES = Path("docs")
MODELS = {"qwen": "Qwen3 30B-A3B", "haiku": "Claude Haiku 5.5", "sonnet": "Claude Sonnet 5.5"}
RETRIEVERS = {"bm25": "BM25", "BAAI__bge-m3": "bge-m3", "bge-m3-ft": "bge-m3 fine-tuned"}
MODES = {"closed_book": "closed book", "one_shot": "one shot (top 20)", "agent": "agent"}

THEMES = {
    "light": {"surface": "#fcfcfb", "ink": "#0b0b0b", "muted": "#52514e", "grid": "#e1e0d9", "axis": "#c3c2b7",
              "series": ["#2a78d6", "#eb6834", "#1baf7a"], "neutral": "#8a8983", "faint": "#c3c2b7"},
    "dark": {"surface": "#1a1a19", "ink": "#f0efec", "muted": "#c3c2b7", "grid": "#2c2c2a", "axis": "#383835",
             "series": ["#3987e5", "#d95926", "#199e70"], "neutral": "#8a8983", "faint": "#52514e"},
}


def parse(stem: str) -> tuple[str, str, str | None]:
    mode, model, *rest = stem.split("__")
    return mode, model, "__".join(rest) or None


def load() -> dict[str, dict]:
    out = {}
    for path in sorted(RUNS.glob("*.jsonl")):
        rows = [json.loads(line) for line in path.open()]
        if len(rows) < len(data.questions(split)):
            continue  # still running
        out[path.stem] = summarize(path, split)
    return out


def per_question_f1(stem: str) -> dict[str, float]:
    gold = {q.id: q.gold for q in data.questions(split)}
    rows = [json.loads(line) for line in (RUNS / f"{stem}.jsonl").open()]
    return {r["id"]: set_scores(frozenset(r["cited"]), gold[r["id"]])["f1"] for r in rows}


def label(stem: str) -> str:
    mode, model, retriever = parse(stem)
    return f"{MODES[mode]}, {MODELS[model]}" + (f", {RETRIEVERS[retriever]}" if retriever else "")


def table(summaries: dict[str, dict]) -> str:
    head = ("| Mode | Model | Retriever | F1 [95% CI] | Precision | Recall "
            "| Gold cited / shown, not cited / never shown | $ per 1k questions | Median latency | Searches |\n"
            "|---|---|---|---|---|---|---|---|---|---|")
    lines = [head]
    order = sorted(summaries, key=lambda s: (list(MODES).index(parse(s)[0]), list(MODELS).index(parse(s)[1]),
                                             list(RETRIEVERS).index(parse(s)[2]) if parse(s)[2] else -1))
    for stem in order:
        s = summaries[stem]
        mode, model, retriever = parse(stem)
        f1, (lo, hi) = 100 * s["f1"]["mean"], (100 * x for x in s["f1"]["ci"])
        breakdown = (f"{100 * s['gold_cited']:.0f} / {100 * s['gold_shown_not_cited']:.0f} / "
                     f"{100 * s['gold_never_shown']:.0f}")
        lines.append(f"| {MODES[mode]} | {MODELS[model]} | {RETRIEVERS.get(retriever, '—')} | "
                     f"**{f1:.1f}** [{lo:.1f}, {hi:.1f}] | {100 * s['precision']['mean']:.1f} | "
                     f"{100 * s['recall']['mean']:.1f} | {breakdown} | ${s['cost_per_1k']:.2f} | "
                     f"{s['latency_s']:.1f} s | {s['searches']:.1f} |")
    return "\n".join(lines)


def diffs(summaries: dict[str, dict]) -> str:
    """F1 differences on the same questions: agent vs one shot, and fine-tuned vs base retriever."""
    pairs = []
    for model, name in MODELS.items():
        for r, rname in RETRIEVERS.items():
            pairs.append((f"one_shot__{model}__{r}", f"agent__{model}__{r}", f"{name}, {rname}: one shot → agent"))
        for mode in ("one_shot", "agent"):
            pairs.append((f"{mode}__{model}__BAAI__bge-m3", f"{mode}__{model}__bge-m3-ft",
                          f"{name}, {MODES[mode]}: bge-m3 → fine-tuned"))
    pairs.append(("one_shot__haiku__bge-m3-ft", "one_shot__sonnet__bge-m3-ft",
                  "One shot, bge-m3 fine-tuned: Haiku → Sonnet"))
    lines = ["| Change | ΔF1 [95% CI] |\n|---|---|"]
    for a, b, change in pairs:
        if a in summaries and b in summaries:
            fa, fb = per_question_f1(a), per_question_f1(b)
            ids = sorted(fa)
            d, lo, hi = paired_diff(np.array([fa[i] for i in ids]), np.array([fb[i] for i in ids]))
            lines.append(f"| {change} | {100 * d:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}] |")
    return "\n".join(lines)


def _style(ax, t, grid_axis="both"):
    ax.set_facecolor(t["surface"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("bottom", "left"):
        ax.spines[side].set_color(t["axis"])
    ax.tick_params(colors=t["muted"], labelsize=8, length=0)
    ax.grid(axis=grid_axis, color=t["grid"], linewidth=0.8, zorder=0)


def _save(fig, stem: str, mode: str, t) -> Path:
    FIGURES.mkdir(exist_ok=True)
    path = FIGURES / f"{stem}-{mode}.png"
    fig.savefig(path, facecolor=t["surface"])
    plt.close(fig)
    return path


def f1_vs_cost(summaries: dict[str, dict]) -> None:
    """One panel per model: an arrow from one shot to agent for each retriever, closed book as a gray point."""
    models = ["qwen", "haiku"]  # Sonnet has a single run: a labeled point on the Haiku panel
    sonnet = summaries.get("one_shot__sonnet__bge-m3-ft")
    for mode, t in THEMES.items():
        fig, axes = plt.subplots(1, len(models), figsize=(4.4 * len(models), 4.0), dpi=200, sharey=True,
                                 squeeze=False)
        fig.patch.set_facecolor(t["surface"])
        for ax, model in zip(axes[0], models, strict=True):
            _style(ax, t)
            ax.set_title(MODELS[model], color=t["ink"], fontsize=9.5, loc="left")
            closed = summaries.get(f"closed_book__{model}")
            if closed:
                ax.scatter([closed["cost_per_1k"]], [100 * closed["f1"]["mean"]], s=44, color=t["neutral"],
                           edgecolor=t["surface"], linewidth=1.5, zorder=4, label="closed book")
            for i, r in enumerate(RETRIEVERS):
                one, agent = summaries.get(f"one_shot__{model}__{r}"), summaries.get(f"agent__{model}__{r}")
                color = t["series"][i]
                if one:
                    ax.scatter([one["cost_per_1k"]], [100 * one["f1"]["mean"]], s=44, marker="o", color=color,
                               edgecolor=t["surface"], linewidth=1.5, zorder=4, label=RETRIEVERS[r])
                if agent:
                    ax.scatter([agent["cost_per_1k"]], [100 * agent["f1"]["mean"]], s=56, marker="^", color=color,
                               edgecolor=t["surface"], linewidth=1.5, zorder=4)
                if one and agent:
                    ax.annotate("", (agent["cost_per_1k"], 100 * agent["f1"]["mean"]),
                                (one["cost_per_1k"], 100 * one["f1"]["mean"]),
                                arrowprops={"arrowstyle": "-|>", "color": color, "linewidth": 1.2,
                                            "shrinkA": 5, "shrinkB": 6}, zorder=3)
            if model == "haiku" and sonnet:
                x, y = sonnet["cost_per_1k"], 100 * sonnet["f1"]["mean"]
                ax.scatter([x], [y], s=44, marker="D", facecolor=t["surface"], edgecolor=t["series"][2],
                           linewidth=1.5, zorder=4)
                ax.annotate("Sonnet 5.5, one shot,\nbge-m3 fine-tuned", (x, y), xytext=(0, -9),
                            textcoords="offset points", fontsize=7.5, color=t["ink"], ha="center", va="top")
            ax.set_xscale("log")
            lo, hi = ax.get_xlim()
            ticks = [x for x in (0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128) if lo <= x <= hi]
            ax.set_xticks(ticks, [f"${x:g}" for x in ticks])
            ax.minorticks_off()
            ax.set_xlabel("Cost per 1,000 questions ($, log scale)", color=t["muted"], fontsize=8.5)
        axes[0][0].set_ylabel("F1 of cited articles (%)", color=t["muted"], fontsize=8.5)
        axes[0][0].set_ylim(bottom=0)
        handles, labels = axes[0][0].get_legend_handles_labels()
        handles.append(plt.Line2D([], [], marker="o", linestyle="", color=t["muted"]))
        handles.append(plt.Line2D([], [], marker="^", linestyle="", color=t["muted"]))
        labels += ["one shot (top 20)", "agent"]
        legend = fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False, fontsize=8)
        for text in legend.get_texts():
            text.set_color(t["ink"])
        fig.tight_layout(rect=(0, 0.07, 1, 1))
        _save(fig, "f1_vs_cost", mode, t)


def gold_breakdown(summaries: dict[str, dict]) -> None:
    """Where each gold article ends up, for every run that searches."""
    stems = [s for s in summaries if parse(s)[0] != "closed_book"]
    stems.sort(key=lambda s: (list(MODELS).index(parse(s)[1]), list(RETRIEVERS).index(parse(s)[2]),
                              list(MODES).index(parse(s)[0])))
    parts = [("gold_cited", "cited"), ("gold_shown_not_cited", "shown, not cited (reading)"),
             ("gold_never_shown", "never shown (search)")]
    for mode, t in THEMES.items():
        colors = [t["series"][0], t["faint"], t["neutral"]]
        fig, ax = plt.subplots(figsize=(8.4, 0.32 * len(stems) + 1.2), dpi=200)
        fig.patch.set_facecolor(t["surface"])
        _style(ax, t, grid_axis="x")
        ax.spines["left"].set_visible(False)
        y = np.arange(len(stems))[::-1]
        left = np.zeros(len(stems))
        for (key, name), color in zip(parts, colors, strict=True):
            values = np.array([100 * summaries[s][key] for s in stems])
            ax.barh(y, values, left=left, height=0.62, color=color, edgecolor=t["surface"], linewidth=1.5,
                    label=name, zorder=3)
            for yi, xi, v in zip(y, left, values, strict=True):
                if v >= 7:
                    ax.text(xi + v / 2, yi, f"{v:.0f}", ha="center", va="center", fontsize=7,
                            color="#ffffff" if key == "gold_cited" else t["ink"], zorder=4)
            left += values
        ax.set_yticks(y, [label(s) for s in stems], fontsize=7.5, color=t["ink"])
        ax.set_xlim(0, 100)
        ax.set_xlabel("Gold articles (%)", color=t["muted"], fontsize=8.5)
        legend = ax.legend(frameon=False, fontsize=8, ncol=3, loc="lower left", bbox_to_anchor=(0, 1.0))
        for text in legend.get_texts():
            text.set_color(t["ink"])
        fig.tight_layout()
        _save(fig, "gold_breakdown", mode, t)


summaries = load()
md = table(summaries) + "\n\n" + diffs(summaries) + "\n"
(RUNS / "table.md").write_text(md)
print(md)
f1_vs_cost(summaries)
gold_breakdown(summaries)
