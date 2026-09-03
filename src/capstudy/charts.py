"""Figures for the report."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

INK = "#1f2933"
ACCENT = "#1d4ed8"
WARM = "#c2410c"
MUTED = "#94a3b8"


def _style(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(colors=INK, labelsize=8)
    ax.grid(axis="y", color=MUTED, alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)


def quality_vs_diversity(results: pd.DataFrame, out: Path) -> Path:
    """The trade-off, on one pair of axes: BLEU-4 against caption diversity.

    Diversity declines monotonically with beam width but quality does not rise
    monotonically, so the points do not form a frontier - the widest beam sits
    below and to the left of beam 5, worse on both axes.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(results["distinct_2"], results["bleu_4"], color=MUTED,
            linewidth=1, linestyle=":", zorder=1)
    ax.scatter(results["distinct_2"], results["bleu_4"], s=70, color=ACCENT, zorder=2)
    for _, row in results.iterrows():
        ax.annotate(row["strategy"], (row["distinct_2"], row["bleu_4"]),
                    textcoords="offset points", xytext=(7, 4), fontsize=8, color=INK)
    _style(ax)
    ax.set_xlabel("Distinct-2 (higher = more varied wording)", fontsize=8)
    ax.set_ylabel("BLEU-4", fontsize=8)
    ax.set_title("Quality peaks at beam 5; diversity falls all the way",
                 fontsize=10, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def bleu_by_strategy(results: pd.DataFrame, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 3.6))
    width = 0.2
    positions = range(len(results))
    for offset, (column, label) in enumerate(
            [("bleu_1", "BLEU-1"), ("bleu_2", "BLEU-2"),
             ("bleu_3", "BLEU-3"), ("bleu_4", "BLEU-4")]):
        ax.bar([p + offset * width for p in positions], results[column],
               width=width, label=label)
    _style(ax)
    ax.set_xticks([p + 1.5 * width for p in positions], results["strategy"], fontsize=8)
    ax.set_ylabel("Score", fontsize=8)
    ax.set_title("BLEU by decoding strategy", fontsize=10, color=INK, loc="left")
    ax.legend(fontsize=7, frameon=False, ncols=4)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def length_and_repetition(results: pd.DataFrame, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    ax.bar(results["strategy"], results["unique_caption_pct"], color=ACCENT, width=0.55)
    for index, value in enumerate(results["unique_caption_pct"]):
        ax.text(index, value, f"{value:.0f}%", ha="center", va="bottom",
                fontsize=8, color=INK)
    _style(ax)
    ax.set_ylabel("Captions that are unique (%)", fontsize=8)
    ax.set_ylim(0, 105)

    twin = ax.twinx()
    twin.plot(results["strategy"], results["mean_length"], color=WARM,
              marker="o", linewidth=1.8)
    twin.set_ylabel("Mean caption length (words)", fontsize=8, color=WARM)
    twin.tick_params(colors=WARM, labelsize=8)
    twin.spines[["top", "left"]].set_visible(False)

    ax.set_title("Uniqueness falls and captions shorten as the beam widens",
                 fontsize=10, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def training_curve(history: pd.DataFrame, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.plot(history["epoch"], history["train_loss"], color=ACCENT,
            linewidth=1.8, label="train")
    ax.plot(history["epoch"], history["val_loss"], color=WARM,
            linewidth=1.8, label="validation")
    _style(ax)
    ax.set_xlabel("Epoch", fontsize=8)
    ax.set_ylabel("Cross-entropy loss", fontsize=8)
    ax.set_title("Decoder training", fontsize=10, color=INK, loc="left")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out
