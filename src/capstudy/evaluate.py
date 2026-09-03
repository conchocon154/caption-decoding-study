"""Caption metrics: how good, and how varied.

BLEU alone cannot answer the question this study asks.  A decoder that emits
"a man riding a wave on a surfboard" for every beach photo can score well on
BLEU while being useless, so diversity is measured alongside it.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu, sentence_bleu

from .vocab import tokenize

_SMOOTH = SmoothingFunction().method1


def corpus_bleu_scores(references: list[list[str]], hypotheses: list[str]) -> dict[str, float]:
    """Corpus BLEU-1 to BLEU-4 against the five COCO references per image."""
    tokenised_refs = [[tokenize(r) for r in refs] for refs in references]
    tokenised_hyps = [tokenize(h) for h in hypotheses]
    weights = {
        "bleu_1": (1.0, 0, 0, 0),
        "bleu_2": (0.5, 0.5, 0, 0),
        "bleu_3": (1 / 3, 1 / 3, 1 / 3, 0),
        "bleu_4": (0.25, 0.25, 0.25, 0.25),
    }
    return {
        name: float(corpus_bleu(tokenised_refs, tokenised_hyps, weights=weight,
                                smoothing_function=_SMOOTH))
        for name, weight in weights.items()
    }


def per_image_bleu4(references: list[list[str]], hypotheses: list[str]) -> np.ndarray:
    """BLEU-4 for each image separately, for the paired significance test."""
    scores = []
    for refs, hypothesis in zip(references, hypotheses):
        scores.append(
            sentence_bleu([tokenize(r) for r in refs], tokenize(hypothesis),
                          smoothing_function=_SMOOTH)
        )
    return np.array(scores, dtype=float)


def diversity(hypotheses: list[str]) -> dict[str, float]:
    """How much of the output is actually different from the rest of it."""
    tokenised = [tokenize(h) for h in hypotheses]
    unigrams = [token for caption in tokenised for token in caption]
    bigrams = [
        (caption[i], caption[i + 1])
        for caption in tokenised
        for i in range(len(caption) - 1)
    ]
    counts = Counter(hypotheses)
    most_common = counts.most_common(10)
    top10_share = sum(count for _, count in most_common) / max(len(hypotheses), 1)

    return {
        "mean_length": float(np.mean([len(c) for c in tokenised])) if tokenised else 0.0,
        "distinct_1": len(set(unigrams)) / max(len(unigrams), 1),
        "distinct_2": len(set(bigrams)) / max(len(bigrams), 1),
        "unique_caption_pct": 100 * len(counts) / max(len(hypotheses), 1),
        "top10_caption_share_pct": 100 * top10_share,
        "vocabulary_used": len(set(unigrams)),
    }


def evaluate_all(references: list[list[str]], hypotheses: list[str]) -> dict[str, float]:
    scores = corpus_bleu_scores(references, hypotheses)
    scores.update(diversity(hypotheses))
    return scores
