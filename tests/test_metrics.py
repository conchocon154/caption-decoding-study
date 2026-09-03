import numpy as np

from capstudy.evaluate import corpus_bleu_scores, diversity, per_image_bleu4
from capstudy.stats import paired_bootstrap, win_rate


def test_perfect_match_scores_near_one():
    refs = [["a cat on a table"], ["a dog in a park"]]
    scores = corpus_bleu_scores(refs, ["a cat on a table", "a dog in a park"])
    assert scores["bleu_1"] > 0.99
    assert scores["bleu_4"] > 0.9


def test_unrelated_output_scores_near_zero():
    refs = [["a cat on a table"]]
    scores = corpus_bleu_scores(refs, ["completely different words entirely"])
    assert scores["bleu_4"] < 0.05


def test_bleu_rises_with_overlap():
    refs = [["a man riding a wave on a surfboard"]]
    poor = corpus_bleu_scores(refs, ["a dog"])["bleu_2"]
    better = corpus_bleu_scores(refs, ["a man riding a wave"])["bleu_2"]
    assert better > poor


def test_diversity_detects_a_repetitive_decoder():
    identical = diversity(["a man riding a wave"] * 100)
    varied = diversity([f"a man number {i} riding" for i in range(100)])
    assert identical["unique_caption_pct"] == 1.0
    assert varied["unique_caption_pct"] == 100.0
    assert varied["distinct_2"] > identical["distinct_2"]


def test_top10_share_is_one_hundred_when_output_collapses():
    scores = diversity(["a cat sat"] * 50)
    assert scores["top10_caption_share_pct"] == 100.0


def test_per_image_bleu_returns_one_score_per_image():
    refs = [["a cat on a table"], ["a dog in a park"], ["a bird in a tree"]]
    scores = per_image_bleu4(refs, ["a cat on a table", "a dog", "a bird in a tree"])
    assert len(scores) == 3
    assert scores[0] > scores[1]


def test_bootstrap_finds_no_difference_between_identical_systems():
    scores = np.random.default_rng(0).random(300)
    result = paired_bootstrap(scores, scores.copy(), rounds=500)
    assert result["observed_difference"] == 0.0
    assert not result["significant"]


def test_bootstrap_detects_a_real_improvement():
    rng = np.random.default_rng(1)
    baseline = rng.random(500)
    contender = baseline + 0.05
    result = paired_bootstrap(baseline, contender, rounds=500)
    assert result["significant"]
    assert result["ci_low"] > 0


def test_win_rate_counts_ties():
    baseline = np.array([0.1, 0.2, 0.3])
    contender = np.array([0.2, 0.2, 0.1])
    result = win_rate(baseline, contender)
    assert result["contender_wins"] == 1
    assert result["baseline_wins"] == 1
    assert result["ties"] == 1
