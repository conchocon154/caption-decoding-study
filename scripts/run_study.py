"""Train the decoder, decode the test split four ways, and compare the results."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402
import torch  # noqa: E402
from tqdm import tqdm  # noqa: E402

from capstudy import charts, features as feature_tools  # noqa: E402
from capstudy.data import Split, load_annotations  # noqa: E402
from capstudy.evaluate import evaluate_all, per_image_bleu4  # noqa: E402
from capstudy.stats import paired_bootstrap, win_rate  # noqa: E402
from capstudy.train import TrainConfig, load_decoder, train_decoder  # noqa: E402
from capstudy.vocab import Vocabulary  # noqa: E402

DATA = ROOT / "data"
PROCESSED = DATA / "processed"
REPORTS = ROOT / "reports"

# Greedy decoding is beam search with a width of one; it is listed separately
# because it is the baseline every wider beam has to beat.
STRATEGIES = [
    ("greedy", 1),
    ("beam-3", 3),
    ("beam-5", 5),
    ("beam-10", 10),
]


def decode_test_set(model, features, feature_index, image_ids, vocab,
                    beam_width: int, max_len: int, device, limit: int | None):
    """Produce one caption per test image with the given strategy."""
    rows = [feature_index[i] for i in image_ids if i in feature_index]
    used_ids = [i for i in image_ids if i in feature_index]
    if limit:
        rows, used_ids = rows[:limit], used_ids[:limit]
    matrix = features[rows].to(device)

    if beam_width == 1:
        token_ids = model.greedy(matrix, vocab.start_id, vocab.end_id, max_len)
        return used_ids, [vocab.decode(ids) for ids in token_ids]

    captions = []
    for row in tqdm(range(matrix.size(0)), desc=f"beam-{beam_width}", unit="img",
                    leave=False):
        ids = model.beam_search(matrix[row], vocab.start_id, vocab.end_id,
                                beam_width=beam_width, max_len=max_len)
        captions.append(vocab.decode(ids))
    return used_ids, captions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--limit-test", type=int, default=1500,
                        help="images to decode per strategy; beam search is slow")
    parser.add_argument("--retrain", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--decode-device", default="cpu",
                        help="beam search runs on CPU by default: it is a "
                             "sequential loop over tiny tensors, and each "
                             "score read back from the GPU costs a sync that "
                             "dwarfs the arithmetic. Measured 5x faster than MPS.")
    args = parser.parse_args()

    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "charts").mkdir(exist_ok=True)
    (REPORTS / "tables").mkdir(exist_ok=True)

    vocab = Vocabulary.load(PROCESSED / "vocab.json")
    split_ids = json.loads((PROCESSED / "splits.json").read_text(encoding="utf-8"))
    splits = {name: Split(name, ids) for name, ids in split_ids.items()}
    features, feature_index = feature_tools.load(PROCESSED / "features.pt")
    _, captions = load_annotations(DATA / "annotations" / "captions_val2014.json")
    print(f"features {tuple(features.shape)}, vocabulary {len(vocab):,}")

    checkpoint = ROOT / "models" / "decoder.pt"
    config = TrainConfig(epochs=args.epochs, seed=args.seed)

    if args.retrain or not checkpoint.exists():
        def report(epoch, train_loss, val_loss):
            print(f"  epoch {epoch:2d}  train {train_loss:.4f}  val {val_loss:.4f}")

        print("training decoder ...")
        _, history, best = train_decoder(features, feature_index, captions, splits,
                                         vocab, config, checkpoint, progress=report)
        pd.DataFrame(history).to_csv(REPORTS / "tables" / "training_history.csv", index=False)
        charts.training_curve(pd.DataFrame(history), REPORTS / "charts" / "training.png")
        print(f"best validation loss {best:.4f}")

    device = torch.device(args.decode_device)
    model = load_decoder(checkpoint, device)

    references, results, per_image, examples = None, [], {}, {}
    for name, width in STRATEGIES:
        started = time.perf_counter()
        used_ids, hypotheses = decode_test_set(
            model, features, feature_index, splits["test"].image_ids, vocab,
            width, config.max_len, device, args.limit_test)
        elapsed = time.perf_counter() - started

        if references is None:
            references = [captions[i] for i in used_ids]
        scores = evaluate_all(references, hypotheses)
        scores.update({"strategy": name, "beam_width": width,
                       "seconds_total": round(elapsed, 1),
                       "ms_per_image": round(1000 * elapsed / len(hypotheses), 1)})
        results.append(scores)
        per_image[name] = per_image_bleu4(references, hypotheses)
        examples[name] = hypotheses[:8]
        print(f"  {name:8} BLEU-4 {scores['bleu_4']:.4f}  distinct-2 {scores['distinct_2']:.4f}"
              f"  unique {scores['unique_caption_pct']:.1f}%  {scores['ms_per_image']:.1f} ms/img")

    frame = pd.DataFrame(results)[
        ["strategy", "beam_width", "bleu_1", "bleu_2", "bleu_3", "bleu_4",
         "mean_length", "distinct_1", "distinct_2", "unique_caption_pct",
         "top10_caption_share_pct", "vocabulary_used", "ms_per_image"]
    ]
    frame.to_csv(REPORTS / "tables" / "decoding_results.csv", index=False)

    # Every strategy against greedy, plus the pair that decides the
    # recommendation: if BLEU peaks mid-range, the drop from the best beam to
    # the widest one has to be tested directly rather than inferred from two
    # separate comparisons against a third system.
    best = max(STRATEGIES, key=lambda pair: float(
        next(r["bleu_4"] for r in results if r["strategy"] == pair[0])))[0]
    pairs = [("greedy", name) for name, _ in STRATEGIES[1:]]
    widest = STRATEGIES[-1][0]
    if best != widest:
        pairs.append((best, widest))

    comparisons = []
    for baseline_name, contender_name in pairs:
        test = paired_bootstrap(per_image[baseline_name], per_image[contender_name],
                                seed=args.seed)
        test.update(win_rate(per_image[baseline_name], per_image[contender_name]))
        test["baseline"] = baseline_name
        test["strategy"] = contender_name
        comparisons.append(test)
        print(f"  {contender_name:8} vs {baseline_name:8}: {test['observed_difference']:+.4f} "
              f"BLEU-4 CI[{test['ci_low']:+.4f},{test['ci_high']:+.4f}] "
              f"p={test['p_value']:.4f} win {test['contender_win_pct']}%")
    comparison_frame = pd.DataFrame(comparisons)[
        ["baseline", "strategy", "observed_difference", "ci_low", "ci_high",
         "p_value", "significant", "contender_win_pct", "identical_output_pct"]
    ]
    comparison_frame.to_csv(REPORTS / "tables" / "significance.csv", index=False)

    charts.quality_vs_diversity(frame, REPORTS / "charts" / "tradeoff.png")
    charts.bleu_by_strategy(frame, REPORTS / "charts" / "bleu.png")
    charts.length_and_repetition(frame, REPORTS / "charts" / "diversity.png")

    (REPORTS / "results.json").write_text(json.dumps({
        "test_images": len(references),
        "vocabulary": len(vocab),
        "splits": {k: len(v.image_ids) for k, v in splits.items()},
        "results": frame.to_dict("records"),
        "significance": comparison_frame.to_dict("records"),
        "examples": examples,
    }, indent=2), encoding="utf-8")
    print(f"\nresults -> {REPORTS / 'results.json'}")


if __name__ == "__main__":
    main()
