import torch

from capstudy.model import CaptionDecoder


def test_forward_predicts_one_step_per_input_token(tiny_model, vocab):
    features = torch.randn(4, 16)
    captions = torch.randint(0, len(vocab), (4, 7))
    logits = tiny_model(features, captions)
    # The image vector replaces the last caption token, so the sequence length
    # stays the same: 1 image token + (7 - 1) words.
    assert logits.shape == (4, 7, len(vocab))


def test_greedy_returns_one_caption_per_image(tiny_model, vocab):
    captions = tiny_model.greedy(torch.randn(5, 16), vocab.start_id, vocab.end_id,
                                 max_len=12)
    assert len(captions) == 5
    assert all(len(c) <= 12 for c in captions)


def test_greedy_is_deterministic(tiny_model, vocab):
    features = torch.randn(3, 16)
    first = tiny_model.greedy(features, vocab.start_id, vocab.end_id, max_len=10)
    second = tiny_model.greedy(features, vocab.start_id, vocab.end_id, max_len=10)
    assert first == second


def test_greedy_stops_at_end_token(tiny_model, vocab):
    """Once <end> is emitted nothing further is appended for that row."""
    captions = tiny_model.greedy(torch.randn(8, 16), vocab.start_id, vocab.end_id,
                                 max_len=15)
    for caption in captions:
        if vocab.end_id in caption:
            assert caption.index(vocab.end_id) == len(caption) - 1


def test_beam_width_one_matches_greedy(tiny_model, vocab):
    """A beam of one is greedy decoding - if it is not, the beam code is wrong.

    This is the sharpest available check on the beam implementation: the two are
    different code paths that must agree in the one case where they describe the
    same algorithm.
    """
    feature = torch.randn(16)
    greedy = tiny_model.greedy(feature.unsqueeze(0), vocab.start_id, vocab.end_id,
                               max_len=12)[0]
    beam = tiny_model.beam_search(feature, vocab.start_id, vocab.end_id,
                                  beam_width=1, max_len=12)
    assert beam == greedy


def test_beam_search_is_deterministic(tiny_model, vocab):
    feature = torch.randn(16)
    first = tiny_model.beam_search(feature, vocab.start_id, vocab.end_id, beam_width=3)
    second = tiny_model.beam_search(feature, vocab.start_id, vocab.end_id, beam_width=3)
    assert first == second


def test_beam_search_respects_max_len(tiny_model, vocab):
    caption = tiny_model.beam_search(torch.randn(16), vocab.start_id, vocab.end_id,
                                     beam_width=4, max_len=6)
    assert len(caption) <= 6


def test_wider_beam_never_scores_worse_than_narrower(vocab):
    """Beam search maximises joint log probability, so a wider beam explores a
    superset of the narrower beam's candidates and cannot end up with a lower
    scoring sequence."""
    torch.manual_seed(3)
    model = CaptionDecoder(16, 12, 16, len(vocab), dropout=0.0).eval()
    feature = torch.randn(16)

    def score(tokens: list[int]) -> float:
        import torch.nn.functional as F
        total, state = 0.0, None
        inputs = model.project(feature.unsqueeze(0)).unsqueeze(1)
        for token in tokens:
            hidden, state = model.lstm(inputs, state)
            log_probs = F.log_softmax(model.output(hidden.squeeze(1)), dim=-1)
            total += float(log_probs[0, token])
            inputs = model.embed(torch.tensor([token])).unsqueeze(1)
        return total

    narrow = model.beam_search(feature, vocab.start_id, vocab.end_id, beam_width=1, max_len=10)
    wide = model.beam_search(feature, vocab.start_id, vocab.end_id, beam_width=5, max_len=10)
    assert score(wide) >= score(narrow) - 1e-4
