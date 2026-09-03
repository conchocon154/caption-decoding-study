"""Encoder-decoder captioner, and the decoding strategies under comparison.

Architecture follows Show and Tell (Vinyals et al., 2015): a CNN encodes the
image into a single vector, which is fed to an LSTM as its first input token,
and the LSTM emits the caption word by word.  The encoder is frozen throughout
this study — the question here is about decoding, so the image representation is
held fixed to keep the comparison controlled.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CaptionDecoder(nn.Module):
    def __init__(self, feature_dim: int, embed_size: int, hidden_size: int,
                 vocab_size: int, num_layers: int = 1, dropout: float = 0.3) -> None:
        super().__init__()
        self.project = nn.Linear(feature_dim, embed_size)
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.output = nn.Linear(hidden_size, vocab_size)

    def forward(self, features: torch.Tensor, captions: torch.Tensor) -> torch.Tensor:
        """Teacher forcing: the image vector, then the caption minus its last token.

        The final token is dropped because the target at each step is the *next*
        word; feeding `<end>` in would ask the model to predict past the end.
        """
        image_token = self.project(features).unsqueeze(1)
        words = self.dropout(self.embed(captions[:, :-1]))
        sequence = torch.cat([image_token, words], dim=1)
        hidden, _ = self.lstm(sequence)
        return self.output(hidden)

    @torch.no_grad()
    def greedy(self, features: torch.Tensor, start_id: int, end_id: int,
               max_len: int = 20) -> list[list[int]]:
        """Take the single most likely word at every step.

        Batched: greedy decoding is deterministic per row, so the whole test set
        runs in one pass rather than one image at a time.
        """
        self.eval()
        batch = features.size(0)
        device = features.device
        inputs = self.project(features).unsqueeze(1)
        state = None
        finished = torch.zeros(batch, dtype=torch.bool, device=device)
        outputs = [[] for _ in range(batch)]

        for _ in range(max_len):
            hidden, state = self.lstm(inputs, state)
            logits = self.output(hidden.squeeze(1))
            predicted = logits.argmax(dim=1)
            for row in range(batch):
                if not finished[row]:
                    outputs[row].append(int(predicted[row]))
            finished |= predicted == end_id
            if bool(finished.all()):
                break
            inputs = self.embed(predicted).unsqueeze(1)
        return outputs

    @torch.no_grad()
    def beam_search(self, feature: torch.Tensor, start_id: int, end_id: int,
                    beam_width: int = 5, max_len: int = 20,
                    length_penalty: float = 0.0) -> list[int]:
        """Keep the `beam_width` highest-scoring partial captions at each step.

        Scores are summed log probabilities.  With `length_penalty` at 0 the
        score is the raw joint log probability, which systematically favours
        short captions — every extra word multiplies in another probability
        below one.  The study reports that bias rather than hiding it behind a
        default penalty, and `length_penalty` exposes the knob (Wu et al., 2016).
        """
        self.eval()
        device = feature.device
        inputs = self.project(feature.unsqueeze(0)).unsqueeze(1)

        # Each beam: (token ids, cumulative log prob, lstm state, finished)
        beams = [([], 0.0, None, False)]
        for _ in range(max_len):
            if all(beam[3] for beam in beams):
                break
            candidates = []
            for tokens, score, state, done in beams:
                if done:
                    candidates.append((tokens, score, state, True))
                    continue
                step_input = inputs if not tokens else self.embed(
                    torch.tensor([tokens[-1]], device=device)).unsqueeze(1)
                hidden, next_state = self.lstm(step_input, state)
                log_probs = F.log_softmax(self.output(hidden.squeeze(1)), dim=-1)
                top_scores, top_ids = log_probs.topk(beam_width, dim=-1)
                for rank in range(beam_width):
                    token = int(top_ids[0, rank])
                    candidates.append((
                        tokens + [token],
                        score + float(top_scores[0, rank]),
                        next_state,
                        token == end_id,
                    ))

            def rank_key(candidate):
                tokens, score, _, _ = candidate
                if length_penalty == 0.0:
                    return score
                # Google NMT length normalisation.
                norm = ((5 + len(tokens)) / 6) ** length_penalty
                return score / norm

            beams = sorted(candidates, key=rank_key, reverse=True)[:beam_width]

        return max(beams, key=lambda b: b[1])[0]
