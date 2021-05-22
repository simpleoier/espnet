"""Beam search module."""

from itertools import chain
import logging
from typing import Any
from typing import Dict
from typing import List
from typing import NamedTuple
from typing import Tuple
from typing import Union

import torch

from espnet.nets.beam_search import Hypothesis


class BeamSearch(torch.nn.Module):
    """Beamsearch for hybrid models"""
    def __init__(
        self,
        lm_model,
        weights,
        beam_size,
        token_list,
        sos = None,
    ):
        super().__init__()
        self.weights = weights
        self.lm_model = lm_model
        self.beam_size = beam_size
        self.token_list = token_list
        self.n_vocab = len(token_list)
        self.sos = sos

    
    def init_hyp(self, x: torch.Tensor) -> List[Hypothesis]:
        """Get an initial hypothesis data.

        Args:
            x (torch.Tensor): The encoder output feature

        Returns:
            Hypothesis: The initial hypothesis.

        """
        init_states = dict(asr=None, lm=self.lm_model.init_state(x))
        init_scores = dict(asr=0.0, lm=0.0)
        return [
            Hypothesis(
                score=0.0,
                scores=init_scores,
                states=init_states,
                yseq=torch.tensor([self.sos], dtype=torch.long, device=x.device) if self.sos is not None else torch.tensor([], dtype=torch.long, device=x.device),
            )
        ]

    @staticmethod
    def append_token(xs: torch.Tensor, x: int) -> torch.Tensor:
        """Append new token to prefix tokens.

        Args:
            xs (torch.Tensor): The prefix token
            x (int): The new token to append

        Returns:
            torch.Tensor: New tensor contains: xs + [x] with xs.dtype and xs.device

        """
        x = torch.tensor([x], dtype=xs.dtype, device=xs.device)
        return torch.cat((xs, x))

    def beam(
        self,
        weighted_scores,
    ):
        top_ids = weighted_scores.topk(self.beam_size)[1]
        return top_ids, top_ids

    def score_full(
        self, hyp: Hypothesis, asr_prob: torch.Tensor
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """Score new hypothesis by `self.full_scorers`.

        Args:
            hyp (Hypothesis): Hypothesis with prefix tokens to score
            x (torch.Tensor): Corresponding input feature

        Returns:
            Tuple[Dict[str, torch.Tensor], Dict[str, Any]]: Tuple of
                score dict of `hyp` that has string keys of `self.full_scorers`
                and tensor score values of shape: `(self.n_vocab,)`,
                and state dict that has string keys
                and state values of `self.full_scorers`

        """
        scores = dict()
        states = dict()
        scores['asr'], states['asr'] = asr_prob, None
        scores['lm'], states['lm'] = self.lm_model.score(hyp.yseq, hyp.states['lm'], None)
        scores['lm'] = scores['lm'][:self.n_vocab]  # To remove the sos/eos in RNNLM
        return scores, states

    @staticmethod
    def merge_scores(
        prev_scores: Dict[str, float],
        next_full_scores: Dict[str, torch.Tensor],
        full_idx: int,
    ) -> Dict[str, torch.Tensor]:
        """Merge scores for new hypothesis.

        Args:
            prev_scores (Dict[str, float]):
                The previous hypothesis scores by `self.scorers`
            next_full_scores (Dict[str, torch.Tensor]): scores by `self.full_scorers`
            full_idx (int): The next token id for `next_full_scores`
            next_part_scores (Dict[str, torch.Tensor]):
                scores of partial tokens by `self.part_scorers`
            part_idx (int): The new token id for `next_part_scores`

        Returns:
            Dict[str, torch.Tensor]: The new score dict.
                Its keys are names of `self.full_scorers` and `self.part_scorers`.
                Its values are scalar tensors by the scorers.

        """
        new_scores = dict()
        for k, v in next_full_scores.items():
            new_scores[k] = prev_scores[k] + v[full_idx]
        return new_scores

    def search(
        self,
        running_hyps: List[Hypothesis],
        asr_probs: torch.Tensor,
    ) -> List[Hypothesis]:
        best_hyps = []
        for hyp in running_hyps:
            # scoring
            weighted_scores = torch.zeros(self.n_vocab, dtype=asr_probs.dtype, device=asr_probs.device)
            scores, states = self.score_full(hyp, asr_probs)
            weighted_scores = self.weights['asr'] * scores['asr'] + self.weights['lm'] * scores['lm']
            # add previous hyp score
            weighted_scores += hyp.score

            # update hyps
            for j, part_j in zip(*self.beam(weighted_scores)):
                # will be (2 x beam at most)
                best_hyps.append(
                    Hypothesis(
                        score=weighted_scores[j],
                        yseq=self.append_token(hyp.yseq, j),
                        scores=self.merge_scores(
                            hyp.scores, scores, j
                        ),
                        states=states
                    )
                )

            # sort and prune 2 x beam -> beam
            best_hyps = sorted(best_hyps, key=lambda x: x.score, reverse=True)[
                : min(len(best_hyps), self.beam_size)
            ]
        return best_hyps

    def forward(
        self,
        asr_outputs: torch.Tensor,  # ASR negative log likelihood / posterior
    ):
        assert len(asr_outputs.shape) == 2
        seq_len, odim = asr_outputs.shape
        assert odim == self.n_vocab, f'{odim} vs. {self.n_vocab}'

        # Compute log likelihood
        asr_probs = torch.nn.functional.log_softmax(asr_outputs, dim=1)

        running_hyps = self.init_hyp(asr_probs)
        for i in range(seq_len):
            # pdb.set_trace()
            logging.debug("position " + str(i))
            best_hyps = self.search(running_hyps, asr_probs[i])
            running_hyps = best_hyps
        
        nbest_hyps = sorted(best_hyps, key=lambda x: x.score, reverse=True)
        return nbest_hyps
