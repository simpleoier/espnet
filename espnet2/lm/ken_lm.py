"""Implementation of kenLM Language Model."""
from typing import List
from typing import Tuple
from typing import Union

import kenlm
import torch
import torch.nn as nn
from typeguard import check_argument_types

from espnet2.lm.abs_model import AbsLM


class kenlmLM(AbsLM):
    """kenlm LM.

    See also:
        https://github.com/pytorch/examples/blob/4581968193699de14b56527296262dd76ab43557/word_language_model/model.py

    """

    def __init__(
        self,
        token_list: List[str] = None,
        lm_file: str = None,
        ngram: int = None,
    ):
        assert check_argument_types()
        super().__init__()

        self.token_list = token_list
        self.n_vocab = len(token_list)
        self.lm_model = kenlm.Model(lm_file)
        self.ngram = ngram

    def forward(
        self, input: torch.Tensor, hidden: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def init_state(self, x):
        return None

    def score(
        self,
        y: torch.Tensor,
        state: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:
        """Score new token.

        Args:
            y: 1D torch.int64 prefix tokens.
            state: Scorer state for prefix tokens
            x: 2D encoder feature that generates ys.

        Returns:
            Tuple of
                torch.float32 scores for next token (n_vocab)
                and next state for ys

        """
        label_seq = [self.token_list[t] for t in y][-self.ngram:]
        logp = []
        for i in range(self.n_vocab):
            new_label_seq = ' '.join(label_seq + [self.token_list[i]])
            logp.append(
                self.lm_model.score(new_label_seq, bos=(len(label_seq)<self.ngram), eos=False)
            )
        return torch.tensor(logp), None

    def batch_score(
        self, ys: torch.Tensor, states: torch.Tensor, xs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Score new token batch.

        Args:
            ys (torch.Tensor): torch.int64 prefix tokens (n_batch, ylen).
            states (List[Any]): Scorer states for prefix tokens.
            xs (torch.Tensor):
                The encoder feature that generates ys (n_batch, xlen, n_feat).

        Returns:
            tuple[torch.Tensor, List[Any]]: Tuple of
                batchfied scores for next token with shape of `(n_batch, n_vocab)`
                and next state list for ys.

        """
        raise NotImplementedError
