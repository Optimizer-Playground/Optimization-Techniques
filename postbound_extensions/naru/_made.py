"""
MADE and ResMADE autoregressive models
    provided in the original Naru repository

Copyright 2019 The Naru Authors (Zongheng Yang, Eric Liang, Amog Kamsetty, Chenggang Wu)
Modifications Copyright 2026 Rico Bergmann

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Original source: https://github.com/naru-project/naru (made.py)
Modified by: Claude Opus 5.5 (Anthropic), supervised by Rico Bergmann. Summary of the changes:

- removed support for multiple masks/orderings (order-agnostic training is not used by Naru)
- the column ordering is now passed explicitly as a list of column indices
- per-tuple column masking that follows the paper (instead of one masking probability per batch and column)
- fixed the direct input/output connections for the natural ordering (the reference wired only a subset of inputs)
- consistent small-domain threshold for embeddings (the reference crashed for domains of exactly `embed_size` values)
- dedicated helpers to encode single columns during progressive sampling
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from typing import Literal, override

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

InputEncoding = Literal["embed", "one_hot", "binary"]
"""How column values are fed into the network.

- *embed*: learnable embeddings for large domains, one-hot vectors for small domains (paper default, Section 4.2)
- *one_hot*: one-hot vectors for all columns
- *binary*: binary representation of the dictionary code (default of the reference implementation)
"""

OutputEncoding = Literal["embed", "one_hot"]
"""How the conditional distributions are decoded.

- *embed*: embedding reuse for large domains (paper Section 4.2). Requires the *embed* input encoding.
- *one_hot*: a dedicated output unit for each domain value
"""


class MaskedLinear(nn.Linear):
    """A linear layer whose connectivity is restricted by a binary mask."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__(in_features, out_features, bias)
        self.register_buffer("mask", torch.ones(out_features, in_features))

    def set_mask(self, mask: np.ndarray) -> None:
        """Accepts a mask of shape [in_features, out_features]."""
        mask_tensor = torch.from_numpy(mask.astype(np.float32).T)
        self.get_buffer("mask").copy_(mask_tensor)

    @override
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, self.get_buffer("mask") * self.weight, self.bias)


class MaskedResidualBlock(nn.Module):
    """Pre-activation residual block of two masked linear layers (ResMADE)."""

    def __init__(self, in_features: int, out_features: int, activation: nn.Module) -> None:
        if in_features != out_features:
            raise ValueError(f"Residual blocks need equal in and out features, not {in_features} and {out_features}")
        super().__init__()
        self.layers = nn.ModuleList(
            [
                MaskedLinear(in_features, out_features, bias=True),
                MaskedLinear(in_features, out_features, bias=True),
            ]
        )
        self.activation = activation

    def set_mask(self, mask: np.ndarray) -> None:
        for layer in self.layers:
            assert isinstance(layer, MaskedLinear)
            layer.set_mask(mask)

    @override
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out = self.activation(input)
        out = self.layers[0](out)
        out = self.activation(out)
        out = self.layers[1](out)
        return input + out


class MADE(nn.Module):
    """Masked autoencoder for distribution estimation (MADE), optionally with residual connections (ResMADE).

    The model receives (dictionary-encoded) tuples and produces the logits of all conditional distributions
    *P(X_i | X_<i)* in a single forward pass. Use `logits_for_col` to extract the logits of a specific column.

    Parameters
    ----------
    domain_sizes : Sequence[int]
        The number of distinct tokens of each column (in natural column order).
    hidden_sizes : Sequence[int]
        Number of units in the hidden layers.
    ordering : Sequence[int] | None, optional
        The autoregressive ordering as a list of column indices. ``ordering[0]`` is the first column in the
        factorization, ``ordering[1]`` the second, and so on. Defaults to the natural column order.
    input_encoding : InputEncoding, optional
        How column values are encoded as network inputs. Defaults to *embed*.
    output_encoding : OutputEncoding, optional
        How the output distributions are decoded. Defaults to *embed* (i.e. embedding reuse).
    embed_size : int, optional
        Dimensionality of the embeddings. Domains with at most this many tokens are one-hot encoded instead.
    residual : bool, optional
        Whether to use residual blocks (ResMADE). Requires hidden layers of equal size to be effective.
    direct_io : bool, optional
        Whether to add direct (masked) connections from the input layer to the output layer.
    column_masking : bool, optional
        Whether to train with randomly masked input columns. This enables wildcard-skipping during inference.
    seed : int, optional
        Seed for the connectivity of a non-residual MADE.
    """

    def __init__(
        self,
        domain_sizes: Sequence[int],
        *,
        hidden_sizes: Sequence[int],
        ordering: Sequence[int] | None = None,
        input_encoding: InputEncoding = "embed",
        output_encoding: OutputEncoding = "embed",
        embed_size: int = 64,
        residual: bool = True,
        direct_io: bool = True,
        column_masking: bool = True,
        seed: int = 11123,
    ) -> None:
        super().__init__()
        if not domain_sizes:
            raise ValueError("MADE requires at least one column")
        if output_encoding == "embed" and input_encoding != "embed":
            raise ValueError("Embedding reuse for the output requires the 'embed' input encoding")

        self.nin = len(domain_sizes)
        self.domain_sizes = [int(d) for d in domain_sizes]
        self.hidden_sizes = list(hidden_sizes)
        self.input_encoding: InputEncoding = input_encoding
        self.output_encoding: OutputEncoding = output_encoding
        self.embed_size = embed_size
        self.residual = residual
        self.direct_io = direct_io
        self.column_masking = column_masking

        ordering = list(range(self.nin)) if ordering is None else [int(i) for i in ordering]
        if sorted(ordering) != list(range(self.nin)):
            raise ValueError(f"Ordering {ordering} is not a permutation of the columns 0..{self.nin - 1}")
        self.ordering = ordering
        # position[i] is the position of the column with natural index i in the autoregressive ordering
        self.position = np.empty(self.nin, dtype=np.int64)
        self.position[ordering] = np.arange(self.nin)

        self.input_sizes = [self._input_size(d) for d in self.domain_sizes]
        self.output_sizes = [self._output_size(d) for d in self.domain_sizes]
        self.input_offsets = np.concatenate([[0], np.cumsum(self.input_sizes)]).astype(np.int64)
        self.output_offsets = np.concatenate([[0], np.cumsum(self.output_sizes)]).astype(np.int64)
        n_inputs = int(self.input_offsets[-1])
        n_outputs = int(self.output_offsets[-1])

        layer_sizes = [n_inputs, *self.hidden_sizes, n_outputs]
        layers: list[nn.Module] = []
        n_layers = len(layer_sizes) - 1
        for i, (h0, h1) in enumerate(itertools.pairwise(layer_sizes)):
            is_inner = 0 < i < n_layers - 1
            if residual and is_inner and h0 == h1:
                layers.append(MaskedResidualBlock(h0, h1, activation=nn.ReLU(inplace=False)))
            elif residual:
                layers.append(MaskedLinear(h0, h1))
            else:
                layers.append(MaskedLinear(h0, h1))
                if i < n_layers - 1:
                    layers.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*layers)

        self.embeddings = nn.ModuleDict()
        if input_encoding == "embed":
            for i, dom_size in enumerate(self.domain_sizes):
                if dom_size > embed_size:
                    self.embeddings[str(i)] = nn.Embedding(dom_size, embed_size)

        # learnable [MASK] representation for each column, used for wildcard skipping
        self.unk_embeddings = nn.ParameterList(
            [nn.Parameter(torch.zeros(1, size), requires_grad=column_masking) for size in self.input_sizes]
        )

        self.direct_io_layer = MaskedLinear(n_inputs, n_outputs) if direct_io and self.nin > 1 else None

        self._update_masks(np.random.RandomState(seed))
        self.apply(_init_weights)

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def has_embedding(self, col: int) -> bool:
        return str(col) in self.embeddings

    def encode_column(self, tokens: torch.Tensor | None, col: int) -> torch.Tensor:
        """Encodes the tokens of a single column. Passing *None* produces the [MASK] encoding.

        Parameters
        ----------
        tokens : torch.Tensor | None
            A 1D tensor of dictionary codes.
        col : int
            The (natural) index of the column.

        Returns
        -------
        torch.Tensor
            A tensor of shape [len(tokens), input_size] (or [1, input_size] for the [MASK] encoding).
        """
        if tokens is None:
            return self.unk_embeddings[col]

        tokens = tokens.long().view(-1)
        dom_size = self.domain_sizes[col]
        match self.input_encoding:
            case "embed" if self.has_embedding(col):
                return self.embeddings[str(col)](tokens)
            case "embed" | "one_hot":
                return F.one_hot(tokens, num_classes=dom_size).to(torch.float32)
            case "binary":
                n_bits = self.input_sizes[col]
                shifts = torch.arange(n_bits, device=tokens.device)
                return ((tokens.view(-1, 1) >> shifts) & 1).to(torch.float32)

    def encode_input(self, data: torch.Tensor) -> torch.Tensor:
        """Encodes a batch of tuples of shape [bs, nin].

        During training (and if column masking is enabled) each column of each tuple is replaced by its [MASK]
        representation with probability *w/n*, where *w* is drawn uniformly from *[0, n)* for each tuple (Section 5.2
        of the paper).
        """
        bs = data.shape[0]
        encoded = [self.encode_column(data[:, i], i) for i in range(self.nin)]

        if self.training and self.column_masking:
            w = torch.randint(0, self.nin, (bs, 1), device=data.device)
            masked = torch.rand(bs, self.nin, device=data.device) < (w.to(torch.float32) / self.nin)
            encoded = [
                torch.where(masked[:, i : i + 1], self.unk_embeddings[i].expand(bs, -1), enc)
                for i, enc in enumerate(encoded)
            ]

        return torch.cat(encoded, dim=1)

    @override
    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """Computes the (unnormalized) logits of all conditionals for a batch of tuples of shape [bs, nin]."""
        return self.forward_with_encoded_input(self.encode_input(data))

    def forward_with_encoded_input(self, encoded: torch.Tensor) -> torch.Tensor:
        out = self.net(encoded)
        if self.direct_io_layer is not None:
            out = out + self.direct_io_layer(encoded)
        return out

    def logits_for_col(self, col: int, logits: torch.Tensor) -> torch.Tensor:
        """Extracts the logits of *P(X_col | X_<col)* from the network output.

        Returns a tensor of shape [bs, domain size of the column].
        """
        lo, hi = int(self.output_offsets[col]), int(self.output_offsets[col + 1])
        logits_for_var = logits[:, lo:hi]
        if self.output_encoding != "embed" or not self.has_embedding(col):
            return logits_for_var

        # Embedding reuse: score each domain value by the dot product with its embedding
        embedding = self.embeddings[str(col)]
        assert isinstance(embedding, nn.Embedding)
        return torch.matmul(logits_for_var, embedding.weight.t())

    def nll(self, logits: torch.Tensor, data: torch.Tensor) -> torch.Tensor:
        """Computes the negative log-likelihood *-log P(x)* of each tuple (in nats)."""
        data = data.long()
        nll = torch.zeros(logits.shape[0], device=logits.device)
        for i in range(self.nin):
            nll += F.cross_entropy(self.logits_for_col(i, logits), data[:, i], reduction="none")
        return nll

    def _input_size(self, dom_size: int) -> int:
        match self.input_encoding:
            case "embed":
                return min(dom_size, self.embed_size)
            case "one_hot":
                return dom_size
            case "binary":
                return max(1, int(np.ceil(np.log2(dom_size))))

    def _output_size(self, dom_size: int) -> int:
        match self.output_encoding:
            case "embed":
                return min(dom_size, self.embed_size)
            case "one_hot":
                return dom_size

    def _update_masks(self, rng: np.random.RandomState) -> None:
        """Computes the connectivity masks that guarantee the autoregressive property."""
        n_hidden = len(self.hidden_sizes)

        # Degrees of the units. Inputs (and outputs) have the position of their column in the ordering,
        # hidden units have a degree in [0, nin - 1) that determines which inputs they can see.
        degrees: list[np.ndarray] = [self.position]
        for size in self.hidden_sizes:
            if self.nin == 1:
                # disconnect all hidden units from the input. The single output then only depends on the bias.
                degrees.append(np.full(size, -1))
            elif self.residual:
                # sequential assignment for ResMADE, see https://arxiv.org/pdf/1904.05626.pdf
                degrees.append(np.array([(k - 1) % (self.nin - 1) for k in range(size)]))
            else:
                degrees.append(rng.randint(degrees[-1].min(), self.nin - 1, size=size))

        masks = [degrees[i][:, None] <= degrees[i + 1][None, :] for i in range(n_hidden)]
        masks.append(degrees[-1][:, None] < self.position[None, :])

        # The input and output layers use multiple units per column, so we need to repeat the masks accordingly.
        masks[0] = np.repeat(masks[0], self.input_sizes, axis=0)
        masks[-1] = np.repeat(masks[-1], self.output_sizes, axis=1)

        masked_layers = [layer for layer in self.net if isinstance(layer, (MaskedLinear, MaskedResidualBlock))]
        assert len(masked_layers) == len(masks), (len(masked_layers), len(masks))
        for layer, mask in zip(masked_layers, masks):
            layer.set_mask(mask)

        if self.direct_io_layer is not None:
            # input block of column i connects to the output block of column j iff i comes before j
            direct = self.position[:, None] < self.position[None, :]
            direct = np.repeat(direct, self.input_sizes, axis=0)
            direct = np.repeat(direct, self.output_sizes, axis=1)
            self.direct_io_layer.set_mask(direct)


def _init_weights(module: nn.Module) -> None:
    """Weight initialization of the reference implementation."""
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=0.02)
