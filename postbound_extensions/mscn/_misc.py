"""
Utility functions
    provided in the original MSCN repository

The MIT License

Copyright (c) 2019 Andreas Kipf, 2026 Rico Bergmann

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

Original author: Andreas Kipf
Original source: https://github.com/andreaskipf/learnedcardinalities
Modified by: Rico Bergmann
"""

from collections.abc import Iterable
from typing import overload

import numpy as np
import torch


def qerror_loss(preds: torch.Tensor, targets: torch.Tensor, min_val: float, max_val: float) -> torch.Tensor:
    """Computes the mean q-error between normalized predictions and targets.

    Both predictions and targets are mapped back to cardinalities (inverting the min-max scaling
    and the ``log1p`` normalization). Following the usual q-error definition, cardinalities are
    floored at 1. This only matters if ``min_card`` is 0: without the floor, zero cardinalities
    would cause a division by zero (inf/NaN loss and gradients). If ``min_card >= 1`` (as in the
    original MSCN setup), the floor never applies and the loss is identical to the reference.
    """
    preds = torch.expm1(preds.reshape(-1) * (max_val - min_val) + min_val).clamp_min(1.0)
    targets = torch.expm1(targets.reshape(-1) * (max_val - min_val) + min_val).clamp_min(1.0)
    qerror = torch.where(preds > targets, preds / targets, targets / preds)
    return torch.mean(qerror)


@overload
def normalize_labels(labels: Iterable[float]) -> tuple[np.ndarray, float, float]: ...


@overload
def normalize_labels(labels: Iterable[float], min_val: float, max_val: float) -> np.ndarray: ...


def normalize_labels(
    labels,
    min_val=None,
    max_val=None,
):
    infer_min_max = min_val is None or max_val is None
    labels = np.log1p(np.asarray(labels))

    if infer_min_max:
        min_val = labels.min()
        max_val = labels.max()
    labels_norm = (labels - min_val) / (max_val - min_val)
    # Threshold labels
    labels_norm = np.minimum(labels_norm, 1)
    labels_norm = np.maximum(labels_norm, 0)

    if infer_min_max:
        return labels_norm, min_val, max_val

    return labels_norm


def unnormalize_labels(labels: torch.Tensor, min_val: float, max_val: float) -> torch.Tensor:
    labels = (labels * (max_val - min_val)) + min_val
    return torch.expm1(labels)


def collate_featurized(batch: list[list]) -> list[torch.Tensor]:
    """Batches featurized queries whose sets have different numbers of rows.

    Each batch item is ``[tables, joins, predicates, tables_mask, joins_mask, predicates_mask, label]``
    (the row layout of the training data frame). All set tensors and their masks are zero-padded
    along the set dimension to the largest set in the batch. Since padded rows have a mask of 0,
    they do not influence the pooled set representation.
    """
    *features, labels = zip(*batch)
    padded = [torch.nn.utils.rnn.pad_sequence(list(feature), batch_first=True) for feature in features]
    return [*padded, torch.as_tensor(labels)]


def expand_dims(tensor: torch.Tensor) -> torch.Tensor:
    shape = tensor.shape
    return tensor.reshape(-1, *shape)
