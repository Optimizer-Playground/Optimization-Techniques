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


def unnormalize_torch(
    vals: torch.Tensor, min_val: float, max_val: float
) -> torch.Tensor:
    vals = (vals * (max_val - min_val)) + min_val
    return torch.expm1(vals)


def qerror_loss(
    preds: torch.Tensor, targets: torch.Tensor, min_val: float, max_val: float
) -> torch.Tensor:
    qerror = []
    preds = unnormalize_torch(preds, min_val, max_val)
    targets = unnormalize_torch(targets, min_val, max_val)

    for i in range(len(targets)):
        if (preds[i] > targets[i]).cpu().data.numpy()[0]:
            qerror.append(preds[i] / targets[i])
        else:
            qerror.append(targets[i] / preds[i])
    return torch.mean(torch.cat(qerror))


@overload
def normalize_labels(labels: Iterable[float]) -> tuple[np.ndarray, float, float]: ...


@overload
def normalize_labels(
    labels: Iterable[float], *, min_val: float, max_val: float
) -> np.ndarray: ...


def normalize_labels(
    labels,
    min_val,
    max_val,
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


def unnormalize_labels(
    labels: torch.Tensor, min_val: float, max_val: float
) -> torch.Tensor:
    labels = (labels * (max_val - min_val)) + min_val
    return torch.exp(labels)


def expand_dims(tensor: torch.Tensor) -> torch.Tensor:
    shape = tensor.shape
    return tensor.reshape(-1, *shape)
