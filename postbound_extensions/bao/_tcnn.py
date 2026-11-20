# < begin copyright >
# Copyright Ryan Marcus 2019
#
# This file is part of TreeConvolution.
#
# TreeConvolution is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# TreeConvolution is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with TreeConvolution.  If not, see <http://www.gnu.org/licenses/>.
#
# < end copyright >

from __future__ import annotations

import torch
from torch import nn


class BinaryTreeConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.__in_channels = in_channels
        self.__out_channels = out_channels
        # we can think of the tree conv as a single dense layer
        # that we "drag" across the tree.
        self.weights = nn.Conv1d(in_channels, out_channels, stride=3, kernel_size=3)

    def forward(self, flat_data):
        trees, idxes, *mask = flat_data  # the optional padding mask is passed through untouched
        orig_idxes = idxes
        idxes = idxes.expand(-1, -1, self.__in_channels).transpose(1, 2)
        expanded = torch.gather(trees, 2, idxes)

        results = self.weights(expanded)

        # add a zero vector back on
        zero_vec = torch.zeros((trees.shape[0], self.__out_channels)).unsqueeze(2)
        results = torch.cat((zero_vec, results), dim=2)
        return (results, orig_idxes, *mask)


class TreeActivation(nn.Module):
    def __init__(self, activation):
        super().__init__()
        self.activation = activation

    def forward(self, x):
        return (self.activation(x[0]), *x[1:])


class TreeLayerNorm(nn.Module):
    """Normalizes each tree over all of its channels and positions.

    If the input carries a padding mask (third tuple element, see `prepare_trees`), padding positions are excluded
    from the statistics. Otherwise, the statistics are computed over all positions (original behavior), which makes
    the result depend on the other trees in the batch.
    """

    def forward(self, x):
        data, idxes, *mask = x
        if not mask:
            mean = torch.mean(data, dim=(1, 2)).unsqueeze(1).unsqueeze(1)
            std = torch.std(data, dim=(1, 2)).unsqueeze(1).unsqueeze(1)
            normd = (data - mean) / (std + 0.00001)
            return (normd, idxes)

        m = mask[0].to(data.dtype)  # (batch, 1, positions), broadcasts over the channels
        count = m.sum(dim=(1, 2), keepdim=True) * data.shape[1]
        mean = (data * m).sum(dim=(1, 2), keepdim=True) / count
        var = (((data - mean) * m) ** 2).sum(dim=(1, 2), keepdim=True) / (count - 1)  # unbiased, like torch.std
        normd = (data - mean) / (var.sqrt() + 0.00001)
        return (normd, idxes, mask[0])


class DynamicPooling(nn.Module):
    """Max-pools each tree over its positions, ignoring padding positions if a mask is given."""

    def forward(self, x):
        data = x[0]
        if len(x) > 2:
            data = data.masked_fill(~x[2], float("-inf"))
        return torch.max(data, dim=2).values
