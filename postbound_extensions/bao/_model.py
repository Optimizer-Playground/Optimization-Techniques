import torch

from ._featurizer import FeaturizedNode, inner_child, node_features, outer_child
from ._tcnn import BinaryTreeConv, DynamicPooling, TreeActivation, TreeLayerNorm
from ._util import prepare_trees


class BaoModel(torch.nn.Module):
    def __init__(self, in_channels: int) -> None:
        super(BaoModel, self).__init__()

        self._in_channels = in_channels
        self._model = torch.nn.Sequential(
            BinaryTreeConv(self._in_channels, 256),
            TreeLayerNorm(),
            TreeActivation(torch.nn.LeakyReLU()),
            BinaryTreeConv(256, 128),
            TreeLayerNorm(),
            TreeActivation(torch.nn.LeakyReLU()),
            BinaryTreeConv(128, 64),
            TreeLayerNorm(),
            DynamicPooling(),
            torch.nn.Linear(64, 32),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(32, 1),
        )

    def forward(self, x):
        if isinstance(x, FeaturizedNode):
            x = [x]
        x = prepare_trees(x, node_features, outer_child, inner_child)
        return self._model(x)
