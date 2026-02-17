from torch import nn

from ._tcnn import BinaryTreeConv, DynamicPooling, TreeActivation, TreeLayerNorm


class BaoModel(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super(BaoModel, self).__init__()

        self._model = nn.Sequential(
            BinaryTreeConv(in_channels, 256),
            TreeLayerNorm(),
            TreeActivation(nn.LeakyReLU()),
            BinaryTreeConv(256, 128),
            TreeLayerNorm(),
            TreeActivation(nn.LeakyReLU()),
            BinaryTreeConv(128, 64),
            TreeLayerNorm(),
            DynamicPooling(),
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self._model(x)
