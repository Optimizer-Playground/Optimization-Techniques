from __future__ import annotations

import postbound as pb

from ._compress import PiecewiseLinearFn


class AlphaStep:
    def __init__(self) -> None:
        pass

    def cardinality(self) -> int:
        pass


class BetaStep:
    def __init__(self) -> None:
        pass

    def cardinality(self) -> int:
        pass


def decompose_query(
    query: pb.SqlQuery, *, statistics: dict[pb.ColumnReference, PiecewiseLinearFn]
) -> AlphaStep | BetaStep:
    pass


def fdsb(root: AlphaStep | BetaStep) -> pb.Cardinality:
    card = root.cardinality()
    return pb.Cardinality(card)
