from __future__ import annotations

import functools
import operator
from collections.abc import Iterable
from dataclasses import dataclass

import postbound as pb

from ._piecewise_fns import PiecewiseConstantFn, PiecewiseLinearFn


class AlphaStep:
    def __init__(self, functions: Iterable[PiecewiseConstantFn]) -> None:
        self._fns = list(functions)

    def cardinality(self) -> int:
        final_fn = self()
        return final_fn.cardinality()

    def __call__(self) -> PiecewiseConstantFn:
        return functools.reduce(operator.mul, self._fns)


@dataclass
class _StarFn:
    fn: PiecewiseConstantFn
    cdf: PiecewiseLinearFn
    inv: PiecewiseLinearFn

    @staticmethod
    def of(fn: PiecewiseConstantFn):
        cdf = fn.integrate()
        return _StarFn(fn, cdf, cdf.invert())


class BetaStep:
    def __init__(
        self,
        star_joins: Iterable[PiecewiseConstantFn],
        *,
        projection_fn: PiecewiseConstantFn,
    ) -> None:
        self._proj = projection_fn
        self._stars: list[_StarFn] = [_StarFn.of(fn) for fn in star_joins]
        self._cdf = self._proj.integrate()

    def cardinality(self) -> int:
        final_fn = self()
        return final_fn.cardinality()

    def __call__(self) -> PiecewiseConstantFn:
        distincts: list[int] = []

        for i in range(self._proj.n_distinct):
            freq = self._proj(i)

            for star_fn in self._stars:
                star, inv = star_fn.fn, star_fn.inv
                freq *= star(inv(self._cdf(i)))

            distincts.append(freq)

        return PiecewiseConstantFn.unit_segments(distincts)


def decompose_query(
    query: pb.SqlQuery, *, statistics: dict[pb.ColumnReference, PiecewiseLinearFn]
) -> AlphaStep | BetaStep:
    pass


def fdsb(root: AlphaStep | BetaStep) -> pb.Cardinality:
    card = root.cardinality()
    return pb.Cardinality(card)
