from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np


@dataclass
class Segment:
    lower: int
    higher: int
    slope: float
    intercept: float

    @staticmethod
    def constant(intercept: float, *, lower: int, higher: int) -> Segment:
        return Segment(lower, higher, 0.0, intercept)

    @staticmethod
    def initial(slope: int) -> Segment:
        return Segment(0, 0, slope, 0)

    @staticmethod
    def after(last: Segment, *, slope: float) -> Segment:
        intercept = last.intercept + last.slope * (last.higher - last.lower)
        return Segment(last.higher, last.higher, slope, intercept)

    @property
    def width(self) -> int:
        return self.higher - self.lower

    def is_constant(self) -> bool:
        return self.slope == 0.0

    def final_freq(self) -> float:
        span = self.higher - self.lower
        return self.slope * span + self.intercept

    def invert(self) -> Segment:
        return Segment(self.lower, self.higher, 1 / self.slope, -self.intercept)

    def __call__(self, i: int) -> int:
        if not self.lower <= i < self.higher:
            raise ValueError(
                f"i={i} is out of bounds for segment [{self.lower}, {self.higher})"
            )
        return round(self.slope * i + self.intercept)


class PiecewiseConstantFn:
    @staticmethod
    def from_segments(segments: Iterable[Segment]) -> PiecewiseConstantFn:
        values: list[float] = []
        bounds: list[int] = []
        for seg in segments:
            if not seg.is_constant():
                raise ValueError("All segments must be constant")
            values.append(seg.intercept)
            bounds.append(seg.higher)
        return PiecewiseConstantFn(values, bounds)

    def __init__(self, values: Iterable[float], bounds: Iterable[int]) -> None:
        self._values = np.asarray(values)
        self._bounds = np.asarray(bounds)
        if len(self._values) != len(self._bounds):
            raise ValueError("values and bounds must be the same length")

        self._widths = np.diff(np.concat(([0], self._bounds)))
        self._num_distinct = self._bounds[-1]

    @property
    def values(self) -> np.ndarray:
        return self._values

    @property
    def bounds(self) -> np.ndarray:
        return self._bounds

    @property
    def n_distinct(self) -> int:
        return self._num_distinct

    def cardinality(self) -> int:
        return self._values @ self._widths

    def integ(self) -> PiecewiseLinearFn:
        widths = np.diff(np.concat(([0], self._bounds)))
        intercepts = np.cumsum(widths * self._values)
        return PiecewiseLinearFn(
            slopes=self._values, intercepts=intercepts, bounds=self._bounds
        )

    def cut_at(self, n_distinct: int) -> PiecewiseConstantFn:
        if self._num_distinct < n_distinct:
            return self

        cum_widths = np.cumsum(self._widths)
        cutoff = np.searchsorted(cum_widths, n_distinct)
        values = self._values[cutoff + 1]
        bounds = np.concat((self._bounds[:cutoff], [n_distinct]))
        return PiecewiseConstantFn(values, bounds)

    def align_with(
        self, other: PiecewiseConstantFn
    ) -> tuple[PiecewiseConstantFn, PiecewiseConstantFn]:
        return align_pcfs(self, other)

    def __len__(self) -> int:
        return len(self._values)

    def __mul__(self, other: PiecewiseConstantFn) -> PiecewiseConstantFn:
        aligned_self, aligned_other = self.align_with(other)
        frequencies = aligned_self._values * aligned_other._values
        return PiecewiseConstantFn(frequencies, aligned_self._bounds)

    def __call__(self, i: int) -> int:
        idx = np.searchsorted(self._bounds, i)
        if idx >= self._num_distinct:
            return 0
        return self._values[idx]


class PiecewiseLinearFn:
    @staticmethod
    def from_segments(segments: Iterable[Segment]) -> PiecewiseLinearFn:
        slopes: list[float] = []
        intercepts: list[float] = []
        bounds: list[int] = []
        for seg in segments:
            slopes.append(seg.slope)
            intercepts.append(seg.intercept)
            bounds.append(seg.higher)

        return PiecewiseLinearFn(slopes=slopes, intercepts=intercepts, bounds=bounds)

    def __init__(
        self,
        *,
        slopes: Iterable[float],
        intercepts: Iterable[float],
        bounds: Iterable[int],
    ) -> None:
        self._slopes = np.asarray(slopes)
        self._intercepts = np.asarray(intercepts)
        self._bounds = np.asarray(bounds)
        if len(self._slopes) != len(self._intercepts) != len(self._bounds):
            raise ValueError("slopes, intercepts and bounds must be the same length")

        self._widths = np.diff(np.concat(([0], self._bounds)))
        self._num_distinct = self._bounds[-1]

    def deriv(self) -> PiecewiseConstantFn:
        return PiecewiseConstantFn(self._slopes, self._bounds)

    def invert_at(self, y: float) -> int:
        idx = np.searchsorted(self._intercepts, y)
        if idx < len(self._intercepts):
            return (y - self._intercepts[idx]) / self._slopes[idx]

        # y is larger than our largest intercept.
        # It might still be part of the very last segment, though.
        x = (y - self._intercepts[-1]) / self._slopes[-1]
        return 0 if x > self._widths[-1] else x

    def __call__(self, i: int) -> int:
        idx = np.searchsorted(self._bounds, i)
        if idx >= self._num_distinct:
            # too few distinct values
            return 0

        return self._slopes[idx] * i + self._intercepts[idx]


def align_pcfs(
    a: PiecewiseConstantFn, b: PiecewiseConstantFn
) -> tuple[PiecewiseConstantFn, PiecewiseConstantFn]:
    values_a, values_b = [], []
    bounds_a, bounds_b = [], []

    steps = min(len(a), len(b))
    for i in range(steps):
        val_a, val_b = a.values[i], b.values[i]
        bound_a, bound_b = a.bounds[i], b.bounds[i]

        if bound_a == bound_b:
            values_a.append(val_a)
            bounds_a.append(bound_a)
            values_b.append(val_b)
            bounds_b.append(bound_b)

        elif bound_a < bound_b:
            values_a.append(val_a)
            bounds_a.append(bound_a)

            # we need to split the other fn into two equal fns
            values_b.append(val_b)
            values_b.append(val_b)
            bounds_b.append(bound_a)
            bounds_b.append(bound_b)

        else:
            assert bound_a > bound_b
            # we need to split our own fn into two equal fns
            values_a.append(val_a)
            values_a.append(val_a)
            bounds_a.append(bound_b)
            bounds_a.append(bound_a)

            values_b.append(val_b)
            bounds_b.append(bound_b)

    return (
        PiecewiseConstantFn(values_a, bounds_a),
        PiecewiseConstantFn(values_b, bounds_b),
    )
