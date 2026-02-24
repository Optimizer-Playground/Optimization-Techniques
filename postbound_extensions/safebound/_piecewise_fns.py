from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol, overload

import numpy as np


class FunctionLike(Protocol):
    @property
    def n_distinct(self) -> int: ...

    def evaluate_at(self, vals: np.ndarray) -> np.ndarray: ...

    def cumulative_at(self, vals: np.ndarray) -> np.ndarray: ...

    def invert_cumulative_at(self, vals: np.ndarray) -> np.ndarray: ...

    def cardinality(self) -> int: ...


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
        self._cumulative = np.cumsum(self._values * self._widths)
        self._cum_widths = np.cumsum(self._widths)

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
        return align_functions(self, other)

    def evaluate_at(self, vals: np.ndarray) -> np.ndarray:
        if not isinstance(vals, np.ndarray):
            vals = np.asarray(vals)

        idx = np.searchsorted(self._bounds, vals)
        out_of_bounds = idx >= len(self._values)
        clipped = np.where(out_of_bounds, 0, idx)
        return np.where(
            out_of_bounds,
            0,  # too few distinct values
            self._values[clipped],
        )

    def cumulative_at(self, vals: np.ndarray) -> np.ndarray:
        if not isinstance(vals, np.ndarray):
            vals = np.asarray(vals)

        idx = np.searchsorted(self._bounds, vals)
        out_of_bounds = idx >= len(self._values)
        clipped_upper = np.where(out_of_bounds, 0, idx)
        in_initial_bucket = idx == 0
        clipped_lower = np.where(in_initial_bucket, 1, clipped_upper)

        cumulative_until_idx = np.where(
            in_initial_bucket | out_of_bounds,
            0,  # out of bounds
            self._cumulative[clipped_lower - 1],
        )

        bucket_vals = np.where(out_of_bounds, 0, self._values[clipped_upper])
        in_bucket = np.where(
            out_of_bounds,
            0,  # mask
            (self._cum_widths[clipped_upper] - vals) * bucket_vals,
        )

        return cumulative_until_idx + in_bucket

    def invert_cumulative_at(self, vals: np.ndarray) -> np.ndarray:
        if not isinstance(vals, np.ndarray):
            vals = np.asarray(vals)

        idx = np.searchsorted(self._cumulative, vals)
        out_of_bounds = idx >= len(self._values)
        clipped_upper = np.where(out_of_bounds, 0, idx)
        in_initial_bucket = idx == 0
        clipped_lower = np.where(in_initial_bucket, 1, clipped_upper)

        freq_until_bucket = np.where(
            in_initial_bucket | out_of_bounds, 0, self._cum_widths[clipped_lower - 1]
        )

        prev_bucket_freq = np.where(
            in_initial_bucket, 0, self._cumulative[clipped_lower - 1]
        )
        per_elem_freq = np.where(
            out_of_bounds, 1, self._values[clipped_upper]
        )  # use 1 to prevent division by 0
        in_bucket_freq = (vals - prev_bucket_freq) / per_elem_freq

        return np.where(out_of_bounds, 0, freq_until_bucket + in_bucket_freq)

    def inspect(self) -> str:
        lines: list[str] = [
            f"PCF ({len(self._values)} segments, {self._num_distinct} distinct values)"
        ]

        prev_bound = 0
        max_bound = self._bounds[-1]
        max_freq = np.max(self._values)
        max_total = self._cumulative[-1]

        bound_padding = len(str(max_bound))
        freq_padding = len(str(max_freq))
        total_padding = len(str(max_total))

        for i in range(len(self._values)):
            freq = self._values[i]
            bound = self._bounds[i]
            total = self._cumulative[i]
            lines.append(
                f" +-- Segment {i}: "
                f"range=[{prev_bound:>{bound_padding}}, {bound:>{bound_padding}}), "
                f"value={freq:>{freq_padding}} "
                f"({total:>{total_padding}} total)"
            )

            prev_bound = bound

        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self._values)

    def __mul__(self, other: PiecewiseConstantFn) -> PiecewiseConstantFn:
        aligned_self, aligned_other = self.align_with(other)
        frequencies = aligned_self._values * aligned_other._values
        return PiecewiseConstantFn(frequencies, aligned_self._bounds)

    def __call__(self, vals: np.ndarray) -> np.ndarray:
        return self.evaluate_at(vals)


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

    @property
    def n_distinct(self) -> int:
        return self._num_distinct

    @property
    def slopes(self) -> np.ndarray:
        return self._slopes

    @property
    def intercepts(self) -> np.ndarray:
        return self._intercepts

    @property
    def bounds(self) -> np.ndarray:
        return self._bounds

    def deriv(self) -> PiecewiseConstantFn:
        return PiecewiseConstantFn(self._slopes, self._bounds)

    def invert(self) -> PiecewiseLinearFn:
        return PiecewiseLinearFn(
            slopes=1 / self._slopes,
            intercepts=-1 * self._intercepts,
            bounds=self._bounds,
        )

    def compose_with(self, other: PiecewiseLinearFn) -> PiecewiseLinearFn:
        aligned_self, aligned_other = align_functions(self, other)
        slopes = aligned_self.slopes * aligned_other.slopes
        intercepts = (
            aligned_self.slopes * aligned_other.intercepts + aligned_other.intercepts
        )

        return PiecewiseLinearFn(
            slopes=slopes, intercepts=intercepts, bounds=aligned_self.bounds
        )

    def __len__(self) -> int:
        return len(self._slopes)

    def __call__(self, vals: np.ndarray) -> np.ndarray:
        idx = np.searchsorted(self._bounds, vals)
        return np.where(
            idx >= self._num_distinct,
            0,  # too few distinct values
            self._slopes[idx] * vals + self._intercepts[idx],
        )


@overload
def align_functions(
    a: PiecewiseConstantFn, b: PiecewiseConstantFn
) -> tuple[PiecewiseConstantFn, PiecewiseConstantFn]: ...


@overload
def align_functions(
    a: PiecewiseLinearFn, b: PiecewiseLinearFn
) -> tuple[PiecewiseLinearFn, PiecewiseLinearFn]:
    pass


def _align_constant_fns(
    a: PiecewiseConstantFn, b: PiecewiseConstantFn
) -> tuple[PiecewiseConstantFn, PiecewiseConstantFn]:
    values_a, values_b = [], []
    bounds_a, bounds_b = [], []

    # FIXME: the iteration loop is complete bullsh*t

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

            # we need to split b into two equal fns
            values_b.append(val_b)
            values_b.append(val_b)
            bounds_b.append(bound_a)
            bounds_b.append(bound_b)

        else:
            assert bound_a > bound_b
            # we need to split a into two equal fns
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


def _align_linear_fns(
    a: PiecewiseLinearFn, b: PiecewiseLinearFn
) -> tuple[PiecewiseLinearFn, PiecewiseLinearFn]:
    slopes_a, slopes_b = [], []
    intercepts_a, intercepts_b = [], []
    bounds_a, bounds_b = [], []

    # FIXME: the iteration loop is complete bullsh*t

    steps = min(len(a), len(b))
    for i in range(steps):
        bound_a, bound_b = a.bounds[i], b.bounds[i]

        if bound_a == bound_b:
            slopes_a.append(a.slopes[i])
            intercepts_a.append(a.intercepts[i])
            bounds_a.append(bound_a)

            slopes_b.append(b.slopes[i])
            intercepts_b.append(b.intercepts[i])
            bounds_b.append(bound_b)

        elif bound_a < bound_b:
            # we need to split b into two equal fns
            slopes_a.append(a.slopes[i])
            intercepts_a.append(a.intercepts[i])
            bounds_a.append(bound_a)

            slopes_b.append(b.slopes[i])
            intercepts_b.append(b.intercepts[i])
            bounds_b.append(bound_a)

            used_interval = bound_b - bound_a
            new_intercept = b.slopes[i] * used_interval + b.intercepts[i]
            slopes_b.append(b.slopes[i])
            intercepts_b.append(new_intercept)
            bounds_b.append(bound_b)

        else:
            assert bound_a > bound_b

            slopes_a.append(a.slopes[i])
            intercepts_a.append(a.intercepts[i])
            bounds_a.append(bound_b)

            used_interval = bound_a - bound_b
            new_intercept = a.slopes[i] * used_interval + a.intercepts[i]
            slopes_a.append(a.slopes[i])
            intercepts_a.append(new_intercept)
            bound_a.append(bound_a)

            slopes_b.append(b.slopes[i])
            intercepts_b.append(b.intercepts[i])
            bounds_b.append(bound_b)

    return (
        PiecewiseLinearFn(slopes=slopes_a, intercepts=intercepts_a, bounds=bounds_a),
        PiecewiseLinearFn(slopes=slopes_b, intercepts=intercepts_b, bounds=bounds_b),
    )


def align_functions(
    a: PiecewiseConstantFn, b: PiecewiseConstantFn
) -> tuple[PiecewiseConstantFn, PiecewiseConstantFn]:
    if isinstance(a, PiecewiseConstantFn) and isinstance(b, PiecewiseConstantFn):
        return _align_constant_fns(a, b)
    elif isinstance(a, PiecewiseLinearFn) and isinstance(b, PiecewiseLinearFn):
        return _align_linear_fns(a, b)
    else:
        raise TypeError("Both functions must be of the same type")
