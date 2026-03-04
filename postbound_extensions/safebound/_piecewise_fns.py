from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Optional, Protocol

import numpy as np
import postbound as pb


class FunctionLike(Protocol):
    @property
    def n_distinct(self) -> int: ...

    def columns(self) -> set[pb.ColumnReference]: ...

    def evaluate_at(self, vals: np.ndarray) -> np.ndarray: ...

    def cumulative_at(self, vals: np.ndarray) -> np.ndarray: ...

    def invert_cumulative_at(self, vals: np.ndarray) -> np.ndarray: ...

    def cardinality(self) -> int: ...

    def __hash__(self) -> int: ...

    def __eq__(self, other: object) -> bool: ...


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
    def from_segments(
        segments: Iterable[Segment], *, column: Optional[pb.ColumnReference] = None
    ) -> PiecewiseConstantFn:
        values: list[float] = []
        bounds: list[int] = []
        for seg in segments:
            if not seg.is_constant():
                raise ValueError("All segments must be constant")
            values.append(seg.intercept)
            bounds.append(seg.higher)
        return PiecewiseConstantFn(values, bounds, column=column)

    @staticmethod
    def zero(column: Optional[pb.ColumnReference] = None) -> PiecewiseConstantFn:
        return PiecewiseConstantFn([0], [0], column=column)

    def __init__(
        self,
        values: Iterable[float],
        bounds: Iterable[int],
        *,
        column: Optional[pb.ColumnReference] = None,
    ) -> None:
        self.column = column

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

    def columns(self) -> set[pb.ColumnReference]:
        return set() if self.column is None else {self.column}

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

    def min_with(self, other: PiecewiseConstantFn) -> PiecewiseConstantFn:
        aligned_self, aligned_other = align_functions(self, other)
        return PiecewiseConstantFn(
            np.min([aligned_self._values, aligned_other._values], axis=0),
            aligned_self._bounds,
        )

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
        if self.column is not None:
            col_desc = f"for column {self.column}"
        else:
            col_desc = ""

        lines: list[str] = [
            f"PCF {col_desc}({len(self._values)} segments, {self._num_distinct} distinct values)"
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

    def __iter__(self) -> Iterator[tuple[float, int]]:
        return ((self._values[i], self._bounds[i]) for i in range(len(self)))

    def __add__(self, other: PiecewiseConstantFn) -> PiecewiseConstantFn:
        aligned_self, aligned_other = align_functions(self, other)
        values = aligned_self._values + aligned_other._values
        return PiecewiseConstantFn(values, aligned_self._bounds, column=self.column)

    def __mul__(self, other: PiecewiseConstantFn) -> PiecewiseConstantFn:
        aligned_self, aligned_other = align_functions(self, other)
        values = aligned_self._values * aligned_other._values
        return PiecewiseConstantFn(values, aligned_self._bounds)

    def __call__(self, vals: np.ndarray) -> np.ndarray:
        return self.evaluate_at(vals)

    def __hash__(self) -> int:
        return hash((self._values.tobytes(), self._bounds.tobytes()))

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, type(self))
            and np.array_equal(self._values, other._values)
            and np.array_equal(self._bounds, other._bounds)
        )


class PiecewiseLinearFn:
    @staticmethod
    def from_segments(
        segments: Iterable[Segment], column: Optional[pb.ColumnReference] = None
    ) -> PiecewiseLinearFn:
        slopes: list[float] = []
        intercepts: list[float] = []
        bounds: list[int] = []
        for seg in segments:
            slopes.append(seg.slope)
            intercepts.append(seg.intercept)
            bounds.append(seg.higher)

        return PiecewiseLinearFn(
            slopes=slopes, intercepts=intercepts, bounds=bounds, column=column
        )

    def __init__(
        self,
        *,
        slopes: Iterable[float],
        intercepts: Iterable[float],
        bounds: Iterable[int],
        column: Optional[pb.ColumnReference] = None,
    ) -> None:
        self.column = column

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
        return PiecewiseConstantFn(self._slopes, self._bounds, column=self.column)

    def invert(self) -> PiecewiseLinearFn:
        return PiecewiseLinearFn(
            slopes=1 / self._slopes,
            intercepts=-1 * self._intercepts,
            bounds=self._bounds,
            column=self.column,
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


def align_functions(
    a: PiecewiseConstantFn, b: PiecewiseConstantFn
) -> tuple[PiecewiseConstantFn, PiecewiseConstantFn]:
    values_a, values_b = [], []
    bounds_a, bounds_b = [], []

    iter_a, iter_b = iter(a), iter(b)
    cur_a, cur_b = next(iter_a, None), next(iter_b, None)
    while cur_a is not None or cur_b is not None:
        if cur_a is None:
            a_val, a_bound = None, None
        else:
            a_val, a_bound = cur_a

        if cur_b is None:
            b_val, b_bound = None, None
        else:
            b_val, b_bound = cur_b

        if cur_a is None:
            values_a.append(0)
            bounds_a.append(b_bound)

            values_b.append(b_val)
            bounds_b.append(b_bound)

            cur_b = next(iter_b, None)
            continue

        if cur_b is None:
            values_a.append(a_val)
            bounds_a.append(a_bound)

            values_b.append(0)
            bounds_b.append(a_bound)

            cur_a = next(iter_a, None)
            continue

        assert (
            a_val is not None
            and b_val is not None
            and a_bound is not None
            and b_bound is not None
        )

        if a_bound == b_bound:
            values_a.append(a_val)
            bounds_a.append(a_bound)

            values_b.append(b_val)
            bounds_b.append(b_bound)

            cur_a = next(iter_a, None)
            cur_b = next(iter_b, None)

        elif a_bound < b_bound:
            values_a.append(a_val)
            bounds_a.append(a_val)

            values_b.append(b_val)
            bounds_b.append(a_bound)

            cur_a = next(iter_a, None)

        else:
            assert a_bound > b_bound
            values_a.append(a_val)
            values_b.append(b_bound)

            values_b.append(b_val)
            values_b.append(b_bound)

            cur_b = next(iter_b, None)

    return (
        PiecewiseConstantFn(values_a, bounds_a, column=a.column),
        PiecewiseConstantFn(values_b, bounds_b, column=b.column),
    )
