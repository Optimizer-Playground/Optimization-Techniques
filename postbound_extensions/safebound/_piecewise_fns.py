from __future__ import annotations

import bisect
import itertools
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import overload

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
    def __init__(self, segments: Iterable[Segment], *, num_distinct: int) -> None:
        if any(not seg.is_constant() for seg in segments):
            raise ValueError("All segments must be constant")
        self._segments = list(segments)
        self._num_distinct = num_distinct

    @property
    def n_distinct(self) -> int:
        return self._num_distinct

    @property
    def segments(self) -> Sequence[Segment]:
        return self._segments

    def cardinality(self) -> int:
        widths = np.asarray(seg.width for seg in self._segments)
        degrees = np.asarray(seg.intercept for seg in self._segments)
        return degrees @ widths

    def integrate(self) -> PiecewiseLinearFn:
        first_seg = self._segments[0]
        linear_segments: list[Segment] = [
            Segment(
                first_seg.lower,
                first_seg.higher,
                first_seg.intercept,
                first_seg.intercept,
            )
        ]
        for seg in self._segments:
            prev = linear_segments[-1]
            linear_segments.append(
                Segment(seg.lower, seg.higher, seg.intercept, prev.final_freq())
            )

        return PiecewiseLinearFn(linear_segments, num_distinct=self._num_distinct)

    def __call__(self, i: int) -> int:
        # XXX: should we rather use np.searchsorted and maintain an ndarray with the upper bounds directly?
        idx = bisect.bisect_left(self._segments, x=i, key=lambda s: s.higher)
        if idx == len(self._segments):
            # out of bounds -> no distinct values left -> frequency is 0
            return 0

        segment = self._segments[idx]
        return round(segment.intercept)


class PiecewiseLinearFn:
    def __init__(self, segments: Iterable[Segment], *, num_distinct: int) -> None:
        self._segments = list(segments)
        self._num_distinct = num_distinct

    @property
    def n_distinct(self) -> int:
        return self._num_distinct

    @property
    def segments(self) -> Sequence[Segment]:
        return self._segments

    def cardinality(self) -> int:
        slopes: list[np.ndarray] = []
        intercepts: list[np.ndarray] = []
        for seg in self._segments:
            slopes.append(np.full(seg.width, seg.slope))
            intercepts.append(np.full(seg.width, seg.intercept))

        slope = np.concat(slopes)
        intercept = np.concat(intercepts)
        frequencies = slope * np.arange(1, self._num_distinct) + intercept
        return np.sum(frequencies)

    def derivative(self) -> PiecewiseConstantFn:
        constant_segments: list[Segment] = []
        for seg, followup in itertools.pairwise(self._segments):
            constant_segments.append(
                Segment.constant(followup.intercept, lower=seg.lower, higher=seg.higher)
            )

        last_seg = self._segments[-1]
        constant_segments.append(
            Segment.constant(
                last_seg.final_freq(), lower=last_seg.higher, higher=self._num_distinct
            )
        )

        return PiecewiseConstantFn(constant_segments, num_distinct=self._num_distinct)

    def invert(self) -> PiecewiseLinearFn:
        inverted_segments = [seg.invert() for seg in self._segments]
        return PiecewiseLinearFn(
            inverted_segments,
            num_distinct=self._num_distinct,
        )

    def _fuse_with(self, inner: PiecewiseLinearFn) -> PiecewiseLinearFn:
        pass

    def __call__(self, i: int) -> int:
        # XXX: should we rather use np.searchsorted and maintain an ndarray with the upper bounds directly?
        idx = bisect.bisect_left(self._segments, x=i, key=lambda s: s.higher)
        if idx == len(self._segments):
            # out of bounds -> no distinct values left -> frequency is 0
            return 0

        segment = self._segments[idx]
        return segment(i)


@overload
def align_functions(
    a: PiecewiseConstantFn, b: PiecewiseConstantFn
) -> tuple[PiecewiseConstantFn, PiecewiseConstantFn]: ...


@overload
def align_functions(
    a: PiecewiseLinearFn, b: PiecewiseLinearFn
) -> tuple[PiecewiseLinearFn, PiecewiseLinearFn]: ...


def align_functions(
    a: PiecewiseLinearFn | PiecewiseConstantFn,
    b: PiecewiseLinearFn | PiecewiseConstantFn,
) -> tuple[
    PiecewiseLinearFn | PiecewiseConstantFn, PiecewiseLinearFn | PiecewiseConstantFn
]:
    if not isinstance(a, type(b)) or not isinstance(b, type(a)):
        raise TypeError("Both functions must be of the same type")

    target = type(a)
    num_distinct = min(a.n_distinct, b.n_distinct)
    a_iter = iter(a.segments)
    b_iter = iter(b.segments)

    a_segments: list[Segment] = []
    b_segments: list[Segment] = []
    last_cutoff = 0
    current_a = next(a_iter, None)
    current_b = next(b_iter, None)
    while True:
        if current_a is None or current_b is None:
            break

        if current_a.higher == current_b.higher:
            a_segments.append(current_a)
            b_segments.append(current_b)
            last_cutoff = current_a.higher
            current_a = next(a_iter, None)
            current_b = next(b_iter, None)
        elif current_a.higher < current_b.higher:
            a_segments.append(current_a)
            b_segments.append(
                Segment(
                    lower=last_cutoff,
                    higher=current_a.higher,
                    slope=current_b.slope,
                    intercept=current_b(current_a.higher),
                )
            )
            last_cutoff = current_a.higher
            current_a = next(a_iter, None)
        else:
            a_segments.append(
                Segment(
                    lower=last_cutoff,
                    higher=current_b.higher,
                    slope=current_a.slope,
                    intercept=current_a(current_b.higher),
                )
            )
            b_segments.append(current_b)
            last_cutoff = current_b.higher
            current_b = next(b_iter, None)

    aligned_a = target(a_segments, num_distinct=num_distinct)
    aligned_b = target(b_segments, num_distinct=num_distinct)
    return aligned_a, aligned_b
