from __future__ import annotations

import bisect
import itertools
from collections.abc import Iterable
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
        return round(self.slope * i + self.intercept)


class PiecewiseConstantFn:
    def __init__(self, segments: Iterable[Segment], *, num_distinct: int) -> None:
        if any(not seg.is_constant() for seg in segments):
            raise ValueError("All segments must be constant")
        self._segments = list(segments)
        self._num_distinct = num_distinct

    def upper_bound(self, other: PiecewiseConstantFn) -> int:
        degrees, widths = self._align_segments(other)
        return degrees @ widths

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

    def _align_segments(
        self, other: PiecewiseConstantFn
    ) -> tuple[np.ndarray, np.ndarray]:
        degrees: list[int] = []
        widths: list[int] = []

        # TODO

        return np.asarray(degrees), np.asarray(widths)

    def __call__(self, i: int) -> int:
        # XXX: should we rather use np.searchsorted and maintain an ndarray with the upper bounds directly?
        idx = bisect.bisect_left(self._segments, x=i, key=lambda s: s.higher)
        if idx == len(self._segments):
            return round(self._segments[-1].intercept)

        segment = self._segments[idx]
        return round(segment.intercept)


class PiecewiseLinearFn:
    def __init__(self, segments: Iterable[Segment], *, num_distinct: int) -> None:
        self._segments = list(segments)
        self._num_distinct = num_distinct

    def upper_bound(self) -> int:
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

    @overload
    def __call__(self, i: int) -> int: ...

    @overload
    def __call__(self, i: PiecewiseLinearFn) -> PiecewiseLinearFn: ...

    def __call__(self, i):
        if isinstance(i, PiecewiseLinearFn):
            return self._fuse_with(i)

        # XXX: should we rather use np.searchsorted and maintain an ndarray with the upper bounds directly?
        idx = bisect.bisect_left(self._segments, x=i, key=lambda s: s.higher)
        if idx == len(self._segments):
            idx = -1

        segment = self._segments[idx]
        return segment(i)
