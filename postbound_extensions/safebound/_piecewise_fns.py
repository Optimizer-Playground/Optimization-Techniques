from __future__ import annotations

import itertools
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

    def is_constant(self) -> bool:
        return self.slope == 0.0


class PiecewiseConstantFn:
    def __init__(self, segments: Iterable[Segment]) -> None:
        if any(not seg.is_constant() for seg in segments):
            raise ValueError("All segments must be constant")
        self._segments = list(segments)

    def upper_bound(self, other: PiecewiseConstantFn) -> int:
        degrees, widths = self._align_segments(other)
        return degrees @ widths

    def _align_segments(
        self, other: PiecewiseConstantFn
    ) -> tuple[np.ndarray, np.ndarray]:
        degrees: list[int] = []
        widths: list[int] = []

        # TODO

        return np.asarray(degrees), np.asarray(widths)


class PiecewiseLinearFn:
    def __init__(
        self, segments: Iterable[Segment], *, num_distinct: int, final_card: int
    ) -> None:
        self._segments = list(segments)
        self._num_distinct = num_distinct
        self._final_card = final_card

    def derivative(self) -> PiecewiseConstantFn:
        constant_segments: list[Segment] = []
        for seg, followup in itertools.pairwise(self._segments):
            constant_segments.append(
                Segment.constant(followup.intercept, lower=seg.lower, higher=seg.higher)
            )

        last_seg = self._segments[-1]
        constant_segments.append(
            Segment.constant(
                self._final_card, lower=last_seg.higher, higher=self._num_distinct
            )
        )

        return PiecewiseConstantFn(constant_segments)
