from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import Optional, Protocol

import numpy as np
import postbound as pb
from numpy.typing import NDArray


class FunctionLike(Protocol):
    """Function-like is the basic building block of the bound estimation algorithm.

    The interface is shared between plain piecewise constant functions, alpha steps, and beta steps.

    See Also
    --------
    PiecewiseConstantFn
    AlphaStep
    BetaStep
    """

    @property
    def n_distinct(self) -> int:
        """Get the number of distinct values in the output."""
        ...

    def columns(self) -> set[pb.ColumnReference]:
        """Provides all join columns that are part of this function, including nested steps."""
        ...

    def evaluate_at(self, vals: np.ndarray) -> np.ndarray:
        """Computes the output frequencies of the given PCF indexes.

        While this description might seem a bit cryptic, remember that SafeBound assumes that the
        i-th most frequent value of one join column is always joined with the i-th most frequent
        value of the partner join column. Using this function, we compute the output frequencies
        of a batch of PCF indexes (the most frequent values at different positions).
        """
        ...

    def cumulative_at(self, vals: np.ndarray) -> np.ndarray:
        """Computes the cumulative output frequencies of the given PCF indexes.

        This method is similar to `evaluate_at` with the key difference that we compute the
        cumulative frequencies (i.e. including all higher-frequency values) instead of the
        frequencies at the specific indexes.
        """
        ...

    def invert_cumulative_at(self, vals: np.ndarray) -> np.ndarray:
        """Computes the PCF indexes that reach the given cumulative frequencies.

        Basically, this method functions as the inverse to `cumulative_at`. Instead of computing the
        cumulative frequencies at specific indexes, it computes the indexes at specific cumulative
        frequencies.
        """
        ...

    def cardinality(self) -> int:
        """Computes the upper bound of the output relation's cardinality."""
        ...

    def __hash__(self) -> int: ...

    def __eq__(self, other: object) -> bool: ...


class DegreeSequence:
    """A degree sequence is the core building block of SafeBound.

    Essentially, each sequence is an ordered list of column frequencies. SafeBound uses these
    frequencies to estimate the cardinality of joins. This works under the pessimistic assumption
    that the i-th most frequent value of one join column joins with the i-th most frequent value of
    the other join column. Mathematically, this boils down to a dot product of two (equal length)
    degree sequences.

    One key disadvantage of degree sequences is their size: a degree sequence is as long as there
    are distinct values in a column. This makes them impractical in their raw form. Therefore,
    SafeBound uses a much cheaper approximation, the piecewise constant function. All practical
    estimation happens on this type, we just use the degree sequence as raw input for the higher-up
    processing (compression, query decomposition, bound estimation).

    We provide three ways to create new degree sequences:

    1. converting an existing most common values list using `from_mcv`,
    2. direct creation for a primary key (or UNIQUE-constrained) column using `for_primary_key`,
    3. passing the frequencies directly via the constructor

    Parameters
    ----------
    degrees : Iterable[int | pb.Cardinality] | NDArray[np.int_]
        The column frequencies. These do not need to be ordered up front. Invalid cardinalities
        are skipped automatically.
    column : Optional[ColumnReference], optional
        The column for which the degree sequence is created. This is used by the higher-up
        processing and should only be omitted for testing/debugging purposes.

    See Also
    --------
    PiecewiseConstantFn
    valid_compress
    """

    @staticmethod
    def from_mcv(
        mcv: pb.db.MostCommonValues, *, column: Optional[pb.ColumnReference] = None
    ) -> DegreeSequence:
        """Creates a new degree sequence for a specific most-common values list.

        The column is used by the higher-up processing and should only be omitted for
        testing/debugging purposes.
        """
        return DegreeSequence(mcv.frequencies, column=column)

    @staticmethod
    def for_primary_key(
        n_values: int, *, column: Optional[pb.ColumnReference] = None
    ) -> DegreeSequence:
        """Creates a new degree sequence for a PRIMARY KEY/UNIQUE column.

        For such a column, all frequencies are by definition 1. Therefore, the degree sequence can
        be derived directly from the number of distinct values.

        The column is used by the higher-up processing and should only be omitted for
        testing/debugging purposes.
        """
        return DegreeSequence(np.ones(n_values), column=column)

    def __init__(
        self,
        degrees: Iterable[int | pb.Cardinality] | NDArray[np.int_],
        *,
        column: Optional[pb.ColumnReference] = None,
    ) -> None:
        if not isinstance(degrees, np.ndarray):
            degrees = [
                int(deg)
                for deg in degrees
                if not isinstance(deg, pb.Cardinality) or deg.is_valid()
            ]
            degrees = np.asarray(degrees)
        degrees = np.sort(degrees)[::-1]  # see https://stackoverflow.com/q/26984414
        self._degrees: NDArray[np.int_] = np.array(degrees)
        self._column = column

    @property
    def column(self) -> Optional[pb.ColumnReference]:
        """Get the column to which this degree sequence belongs.

        The column should always be set and only omitted during debugging/testing.
        """
        return self._column

    @property
    def max_deg(self) -> int:
        """Get the maximum degree stored in this sequence."""
        return self._degrees[0]

    @property
    def min_deg(self) -> int:
        """Get the minimum degree stored in this sequence."""
        return self._degrees[-1]

    @property
    def max_freq(self) -> int:
        """Get the maximum degree stored in this sequence.

        This is just an alias for `max_deg`.
        """
        return self._degrees[0]

    @property
    def degrees(self) -> Sequence[int]:
        """Get all frequencies/degrees stored in this sequence.

        The frequencies are ordered with the highest values appearing first.
        """
        return self._degrees.tolist()

    @property
    def distinct_values(self) -> int:
        """Get the number of elements in this sequence (i.e. its length)."""
        return len(self)

    @property
    def cardinality(self) -> int:
        """Get the total number of rows stored in this sequence."""
        return int(np.sum(self._degrees))

    def join_bound(self, other: DegreeSequence) -> int:
        """Calculates the cardinality of the join result between this sequence and the `other` one.

        The cardinality is computed under the pessimistic assumption that the i-th most frequent
        value from this sequence joins with the i-th most frequent value from `other`. Both
        sequences must have the same number of elements.
        """
        if len(self._degrees) != len(other._degrees):
            raise ValueError(
                "Degree sequences must have the same length for join bound calculation"
            )
        return int(self._degrees @ other._degrees)

    def expand_to(self, n_elems: int) -> DegreeSequence:
        """Modifies the length of this degree sequence to contain exactly `n_elems` entries.

        If the sequence currently contains more elements, only the `n_elems` most frequent ones are
        kept. Otherwise, elements with 0 frequency are inserted up to `n_elems`.
        """
        current_len = len(self)
        if current_len < n_elems:
            return DegreeSequence(self._degrees[:n_elems], column=self._column)

        padding = np.zeros(n_elems - current_len)
        adjusted = np.concat([self._degrees, padding])
        return DegreeSequence(adjusted, column=self._column)

    def __len__(self) -> int:
        return len(self._degrees)

    def __getitem__(self, i) -> int:
        return self._degrees[i]

    def __le__(self, other: DegreeSequence) -> bool:
        if len(self) != len(other):
            raise ValueError(
                "Degree sequences must have the same length for comparison"
            )
        return bool(np.min(other._degrees - self._degrees) >= 0)

    def __repr__(self) -> str:
        degrees = repr(self._degrees.tolist())
        return f"DegreeSequence({degrees})"

    def __str__(self) -> str:
        return str(self._degrees.tolist())


@dataclass
class Segment:
    """A segment models a part of a piecewise linear function or piecewise constant function (PCF).

    To check whether this segment might belong to a PCF, use `is_constant`. If this is true, the
    slope is guaranteed to be 0.

    Whether segments are modelled as closed intervals or (half) open intervals (and which half is
    open) depends entirely on the user. This class does not assume nor restrict anything about the
    semantics.

    A segment can be called like a normal function to evaluate the y value for a specific x.
    This assumes that the requested x value falls in the interval [lower, higher).
    """

    lower: int
    """The smallest x value in this segment.

    See the general documentation on Segment for open vs. closed intervals.
    """

    higher: int
    """The largest x value in this segment.

    See the general documentation on Segment for open vs. closed intervals.
    """

    slope: float
    """The slope of this segment.

    For segments belonging to a piecewise constant function, this will be 0.
    """

    intercept: float
    """The intercept of this segment at `lower`."""

    @staticmethod
    def constant(intercept: float, *, lower: int, higher: int) -> Segment:
        """Creates a new segment for a piecewise constant function."""
        return Segment(lower, higher, 0.0, intercept)

    @staticmethod
    def initial(slope: int = 0, *, intercept: int = 0) -> Segment:
        """Creates the first segment of a new piecewise function (linear or constant).

        By default, both slope and intercept are set to 0.
        """
        return Segment(0, 0, slope, intercept)

    @staticmethod
    def after(last: Segment, *, slope: float, higher: Optional[int] = 0) -> Segment:
        """Creates a new segment that "picks up" after a previous segment.

        The lower bound of the segment as well as the intercept are inferred directly based on the
        previous segment. The slope must be given explicitly.

        If the upper bound is omitted, it is set to the previous segment's upper bound. This needs
        to be adjusted later on.
        """
        higher = higher or last.higher
        intercept = last.intercept + last.slope * (last.higher - last.lower)
        return Segment(last.higher, higher, slope, intercept)

    @property
    def width(self) -> int:
        """Get the width of the interval [lower, higher]."""
        return self.higher - self.lower

    def is_constant(self) -> bool:
        """Checks, whether this segment is constant, i.e. does not have any width."""
        return self.slope == 0.0

    def final_freq(self) -> float:
        """Computes the frequency (i.e. y value) at the segment's upper bound."""
        span = self.higher - self.lower
        return self.slope * span + self.intercept

    def invert(self) -> Segment:
        """Computes the inverse segment.

        This segment uses the same interval, but slope and and intercept are inverted.
        """
        return Segment(self.lower, self.higher, 1 / self.slope, -self.intercept)

    def __call__(self, i: int) -> int:
        if not self.lower <= i < self.higher:
            raise ValueError(
                f"i={i} is out of bounds for segment [{self.lower}, {self.higher})"
            )
        return round(self.slope * i + self.intercept)


class PiecewiseConstantFn:
    """A piecewise constant function (PCF) is a complex function composed of constant functions.

    The segments are aligned in such a way that there are no gaps between segments and no segments
    overlap.

    PCFs are great for compression since we only need to store two values
    per segment: its upper bound and the intercept of the segment. This makes them the central data
    structure for bound estimation and statistics storage in SafeBound.

    A PCF instance implements the standard `FunctionLike` protocol. This allows for easy integration
    into the bound estimation algorithm.

    Parameters
    ----------
    values : Iterable[float]
        The intercepts of the individual segments
    bounds : Iterable[int]
        The upper bounds of the individual segments. It is assumed that these are strictly
        monotonic. The first segment is assumed to have a lower bound of 0. The lower bounds of all
        other segments are implictly defined by the upper bound of their predecessor (as a half-open
        range).
    column : Optional[ColumnReference], optional
        The column for which the PCF is created. This is used by the higher-up processing and should
        only be omitted for testing/debugging purposes.
    """

    @staticmethod
    def from_segments(
        segments: Iterable[Segment], *, column: Optional[pb.ColumnReference] = None
    ) -> PiecewiseConstantFn:
        """Creates a new PCF.

        All segments must be constant. Furthermore, it is assumed that segments are already ordered
        according to their interval and that there are no gaps in between. Otherwise, such gaps
        will be closed using the intercept of the current segment.
        """
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
        """Creates a new 0-PCF.

        This PCF contains a single segment with an intercept of 0 and an upper bound of 0. Use
        `align_functions` to extend the shape of the PCF to match the shape of a different function.
        """
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
        if len(self._values) == 0:
            raise ValueError("Empty PCFs are not allowed")

        self._bounds = np.asarray(bounds)
        if len(self._values) != len(self._bounds):
            raise ValueError(
                "values and bounds must be the same length "
                f"({len(self._values)} vs. {len(self._bounds)}): "
                f"values = {self._values}, bounds = {self._bounds}"
            )

        self._widths = np.diff(np.concat(([0], self._bounds)))
        self._num_distinct = self._bounds[-1]
        self._cumulative = np.cumsum(self._values * self._widths)
        self._cum_widths = np.cumsum(self._widths)

    @property
    def values(self) -> np.ndarray:
        """Get the intercepts/values of each segment."""
        return self._values

    @property
    def bounds(self) -> np.ndarray:
        """Get the upper bounds of each segment.

        The first segment is assumed to have a lower bound of 0. The lower bounds of all
        other segments are implictly defined by the upper bound of their predecessor (as a half-open
        range).
        """
        return self._bounds

    @property
    def n_distinct(self) -> int:
        """Get the number of distinct values in the output PCF."""
        return self._num_distinct

    def columns(self) -> set[pb.ColumnReference]:
        """Provides all join columns that are part of this alpha step, including nested steps."""
        return set() if self.column is None else {self.column}

    def cardinality(self) -> int:
        """Computes the upper bound of the output relation's cardinality."""
        return self._values @ self._widths

    def integ(self) -> PiecewiseLinearFn:
        """Integrates the PCF, which yields a piecewise linear function.

        Integration works by integrating each segment individually and aligning their intercepts.
        """
        widths = np.diff(np.concat(([0], self._bounds)))
        intercepts = np.cumsum(widths * self._values)
        return PiecewiseLinearFn(
            slopes=self._values, intercepts=intercepts, bounds=self._bounds
        )

    def cut_at(self, n_distinct: int) -> PiecewiseConstantFn:
        """Cuts the PCF after the first `n_distinct` values.

        If the PCF contains less than `n_distinct` values in total, it is returned as-is.
        Note that we are using *values* here, not *segments*. Each segment typically spans multiple
        values.
        """
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
        """Provides two aligned versions of the PCFs as (algined_self, aligned_other).

        See Also
        --------
        align_functions : for details about the alginment logic and its use-cases
        """
        return align_functions(self, other)

    def min_with(self, other: PiecewiseConstantFn) -> PiecewiseConstantFn:
        """Provides a new PCF that contains the minimum intercept at each value.

        The PCFs are algined as necessary. Note that the resulting PCF might contain neighboring
        segments with the same intercepts. While those could be merged into a single segment, we
        currently do not perform such an optimization.
        """
        aligned_self, aligned_other = align_functions(self, other, cut_early=True)
        if self.column is None:
            col = other.column
        elif other.column is None:
            col = self.column
        elif self.column == other.column:
            col = self.column
        else:
            col = None
        return PiecewiseConstantFn(
            np.min([aligned_self._values, aligned_other._values], axis=0),
            aligned_self._bounds,
            column=col,
        )

    def evaluate_at(self, vals: np.ndarray) -> np.ndarray:
        """Computes the output frequencies of the given PCF indexes.

        While this description might seem a bit cryptic, remember that SafeBound assumes that the
        i-th most frequent value of one join column is always joined with the i-th most frequent
        value of the partner join column. Using this function, we compute the output frequencies
        of a batch of PCF indexes (the most frequent values at different positions).
        """
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
        """Computes the cumulative output frequencies of the given PCF indexes.

        This method is similar to `evaluate_at` with the key difference that we compute the
        cumulative frequencies (i.e. including all higher-frequency values) instead of the
        frequencies at the specific indexes.
        """
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
        """Computes the PCF indexes that reach the given cumulative frequencies.

        Basically, this method functions as the inverse to `cumulative_at`. Instead of computing the
        cumulative frequencies at specific indexes, it computes the indexes at specific cumulative
        frequencies.
        """
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
        """Provides a human-readable representation of the PCF."""
        lines = self._inspect_internal()
        return "\n".join(lines)

    def _inspect_internal(self) -> list[str]:
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

        segno_padding = len(str(len(self._values) - 1))
        bound_padding = len(str(round(max_bound, 3)))
        freq_padding = len(str(max_freq))
        total_padding = len(str(round(max_total)))

        for i in range(len(self._values)):
            freq = self._values[i]
            bound = round(self._bounds[i], 3)
            total = round(self._cumulative[i])
            lines.append(
                f" +-- Segment {i:>{segno_padding}}: "
                f"range=[{prev_bound:>{bound_padding}}, {bound:>{bound_padding}}), "
                f"value={freq:>{freq_padding}} "
                f"({total:>{total_padding}} total)"
            )

            prev_bound = bound

        return lines

    def __json__(self) -> pb.util.jsondict:
        return {
            "values": self._values.tolist(),
            "bounds": self._bounds.tolist(),
            "column": self.column,
        }

    def __len__(self) -> int:
        return len(self._values)

    def __iter__(self) -> Iterator[tuple[float, int]]:
        return ((self._values[i], self._bounds[i]) for i in range(len(self)))

    def __add__(self, other: PiecewiseConstantFn) -> PiecewiseConstantFn:
        aligned_self, aligned_other = align_functions(self, other, cut_early=False)
        values = aligned_self._values + aligned_other._values
        return PiecewiseConstantFn(values, aligned_self._bounds, column=self.column)

    def __mul__(self, other: PiecewiseConstantFn) -> PiecewiseConstantFn:
        aligned_self, aligned_other = align_functions(self, other, cut_early=True)
        values = aligned_self._values * aligned_other._values
        if self.column is None:
            col = other.column
        elif other.column is None:
            col = self.column
        elif self.column == other.column:
            col = self.column
        else:
            col = None
        return PiecewiseConstantFn(values, aligned_self._bounds, column=col)

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

    def _repr_pretty_(self, p, cycle):
        if cycle:
            p.text("PiecewiseConstantFn(...)")
            return

        lines = self._inspect_internal()
        head, tail = lines[0], lines[1:]
        p.text(head)
        for line in tail:
            p.breakable()
            p.text(line)

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return f"PCF({self.column})" if self.column else "PCF(<anonymous>)"


class PiecewiseLinearFn:
    """A piecewise linear functions is divided into segments of simple linear functions.
    
    The segments are aligned in such a way that there are no gaps between segments and no segments overlap.

    This class differs from `PiecewiseConstantFn` in that the segments are linear instead of constant. This makes it
    more expressive, but also more expensive to store and evaluate. By computing the derivative of a piecewise linear
    function, we can get a piecewise constant function and by integrating a piecewise constant function we get the
    piecewise linear function back.
    Still, SafeBound explicitly uses PLFs only as the output of the compression algorithm. All other processing (mostly
    the bound calculation) can also be expressed by computing cumulative sums of PCFs. Therefore, the main purpose of this
    class is clarity of our implementation and a close mapping to the definitions in the paper.

    Parameters
    ----------
    slopes : Iterable[float]
        The slopes of the individual segments
    intercepts : Iterable[float]
        The intercepts of the individual segments at their lower bound. Must contain the same number of elements as
        `slopes`.
    bounds : Iterable[int]
        The upper bounds of the individual segments. It is assumed that these are strictly monotonic. The first segment is
        assumed to have a lower bound of 0. The lower bounds of all other segments are implictly defined by the upper bound
        of their predecessor (as a half-open range). Must contain the same number of elements as `slopes`.
    column : Optional[ColumnReference], optional
        The column for which the PLF is created. This is used by the higher-up processing and should only be omitted for
        testing/debugging purposes.
    """

    @staticmethod
    def from_segments(
        segments: Iterable[Segment], column: Optional[pb.ColumnReference] = None
    ) -> PiecewiseLinearFn:
        """Creates a new PCF.

        Segments can be either linear or constant. Furthermore, it is assumed that segments are already ordered
        according to their interval and that there are no gaps in between. Otherwise, such gaps
        will be closed using the intercept of the current segment.
        """
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
        """Get the number of distinct values in the output PLF."""
        return self._num_distinct

    @property
    def slopes(self) -> np.ndarray:
        """Get the slopes of each segment."""
        return self._slopes

    @property
    def intercepts(self) -> np.ndarray:
        """Get the intercepts of each segment."""
        return self._intercepts

    @property
    def bounds(self) -> np.ndarray:
        """Get the bounds of each segment."""
        return self._bounds

    def deriv(self) -> PiecewiseConstantFn:
        """Computes the derivative of the PLF, which yields a piecewise constant function."""
        return PiecewiseConstantFn(self._slopes, self._bounds, column=self.column)

    def invert(self) -> PiecewiseLinearFn:
        """Computes the inverse of the PLF.

        The inverse is computed by inverting each segment individually and aligning their intercepts.
        """
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
    a: PiecewiseConstantFn, b: PiecewiseConstantFn, *, cut_early: bool = False
) -> tuple[PiecewiseConstantFn, PiecewiseConstantFn]:
    """Ensures that two PCFs have the same bounds.

    The bound alignment works by splitting segments as necessary. For example, consider the following two PCFs:

    PCF A: 10 for [0, 10], 5 for (10, 15], 1 for (15, 20]
    PCF B: 5 for [0, 5], 4 for (5, 20]

    The alignment logic splits the first segment of PCF A into two segments with the same value: 10 for [0, 5], and
    10 for (5, 10]. The other segments of PCF A remain unchanged. The first segment of PCF B remains unchanged, while
    the other two segments are split to align with the segments of PCF A: 4 for (10, 15], and 4 for (15, 20].
    
    As a result of the alignment process, the two PCFs have the same bounds and the same number of segments.

    If `cut_early` is enabled, the longer PCF is cut after the last bound of the shorter PCF. All further values are
    dropped. Otherwise, the shorter PCF is extended with additional 0 segments up to the last bound of the longer PCF.

    Depending on the use case, one or the other behavior might improve performance because the number of required
    computations is reduced.
    """
    values_a, values_b = [], []
    bounds_a, bounds_b = [], []

    iter_a, iter_b = iter(a), iter(b)
    cur_a, cur_b = next(iter_a, None), next(iter_b, None)
    while cur_a is not None or cur_b is not None:
        if cut_early and (cur_a is None or cur_b is None):
            break

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
            bounds_a.append(a_bound)

            values_b.append(b_val)
            bounds_b.append(a_bound)

            cur_a = next(iter_a, None)

        else:
            assert a_bound > b_bound
            values_a.append(a_val)
            bounds_a.append(b_bound)

            values_b.append(b_val)
            bounds_b.append(b_bound)

            cur_b = next(iter_b, None)

    aligned_a = PiecewiseConstantFn(values_a, bounds_a, column=a.column)
    aligned_b = PiecewiseConstantFn(values_b, bounds_b, column=b.column)
    return aligned_a, aligned_b


def load_pcf_json(json_data: dict | str) -> PiecewiseConstantFn:
    """Restores a PCF from its JSON representation."""
    if isinstance(json_data, str):
        json_data = json.loads(json_data)

    column = pb.parser.load_column_json(json_data["column"])
    bounds = json_data["bounds"]
    values = json_data["values"]
    return PiecewiseConstantFn(values, bounds, column=column)
