from __future__ import annotations

import bisect
import collections
import itertools
import json
import lzma
from collections.abc import Iterable, Iterator, Mapping, Sequence
from datetime import datetime, timedelta
from pathlib import Path
from typing import Protocol, override

import numpy as np
import postbound as pb

from postbound_extensions.preprocessing import cast_value

from ..safebound import (
    CatalogSpec,
    DegreeSequence,
    catalog_from_schema,
    catalog_from_workload,
    fetch_column_distribution,
    fetch_correlated_ds,
    fetch_mcv_ds,
    fetch_non_mcv_ds,
    fetch_raw_ds,
)
from ..util import make_json_parser, wrap_logger
from ._core import LpBoundSpec, NormSet


class LpNorms:
    """Collects all Lp norms that are computed for a specific join key.

    The norms are stored as a mapping from the p-value (0, 1, ..., inf) to the actual norm. The p-values can be iterated
    over and individual norms can be accessed with *__getitem__* semantics. In addition, `norms` provides quick access
    to all norms of p-values with p > 0. Those are the norms that actually feed into the linear programs.

    Different sets of Lp norms can be combined using `max_with`, `min_with`, and `sum_with`. Note that these operations
    are only defined for collections of Lp norms with the same p-values and raise otherwise. An empty collection is
    the only exception and can be combined with any other instance.
    """

    @staticmethod
    def zeros() -> LpNorms:
        """Constructs a collection of all-zero Lp norms."""
        return LpNorms({})

    def __init__(self, norms: Mapping[float, float]) -> None:
        self._entries = dict(norms)
        self._norms = set(self._entries.keys())

    @property
    def norms(self) -> Mapping[float, float]:
        """Get all Lp norms with p > 0, mapping p -> norm value."""
        return {norm: val for norm, val in self._entries.items() if norm > 0}

    def is_zero(self) -> bool:
        """Checks, whether all norms are zero."""
        return all(val == 0 for val in self._entries.values())

    def max_with(self, other: LpNorms) -> LpNorms:
        """Computes the element-wise maximum of two collections of Lp norms.

        If the collections contain different p-values, an error is raised.
        """
        if self.is_zero():
            return other
        if other.is_zero():
            return self

        diff = self._norms.symmetric_difference(other._norms)
        if diff:
            raise ValueError(f"Cannot compute max_with for collections with different norms: {diff}")
        return LpNorms({norm: max(self[norm], other[norm]) for norm in self._norms})

    def min_with(self, other: LpNorms) -> LpNorms:
        """Computes the element-wise minimum of two collections of Lp norms.

        If the collections contain different p-values, an error is raised.
        """
        if self.is_zero() or other.is_zero():
            return LpNorms.zeros()

        diff = self._norms.symmetric_difference(other._norms)
        if diff:
            raise ValueError(f"Cannot compute min_with for collections with different norms: {diff}")
        return LpNorms({norm: min(self[norm], other[norm]) for norm in self._norms})

    def sum_with(self, other: LpNorms) -> LpNorms:
        """Computes the element-wise sum of two collections of Lp norms.

        If the collections contain different p-values, an error is raised.
        """
        if self.is_zero():
            return other
        if other.is_zero():
            return self

        diff = self._norms.symmetric_difference(other._norms)
        if diff:
            raise ValueError(f"Cannot compute sum_with for collections with different norms: {diff}")
        return LpNorms({norm: self[norm] + other[norm] for norm in self._norms})

    def get(self, norm: float, default: float | None) -> float | None:
        """Provides the norm for a specific p-value, or a default if it does not exist."""
        return self._entries.get(norm, default)

    def __json__(self) -> pb.util.jsondict:
        # JSON represents all keys as string.
        # Let's avoid all parsing shenanigans (i.e. forgetting to parse to float) and just use tuples.
        return [(norm, val) for norm, val in self._entries.items()]

    def __iter__(self) -> Iterator[float]:
        return iter(self._entries)

    def __getitem__(self, norm: float) -> float:
        if not self._entries:
            return 0
        return self._entries[norm]


def compute_norms(ds: DegreeSequence, *, norms: NormSet) -> LpNorms:
    """Computes the Lp norms for a given degree sequence."""
    degrees = ds.deg_array
    lpcoll: dict[float, float] = {}
    for norm in norms:
        lpcoll[norm] = np.linalg.norm(degrees, ord=norm)
    return LpNorms(lpcoll)


def load_norm_json(json_data: dict | str) -> LpNorms:
    """Restores a Lp norms collection from its JSON representation."""
    if isinstance(json_data, str):
        json_data = json.loads(json_data)
    return LpNorms({float(norm): float(val) for norm, val in json_data})


class EqualityConditionedNorms[T]:
    """Represents the Lp norms for equality predicates over an MCV list.

    The norms for each MCV value are stored separately, and there is an additional "unconditioned" norm
    that represents the frequencies of all non-MCV values combined.

    The column that the norms are conditioned on (i.e. the column containing the MCV values) is stored
    in `filter_col`. It is the client's responsibility to keep track of the join column that the norms
    are associated with.

    Use the `get` method to obtain the norms for a specific value.
    """

    def __init__(
        self,
        norms: Mapping[T | None, LpNorms],
        *,
        filter_col: pb.BoundColumnReference,
        fallback: LpNorms,
    ) -> None:
        self._norms = dict(norms)
        self._fallback = fallback
        self._col = filter_col

    @property
    def filter_col(self) -> pb.BoundColumnReference:
        return self._col

    def get(self, key: T | None) -> LpNorms:
        """Returns the norms for the given filter key if it exists, otherwise returns the unconditioned norms."""
        norm = self._norms.get(key)
        if norm is None:
            return self._fallback
        return norm

    def __json__(self) -> pb.util.jsondict:
        norms_json = [{"value": val, "norms": norms} for val, norms in self._norms.items()]
        return {"filter_column": self._col, "norms": norms_json, "fallback": self._fallback}


def load_eq_norm_json(json_data: dict | str, *, database: pb.Database) -> EqualityConditionedNorms:
    """Reconstructs equality-conditioned norms from their JSON representation.

    Access to the database is required to parse the column values based on their data type.
    """
    if isinstance(json_data, str):
        json_data = json.loads(json_data)

    filter_col = pb.parser.load_column_json(json_data["filter_column"])
    dtype = database.schema().datatype(filter_col)
    val_parser = make_json_parser(dtype)

    fallback = load_norm_json(json_data["fallback"])
    norms = {val_parser(norm_json["value"]): load_norm_json(norm_json["norms"]) for norm_json in json_data["norms"]}

    return EqualityConditionedNorms(norms, filter_col=filter_col, fallback=fallback)


class EqualityConditionedRepo:
    """Maintains the equality-conditioned norms for a set of join columns.

    Use the `lookup` method to obtain the norms for a specific join column and filter value.

    Parameters
    ----------
    join_norms : dict[pb.ColumnReference, Sequence[EqualityConditionedNorms]]
        A mapping from join column to the equality-conditioned norms for that column. Each set of norms should
        be conditioned on a different filter column.
    """

    def __init__(self, join_norms: Mapping[pb.ColumnReference, Sequence[EqualityConditionedNorms]]) -> None:
        self._norms = dict(join_norms)

    def lookup(
        self, join_col: pb.ColumnReference, *, filter_col: pb.ColumnReference, filter_val: object
    ) -> LpNorms | None:
        """Retrieves the Lp norms set for a join column based on the condition *filter_col = filter_val*.

        If the repository does not contain norms conditioned on the filter column, *None* is
        returned. However, if there is a matching conditioned Lp norm, but it does not contain the
        filter value (e.g. because the value is not in the MCV list), its unconditioned Lp norm is
        returned.
        """
        join_col = join_col.drop_table_alias()
        filter_col = filter_col.drop_table_alias()

        norms = self._norms.get(join_col)
        if norms is None:
            return None

        for norm in norms:
            if norm.filter_col != filter_col:
                continue
            return norm.get(filter_val)
        return None

    def __json__(self) -> pb.util.jsondict:
        norms_json = [{"join_column": join_col, "norms": norms} for join_col, norms in self._norms.items()]
        return {"norms": norms_json}


def build_equality_norms(
    spec: CatalogSpec, *, mcv_size: int, norms: NormSet, database: pb.Database, verbose: bool | pb.util.Logger = False
) -> EqualityConditionedRepo:
    """Fetches all equality-conditioned Lp norms for the given catalog.

    Parameters
    ----------
    spec : CatalogSpec
        Specification that contains the join column/equality column pairs to build.
    mcv_size : int
        The number of entries for each MCV list. All entries in the MCV list are used to build
        separate conditioned norms, as well as one additional "unconditioned" Lp norm that combines all
        non-MCV values.
    norms : NormSet
        The Lp norms that should be computed for each join key.
    database : pb.Database
        The database to fetch the degree sequences from.
    verbose : bool | pb.util.Logger, optional
        Either a pre-defined logger, or a boolean that indicates whether to log the progress using a
        default logger.

    Returns
    -------
    EqualityConditionedRepo
        A repository that contains all specified Lp norms.
    """
    log = wrap_logger(verbose)
    repo = collections.defaultdict(list)

    for join_col, filter_cols in spec.equality_cols.items():
        for filter_col in filter_cols:
            log(f"Building MCV norms for {join_col} on {filter_col}")
            correlated_ds = fetch_mcv_ds(filter_col, on=join_col, database=database, num_mcv=mcv_size)
            correlated_norms = {val: compute_norms(ds, norms=norms) for val, ds in correlated_ds.items()}

            log(f"Building non-mcv norms for {join_col} on {filter_col}")
            non_mcv_ds = fetch_non_mcv_ds(filter_col, on=join_col, database=database, num_mcv=mcv_size)
            non_mcv_norms = compute_norms(non_mcv_ds, norms=norms)

            repo[join_col].append(
                EqualityConditionedNorms(correlated_norms, filter_col=filter_col, fallback=non_mcv_norms)
            )

    return EqualityConditionedRepo(repo)


def load_eq_repo_json(json_data: dict | str, *, database: pb.Database) -> EqualityConditionedRepo:
    """Reconstructs an equality-conditioned Lp norms repository from its JSON representation.

    Access to the database is required to parse the column values based on their data type.
    """
    if isinstance(json_data, str):
        json_data = json.loads(json_data)

    repo: dict[pb.ColumnReference, Sequence[EqualityConditionedNorms]] = {}
    for entry in json_data["norms"]:
        join_col = pb.parser.load_column_json(entry["join_column"])
        norms = [load_eq_norm_json(norm_json, database=database) for norm_json in entry["norms"]]
        repo[join_col] = norms
    return EqualityConditionedRepo(repo)


class _HistogramKey(Protocol):
    def __lt__(self, other: _HistogramKey) -> bool: ...

    def __le__(self, other: _HistogramKey) -> bool: ...

    def __ge__(self, other: _HistogramKey) -> bool: ...


class RangeConditionedNorms[T: _HistogramKey]:
    """Represents the Lp norms for range predicates over a (hierarchy of) histogram(s).

    Each bucket in the histogram contains its own set of Lp norms. These norms model the
    join column conditioned on the filter column taking values in the bucket's range.

    Optionally, each histogram can have a "lower resolution" child histogram that models the same
    join column conditioned on the same filter column, but with a coarser granularity (i.e. less
    buckets). This results in additional bucket boundaries that might over-estimate the true
    frequencies by a smaller margin.

    Use the `get_range`, `get_less` and `get_greater` methods to obtain the norms for specific
    predicates.

    Notes
    -----
    The original LpBound paper does not specify how to handle half-open ranges (e.g. ``R.a < 42``),
    but only closed ranges (e.g. ``R.a BETWEEN 24 AND 42``). Our implementation supports both. See
    the documentation of `get_less` for details.
    """

    def __init__(
        self,
        norms: Sequence[LpNorms],
        bounds: Sequence[T],
        *,
        range_col: pb.ColumnReference,
        min_val: T,
        coarser: RangeConditionedNorms[T] | None,
    ) -> None:
        if len(norms) != len(bounds):
            raise ValueError(f"Number of norms ({len(norms)}) must match number of bounds ({len(bounds)})")
        if not norms:
            raise ValueError("At least one norm must be provided")

        self._norms = list(norms)
        self._bounds = list(bounds)
        self._min_val = min_val
        self._coarser = coarser
        self._col = range_col

    @property
    def range_col(self) -> pb.ColumnReference:
        return self._col

    def get_range(self, lower: T, upper: T) -> LpNorms:
        """Retrieves the Lp norms for a specific range of values.

        Matching norms must be enclosed by a single bucket of the histogram. If there exists no such
        bucket, a coarser histogram is queried..

        This function automatically traverses the histogram hierarchy to find the smallest bucket
        that encompasses the entire range (and thus the norms that over-estimates the true frequencies
        by the smallest margin).
        """
        if lower > upper:
            # SQL allows BETWEEN 42 AND 24 which produces an empty result set.
            return LpNorms.zeros()
        if upper < self._min_val or lower > self._bounds[-1]:
            return LpNorms.zeros()

        idx = bisect.bisect_left(self._bounds, upper)

        # NB: we do not need a special handler for idx == len(bounds). In this case, the upper bound would be greater
        # than our largest bound. However, all histograms are exactly the same width and span the entire domain of the
        # column. Therefore, it is safe to effectively clip the upper bound to the largest bound: there cannot be any
        # values outside of it.
        idx = min(idx, len(self._bounds) - 1)

        if idx == 0:
            # Our upper bound is already less than our smallest bound. Since all "higher up" histograms are coarser,
            # their upper bounds can only increase. Therefore, they must match an even broader range than the one we
            # are looking for. To obtain the smallest bucket that fully encopasses our range, we need to stop right
            # here.
            return self._norms[0]

        prev = self._bounds[idx - 1]
        if prev <= lower:
            return self._norms[idx]

        if self._coarser is None:
            raise AssertionError(f"Cannot find a histogram bucket for range [{lower}, {upper}]")
        return self._coarser.get_range(lower, upper)

    def get_less(self, value: T, *, inclusive: bool) -> LpNorms:
        """Obtains the Lp norms for all values less than (or equal to) a given value.

        Use the `inclusive` flag to distinguish between open and closed ranges (i.e. strict less-than
        vs. less-or-equal). But see the detailed notes below!

        Notes
        -----
        The original LpBound paper did not specify how to obtain conditioned Lp norms for half-open
        ranges. Our algorithm simply maps a filter *col < v* to the range *col between MIN and v*.

        Our implementation currently does not distinguish between open and closed intervals. This is
        due to limitations of the histogram-based approach. If the value falls somewhere in the
        middle of a bucket, we can only default to the entire bucket. Likewise, if the value falls
        exactly on a bucket boundary, we do not know what the first smaller value is and thus have to
        default once again to the entire bucket (in case of an open interval). Lastly, for closed
        intervals, we have to include the boundary value anyway and thus use the entire bucket. To
        representation for range predicates. We argue that this would no longer be a faithful
        adaptation of SafeBound. In case we come up with a better solution in the future, we already
        include the `inclusive` flag in the method signature. This avoids breaking changes - even if
        this flag has no effect for now.
        """
        return self.get_range(self._min_val, value)

    def get_greater(self, value: T, *, inclusive: bool) -> LpNorms:
        """Obtains the Lp norms for all values greater than (or equal to) a given value.

        Use the `inclusive` flag to distinguish between open and closed ranges (i.e. strict
        greater-than vs. greater-or-equal).
        But see the notes below!

        See Also
        --------
        get_less : for details on the underyling algorithm and the handling of open vs. closed
                   intervals.
        """
        global_max = self._bounds[-1]
        return self.get_range(value, global_max)

    def __json__(self) -> pb.util.jsondict:
        return {
            "filter_column": self._col,
            "norms": self._norms,
            "bounds": self._bounds,
            "min_val": self._min_val,
            "coarser_histogram": self._coarser,
        }


def load_range_norms_json(json_data: dict | str, *, database: pb.Database) -> RangeConditionedNorms:
    """Reconstructs range-conditioned norms from their JSON representation.

    Access to the database is required to parse the column values based on their data type.
    """
    if isinstance(json_data, str):
        json_data = json.loads(json_data)

    filter_col = pb.parser.load_column_json(json_data["filter_column"])
    dtype = database.schema().datatype(filter_col)
    val_parser = make_json_parser(dtype)

    norms = [load_norm_json(norm_json) for norm_json in json_data["norms"]]
    bounds = [val_parser(bound) for bound in json_data["bounds"]]
    min_val = val_parser(json_data["min_val"])

    coarser_json = json_data.get("coarser_histogram")
    coarser = load_range_norms_json(coarser_json, database=database) if coarser_json is not None else None

    return RangeConditionedNorms(norms, bounds, range_col=filter_col, min_val=min_val, coarser=coarser)


class RangeConditionedRepo:
    """Maintains the range-conditioned Lp norms for a set of join columns.

    Depending on the current filter predicate, different lookup methods exist. Each method requires
    the join column as well as the column being filtered. In addition, the filter value must be
    supplied in a predicate-specific form. Lp norms can be looked up using the following methods:

    - `lookup_range`: for predicates of the form *filter_col BETWEEN lower AND upper*
    - `lookup_less_equal`: for predicates of the form *filter_col <= bound*
    - `lookup_strict_less`: for predicates of the form *filter_col < bound*
    - `lookup_greater_equal`: for predicates of the form *filter_col >= bound*
    - `lookup_strict_greater`: for predicates of the form *filter_col > bound*

    While there exist different lookup methods for open and closed intervals, see the documentation
    on `RangeConditionedNorms.get_less` for details on how we currently handle this distinction.

    Parameters
    ----------
    join_pcfs : dict[pb.ColumnReference, Sequence[RangeConditionedNorms]]
        A mapping from join column to the range-conditioned norms for that column. Each set of Lp norms
        should be conditioned on a different filter column.
    """

    def __init__(self, join_norms: Mapping[pb.ColumnReference, Sequence[RangeConditionedNorms]]) -> None:
        self._norms = dict(join_norms)

    def lookup_range[T](
        self, join_col: pb.ColumnReference, *, range_col: pb.ColumnReference, between: tuple[T, T]
    ) -> LpNorms | None:
        """Retrieves the Lp norms for a join column based on the condition *range_col BETWEEN lo AND hi*.

        The range must be given as a tuple of the form *(lo, hi)*.

        If the repository does not contain Lp norms conditioned on the filter column, *None* is
        returned.
        """
        join_col = join_col.drop_table_alias()
        range_col = range_col.drop_table_alias()

        norms = self._norms.get(join_col)
        if norms is None:
            return None

        for norm in norms:
            if norm.range_col != range_col:
                continue
            lo, hi = between
            return norm.get_range(lo, hi)
        return None

    def lookup_less_equal[T](
        self, join_col: pb.ColumnReference, *, range_col: pb.ColumnReference, bound: T
    ) -> LpNorms | None:
        """Retrieves the Lp norms for a join column based on the condition *range_col <= bound*.

        If the repository does not contain Lp norms conditioned on the filter column, *None* is
        returned.

        See the documentation on `RangeConditionedNorms.get_less` for details on how we currently
        handle the distinction between open and closed intervals.
        """
        join_col = join_col.drop_table_alias()
        range_col = range_col.drop_table_alias()

        norms = self._norms.get(join_col)
        if norms is None:
            return None

        for norm in norms:
            if norm.range_col != range_col:
                continue
            return norm.get_less(bound, inclusive=True)
        return None

    def lookup_strict_less[T](
        self, join_col: pb.ColumnReference, *, range_col: pb.ColumnReference, bound: T
    ) -> LpNorms | None:
        """Retrieves the Lp norms for a join column based on the condition *range_col < bound*.

        If the repository does not contain Lp norms conditioned on the filter column, *None* is
        returned.

        See the documentation on `RangeConditionedNorms.get_less` for details on how we currently
        handle the distinction between open and closed intervals.
        """
        join_col = join_col.drop_table_alias()
        range_col = range_col.drop_table_alias()

        norms = self._norms.get(join_col)
        if norms is None:
            return None

        for norm in norms:
            if norm.range_col != range_col:
                continue
            return norm.get_less(bound, inclusive=False)
        return None

    def lookup_greater_equal[T](
        self, join_col: pb.ColumnReference, *, range_col: pb.ColumnReference, bound: T
    ) -> LpNorms | None:
        """Retrieves the Lp norms for a join column based on the condition *range_col >= bound*.

        If the repository does not contain Lp norms conditioned on the filter column, *None* is
        returned.

        See the documentation on `RangeConditionedNorms.get_less` (sic!) for details on how we
        currently handle the distinction between open and closed intervals.
        """
        join_col = join_col.drop_table_alias()
        range_col = range_col.drop_table_alias()

        norms = self._norms.get(join_col)
        if norms is None:
            return None

        for norm in norms:
            if norm.range_col != range_col:
                continue
            return norm.get_greater(bound, inclusive=True)
        return None

    def lookup_strict_greater[T](
        self, join_col: pb.ColumnReference, *, range_col: pb.ColumnReference, bound: T
    ) -> LpNorms | None:
        """Retrieves the Lp norms for a join column based on the condition *range_col > bound*.

        If the repository does not contain Lp norms conditioned on the filter column, *None* is
        returned.

        See the documentation on `RangeConditionedNorms.get_less` (sic!) for details on how we
        currently handle the distinction between open and closed intervals.
        """
        join_col = join_col.drop_table_alias()
        range_col = range_col.drop_table_alias()

        norms = self._norms.get(join_col)
        if norms is None:
            return None

        for norm in norms:
            if norm.range_col != range_col:
                continue
            return norm.get_greater(bound, inclusive=False)
        return None

    def __json__(self) -> pb.util.jsondict:
        norms_json = [{"join_column": join_col, "norms": norms} for join_col, norms in self._norms.items()]
        return {"norms": norms_json}


def build_histogram_hierarchy[T: _HistogramKey](
    frequencies: np.ndarray,
    values: Sequence[T],
    *,
    join_col: pb.BoundColumnReference,
    range_col: pb.ColumnReference,
    n_buckets: int,
    norms: NormSet,
    database: pb.Database,
    log: pb.util.Logger,
) -> RangeConditionedNorms:
    """Recursively builds a histogram hierarchy starting at a specific number of buckets.

    Each lower level histogram is coarser than the current one: it contains exactly 1/2 the number of buckets.
    Recursion stops as soon as only a single bucket is left. This ensures that the each range predicate can be supported.

    Parameters
    ----------
    frequencies : np.ndarray
        The frequencies of the values in the range column. The i-th entry corresponds to the i-th entry in `values`.
    values : Sequence[T]
        The distinct values in the range column, sorted in ascending order. The i-th entry corresponds to the i-th
        entry in `frequencies`.
    join_col : pb.BoundColumnReference
        The join column that the resulting Lp norms should be defined on. This is required to fetch the
        correlated degree sequences from the database and to construct the resulting norms.
    range_col : pb.BoundColumnReference
        The filter column that the resulting Lp norms should be conditioned on. This is required to
        fetch the correlated degree sequences from the database and to construct the resulting norms.
    n_buckets : int
        The number of buckets to use for the current histogram. Must be a power of 2.
    norms : NormSet
        The Lp norms that should be computed for each join key.
    database : pb.Database
        The database to fetch the degree sequences from.
    log : pb.util.Logger
        A logger to log the progress of the histogram construction.
    """
    if n_buckets > 1 and (n_buckets & (n_buckets - 1)) != 0:
        raise ValueError(f"Number of buckets must be a power of 2, got {n_buckets}")

    cumulative = np.cumsum(frequencies)
    total = cumulative[-1]
    if total == 0:
        raise ValueError(f"Cannot build histogram hierarchy for {join_col} conditioned on {range_col}: no values")

    bucket_freq = total / n_buckets
    breaks = [values[0]]
    for i in range(1, n_buckets):
        target = bucket_freq * i
        idx = bisect.bisect_left(cumulative, target)
        bound = values[idx]
        prev_bound = breaks[-1]
        if i > 1 and bound == prev_bound:
            continue
        breaks.append(values[idx])

    if breaks[-1] != values[-1] or len(values) == 1:
        breaks.append(values[-1])

    lp_norms: list[LpNorms] = []
    bounds: list[T] = []
    for lo, hi in itertools.pairwise(breaks):
        lo_pred = pb.qal.as_predicate(range_col, ">=", lo)
        hi_pred = pb.qal.as_predicate(range_col, "<=", hi)
        range_pred = pb.qal.AndPredicate([lo_pred, hi_pred])

        log(f"Loading range-conditioned norms for {join_col} on bucket {range_pred}")
        ds = fetch_correlated_ds(range_pred, on=join_col, database=database)
        lp_norms.append(compute_norms(ds, norms=norms))
        bounds.append(hi)

    coarser = (
        build_histogram_hierarchy(
            frequencies,
            values,
            join_col=join_col,
            range_col=range_col,
            n_buckets=n_buckets // 2,
            norms=norms,
            database=database,
            log=log,
        )
        if n_buckets > 1
        else None
    )

    return RangeConditionedNorms(lp_norms, bounds, min_val=values[0], range_col=range_col, coarser=coarser)


def build_range_norms(
    spec: CatalogSpec,
    *,
    n_buckets: int,
    norms: NormSet,
    database: pb.Database,
    verbose: bool | pb.util.Logger = False,
) -> RangeConditionedRepo:
    """Fetches all range-conditioned Lp norms for the given catalog.

    Parameters
    ----------
    spec : CatalogSpec
        Specification that contains the join column/range column pairs to build.
    n_buckets : int
        The maximum number of buckets to use for the histograms. Must be a power of 2. The histograms are organized in a
        hierarchy of increasingly coarser histograms. Each lower level histogram has exactly 1/2 the number of buckets
        of its parent. Recursion stops as soon as only a single bucket left.
    norms : NormSet
        The Lp norms that should be computed for each join key.
    database : pb.Database
        The database to fetch the degree sequences from.
    verbose : bool | pb.util.Logger, optional
        Either a pre-defined logger, or a boolean that indicates whether to log the progress using a
        default logger.

    Returns
    -------
    RangeConditionsRepo
        A repository that contains all specified PCFs.
    """
    log = wrap_logger(verbose)
    repo = collections.defaultdict(list)

    for join_col, range_cols in spec.range_cols.items():
        for range_col in range_cols:
            log(f"Loading column distribution for {join_col} conditioned on {range_col}")
            filter_distribution = fetch_column_distribution(range_col, database, exclude_nulls=True)
            values, frequencies = zip(*filter_distribution)
            histogram = build_histogram_hierarchy(
                np.array(frequencies),
                values,
                join_col=join_col,
                range_col=range_col,
                n_buckets=n_buckets,
                norms=norms,
                database=database,
                log=log,
            )
            repo[join_col].append(histogram)

    return RangeConditionedRepo(repo)


def load_range_repo_json(json_data: dict | str, *, database: pb.Database) -> RangeConditionedRepo:
    """Reconstructs a range-conditioned Lp norms repository from its JSON representation.

    Access to the database is required to parse the column values based on their data type.
    """
    if isinstance(json_data, str):
        json_data = json.loads(json_data)

    repo: dict[pb.ColumnReference, Sequence[RangeConditionedNorms]] = {}
    for entry in json_data["norms"]:
        join_col = pb.parser.load_column_json(entry["join_column"])
        norms = [load_range_norms_json(norm_json, database=database) for norm_json in entry["norms"]]
        repo[join_col] = norms
    return RangeConditionedRepo(repo)


def build_unconditioned_norms(
    spec: CatalogSpec,
    *,
    norms: NormSet,
    database: pb.Database,
    verbose: bool | pb.util.Logger = False,
) -> Mapping[pb.ColumnReference, LpNorms]:
    """Fetches all unconditioned Lp norms for the given catalog.

    Parameters
    ----------
    spec : CatalogSpec
        Specification that contains the columns to build unconditioned Lp norms for.
    norms : NormSet
        The Lp norms that should be computed for each join key.
    database : pb.Database
        The database to fetch the degree sequences from.
    verbose : bool | pb.util.Logger, optional
        Either a pre-defined logger, or a boolean that indicates whether to log the progress using a
        default logger.

    Returns
    -------
    Mapping[pb.ColumnReference, LpNorms]
        A mapping from column reference to their corresponding unconditioned set of Lp norms.
    """
    log = wrap_logger(verbose)
    repo: dict[pb.ColumnReference, LpNorms] = {}
    for join_col in spec.unconditioned_cols:
        log(f"Loading unconditioned norms for {join_col}")
        ds = fetch_raw_ds(join_col, database=database)
        repo[join_col] = compute_norms(ds, norms=norms)
    return repo


class LpBoundCatalog(pb.qal.PredicateVisitor[LpNorms]):
    """The LpBound catalog acts as the central storage for all conditioned and unconditioned Lp norms.

    Each catalog works on two specs: a `CatalogSpec` which defines the Lp norms to build, and a
    `LpBoundSpec` which defines the "hyperparameters" for the norm calculation (e.g. number of
    MCVs or the p-values). Usually, the catalog spec is inferred automatically based on the
    usage scenario (see below).

    In addition, the catalog provides functionality to retrieve the appropriate Lp norms for a specific
    query. This behavior is encoded in the different ``visit_XYZ()`` methods.

    Creating a Catalog
    ------------------
    Depending on the use case, a catalog can be created using the following strategies:

    - `online` infers the catalog spec directly from the database schema
    - `infer_from` creates a catalog tailored to a specific workload
    - `from_spec` uses an arbitrary (user-supplied) spec
      (NB: the previous methods basically create the spec and then delegate to this function)

    An existing catalog can be exported using the `store` method to prevent re-computation.
    Such a catalog can be re-created via `load`. As a convenience, the `load_or_build` function
    combines the behavior of all previous methods: if the catalog exists, it will be re-loaded.
    Otherwise, online- or workload-based inferrence are used depending on the additional parameters.
    The resulting catalog is then automatically stored for future use.
    """

    @staticmethod
    def online(
        database: pb.Database, *, spec: LpBoundSpec = LpBoundSpec.default(), verbose: bool | pb.util.Logger = False
    ) -> LpBoundCatalog:
        """Infers the catalog spec from the database schema and builds the corresponding catalog.

        Parameters
        ----------
        database : pb.Database
            The database which contains the target schema and data
        spec : LpBoundSpec
            The "hyperparameters" for the Lp norm calculation (e.g. number of MCVs or compression
            accuracy). This defaults to the same values as in the original LpBound paper.
        verbose : bool | pb.util.Logger, optional
            Either a pre-defined logger, or a boolean that indicates whether to log the progress
            using a default logger.

        See Also
        --------
        catalog_from_schema : For details on the inference logic
        LpBoundSpec.default : For details on the default hyperparameters
        """
        catalog_spec = catalog_from_schema(database.schema())
        return LpBoundCatalog.from_spec(catalog_spec, stats_spec=spec, database=database, verbose=verbose)

    @staticmethod
    def infer_from(
        workload: pb.Workload,
        *,
        spec: LpBoundSpec = LpBoundSpec.default(),
        database: pb.Database,
        verbose: bool | pb.util.Logger = False,
    ) -> LpBoundCatalog:
        """Infers the catalog spec from an example workload and builds the corresponding catalog.

        Parameters
        ----------
        workload : pb.Workload
            The workload
        spec : LpBoundSpec
            The "hyperparameters" for the Lp norm calculation (e.g. number of MCVs or compression
            accuracy). This defaults to the same values as in the original LpBound paper.
        database : pb.Database
            The database which contains the actual data
        verbose : bool | pb.util.Logger, optional
            Either a pre-defined logger, or a boolean that indicates whether to log the progress
            using a default logger.

        See Also
        --------
        catalog_from_workload : For details on the inference logic
        LpBoundSpec.default : For details on the default hyperparameters
        """

        catalog_spec = catalog_from_workload(workload, verbose=False)
        return LpBoundCatalog.from_spec(catalog_spec, stats_spec=spec, database=database, verbose=verbose)

    @staticmethod
    def from_spec(
        catalog_spec: CatalogSpec,
        *,
        stats_spec: LpBoundSpec = LpBoundSpec.default(),
        database: pb.Database,
        verbose: bool | pb.util.Logger = False,
    ) -> LpBoundCatalog:
        """Creates a catalog for an arbitrary specification.

        Parameters
        ----------
        catalog_spec : CatalogSpec
            The specification that defines the Lp norms to calculate
        stats_spec : LpBoundSpec
            The "hyperparameters" for the Lp norm calculation (e.g. number of MCVs or compression
            accuracy). This defaults to the same values as in the original LpBound paper.
        database : pb.Database
            The database which contains the actual data
        verbose : bool | pb.util.Logger, optional
            Either a pre-defined logger, or a boolean that indicates whether to log the progress
            using a default logger.

        See Also
        --------
        LpBoundSpec.default : For details on the default hyperparameters
        """
        build_start = datetime.now()  # noqa: DTZ005 - let's not worry about timezones for now

        eq_pcfs_repo = build_equality_norms(
            catalog_spec,
            mcv_size=stats_spec.num_mcvs,
            norms=stats_spec.norms,
            database=database,
            verbose=verbose,
        )
        range_pcfs_repo = build_range_norms(
            catalog_spec,
            n_buckets=stats_spec.hist_buckets,
            norms=stats_spec.norms,
            database=database,
            verbose=verbose,
        )
        unconditioned_pcfs = build_unconditioned_norms(
            catalog_spec,
            norms=stats_spec.norms,
            database=database,
            verbose=verbose,
        )

        build_end = datetime.now()  # noqa: DTZ005 - let's not worry about timezones for now
        construction_time = build_end - build_start

        catalog = LpBoundCatalog(
            eq_pcfs_repo,
            range_pcfs_repo,
            unconditioned_pcfs,
            database=database,
            log=verbose,
        )
        catalog._construction_time = construction_time
        return catalog

    @staticmethod
    def load(archive: Path | str, *, database: pb.Database) -> LpBoundCatalog:
        """Loads a previously built catalog from disk.

        The archive should be either the catalog file itself, or a directory that contains a
        *lpbound-catalog.json*.
        Access to the database is required to parse the column values based on their data type.

        See Also
        --------
        store : inverse method to persist a catalog to disk
        """
        archive = Path(archive)
        if not archive.exists():
            raise FileNotFoundError(f"Norm repository archive {archive} does not exist")
        if archive.is_dir():
            catalog_file = archive / "lpbound-catalog.json.xz"
            if not catalog_file.exists():
                catalog_file = archive / "lpbound-catalog.json"
            if not catalog_file.exists():
                raise FileNotFoundError(
                    "Fallback catalogs 'lpbound-catalog.json' or"
                    f"'lpbound-catalog.json.xz' not found in {archive}. "
                    "Please specify the catalog path explicitly."
                )
            archive = catalog_file

        if not archive.exists():
            raise FileNotFoundError(archive)

        if archive.name.endswith(".json.xz"):
            with open(archive, "rb") as f:
                decompressed = lzma.decompress(f.read()).decode("utf-8")
                catalog = json.loads(decompressed)
        elif archive.name.endswith(".json"):
            with open(archive, "r", encoding="utf-8") as f:
                catalog = json.load(f)
        else:
            raise ValueError(f"No suitable parser for catalog {archive}")

        construction_time = catalog.get("construction_time")

        catalog = load_norm_repo_json(catalog, database=database)
        catalog._construction_time = timedelta(seconds=construction_time) if construction_time is not None else None
        return catalog

    @staticmethod
    def load_or_build(
        archive: Path | str,
        *,
        database: pb.Database,
        workload: pb.Workload | None = None,
        spec: LpBoundSpec = LpBoundSpec.default(),
        verbose: bool | pb.util.Logger = False,
    ) -> LpBoundCatalog:
        """Loads a previously built catalog from disk if it exists, otherwise builds a new one.

        If the catalog does not yet exist, it will be created. The specific inference strategy
        depends on the additional parameters:

        - If a `workload` is given, the catalog will be tailored to it
        - Otherwise, the catalog will be based on the schema of the `database` alone

        If the catalog needs to be built, the hyperparameters can be adjusted via `spec`. It
        defaults to the same values as in the original LpBound paper.

        See Also
        --------
        online : For building a catalog exclusively based on the database schema
        infer_from : For building a catalog tailored to a specific workload
        LpBoundSpec.default : For details on the default hyperparameters
        """
        archive = Path(archive)
        if archive.is_dir() and (archive / "safebound-catalog.json.xz").exists():
            archive = archive / "safebound-catalog.json.xz"
        elif archive.is_dir() and (archive / "safebound-catalog.json").exists():
            archive = archive / "safebound-catalog.json"

        if archive.is_file():
            return LpBoundCatalog.load(archive, database=database)

        if workload is None:
            catalog = LpBoundCatalog.online(database, spec=spec, verbose=verbose)
        else:
            catalog = LpBoundCatalog.infer_from(workload, spec=spec, database=database, verbose=verbose)

        catalog.store(archive)
        return catalog

    def __init__(
        self,
        equality_norms: EqualityConditionedRepo,
        range_norms: RangeConditionedRepo,
        unconditioned_norms: Mapping[pb.ColumnReference, LpNorms],
        *,
        database: pb.Database,
        log: pb.util.Logger | bool = False,
    ) -> None:
        self._eq_norms = equality_norms
        self._range_norms = range_norms
        self._unconditioned = dict(unconditioned_norms)
        self._database = database
        self._log = wrap_logger(log)
        self._construction_time: timedelta | None = None

    def retrieve_norms(self, query: pb.SqlQuery) -> Mapping[pb.ColumnReference, LpNorms]:
        """Fetches all Lp norms for a specific query.

        For each of the join columns in the query, this determines all applicable filter predicates,
        loads the correponding conditioned Lp norms (falling back to unconditioned norms as necessary),
        and combines the Lp norms according to the conjunction/disjunction predicates.

        The result is provided as a mapping from join column to the final Lp norms on that column.
        """
        join_map: dict[pb.ColumnReference, pb.qal.AbstractPredicate] = {}
        join_cols = pb.util.set_union(join.columns() for join in query.joins())
        unconditioned_cols: set[pb.ColumnReference] = set()

        for join_col in join_cols:
            if not pb.ColumnReference.assert_bound(join_col):
                raise pb.UnboundColumnError(join_col)
            filter_preds = query.filters_for(join_col.table)

            if filter_preds:
                join_map[join_col] = filter_preds
            else:
                unconditioned_cols.add(join_col)

        lp_norms = {col: pred.accept_visitor(self, join_col=col.drop_table_alias()) for col, pred in join_map.items()}
        for join_col in unconditioned_cols:
            lp_norms[join_col] = self._unconditioned[join_col.drop_table_alias()]
        return lp_norms

    @override
    def visit_binary_predicate(self, predicate: pb.qal.BinaryPredicate, *args, **kwargs) -> LpNorms:
        join_col: pb.ColumnReference = kwargs["join_col"]
        join_col = join_col.drop_table_alias()

        simplified = pb.qal.SimpleFilter.attempt_wrap(predicate)
        if simplified is None:
            self._log(f"Unsupported predicate {predicate} for {join_col}, falling back to unconditioned norms")
            return self._unconditioned[join_col]

        filter_col = simplified.column.drop_table_alias()
        dtype = self._database.schema().datatype(simplified.column)
        filter_val = cast_value(simplified.value, datatype=dtype, null_placeholder=None)

        match simplified.operator:
            case pb.qal.BinaryOperator.Equal:
                lp_norms = self._eq_norms.lookup(join_col, filter_col=filter_col, filter_val=filter_val)

            case pb.qal.BinaryOperator.Less:
                lp_norms = self._range_norms.lookup_strict_less(join_col, range_col=filter_col, bound=filter_val)
            case pb.qal.BinaryOperator.LessEqual:
                lp_norms = self._range_norms.lookup_less_equal(join_col, range_col=filter_col, bound=filter_val)

            case pb.qal.BinaryOperator.Greater:
                lp_norms = self._range_norms.lookup_strict_greater(join_col, range_col=filter_col, bound=filter_val)
            case pb.qal.BinaryOperator.GreaterEqual:
                lp_norms = self._range_norms.lookup_greater_equal(join_col, range_col=filter_col, bound=filter_val)

            case _:
                self._log(f"Unsupported predicate {predicate} for {join_col}, falling back to unconditioned norms")
                lp_norms = self._unconditioned[join_col]

        if lp_norms is None:
            self._log(
                f"Could not find norms for {join_col} conditioned on {simplified}, falling back to unconditioned norms"
            )
            lp_norms = self._unconditioned[join_col]
        return lp_norms

    @override
    def visit_between_predicate(self, predicate: pb.qal.BetweenPredicate, *args, **kwargs) -> LpNorms:
        join_col: pb.ColumnReference = kwargs["join_col"]
        join_col = join_col.drop_table_alias()

        simplified = pb.qal.SimpleFilter.attempt_wrap(predicate)
        if simplified is None:
            self._log(f"Unsupported predicate {predicate} for {join_col}, falling back to unconditioned norms")
            return self._unconditioned[join_col]
        assert isinstance(simplified.value, tuple)

        filter_col = simplified.column.drop_table_alias()
        dtype = self._database.schema().datatype(simplified.column)
        lo, hi = simplified.value
        lo = cast_value(lo, datatype=dtype, null_placeholder=None)
        hi = cast_value(hi, datatype=dtype, null_placeholder=None)

        lp_norms = self._range_norms.lookup_range(join_col, range_col=filter_col, between=(lo, hi))
        if lp_norms is None:
            self._log(
                f"Could not find norms for {join_col} conditioned on {simplified}, falling back to unconditioned norms"
            )
            lp_norms = self._unconditioned[join_col]
        return lp_norms

    @override
    def visit_in_predicate(self, predicate: pb.qal.InPredicate, *args, **kwargs) -> LpNorms:
        join_col: pb.ColumnReference = kwargs["join_col"]
        join_col = join_col.drop_table_alias()

        simplified = pb.qal.SimpleFilter.attempt_wrap(predicate)
        if simplified is None:
            self._log(f"Unsupported predicate {predicate} for {join_col}, falling back to unconditioned norms")
            return self._unconditioned[join_col]
        assert isinstance(simplified.value, Iterable)

        if simplified.operator == pb.qal.BinaryOperator.NotIn:
            self._log(f"Unsupported predicate {predicate} for {join_col}, falling back to unconditioned norms")
            return self._unconditioned[join_col]

        filter_col = simplified.column.drop_table_alias()
        dtype = self._database.schema().datatype(simplified.column)

        lp_norms = LpNorms.zeros()
        missing_stats = False

        for filter_val in simplified.value:
            filter_val = cast_value(filter_val, datatype=dtype, null_placeholder=None)
            norms = self._eq_norms.lookup(join_col, filter_col=filter_col, filter_val=filter_val)
            if norms is None:
                # No statistics on this value, need to fall-back
                missing_stats = True
                break

            # an obvious optimization would be to bound the sum by the unconditioned norms, but this is not part of the
            # paper so we do not implement it either.
            lp_norms = lp_norms.sum_with(norms)

        if missing_stats:
            self._log(
                f"Could not find norms for {join_col} conditioned on {simplified}, falling back to unconditioned norms"
            )
            lp_norms = self._unconditioned[join_col]

        return lp_norms

    @override
    def visit_unary_predicate(self, predicate: pb.qal.UnaryPredicate, *args, **kwargs) -> LpNorms:
        join_col = kwargs["join_col"]
        self._log(f"Unsupported predicate {predicate} for {join_col}, falling back to unconditioned norms")
        return self._unconditioned[join_col]

    @override
    def visit_not_predicate(self, predicate: pb.qal.NotPredicate, *args, **kwargs) -> LpNorms:
        join_col = kwargs["join_col"]
        self._log(f"Unsupported predicate {predicate} for {join_col}, falling back to unconditioned norms")
        return self._unconditioned[join_col]

    @override
    def visit_or_predicate(self, predicate: pb.qal.OrPredicate, *args, **kwargs) -> LpNorms:
        lp_norms = LpNorms.zeros()
        for child in predicate.children:
            child_norms = child.accept_visitor(self, *args, **kwargs)
            # an obvious optimization would be to bound the sum by the unconditioned norms, but this is not part of the
            # paper so we do not implement it either.
            lp_norms = lp_norms.sum_with(child_norms)
        return lp_norms

    @override
    def visit_and_predicate(self, predicate: pb.qal.AndPredicate, *args, **kwargs) -> LpNorms:
        head, *tail = predicate.children
        lp_norms = head.accept_visitor(self, *args, **kwargs)
        for child in tail:
            child_norms = child.accept_visitor(self, *args, **kwargs)
            lp_norms = lp_norms.min_with(child_norms)
        return lp_norms

    def store(self, archive: Path | str, *, compressed: bool = False) -> Path:
        """Persists the current catalog to disk.

        If `compressed` is enabled, we perform LZMA compression on the catalog.

        `archive` should be a full file path. However, we also support specifying a directory.
        It is handled as follows: if compression is enabled, the catalog is stored as
        *lpbound-catalog.json.xz* in the given directory. Otherwise, it is stored as
        *lpbound-catalog.json*.
        """
        archive = Path(archive)
        if not archive.suffix and not archive.is_dir():
            extension = "json.xz" if compressed else "json"
            archive = archive / f"lpbound-catalog.{extension}"
        archive.parent.mkdir(parents=True, exist_ok=True)

        if not compressed:
            with archive.open("w", encoding="utf-8") as f:
                pb.util.to_json_dump(self, f)
            return archive

        raw_dump = pb.util.to_json(self)
        if raw_dump is None:
            raise RuntimeError("Failed to serialize catalog to JSON")
        compressed_dump = lzma.compress(raw_dump.encode("utf-8"))

        with archive.open("wb") as f:
            f.write(compressed_dump)

        return archive

    def __json__(self) -> pb.util.jsondict:
        unconditioned_json = [
            {"join_column": join_col, "norms": norms} for join_col, norms in self._unconditioned.items()
        ]
        return {
            "equality_norms": self._eq_norms,
            "range_norms": self._range_norms,
            "unconditioned_norms": unconditioned_json,
        }


def load_norm_repo_json(json_data: dict | str, *, database: pb.Database) -> LpBoundCatalog:
    """Reconstructs an LpBound catalog from its JSON representation.

    Access to the database is required to parse the column values based on their data type.
    """

    if isinstance(json_data, str):
        json_data = json.loads(json_data)

    equality_norms = load_eq_repo_json(json_data["equality_norms"], database=database)
    range_norms = load_range_repo_json(json_data["range_norms"], database=database)

    unconditioned: dict[pb.ColumnReference, LpNorms] = {}
    for entry in json_data["unconditioned_norms"]:
        join_col = pb.parser.load_column_json(entry["join_column"])
        norms = load_norm_json(entry["norms"])
        unconditioned[join_col] = norms

    return LpBoundCatalog(equality_norms, range_norms, unconditioned, database=database)
