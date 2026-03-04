from __future__ import annotations

import bisect
import collections
from collections.abc import Generator, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Protocol

import numpy as np
import postbound as pb

from ..util import wrap_logger
from ._compress import valid_compress
from ._core import DegreeSequence
from ._piecewise_fns import PiecewiseConstantFn


class _HistogramKey(Protocol):
    def __lt__(self, other: _HistogramKey) -> bool: ...

    def __add__(self, other: _HistogramKey) -> _HistogramKey: ...

    def __sub__(self, other: _HistogramKey) -> _HistogramKey: ...

    def __hash__(self) -> int: ...


@dataclass
class SafeBoundSpec:
    accuracy: float
    mcv_size: int
    hist_hierarchy_depth: int

    @staticmethod
    def default() -> SafeBoundSpec:
        return SafeBoundSpec(accuracy=0.01, mcv_size=1000, hist_hierarchy_depth=7)


@dataclass
class CatalogSpec:
    simple_cols: set[pb.BoundColumnReference]
    equality_cols: dict[pb.BoundColumnReference, set[pb.BoundColumnReference]]
    range_cols: dict[pb.BoundColumnReference, set[pb.BoundColumnReference]]
    like_cols: dict[pb.BoundColumnReference, set[pb.BoundColumnReference]]

    @staticmethod
    def empty() -> CatalogSpec:
        return CatalogSpec(
            set(),
            collections.defaultdict(set),
            collections.defaultdict(set),
            collections.defaultdict(set),
        )

    def columns(self) -> set[pb.ColumnReference]:
        cols = self.simple_cols
        for join_col, filter_cols in self.equality_cols.items():
            cols |= filter_cols
            cols.add(join_col)
        for join_col, range_cols in self.range_cols.items():
            cols |= range_cols
            cols.add(join_col)
        for join_col, like_cols in self.like_cols.items():
            cols |= like_cols
            cols.add(join_col)
        return cols  # type: ignore


class _CatalogVisitor(pb.qal.PredicateVisitor[None]):
    def __init__(self, spec: CatalogSpec, *, log: pb.util.Logger) -> None:
        self.spec = spec
        self._log = log

    def visit_binary_predicate(
        self, predicate: pb.qal.BinaryPredicate, *args, **kwargs
    ) -> None:
        if not pb.qal.SimpleFilter.can_wrap(predicate):
            return
        simplified = pb.qal.SimpleFilter.wrap(predicate)
        assert pb.ColumnReference.assert_bound(simplified.column)
        join_map = kwargs["join_map"]

        match simplified.operation:
            case pb.qal.LogicalOperator.Equal:
                for join_col in join_map[simplified.column.table]:
                    self._log(
                        f"Detected equality-conditioned PCF on {join_col} for {simplified}"
                    )
                    self.spec.equality_cols[join_col].add(simplified.column)

            case (
                pb.qal.LogicalOperator.Less
                | pb.qal.LogicalOperator.LessEqual
                | pb.qal.LogicalOperator.Greater
                | pb.qal.LogicalOperator.GreaterEqual
            ):
                for join_col in join_map[simplified.column.table]:
                    self._log(
                        f"Detected range-conditioned PCF on {join_col} for {simplified}"
                    )
                    self.spec.range_cols[join_col].add(simplified.column)

            case pb.qal.LogicalOperator.Like | pb.qal.LogicalOperator.ILike:
                for join_col in join_map[simplified.column.table]:
                    self._log(
                        f"Detected like-conditioned PCF on {join_col} for {simplified}"
                    )
                    self.spec.like_cols[join_col].add(simplified.column)

            case _:
                # Operator is not supported
                pass

    def visit_between_predicate(
        self, predicate: pb.qal.BetweenPredicate, *args, **kwargs
    ) -> None:
        # even for BETWEEN predicates we perform the simplification check.
        # This makes sure that the predicate contains a simple range, i.e. BETWEEN 24 AND 42
        # instead more sophisticated stuff like BETWEEN 42 * 42 AND R.b
        if not pb.qal.SimpleFilter.can_wrap(predicate):
            return
        simplified = pb.qal.SimpleFilter.wrap(predicate)
        assert pb.ColumnReference.assert_bound(simplified.column)
        join_map = kwargs["join_map"]
        for join_col in join_map[simplified.column.table]:
            self._log(f"Detected range-conditioned PCF on {join_col} for {simplified}")
            self.spec.range_cols[join_col].add(simplified.column)

    def visit_in_predicate(
        self, predicate: pb.qal.InPredicate, *args, **kwargs
    ) -> None:
        # even for IN predicates we perform the simplification check.
        # This makes sure that the predicate contains a simple range, i.e. IN (1, 2, 3)
        # instead more sophisticated stuff like IN(SELECT b FROM R)
        if not pb.qal.SimpleFilter.can_wrap(predicate):
            return
        simplified = pb.qal.SimpleFilter.wrap(predicate)
        assert pb.ColumnReference.assert_bound(simplified.column)
        join_map = kwargs["join_map"]
        for join_col in join_map[simplified.column.table]:
            self._log(
                f"Detected equality-conditioned PCF on {join_col} for {simplified}"
            )
            self.spec.equality_cols[join_col].add(simplified.column)

    def visit_unary_predicate(
        self, predicate: pb.qal.UnaryPredicate, *args, **kwargs
    ) -> None:
        # SafeBound does not handle unary predicates
        pass

    def visit_not_predicate(
        self,
        predicate: pb.qal.CompoundPredicate,
        child_predicate: pb.qal.AbstractPredicate,
        *args,
        **kwargs,
    ) -> None:
        # SafeBound does not handle NOT predicates
        pass

    def visit_or_predicate(
        self,
        predicate: pb.qal.CompoundPredicate,
        components: Sequence[pb.qal.AbstractPredicate],
        *args,
        **kwargs,
    ) -> None:
        for child in components:
            child.accept_visitor(self, *args, **kwargs)

    def visit_and_predicate(
        self,
        predicate: pb.qal.CompoundPredicate,
        components: Sequence[pb.qal.AbstractPredicate],
        *args,
        **kwargs,
    ) -> None:
        for child in components:
            child.accept_visitor(self, *args, **kwargs)


def _build_join_map(
    query: pb.SqlQuery,
) -> dict[pb.TableReference, set[pb.ColumnReference]]:
    join_cols = pb.util.set_union(join.columns() for join in query.joins())
    join_map = collections.defaultdict(set)
    for col in join_cols:
        join_map[col.table].add(col)
    return join_map


def derive_catalog(
    workload: pb.Workload,
    verbose: bool | pb.util.Logger = False,
) -> CatalogSpec:
    logger = wrap_logger(verbose)
    spec = CatalogSpec.empty()
    visitor = _CatalogVisitor(spec, log=logger)
    for query in workload.queries():
        join_map = _build_join_map(query)
        visitor.visit_query_predicates(query, join_map=join_map)
    return spec


def fetch_raw_ds(
    column: pb.ColumnReference, *, database: pb.Database
) -> DegreeSequence:
    mcv_list = database.statistics().most_common_values(column, k=-1)
    return DegreeSequence.from_mcv(mcv_list, column=column)


def fetch_correlated_ds(
    predicate: pb.qal.AbstractPredicate,
    *,
    on: pb.BoundColumnReference,
    database: pb.Database,
    accuracy: float,
) -> PiecewiseConstantFn:
    select_clause = pb.qal.Select.count_star()
    from_clause = pb.qal.From.create_for(on.table)
    where_clause = pb.qal.Where(predicate)
    group_clause = pb.qal.GroupBy.create_for(on)
    sql = pb.SqlQuery(
        select_clause=select_clause,
        from_clause=from_clause,
        where_clause=where_clause,
        groupby_clause=group_clause,
    )

    ds = DegreeSequence(database.execute_query(sql), column=on)
    piecewiese_linear = valid_compress(ds, accuracy=accuracy)
    return piecewiese_linear.deriv()


def fetch_column_values[T](
    column: pb.BoundColumnReference, database: pb.Database
) -> list[T]:
    select_clause = pb.qal.Select.create_for(column)
    from_clause = pb.qal.From.create_for(column.table)
    sql = pb.SqlQuery(select_clause=select_clause, from_clause=from_clause)

    return database.execute_query(sql, raw=False)


def fetch_column_distribution[T](
    column: pb.BoundColumnReference, database: pb.Database
) -> list[tuple[T, int]]:
    """Builds an ordered list of (value, frequency) pairs of all distinct values in the column.

    In contrast to an MCV list, the column distribution is ordered by column value and not by value frequency.
    """
    select_clause = pb.qal.Select(
        [pb.qal.BaseProjection.column(column), pb.qal.BaseProjection.count_star()]
    )
    from_clause = pb.qal.From.create_for(column.table)
    group_clause = pb.qal.GroupBy.create_for(column)
    order_clause = pb.qal.OrderBy.create_for(column, ascending=True)
    sql = pb.SqlQuery(
        select_clause=select_clause,
        from_clause=from_clause,
        groupby_clause=group_clause,
        orderby_clause=order_clause,
    )

    return database.execute_query(sql, raw=True)


class EqualityConditionedPCF[T]:
    def __init__(
        self,
        filtered_column: pb.ColumnReference,
        functions: dict[T, PiecewiseConstantFn],
        fallback: PiecewiseConstantFn,
    ) -> None:
        self._col = filtered_column
        self._functions = functions
        self._fallback = fallback

    @property
    def filter_col(self) -> pb.ColumnReference:
        return self._col

    def get(self, key: T) -> PiecewiseConstantFn:
        fn = self._functions.get(key)
        return fn or self._fallback


class EqualityConditionsRepo:
    def __init__(
        self,
        join_pcfs: dict[pb.ColumnReference, list[EqualityConditionedPCF]],
    ) -> None:
        self._functions = join_pcfs

    def lookup(
        self,
        join_col: pb.ColumnReference,
        *,
        filter_col: pb.ColumnReference,
        filter_val: Any,
    ) -> Optional[PiecewiseConstantFn]:
        candidates = self._functions.get(join_col)
        if candidates is None:
            return None

        target_pcf: EqualityConditionedPCF | None = None
        for current in candidates:
            if current.filter_col != filter_col:
                continue
            target_pcf = current
            break

        if target_pcf is None:
            return None

        return target_pcf.get(filter_val)


def build_equality_mcvs(
    spec: CatalogSpec, *, mcv_size: int, accuracy: float, database: pb.Database
) -> EqualityConditionsRepo:
    pcfs = collections.defaultdict(list)

    for join_col, filter_cols in spec.equality_cols.items():
        for filter_col in filter_cols:
            mcv = database.statistics().most_common_values(filter_col, k=-1)
            correlated_pcfs: dict[Any, PiecewiseConstantFn] = {}

            # TODO: this implementation is incredibly inefficient, likely prohibititely so
            # We can speed it up by computing all of the required frequencies in one go
            # using window functions to compute the ranks and CASEs to label mcv/non-mcv frequencies.
            # Afterwards, it should even be possible to perform a GROUP BY to sum the individual
            # segments. That way, we just need to execute one query that provides the (interleafed)
            # MCVs. We just need to split them up and compress them in Python.

            for val in mcv.values[:mcv_size]:
                pred = pb.qal.as_predicate(filter_col, "=", val)
                pcf = fetch_correlated_ds(
                    pred, on=join_col, accuracy=accuracy, database=database
                )
                correlated_pcfs[val] = pcf

            non_mcv_pcf = PiecewiseConstantFn.zero(column=join_col)
            for val in mcv.values[mcv_size:]:
                pred = pb.qal.as_predicate(filter_col, "=", val)
                pcf = fetch_correlated_ds(
                    pred, on=join_col, accuracy=accuracy, database=database
                )
                non_mcv_pcf += pcf

            repo = EqualityConditionedPCF(filter_col, correlated_pcfs, non_mcv_pcf)
            pcfs[join_col].append(repo)

    return EqualityConditionsRepo(pcfs)


class RangeConditionedPCF[T: _HistogramKey]:
    def __init__(
        self,
        buckets: Sequence[PiecewiseConstantFn],
        bounds: Sequence[T],
        *,
        conditioned_col: pb.ColumnReference,
        higher_res: RangeConditionedPCF[T] | None = None,
    ) -> None:
        if len(buckets) != len(bounds):
            raise ValueError("bounds and buckets have to have the same length")
        self._buckets = list(buckets)
        self._bounds = list(bounds)
        self._conditioned_col = conditioned_col
        self._higher_res = higher_res

    @property
    def conditioned_col(self) -> pb.ColumnReference:
        return self._conditioned_col

    @property
    def higher_res(self) -> Optional[RangeConditionedPCF[T]]:
        return self._higher_res

    @higher_res.setter
    def higher_res(self, value: RangeConditionedPCF[T] | None) -> None:
        self._higher_res = value

    def pcf_from_buckets(self, rng: slice) -> PiecewiseConstantFn:
        buckets = self._buckets[rng]
        pcf = buckets[0]
        for current_pcf in buckets[1:]:
            pcf += current_pcf
        return pcf

    def get_range(self, lower: T, upper: T) -> PiecewiseConstantFn:
        own_range = self.error_range(lower, upper)
        if self._higher_res is None:
            return self.pcf_from_buckets(own_range[1])

        higher_res = self._higher_res.error_range(lower, upper)
        own_err, higher_err = own_range[0], higher_res[0]
        if own_err < higher_err:
            return self.pcf_from_buckets(own_range[1])

        # Instead of calling pcf_from_buckets() directly on higher_res,
        # we intentionally delegate to get_range(). This allows the
        # higher resolution histogram to compute an even better bound
        # through its even higher resolution child.
        # The downside is that we compute the error on the children
        # twice, but since this is a rather cheap operation, we are fine
        # with this for now.
        return self._higher_res.get_range(lower, upper)

    def get_less(self, value: T, *, inclusive: bool = False) -> PiecewiseConstantFn:
        # TODO: support inclusive!

        own_range = self.error_less(value)
        if self._higher_res is None:
            return self.pcf_from_buckets(slice(own_range[1]))

        higher_res = self._higher_res.error_less(value)
        own_err, higher_err = own_range[0], higher_res[0]
        if own_err < higher_err:
            return self.pcf_from_buckets(slice(own_range[1]))

        # See comment in get_range() for why we have to call get_less()
        return self._higher_res.get_less(value, inclusive=inclusive)

    def get_greater(self, value: T, *, inclusive: bool = False) -> PiecewiseConstantFn:
        # TODO: support inclusive!

        own_range = self.error_greater(value)
        if self._higher_res is None:
            return self.pcf_from_buckets(slice(own_range[1] + 1))

        higher_res = self._higher_res.error_greater(value)
        own_err, higher_err = own_range[0], higher_res[0]
        if own_err < higher_err:
            return self.pcf_from_buckets(slice(own_range[1] + 1))

        # See comment in get_range() for why we have to call get_greater()
        return self._higher_res.get_greater(value, inclusive=inclusive)

    def error_range(self, lower: T, upper: T) -> tuple[T, slice]:
        err_lo = self.error_greater(lower)
        err_hi = self.error_less(upper)
        err_total = err_lo[0] + err_hi[0]
        return (err_total, slice(err_lo[1], err_hi[1] + 1))

    def error_less(self, value: T) -> tuple[T, int]:
        idx = bisect.bisect_right(self._bounds, value)
        if idx == 0:
            np.inf, 0
        return self._bounds[idx] - value, idx

    def error_greater(self, value: T) -> tuple[T, int]:
        idx = bisect.bisect_left(self._bounds, value)
        if idx == len(self._bounds):
            np.inf, idx - 1
        return value - self._bounds[idx], idx


class RangeConditionedSequenceRepo:
    def __init__(
        self, join_pcfs: dict[pb.ColumnReference, list[RangeConditionedPCF]]
    ) -> None:
        self._join_pcfs = join_pcfs

    def lookup_range[T](
        self,
        join_col: pb.ColumnReference,
        *,
        range_col: pb.ColumnReference,
        between: tuple[T, T],
    ) -> Optional[PiecewiseConstantFn]:
        candidates = self._join_pcfs.get(join_col)
        if candidates is None:
            return None

        target_pcf: RangeConditionedPCF | None = None
        for current in candidates:
            if current.conditioned_col != range_col:
                continue
            target_pcf = current
            break

        if target_pcf is None:
            return None
        lo, hi = between
        return target_pcf.get_range(lo, hi)

    def lookup_less_equal[T](
        self, join_col: pb.ColumnReference, *, range_col: pb.ColumnReference, bound: T
    ) -> Optional[PiecewiseConstantFn]:
        candidates = self._join_pcfs.get(join_col)
        if candidates is None:
            return None

        target_pcf: RangeConditionedPCF | None = None
        for current in candidates:
            if current.conditioned_col != range_col:
                continue
            target_pcf = current
            break

        if target_pcf is None:
            return None
        return target_pcf.get_less(bound, inclusive=True)

    def lookup_strict_less[T](
        self, join_col: pb.ColumnReference, *, range_col: pb.ColumnReference, bound: T
    ) -> Optional[PiecewiseConstantFn]:
        candidates = self._join_pcfs.get(join_col)
        if candidates is None:
            return None

        target_pcf: RangeConditionedPCF | None = None
        for current in candidates:
            if current.conditioned_col != range_col:
                continue
            target_pcf = current
            break

        if target_pcf is None:
            return None
        return target_pcf.get_less(bound, inclusive=False)

    def lookup_greater_equal[T](
        self, join_col: pb.ColumnReference, *, range_col: pb.ColumnReference, bound: T
    ) -> Optional[PiecewiseConstantFn]:
        candidates = self._join_pcfs.get(join_col)
        if candidates is None:
            return None

        target_pcf: RangeConditionedPCF | None = None
        for current in candidates:
            if current.conditioned_col != range_col:
                continue
            target_pcf = current
            break

        if target_pcf is None:
            return None
        return target_pcf.get_greater(bound, inclusive=True)

    def lookup_strict_greater[T](
        self, join_col: pb.ColumnReference, *, range_col: pb.ColumnReference, bound: T
    ) -> Optional[PiecewiseConstantFn]:
        candidates = self._join_pcfs.get(join_col)
        if candidates is None:
            return None

        target_pcf: RangeConditionedPCF | None = None
        for current in candidates:
            if current.conditioned_col != range_col:
                continue
            target_pcf = current
            break

        if target_pcf is None:
            return None
        return target_pcf.get_greater(bound, inclusive=False)


def histogram_for_bucket[T: _HistogramKey](
    range_distribution: list[tuple[T, int]],
    *,
    k: int,
    total_cardinality: int,
    accuracy: float,
    join_col: pb.BoundColumnReference,
    range_col: pb.BoundColumnReference,
    database: pb.Database,
) -> RangeConditionedPCF[T]:
    freq_per_bucket = total_cardinality // k
    buckets: list[PiecewiseConstantFn] = []
    bounds: list[T] = []

    total_freq = 0
    lower_bound: T | None = None
    for prev, cur, nxt in pb.util.sliding_window(range_distribution, 3):
        cur_val, cur_freq = cur
        prev_val, prev_freq = prev
        nxt_val, nxt_freq = nxt
        updated_freq = total_freq + cur_freq
        if updated_freq < freq_per_bucket:
            # we still have room in our bucket
            total_freq = updated_freq
            continue

        # Our current frequency exceeds the target frequency of each bucket,
        # therefore we need to create a new bucket.
        # To minimize the difference between ideal frequency of each bucket and
        # the actual frequency, we need to compare whether the new bucket should
        # end at the current element, or (if the current element's frequency is huge)
        # if ending at the previous bucket would be even better.

        lower_err = freq_per_bucket - total_freq
        cur_err = updated_freq - freq_per_bucket
        if lower_err < cur_err:
            # stop at the previous value, include the current value in the next bucket
            upper_bound = cur_val  # upper bound is exclusive
            total_freq = cur_freq
        else:
            # include the current value in the bucket
            upper_bound = nxt_val  # upper bound is exclusive
            total_freq = 0

        lower_pred = pb.qal.as_predicate(range_col, ">=", lower_bound)
        upper_pred = pb.qal.as_predicate(range_col, "<", upper_bound)
        range_pred = pb.qal.CompoundPredicate.create_and([lower_pred, upper_pred])

        pcf = fetch_correlated_ds(
            range_pred, on=join_col, database=database, accuracy=accuracy
        )
        buckets.append(pcf)
        bounds.append(upper_bound)

        # since our upper bound is exclusive, the next bucket has to start at our current
        # upper bound to make sure we don't miss any values.
        lower_bound = upper_bound

    return RangeConditionedPCF(buckets, bounds, conditioned_col=join_col)


def build_histograms(
    spec: CatalogSpec, *, hierarchy_depth: int, accuracy: float, database: pb.Database
) -> RangeConditionedSequenceRepo:
    pcfs = collections.defaultdict(list)

    for join_col, filter_cols in spec.range_cols.items():
        for range_col in filter_cols:
            filter_distribution = fetch_column_distribution(range_col, database)
            cardinality = database.statistics().total_rows(join_col.table)
            assert cardinality is not None

            last_histogram: RangeConditionedPCF | None = None
            for i in range(hierarchy_depth, 0, -1):
                # XXX: The current implementation is pretty inefficient:
                # We essentially scan the entire distribution k times, summing
                # up all the frequencies each and every time. Maybe we can use
                # cumulative sums to eliminate some of this?

                current_histogram = histogram_for_bucket(
                    filter_distribution,
                    k=i,
                    total_cardinality=cardinality,
                    accuracy=accuracy,
                    join_col=join_col,
                    range_col=range_col,
                    database=database,
                )
                current_histogram.higher_res = last_histogram
                last_histogram = current_histogram

            pcfs[join_col].append(last_histogram)

    return RangeConditionedSequenceRepo(pcfs)


ThreeGram = str


def three_grams(text: str) -> Generator[ThreeGram, None, None]:
    for i in range(len(text) - 2):
        yield text[i : i + 3]


class LikeConditionedPCF:
    def __init__(
        self,
        three_grams: dict[ThreeGram, PiecewiseConstantFn],
        *,
        default: PiecewiseConstantFn,
        conditioned_col: pb.ColumnReference,
    ) -> None:
        self._grams = three_grams
        self._default = default
        self._column = conditioned_col

    @property
    def conditioned_col(self) -> pb.ColumnReference:
        return self._column

    def get(self, key: str) -> PiecewiseConstantFn:
        current_pcf: PiecewiseConstantFn | None = None
        for gram in three_grams(key):
            pcf = self._grams.get(gram)
            if pcf is None:
                continue

            if current_pcf is None:
                current_pcf = pcf
            else:
                current_pcf = current_pcf.min_with(pcf)

        return current_pcf or self._default


class LikeConditionedSequenceRepo:
    def __init__(
        self, join_pcfs: dict[pb.ColumnReference, list[LikeConditionedPCF]]
    ) -> None:
        self._pcfs = join_pcfs

    def lookup(
        self,
        join_col: pb.ColumnReference,
        *,
        like_col: pb.ColumnReference,
        like_val: str,
    ) -> Optional[PiecewiseConstantFn]:
        candidates = self._pcfs.get(join_col)
        if candidates is None:
            return None

        target_pcf: LikeConditionedPCF | None = None
        for current in candidates:
            if current.conditioned_col != like_col:
                continue
            target_pcf = current
            break

        if target_pcf is None:
            return None

        return target_pcf.get(like_val)


def gram_frequency(values: list[str], *, mcv_size: int) -> tuple[list[str], list[str]]:
    counter = collections.Counter()
    for txt in values:
        for gram in three_grams(txt):
            counter[gram] += 1

    frequent_grams = [elem for elem, _ in counter.most_common(mcv_size)]
    # see https://docs.python.org/3/library/collections.html#counter-objects for the weird syntax
    rare_grams = [elem for elem, _ in counter.most_common()[: -mcv_size - 1 : -1]]

    return frequent_grams, rare_grams


def build_gram_pcf(
    frequent_grams: list[str],
    rare_grams: list[str],
    *,
    join_col: pb.BoundColumnReference,
    like_col: pb.BoundColumnReference,
    accuracy: float,
    database: pb.Database,
) -> LikeConditionedPCF:
    frequent_pcfs: dict[str, PiecewiseConstantFn] = {}

    for gram in frequent_grams:
        like_pred = pb.qal.as_predicate(like_col, "LIKE", f"%{gram}%")
        frequent_pcfs[gram] = fetch_correlated_ds(
            like_pred, on=join_col, accuracy=accuracy, database=database
        )

    rare_pcf = PiecewiseConstantFn.zero()
    for gram in rare_grams:
        like_pred = pb.qal.as_predicate(like_col, "LIKE", f"%{gram}%")
        rare_pcf += fetch_correlated_ds(
            like_pred, on=join_col, accuracy=accuracy, database=database
        )

    return LikeConditionedPCF(frequent_pcfs, default=rare_pcf, conditioned_col=like_col)


def build_3grams(
    spec: CatalogSpec, *, mcv_size: int, accuracy: float, database: pb.Database
) -> LikeConditionedSequenceRepo:
    pcfs = collections.defaultdict(list)

    for join_col, filter_cols in spec.like_cols.items():
        for like_col in filter_cols:
            text_values = fetch_column_values(like_col, database)
            frequent_grams, rare_grams = gram_frequency(text_values, mcv_size=mcv_size)
            pcf = build_gram_pcf(
                frequent_grams,
                rare_grams,
                join_col=join_col,
                like_col=like_col,
                accuracy=accuracy,
                database=database,
            )
            pcfs[join_col].append(pcf)

    return LikeConditionedSequenceRepo(pcfs)


class _PCFCollector(pb.qal.PredicateVisitor[PiecewiseConstantFn]):
    def __init__(self, catalog: SafeBoundCatalog, *, log: pb.util.Logger) -> None:
        self._catalog = catalog
        self._log = log

    def visit_binary_predicate(
        self, predicate: pb.qal.BinaryPredicate, *args, **kwargs
    ) -> PiecewiseConstantFn:
        join_col = kwargs["join_col"]
        if not pb.qal.SimpleFilter.can_wrap(predicate):
            return self._catalog.lookup_unconditioned(join_col)
        simplified = pb.qal.SimpleFilter(predicate)

        match simplified.operation:
            case pb.qal.LogicalOperator.Equal:
                return self._catalog.lookup_eq_conditioned(
                    join_col, filter_col=simplified.column, filter_val=simplified.value
                )

            case pb.qal.LogicalOperator.Less:
                return self._catalog.lookup_range_conditioned(
                    join_col,
                    range_col=simplified.column,
                    upper_bound=simplified.value,
                    upper_inclusive=False,
                )

            case pb.qal.LogicalOperator.LessEqual:
                return self._catalog.lookup_range_conditioned(
                    join_col,
                    range_col=simplified.column,
                    upper_bound=simplified.value,
                    upper_inclusive=True,
                )

            case pb.qal.LogicalOperator.Greater:
                return self._catalog.lookup_range_conditioned(
                    join_col,
                    range_col=simplified.column,
                    lower_bound=simplified.value,
                    lower_inclusive=False,
                )

            case pb.qal.LogicalOperator.GreaterEqual:
                return self._catalog.lookup_range_conditioned(
                    join_col,
                    range_col=simplified.column,
                    lower_bound=simplified.value,
                    lower_inclusive=True,
                )

            case pb.qal.LogicalOperator.Like | pb.qal.LogicalOperator.ILike:
                assert isinstance(simplified.value, str)
                return self._catalog.lookup_like_conditioned(
                    join_col, like_col=simplified.column, like_val=simplified.value
                )

            case _:
                return self._catalog.lookup_unconditioned(join_col)

    def visit_between_predicate(
        self, predicate: pb.qal.BetweenPredicate, *args, **kwargs
    ) -> PiecewiseConstantFn:
        join_col = kwargs["join_col"]
        if not pb.qal.SimpleFilter.can_wrap(predicate):
            return self._catalog.lookup_unconditioned(join_col)
        simplified = pb.qal.SimpleFilter(predicate)
        assert isinstance(simplified.value, tuple)
        lo, hi = simplified.value
        return self._catalog.lookup_range_conditioned(
            join_col, range_col=simplified.column, lower_bound=lo, upper_bound=hi
        )

    def visit_in_predicate(
        self, predicate: pb.qal.InPredicate, *args, **kwargs
    ) -> PiecewiseConstantFn:
        join_col = kwargs["join_col"]
        if not pb.qal.SimpleFilter.can_wrap(predicate):
            return self._catalog.lookup_unconditioned(join_col)
        simplified = pb.qal.SimpleFilter(predicate)
        assert isinstance(simplified.value, Iterable)
        pcf = PiecewiseConstantFn.zero()
        for filter_val in simplified.value:
            filter_pcf = self._catalog.lookup_eq_conditioned(
                join_col, filter_col=simplified.column, filter_val=filter_val
            )
            pcf += filter_pcf
        return pcf

    def visit_unary_predicate(
        self, predicate: pb.qal.UnaryPredicate, *args, **kwargs
    ) -> PiecewiseConstantFn:
        join_col = kwargs["join_col"]
        self._log(
            f"Falling back to unconditioned PCF for unary predicate {predicate} on join column {join_col}"
        )
        return self._catalog.lookup_unconditioned(join_col)

    def visit_not_predicate(
        self,
        predicate: pb.qal.CompoundPredicate,
        child_predicate: pb.qal.AbstractPredicate,
        *args,
        **kwargs,
    ) -> PiecewiseConstantFn:
        join_col = kwargs["join_col"]
        self._log(
            f"Falling back to unconditioned PCF for NOT predicate {predicate} on join column {join_col}"
        )
        return self._catalog.lookup_unconditioned(join_col)

    def visit_or_predicate(
        self,
        predicate: pb.qal.CompoundPredicate,
        components: Sequence[pb.qal.AbstractPredicate],
        *args,
        **kwargs,
    ) -> PiecewiseConstantFn:
        pcf = PiecewiseConstantFn.zero()
        for child in components:
            child_pcf = child.accept_visitor(self, *args, **kwargs)
            pcf += child_pcf
        return pcf

    def visit_and_predicate(
        self,
        predicate: pb.qal.CompoundPredicate,
        components: Sequence[pb.qal.AbstractPredicate],
        *args,
        **kwargs,
    ) -> PiecewiseConstantFn:
        head, tail = components[0], components[1:]
        pcf = head.accept_visitor(self, *args, **kwargs)
        for child in tail:
            child_pcf = child.accept_visitor(self, *args, **kwargs)
            pcf = pcf.min_with(child_pcf)
        return pcf


def build_unconditioned_pcfs(
    spec: CatalogSpec, *, accuracy: float, database: pb.Database
) -> dict[pb.ColumnReference, PiecewiseConstantFn]:
    pcfs: dict[pb.ColumnReference, PiecewiseConstantFn] = {}
    for col in spec.columns():
        mcv = database.statistics().most_common_values(col, k=-1, emulated=True)
        ds = DegreeSequence.from_mcv(mcv, column=col)
        compressed = valid_compress(ds, accuracy=accuracy)
        pcfs[col] = compressed.deriv()
    return pcfs


class SafeBoundCatalog:
    @staticmethod
    def online(
        workload: pb.Workload,
        database: pb.Database,
        *,
        spec: SafeBoundSpec = SafeBoundSpec.default(),
        verbose: bool | pb.util.Logger = False,
    ) -> SafeBoundCatalog:
        catalog_spec = derive_catalog(workload, verbose=verbose)
        eq_pcfs_repo = build_equality_mcvs(
            catalog_spec,
            mcv_size=spec.mcv_size,
            accuracy=spec.accuracy,
            database=database,
        )
        range_pcfs_repo = build_histograms(
            catalog_spec,
            hierarchy_depth=spec.hist_hierarchy_depth,
            accuracy=spec.accuracy,
            database=database,
        )
        like_pcfs_repo = build_3grams(
            catalog_spec,
            mcv_size=spec.mcv_size,
            accuracy=spec.accuracy,
            database=database,
        )
        unconditioned_pcfs = build_unconditioned_pcfs(
            catalog_spec, accuracy=spec.accuracy, database=database
        )
        return SafeBoundCatalog(
            equality_pcfs=eq_pcfs_repo,
            range_pcfs=range_pcfs_repo,
            like_pcfs=like_pcfs_repo,
            unconditioned_pcfs=unconditioned_pcfs,
            verbose=verbose,
        )

    @staticmethod
    def load(archive: Path | str) -> SafeBoundCatalog:
        pass

    @staticmethod
    def load_or_build(
        archive: Path | str,
        *,
        workload: pb.Workload,
        spec: SafeBoundSpec = SafeBoundSpec.default(),
        database: pb.Database,
        verbose: bool | pb.util.Logger = False,
    ) -> SafeBoundCatalog:
        archive = Path(archive)
        if archive.is_file():
            return SafeBoundCatalog.load(archive)

        catalog = SafeBoundCatalog.online(
            workload, database, spec=spec, verbose=verbose
        )
        catalog.store(archive)
        return catalog

    def __init__(
        self,
        *,
        equality_pcfs: EqualityConditionsRepo,
        range_pcfs: RangeConditionedSequenceRepo,
        like_pcfs: LikeConditionedSequenceRepo,
        unconditioned_pcfs: dict[pb.ColumnReference, PiecewiseConstantFn],
        verbose: bool | pb.util.Logger = False,
    ) -> None:
        self._eq_pcfs = equality_pcfs
        self._range_pcfs = range_pcfs
        self._like_pcfs = like_pcfs
        self._unconditioned_pcfs = unconditioned_pcfs
        self._log = wrap_logger(verbose)

    def retrieve_stats(
        self, query: pb.SqlQuery
    ) -> dict[pb.ColumnReference, PiecewiseConstantFn]:
        join_map: dict[pb.ColumnReference, pb.qal.AbstractPredicate] = {}
        join_cols = pb.util.set_union(join.columns() for join in query.joins())
        for join_col in join_cols:
            assert pb.ColumnReference.assert_bound(join_col)
            filter_preds = query.filters_for(join_col.table)
            if not filter_preds:
                continue
            join_map[join_col] = filter_preds

        pred_traversal = _PCFCollector(self, log=self._log)
        return {
            col: pred.accept_visitor(pred_traversal, join_col=col)
            for col, pred in join_map.items()
        }

    def lookup_unconditioned(self, join_col: pb.ColumnReference) -> PiecewiseConstantFn:
        pcf = self._unconditioned_pcfs.get(join_col)
        if pcf is None:
            raise KeyError(
                f"Catalog has no unconditioned PCF for join column {join_col}"
            )
        return pcf

    def lookup_eq_conditioned[T](
        self,
        join_col: pb.ColumnReference,
        *,
        filter_col: pb.ColumnReference,
        filter_val: T,
    ) -> PiecewiseConstantFn:
        pcf = self._eq_pcfs.lookup(
            join_col, filter_col=filter_col, filter_val=filter_val
        )

        if pcf is None:
            self._log(
                f"Falling back to unconditioned PCF for join column {join_col} on equality condition {filter_col} = {filter_val}"
            )
            return self.lookup_unconditioned(join_col)

        return pcf

    def lookup_range_conditioned[T](
        self,
        join_col: pb.ColumnReference,
        *,
        range_col: pb.ColumnReference,
        lower_bound: T | None = None,
        upper_bound: T | None = None,
        lower_inclusive: bool = True,
        upper_inclusive: bool = True,
    ) -> PiecewiseConstantFn:
        if lower_bound is not None and upper_bound is not None:
            pcf = self._range_pcfs.lookup_range(
                join_col, range_col=range_col, between=(lower_bound, upper_bound)
            )

        elif lower_bound is not None and lower_inclusive:
            pcf = self._range_pcfs.lookup_less_equal(
                join_col, range_col=range_col, bound=lower_bound
            )
        elif lower_bound is not None and not lower_inclusive:
            pcf = self._range_pcfs.lookup_strict_less(
                join_col, range_col=range_col, bound=lower_bound
            )

        elif upper_bound is not None and lower_inclusive:
            pcf = self._range_pcfs.lookup_greater_equal(
                join_col, range_col=range_col, bound=upper_bound
            )
        elif upper_bound is not None and not lower_inclusive:
            pcf = self._range_pcfs.lookup_strict_greater(
                join_col, range_col=range_col, bound=upper_bound
            )

        else:
            raise ValueError(
                "At least one of lower_bound and upper_bound has to be non-None"
            )

        if pcf is None:
            self._log(
                f"Falling back to unconditioned PCF for join column {join_col} on range condition {range_col} between {lower_bound} and {upper_bound}"
            )
            return self.lookup_unconditioned(join_col)

        return pcf

    def lookup_like_conditioned(
        self,
        join_col: pb.ColumnReference,
        *,
        like_col: pb.ColumnReference,
        like_val: str,
    ) -> PiecewiseConstantFn:
        pcf = self._like_pcfs.lookup(join_col, like_col=like_col, like_val=like_val)

        if pcf is None:
            self._log(
                f"Falling back to unconditioned PCF for join column {join_col} on like condition {like_col} LIKE {like_val}"
            )
            return self.lookup_unconditioned(join_col)

        return pcf

    def store(self, archive: Path | str) -> None:
        pass
