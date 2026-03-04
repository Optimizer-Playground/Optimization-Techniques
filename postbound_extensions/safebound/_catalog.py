from __future__ import annotations

import bisect
import collections
from collections.abc import Generator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional, Protocol, overload

import numpy as np
import postbound as pb

from ._compress import valid_compress
from ._core import DegreeSequence
from ._piecewise_fns import PiecewiseConstantFn


class _HistogramKey(Protocol):
    def __lt__(self, other: _HistogramKey) -> bool: ...

    def __add__(self, other: _HistogramKey) -> _HistogramKey: ...

    def __sub__(self, other: _HistogramKey) -> _HistogramKey: ...

    def __hash__(self) -> int: ...


@dataclass
class CatalogSpec:
    simple_cols: set[pb.ColumnReference]
    equality_cols: dict[pb.ColumnReference, set[pb.ColumnReference]]
    range_cols: dict[pb.ColumnReference, set[pb.ColumnReference]]
    like_cols: dict[pb.ColumnReference, set[pb.ColumnReference]]

    @staticmethod
    def empty() -> CatalogSpec:
        return CatalogSpec(
            set(),
            collections.defaultdict(set),
            collections.defaultdict(set),
            collections.defaultdict(set),
        )


class _CatalogVisitor(pb.qal.PredicateVisitor[None]):
    def __init__(self, spec: CatalogSpec) -> None:
        self.spec = spec

    def visit_binary_predicate(
        self, predicate: pb.qal.BinaryPredicate, *args, **kwargs
    ) -> None:
        if not pb.qal.SimpleFilter.can_wrap(predicate):
            return
        simplified = pb.qal.SimpleFilter.wrap(predicate)
        join_map = kwargs["join_map"]

        match simplified.operation:
            case pb.qal.LogicalOperator.Equal:
                for join_col in join_map[simplified.column.table]:
                    self.spec.equality_cols[join_col].add(simplified.column)

            case (
                pb.qal.LogicalOperator.Less
                | pb.qal.LogicalOperator.LessEqual
                | pb.qal.LogicalOperator.Greater
                | pb.qal.LogicalOperator.GreaterEqual
            ):
                for join_col in join_map[simplified.column.table]:
                    self.spec.range_cols[join_col].add(simplified.column)

            case pb.qal.LogicalOperator.Like | pb.qal.LogicalOperator.ILike:
                for join_col in join_map[simplified.column.table]:
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
        join_map = kwargs["join_map"]
        for join_col in join_map[simplified.column.table]:
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
        join_map = kwargs["join_map"]
        for join_col in join_map[simplified.column.table]:
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
    join_cols = join_cols = pb.util.set_union(join.columns() for join in query.joins())
    join_map = collections.defaultdict(set)
    for col in join_cols:
        join_map[col.table].add(col)
    return join_map


def derive_catalog(workload: pb.Workload) -> CatalogSpec:
    spec = CatalogSpec.empty()
    visitor = _CatalogVisitor(spec)
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
    on: pb.ColumnReference,
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
        fallbacks: dict[pb.ColumnReference, PiecewiseConstantFn],
    ) -> None:
        self._functions = join_pcfs
        self._fallbacks = fallbacks

    def lookup(
        self,
        join_col: pb.ColumnReference,
        *,
        filter_col: pb.ColumnReference,
        filter_val: Any,
    ) -> PiecewiseConstantFn:
        candidates = self._functions.get(join_col)
        if candidates is None:
            raise KeyError(f"Catalog has no PCF for join column {join_col}")

        target_pcf: EqualityConditionedPCF | None = None
        for current in candidates:
            if current.filter_col != filter_col:
                continue
            target_pcf = current

        if target_pcf is None:
            fallback = self._fallbacks.get(join_col)
            if fallback is None:
                raise KeyError(
                    f"Catalog has no fallback PCF for join column {join_col}"
                )
            return fallback

        return target_pcf.get(filter_val)


def build_equality_mcvs(
    spec: CatalogSpec, *, mcv_size: int, accuracy: float, database: pb.Database
) -> EqualityConditionsRepo:
    pcfs = collections.defaultdict(list)
    fallbacks: dict[pb.ColumnReference, PiecewiseConstantFn] = {}

    for join_col, filter_cols in spec.equality_cols.items():
        fallback_ds = fetch_raw_ds(join_col, database=database)
        fallback_pcf = valid_compress(fallback_ds, accuracy=accuracy)
        fallbacks[join_col] = fallback_pcf.deriv()

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

    return EqualityConditionsRepo(pcfs, fallbacks)


class RangeConditionedPCF[T: _HistogramKey]:
    def __init__(
        self,
        buckets: Sequence[PiecewiseConstantFn],
        bounds: Sequence[T],
        *,
        column: pb.ColumnReference,
        higher_res: RangeConditionedPCF[T] | None = None,
    ) -> None:
        if len(buckets) != len(bounds):
            raise ValueError("bounds and buckets have to have the same length")
        self._buckets = list(buckets)
        self._bounds = list(bounds)
        self._column = column
        self._higher_res = higher_res

    @property
    def column(self) -> pb.ColumnReference:
        return self._column

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

    def get_less(self, value: T) -> PiecewiseConstantFn:
        own_range = self.error_less(value)
        if self._higher_res is None:
            return self.pcf_from_buckets(slice(own_range[1]))

        higher_res = self._higher_res.error_less(value)
        own_err, higher_err = own_range[0], higher_res[0]
        if own_err < higher_err:
            return self.pcf_from_buckets(slice(own_range[1]))

        # See comment in get_range() for why we have to call get_less()
        return self._higher_res.get_less(value)

    def get_greater(self, value: T) -> PiecewiseConstantFn:
        own_range = self.error_greater(value)
        if self._higher_res is None:
            return self.pcf_from_buckets(slice(own_range[1] + 1))

        higher_res = self._higher_res.error_greater(value)
        own_err, higher_err = own_range[0], higher_res[0]
        if own_err < higher_err:
            return self.pcf_from_buckets(slice(own_range[1] + 1))

        # See comment in get_range() for why we have to call get_greater()
        return self._higher_res.get_greater(value)

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


def fetch_column_distribution[T](
    column: pb.ColumnReference, database: pb.Database
) -> list[tuple[T, int]]:
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


def histogram_for_bucket[T: _HistogramKey](
    range_distribution: list[tuple[T, int]],
    *,
    k: int,
    total_cardinality: int,
    accuracy: float,
    join_col: pb.ColumnReference,
    range_col: pb.ColumnReference,
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

    return RangeConditionedPCF(buckets, bounds, column=join_col)


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
        default: PiecewiseConstantFn,
    ) -> None:
        self._grams = three_grams
        self._default = default

    @overload
    def get(self, key: str, default: PiecewiseConstantFn) -> PiecewiseConstantFn: ...

    @overload
    def get(
        self, key: str, default: Literal[None]
    ) -> Optional[PiecewiseConstantFn]: ...

    @overload
    def get(self, key: str) -> Optional[PiecewiseConstantFn]: ...

    def get(
        self, key: str, default: PiecewiseConstantFn | None
    ) -> Optional[PiecewiseConstantFn]:
        current_pcf: PiecewiseConstantFn | None = None
        for gram in three_grams(key):
            pcf = self._grams.get(gram)
            if pcf is None:
                continue

            if current_pcf is None:
                current_pcf = pcf
            else:
                current_pcf = current_pcf.min_with(pcf)

        if current_pcf is None:
            return default

    def __getitem__(self, key: str) -> PiecewiseConstantFn:
        pcf = self.get(key)
        return self._default if pcf is None else pcf


class LikeConditionedSequenceRepo:
    pass


class SafeBoundCatalog:
    @staticmethod
    def online(workload: pb.Workload, database: pb.Database) -> SafeBoundCatalog:
        pass

    @staticmethod
    def load(archive: Path | str) -> SafeBoundCatalog:
        pass

    @staticmethod
    def load_or_build(
        archive: Path | str, *, workload: pb.Workload, database: pb.Database
    ) -> SafeBoundCatalog:
        archive = Path(archive)
        if archive.is_file():
            return SafeBoundCatalog.load(archive)

        catalog = SafeBoundCatalog.online(workload, database)
        catalog.store(archive)
        return catalog

    def __init__(self) -> None:
        pass

    def retrieve_stats(
        self, query: pb.SqlQuery
    ) -> dict[pb.ColumnReference, PiecewiseConstantFn]:
        pass

    def store(self, archive: Path | str) -> None:
        pass
