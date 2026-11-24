from __future__ import annotations

import bisect
import collections
import itertools
import json
import lzma
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Protocol, override

import numpy as np
import postbound as pb

from postbound_extensions.preprocessing import cast_value

from ..safebound import (
    CatalogSpec,
    DegreeSequence,
    fetch_column_distribution,
    fetch_correlated_ds,
    fetch_mcv_ds,
    fetch_non_mcv_ds,
    fetch_raw_ds,
)
from ..util import make_json_parser, wrap_logger
from ._core import NormSet


class LpCollection:
    @staticmethod
    def zeros() -> LpCollection:
        return LpCollection({})

    def __init__(self, norms: Mapping[float, float]) -> None:
        self._entries = dict(norms)
        self._norms = set(self._entries.keys())

    @property
    def norms(self) -> Mapping[float, float]:
        return {norm: val for norm, val in self._entries.items() if norm > 0}

    def is_zero(self) -> bool:
        return all(val == 0 for val in self._entries.values())

    def max_with(self, other: LpCollection) -> LpCollection:
        if self.is_zero():
            return other
        if other.is_zero():
            return self

        diff = self._norms.symmetric_difference(other._norms)
        if diff:
            raise ValueError(f"Cannot compute max_with for collections with different norms: {diff}")
        return LpCollection({norm: max(self[norm], other[norm]) for norm in self._norms})

    def min_with(self, other: LpCollection) -> LpCollection:
        if self.is_zero() or other.is_zero():
            return LpCollection.zeros()

        diff = self._norms.symmetric_difference(other._norms)
        if diff:
            raise ValueError(f"Cannot compute min_with for collections with different norms: {diff}")
        return LpCollection({norm: min(self[norm], other[norm]) for norm in self._norms})

    def sum_with(self, other: LpCollection) -> LpCollection:
        if self.is_zero():
            return other
        if other.is_zero():
            return self

        diff = self._norms.symmetric_difference(other._norms)
        if diff:
            raise ValueError(f"Cannot compute sum_with for collections with different norms: {diff}")
        return LpCollection({norm: self[norm] + other[norm] for norm in self._norms})

    def get(self, norm: float, default: float | None) -> float | None:
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


def load_norm_json(json_data: dict | str) -> LpCollection:
    if isinstance(json_data, str):
        json_data = json.loads(json_data)
    return LpCollection({float(norm): float(val) for norm, val in json_data})


def compute_norms(ds: DegreeSequence, *, norms: NormSet) -> LpCollection:
    degrees = ds.deg_array
    lpcoll: dict[float, float] = {}
    for norm in norms:
        lpcoll[norm] = np.linalg.norm(degrees, ord=norm)
    return LpCollection(lpcoll)


class EqualityConditionedNorms[T]:
    def __init__(
        self,
        norms: Mapping[T | None, LpCollection],
        *,
        filter_col: pb.BoundColumnReference,
        fallback: LpCollection,
    ) -> None:
        self._norms = dict(norms)
        self._fallback = fallback
        self._col = filter_col

    @property
    def filter_col(self) -> pb.BoundColumnReference:
        return self._col

    def get(self, key: T | None) -> LpCollection:
        norm = self._norms.get(key)
        if norm is None:
            return self._fallback
        return norm

    def __json__(self) -> pb.util.jsondict:
        norms_json = [{"value": val, "norms": norms} for val, norms in self._norms.items()]
        return {"filter_column": self._col, "norms": norms_json, "fallback": self._fallback}


def load_eq_norm_json(json_data: dict | str, *, database: pb.Database) -> EqualityConditionedNorms:
    if isinstance(json_data, str):
        json_data = json.loads(json_data)

    filter_col = pb.parser.load_column_json(json_data["filter_column"])
    dtype = database.schema().datatype(filter_col)
    val_parser = make_json_parser(dtype)

    fallback = load_norm_json(json_data["fallback"])
    norms = {val_parser(norm_json["value"]): load_norm_json(norm_json["norms"]) for norm_json in json_data["norms"]}

    return EqualityConditionedNorms(norms, filter_col=filter_col, fallback=fallback)


class EqualityConditionedRepo:
    def __init__(self, join_norms: Mapping[pb.ColumnReference, Sequence[EqualityConditionedNorms]]) -> None:
        self._norms = dict(join_norms)

    def lookup(
        self, join_col: pb.ColumnReference, *, filter_col: pb.ColumnReference, filter_val: object
    ) -> LpCollection | None:
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
    def __init__(
        self,
        norms: Sequence[LpCollection],
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

    def get_range(self, lower: T, upper: T) -> LpCollection:
        if lower > upper:
            # SQL allows BETWEEN 42 AND 24 which produces an empty result set.
            return LpCollection.zeros()
        if upper < self._min_val or lower > self._bounds[-1]:
            return LpCollection.zeros()

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

    def get_less(self, value: T, *, inclusive: bool) -> LpCollection:
        return self.get_range(self._min_val, value)

    def get_greater(self, value: T, *, inclusive: bool) -> LpCollection:
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
    def __init__(self, join_norms: Mapping[pb.ColumnReference, Sequence[RangeConditionedNorms]]) -> None:
        self._norms = dict(join_norms)

    def lookup_range[T](
        self, join_col: pb.ColumnReference, *, range_col: pb.ColumnReference, between: tuple[T, T]
    ) -> LpCollection | None:
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
    ) -> LpCollection | None:
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
    ) -> LpCollection | None:
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
    ) -> LpCollection | None:
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
    ) -> LpCollection | None:
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

    lp_norms: list[LpCollection] = []
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
) -> Mapping[pb.ColumnReference, LpCollection]:
    log = wrap_logger(verbose)
    repo: dict[pb.ColumnReference, LpCollection] = {}
    for join_col in spec.unconditioned_cols:
        log(f"Loading unconditioned norms for {join_col}")
        ds = fetch_raw_ds(join_col, database=database)
        repo[join_col] = compute_norms(ds, norms=norms)
    return repo


class LpNormRepo(pb.qal.PredicateVisitor[LpCollection]):
    @staticmethod
    def load(archive: Path | str, *, database: pb.Database) -> LpNormRepo:
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

        return load_norm_repo_json(catalog, database=database)

    def __init__(
        self,
        equality_norms: EqualityConditionedRepo,
        range_norms: RangeConditionedRepo,
        unconditioned_norms: Mapping[pb.ColumnReference, LpCollection],
        *,
        database: pb.Database,
        log: pb.util.Logger | bool = False,
    ) -> None:
        self._eq_norms = equality_norms
        self._range_norms = range_norms
        self._unconditioned = dict(unconditioned_norms)
        self._database = database
        self._log = wrap_logger(log)

    def retrieve_norms(self, query: pb.SqlQuery) -> Mapping[pb.ColumnReference, LpCollection]:
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
    def visit_binary_predicate(self, predicate: pb.qal.BinaryPredicate, *args, **kwargs) -> LpCollection:
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
    def visit_between_predicate(self, predicate: pb.qal.BetweenPredicate, *args, **kwargs) -> LpCollection:
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
    def visit_in_predicate(self, predicate: pb.qal.InPredicate, *args, **kwargs) -> LpCollection:
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

        lp_norms = LpCollection.zeros()
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
    def visit_unary_predicate(self, predicate: pb.qal.UnaryPredicate, *args, **kwargs) -> LpCollection:
        join_col = kwargs["join_col"]
        self._log(f"Unsupported predicate {predicate} for {join_col}, falling back to unconditioned norms")
        return self._unconditioned[join_col]

    @override
    def visit_not_predicate(self, predicate: pb.qal.NotPredicate, *args, **kwargs) -> LpCollection:
        join_col = kwargs["join_col"]
        self._log(f"Unsupported predicate {predicate} for {join_col}, falling back to unconditioned norms")
        return self._unconditioned[join_col]

    @override
    def visit_or_predicate(self, predicate: pb.qal.OrPredicate, *args, **kwargs) -> LpCollection:
        lp_norms = LpCollection.zeros()
        for child in predicate.children:
            child_norms = child.accept_visitor(self, *args, **kwargs)
            # an obvious optimization would be to bound the sum by the unconditioned norms, but this is not part of the
            # paper so we do not implement it either.
            lp_norms = lp_norms.sum_with(child_norms)
        return lp_norms

    @override
    def visit_and_predicate(self, predicate: pb.qal.AndPredicate, *args, **kwargs) -> LpCollection:
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


def load_norm_repo_json(json_data: dict | str, *, database: pb.Database) -> LpNormRepo:
    if isinstance(json_data, str):
        json_data = json.loads(json_data)

    equality_norms = load_eq_repo_json(json_data["equality_norms"], database=database)
    range_norms = load_range_repo_json(json_data["range_norms"], database=database)

    unconditioned: dict[pb.ColumnReference, LpCollection] = {}
    for entry in json_data["unconditioned_norms"]:
        join_col = pb.parser.load_column_json(entry["join_column"])
        norms = load_norm_json(entry["norms"])
        unconditioned[join_col] = norms

    return LpNormRepo(equality_norms, range_norms, unconditioned, database=database)
