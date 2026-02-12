from __future__ import annotations

import itertools
import json
from collections.abc import Collection, Generator, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import postbound as pb
import sklearn.preprocessing
import torch

from ..preprocessing import ColumnEncoder
from ..util import wrap_logger


def _normalize_column(
    col: pb.ColumnReference, drop_table_aliases: bool
) -> pb.ColumnReference:
    if drop_table_aliases:
        return pb.ColumnReference(col.name, col.table.drop_alias())
    return col


def _normalize_join_key(
    join: pb.qal.AbstractPredicate, drop_table_aliases: bool
) -> pb.qal.BinaryPredicate:
    if not pb.qal.SimpleJoin.can_wrap(join):
        raise ValueError(
            f"Cannot featurized join predicate {join}. Structure is not supported"
        )
    simplified = pb.qal.SimpleJoin(join)

    key1 = _normalize_column(simplified.lhs, drop_table_aliases)
    key2 = _normalize_column(simplified.rhs, drop_table_aliases)
    if key2 < key1:
        key1, key2 = key2, key1
    return pb.qal.as_predicate(key1, pb.qal.LogicalOperator.Equal, key2)


@dataclass
class FeaturizedQuery:
    tables: torch.FloatTensor
    joins: torch.FloatTensor
    predicates: torch.FloatTensor
    tables_mask: torch.FloatTensor
    joins_mask: torch.FloatTensor
    predicates_mask: torch.FloatTensor

    def aslist(self) -> list[torch.FloatTensor]:
        return [
            self.tables,
            self.joins,
            self.predicates,
            self.tables_mask,
            self.joins_mask,
            self.predicates_mask,
        ]


class MscnFeaturizer:
    @staticmethod
    def online(
        database: pb.Database,
        tables: Optional[Iterable[pb.TableReference]] = None,
        verbose: bool | pb.util.Logger = False,
    ) -> MscnFeaturizer:
        schema = database.schema()
        if tables is None:
            tables = [
                tab for tab in schema.tables() if not tab.full_name.startswith("pg_")
            ]
        else:
            tables = [tab.drop_alias() for tab in tables]

        fk_join_classes = schema.join_equivalence_classes()
        join_pairs = pb.util.flatten(
            itertools.combinations(cls, 2) for cls in fk_join_classes
        )
        join_pairs = [tuple(sorted(pair)) for pair in join_pairs]
        joins = [
            pb.qal.as_predicate(key1, pb.qal.LogicalOperator.Equal, key2)
            for key1, key2 in join_pairs
        ]

        columns = pb.util.flatten([schema.columns(table) for table in tables])

        stats = database.statistics()
        cards = [stats.total_rows(tab) for tab in tables]
        cards = [card for card in cards if card is not None]
        cards.sort(reverse=True)
        outer_extent = np.prod(cards[:3])

        column_encoders = {
            col: ColumnEncoder.online(col, database=database, verbose=verbose)
            for col in columns
        }

        return MscnFeaturizer(
            schema_name=database.database_name(),
            tables=tables,
            joins=joins,
            filter_columns=columns,
            comparison_operators=list(pb.qal.LogicalOperator),
            value_encoders=column_encoders,
            min_card=pb.Cardinality(0),
            max_card=pb.Cardinality(outer_extent),
            drop_table_aliases=True,
            verbose=verbose,
        )

    @staticmethod
    def infer_from_workload(
        workload: pb.Workload,
        *,
        min_card: pb.Cardinality | float | int = pb.Cardinality.unknown(),
        max_card: pb.Cardinality | float | int = pb.Cardinality.unknown(),
        database: Optional[pb.Database] = None,
        verbose: bool | pb.util.Logger = False,
    ) -> MscnFeaturizer:
        tables: set[pb.TableReference] = set()
        joins: set[pb.qal.BinaryPredicate] = set()
        filter_columns: set[pb.ColumnReference] = set()
        comparison_operators: set[pb.qal.LogicalOperator] = set()

        for query in workload.queries():
            tables.update(query.tables())

            for join in query.joins():
                simplified = pb.qal.SimpleJoin(join)
                normalized_join = _normalize_join_key(join, drop_table_aliases=True)
                joins.add(normalized_join)

            for pred in query.filters():
                simplified = pb.qal.SimpleFilter(pred)
                col = _normalize_column(simplified.column, drop_table_aliases=True)
                filter_columns.add(col)
                comparison_operators.add(simplified.operation)

        database = database or pb.db.current_database()
        column_encoders = {
            col: ColumnEncoder.online(col, database=database, verbose=verbose)
            for col in filter_columns
        }

        min_card = pb.Cardinality.of(min_card)
        max_card = pb.Cardinality.of(max_card)
        min_card = pb.Cardinality.of(max(min_card, 0))
        if max_card.is_unknown():
            cards = [database.statistics().total_rows(tab) or 0 for tab in tables]
            cards.sort(reverse=True)
            outer_extent = np.prod(cards[:3])
            max_card = pb.Cardinality.of(outer_extent)

        return MscnFeaturizer(
            schema_name=database.database_name(),
            tables=tables,
            joins=joins,
            filter_columns=filter_columns,
            comparison_operators=comparison_operators,
            value_encoders=column_encoders,
            min_card=min_card,
            max_card=max_card,
            drop_table_aliases=True,
            verbose=verbose,
        )

    @staticmethod
    def infer_from_samples(
        df: pd.DataFrame | str | Path,
        *,
        query_col: str = "query",
        cardinality_col: str = "cardinality",
        database: Optional[pb.Database] = None,
        verbose: bool | pb.util.Logger = False,
    ) -> MscnFeaturizer:
        df = pd.read_csv(df) if not isinstance(df, pd.DataFrame) else df
        queries = df[query_col].map(pb.parse_query)
        workload = pb.Workload({i: query for i, query in enumerate(queries, start=1)})
        min_card = df[cardinality_col].min()
        max_card = df[cardinality_col].max()
        return MscnFeaturizer.infer_from_workload(
            workload,
            min_card=min_card,
            max_card=max_card,
            database=database,
            verbose=verbose,
        )

    @staticmethod
    def pre_built(
        catalog_path: Path | str, verbose: bool | pb.util.Logger = False
    ) -> MscnFeaturizer:
        with open(catalog_path, "r") as f:
            catalog = json.load(f)

        schema_name = catalog["schema"]
        tables = [pb.parser.load_table_json(t) for t in catalog.get("tables", [])]
        joins = [pb.parser.load_predicate_json(j) for j in catalog.get("joins", [])]
        filter_columns = [
            pb.parser.load_column_json(c) for c in catalog.get("filter_columns", [])
        ]
        comparison_operators = [
            pb.qal.LogicalOperator(op) for op in catalog.get("comparison_operators", [])
        ]

        column_encoders: dict[pb.ColumnReference, ColumnEncoder] = {}
        for encoder_entry in catalog.get("column_encoders", []):
            col = pb.parser.load_column_json(encoder_entry["column"])
            assert col is not None and col.table is not None
            dtype = encoder_entry["dtype"]
            archive_file = encoder_entry["archive_file"]
            encoder = ColumnEncoder(col.name, col.table.full_name, dtype)
            encoder.load(Path(archive_file))
            column_encoders[col] = encoder

        return MscnFeaturizer(
            schema_name=schema_name,
            tables=tables,
            joins=joins,
            filter_columns=filter_columns,
            comparison_operators=comparison_operators,
            value_encoders=column_encoders,
            min_card=pb.Cardinality(catalog["min_card"]),
            max_card=pb.Cardinality(catalog["max_card"]),
            drop_table_aliases=catalog["drop_table_aliases"],
            verbose=verbose,
        )

    @staticmethod
    def load_or_build(
        catalog_path: Path | str,
        *,
        database: pb.Database,
        verbose: bool | pb.util.Logger = False,
    ) -> MscnFeaturizer:
        log = wrap_logger(verbose)
        catalog_path = Path(catalog_path)
        if catalog_path.exists():
            log("Loading pre-built MSCN featurizer from", catalog_path)
            return MscnFeaturizer.pre_built(catalog_path)
        log("MSCN featurizer not found. Building new one.")
        featurizer = MscnFeaturizer.online(database, verbose=verbose)
        log("Storing MSCN featurizer to", catalog_path)
        featurizer.store(catalog_path, encoder_dir=None)
        return featurizer

    def __init__(
        self,
        schema_name: str,
        tables: Iterable[pb.TableReference],
        joins: Iterable[pb.qal.BinaryPredicate],
        filter_columns: Iterable[pb.ColumnReference],
        comparison_operators: Iterable[pb.qal.LogicalOperator],
        *,
        min_card: pb.Cardinality,
        max_card: pb.Cardinality,
        value_encoders: dict[pb.ColumnReference, ColumnEncoder],
        drop_table_aliases: bool,
        verbose: bool | pb.util.Logger = False,
    ) -> None:
        self._log = wrap_logger(verbose)
        self.schema = schema_name

        self._log("Building table encoder")
        self._tables = list(tables)
        tables_arr = np.asarray([str(t) for t in self._tables]).reshape(-1, 1)
        self._tables_encoder = sklearn.preprocessing.OneHotEncoder(sparse_output=False)
        self._tables_encoder.fit(tables_arr)
        self._drop_table_aliases = drop_table_aliases

        self._log("Building join encoder")
        self._joins = list(joins)
        joins_arr = np.asarray([str(j) for j in self._joins]).reshape(-1, 1)
        self._joins_encoder = sklearn.preprocessing.OneHotEncoder(sparse_output=False)
        self._joins_encoder.fit(joins_arr)

        self._log("Building filter column encoder")
        self._columns = list(filter_columns)
        columns_arr = np.asarray([str(c) for c in self._columns]).reshape(-1, 1)
        self._columns_encoder = sklearn.preprocessing.OneHotEncoder(sparse_output=False)
        self._columns_encoder.fit(columns_arr)

        self._operators = [op.value for op in comparison_operators]
        operators_arr = np.asarray(self._operators).reshape(-1, 1)
        self._operator_encoder = sklearn.preprocessing.OneHotEncoder(
            sparse_output=False
        )
        self._operator_encoder.fit(operators_arr)

        self._value_encoders = value_encoders
        self._min_card = min_card
        self._max_card = max_card

    @property
    def n_tables(self) -> int:
        return len(self._tables)

    @property
    def n_joins(self) -> int:
        return len(self._joins)

    @property
    def n_columns(self) -> int:
        return len(self._columns)

    @property
    def n_operators(self) -> int:
        return len(self._operators)

    @property
    def min_card(self) -> pb.Cardinality:
        return self._min_card

    @property
    def max_card(self) -> pb.Cardinality:
        return self._max_card

    @property
    def norm_min_card(self) -> float:
        return np.log1p(self._min_card.value)

    @property
    def norm_max_card(self) -> float:
        return np.log1p(self._max_card.value)

    def encode_single(self, query: pb.SqlQuery) -> FeaturizedQuery:
        tables = query.tables()
        joins = query.joins()
        predicates = query.filters()

        tables_enc = self.encode_tables(tables)
        joins_enc = self.encode_joins(joins)
        predicates_enc = self.encode_filter_predicates(predicates)

        tables_mask = self.build_mask(len(tables), self.n_tables)
        joins_mask = self.build_mask(len(joins), self.n_joins)
        predicates_mask = self.build_mask(len(predicates), self.n_columns)

        return FeaturizedQuery(
            tables=torch.FloatTensor(tables_enc),
            joins=torch.FloatTensor(joins_enc),
            predicates=torch.FloatTensor(predicates_enc),
            tables_mask=torch.FloatTensor(tables_mask),
            joins_mask=torch.FloatTensor(joins_mask),
            predicates_mask=torch.FloatTensor(predicates_mask),
        )

    def encode_batch(
        self, queries: Iterable[pb.SqlQuery]
    ) -> Generator[FeaturizedQuery, None, None]:
        for query in queries:
            yield self.encode_single(query)

    def encode_tables(self, tables: Collection[pb.TableReference]) -> np.ndarray:
        if self._drop_table_aliases:
            tables = [t.drop_alias() for t in tables]
        tables_arr = np.asarray([str(t) for t in tables]).reshape(-1, 1)
        enc = self._tables_encoder.transform(tables_arr)
        num_pad = self.n_tables - len(tables)
        return np.pad(enc, {0: (0, num_pad)})

    def encode_joins(self, joins: Collection[pb.qal.BinaryPredicate]) -> np.ndarray:
        if not joins:
            return np.zeros((self.n_joins, self.n_joins))

        normalized_joins: list[pb.qal.BinaryPredicate] = []
        for join in joins:
            simplified = pb.qal.SimpleJoin(join)
            key1, key2 = simplified.lhs, simplified.rhs

            key1 = _normalize_column(key1, self._drop_table_aliases)
            key2 = _normalize_column(key2, self._drop_table_aliases)
            if key2 < key1:
                key1, key2 = key2, key1

            normalized_joins.append(
                pb.qal.as_predicate(key1, pb.qal.LogicalOperator.Equal, key2)
            )

        joins_arr = np.asarray([str(j) for j in normalized_joins]).reshape(-1, 1)
        enc = self._joins_encoder.transform(joins_arr)
        num_pad = self.n_joins - len(joins)
        return np.pad(enc, {0: (0, num_pad)})

    def encode_filter_predicates(
        self, predicates: Collection[pb.qal.BinaryPredicate]
    ) -> np.ndarray:
        if not predicates:
            # tensor structure: (column one-hot | operator one-hot | value encoding) x predicates
            n_features = self.n_columns + len(self._operators) + 1
            return np.zeros((self.n_columns, n_features))

        partial_enc = tuple[np.ndarray, np.ndarray, np.ndarray]
        vectors: dict[pb.ColumnReference, partial_enc] = {}

        for pred in predicates:
            simplified = pb.qal.SimpleFilter(pred)
            col = _normalize_column(simplified.column, self._drop_table_aliases)
            value_encoder = self._value_encoders[col]

            column = self._columns_encoder.transform([[str(col)]])
            operator = self._operator_encoder.transform([[simplified.operation.value]])
            assert simplified.operation not in [
                pb.qal.LogicalOperator.Between,
                pb.qal.LogicalOperator.In,
            ]
            value = value_encoder.encode_single(simplified.value)

            existing_vector = vectors.get(col)
            if existing_vector is None:
                vectors[col] = (column, operator, value)
                continue

            _, existing_operator, existing_value = existing_vector
            combined_op = np.maximum(existing_operator, operator)
            combined_value = np.concat([existing_value, value]).mean().reshape(-1, 1)
            vectors[col] = (column, combined_op, combined_value)

        enc = np.concat([np.concat(v, axis=1) for v in vectors.values()])
        num_pad = self.n_columns - len(vectors)
        return np.pad(enc, {0: (0, num_pad)})

    def build_mask(self, n_set: int, n_max: int) -> np.ndarray:
        present = np.ones(n_set, dtype=np.float32)
        padding = np.zeros(n_max - n_set, dtype=np.float32)
        return np.concatenate([present, padding], axis=0).reshape(-1, 1)

    def store(
        self, catalog_path: Path | str, *, encoder_dir: Optional[Path | str] = None
    ) -> None:
        catalog_path = Path(catalog_path)
        encoder_dir = encoder_dir or (catalog_path.parent)
        encoder_dir = Path(encoder_dir)
        catalog_path.parent.mkdir(parents=True, exist_ok=True)
        encoder_dir.mkdir(parents=True, exist_ok=True)

        self._log("Preparing catalog for storage at", catalog_path)
        catalog: dict = {
            "schema": self.schema,
            "tables": self._tables,
            "joins": self._joins,
            "filter_columns": self._columns,
            "comparison_operators": self._operators,
            "min_card": self._min_card,
            "max_card": self._max_card,
            "drop_table_aliases": self._drop_table_aliases,
            "column_encoders": [],
        }

        for col, encoder in self._value_encoders.items():
            self._log("Exporting encoder for", col)
            archive_file = encoder.store(encoder_dir)
            encoder_entry = {
                "column": col,
                "dtype": encoder.dtype,
                "archive_file": archive_file,
            }
            catalog["column_encoders"].append(encoder_entry)

        self._log("Exporting catalog to", catalog_path)
        with open(catalog_path, "w") as f:
            pb.util.to_json_dump(catalog, f)
