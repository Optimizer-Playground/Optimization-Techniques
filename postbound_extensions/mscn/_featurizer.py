"""
MSCN query featurization logic
    added as part of the PostBOUND integration

The MIT License

Copyright (c) 2026 Rico Bergmann

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

Original author: Andreas Kipf
Original source: https://github.com/andreaskipf/learnedcardinalities
Modified by: Rico Bergmann
"""

from __future__ import annotations

import itertools
import json
import warnings
from collections.abc import Collection, Generator, Iterable, Mapping, Sequence
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
        return col.drop_table_alias()
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


class FeaturizationWarning(UserWarning):
    pass


warnings.simplefilter("ignore", category=FeaturizationWarning)


@dataclass
class FeaturizedQuery:
    """Wrapper class for all feature vectors for a single query."""

    tables: torch.Tensor
    joins: torch.Tensor
    predicates: torch.Tensor
    tables_mask: torch.Tensor
    joins_mask: torch.Tensor
    predicates_mask: torch.Tensor

    def aslist(self) -> list[torch.Tensor]:
        return [
            self.tables,
            self.joins,
            self.predicates,
            self.tables_mask,
            self.joins_mask,
            self.predicates_mask,
        ]

    def to(self, device: torch.device) -> FeaturizedQuery:
        """Transfer the tensors of this feature vector to a specific device.

        Returns
        -------
        FeaturizedQuery
            A new featurized query instance that contains the same feature vectors, but with all
            tensors transferred to the specified device.
        """
        return FeaturizedQuery(
            tables=self.tables.to(device),
            joins=self.joins.to(device),
            predicates=self.predicates.to(device),
            tables_mask=self.tables_mask.to(device),
            joins_mask=self.joins_mask.to(device),
            predicates_mask=self.predicates_mask.to(device),
        )


class MscnFeaturizer:
    """The featurizer is used to transform SQL queries into their corresponding feature vectors.

    In addition to the query transformation, we also use the featurizer to store normalization
    data, specifically for the output cardinality.

    MSCN uses an featurization scheme that is very tightly coupled with a specific database instance
    and query workload. For example, MSCN must know in advance, exactly which tables might occur in
    a query, how they can be joined, which columns could be used for filters, which filter
    predicates are present, and what domains the filter values have.

    A mismatch between the featurization of the training data and that of the test data will lead to
    errors during featurization. Therefore, one needs to be careful when deciding how the
    featurization should be selected. As a consequence, we provide a number of different options for
    building a featurization. Each strategy is available as a static method to construct the
    featurizer:

    - `online` builds a featurizer exclusively based on the database schema, i.e. without
      considering the query workload. While this is the most realistic setting, it can lead to very
      large feature vectors. In turn, this might impact the training performance negatively
    - `infer_from_samples` builds a featurization tailored for a specific training set. Users must
      ensure that the test set uses a subset of the features of the training set. Alternatively, it
      is also possible to also pass the test queries to this method. In this case, we make sure that
      the featurization is built over the union of train and test queries.
    - `infer_from_workload` builds a featurization based on a given query workload. Users must
      ensure that both training and test set use a subset of features of this workload.
    - `pre_built` loads a previous featurization

    Once a featurization has been created, use `encode_single` or `encode_batch` to transform
    queries into their corresponding feature vectors. The other ``encode_XYZ`` methods handle the
    different steps of the featurization process. Usually, you do not need to call these explicitly.

    An existing featurization can be persisted using the `store` method. As a convenience function,
    the `load_or_build` function will load a previously stored featurization. If it does not exist,
    it will be created and stored.

    Text Columns
    ------------
    The original MSCN paper did not consider text columns in the featurization. We argue that this
    excludes a large number of interesting workloads. Therefore, we augment the featurization scheme
    to support text columns as follows: For each text column, we build a separate encoder. This
    encoder maintains an ordered dictionary of all distinct values of that column. To featurize
    a filter value, we lookup the insertion index of this value in the dictionary. This value is
    then min-max scaled to obtain a value between 0 and 1, similar to how integer columns are
    featurized in the original MSCN paper.

    Timestamp and Date Columns
    --------------------------
    The original MSCN paper did not explicitly consider time-valued columns in its featurization.
    However, since such column values can be directly mapped to UNIX timestamps (and therefore
    integer values), we perform this mapping and featurize them like normal integer values. Note
    that this discards any timezone information.
    """

    @staticmethod
    def online(
        database: pb.Database,
        tables: Optional[Iterable[pb.TableReference]] = None,
        verbose: bool | pb.util.Logger = False,
    ) -> MscnFeaturizer:
        """Infers the featurization scheme from a database schema.

        This featurization strategy trades off high generality for potentially very large
        (and thus sparse) feature vectors.

        Parameters
        ----------
        database : pb.Database
            The database for which the featurization should be built.
        tables : Optional[Iterable[pb.TableReference]], optional
            The tables to consider for the featurization. If not specified, all tables in the
            database schema are considered. As an optimization, we ignore Postgres system tables.
        verbose : bool | pb.util.Logger, optional
            Whether to print verbose output during the featurization building process.

        Notes
        -----
        This featurization strategy was not present in the original MSCN paper. However, we argue
        that this is the most realistic setting because it does not require any up-front
        knowledege about the query workload.

        We use the same featurization algorithm as standard MSCN. The only thing that changes are
        the parts of the schema that are considered for the one-hot encodings. In particular, we
        use the following logic:

        - All tables in the schema are considered for the table one-hot encoding.
        - All joins that can be inferred from the schema based on primary key/foreign key
          constraints are considered for the join one-hot encoding. This includes joins implied by
          equivalence classes of foreign key joins.
        - All columns of all tables are included in the filter column one-hot-encoding.
        - Filter values are min-max scaled to [0, 1] as outlined in the original MSCN paper (and
          including our extensions)
        - Filter operators are one-hot encoded based on all binary operators known to PostBOUND.
        - The cardinality is log-scaled similar to the original MSCN paper. We determine the maximum
          cardinality using a heuristic (see below)

        To determine the maximum cardinality for the featurization, we use the following bounding
        heuristic: we determine the three largest tables in the schema and calculate the cardinality
        of their cross product. The minimum cardinality is always assumed to be 0.
        """

        schema = database.schema()
        if tables is None:
            # Optimization: drop all Postgres system tables should they still be present.
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

        column_encoders: Mapping[pb.ColumnReference, ColumnEncoder] = {
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
        """Builds a featurizer tailored for a specific workload.

        This featurization strategy trades off generality for smaller (and thus less sparse) feature
        vectors.

        Parameters
        ----------
        workload : pb.Workload
            The workload for which the featurization should be built. See notes below for the
            specific strategy.
        min_card : pb.Cardinality | float | int, optional
            The minimum cardinality for the featurization. If this is omitted, it will be
            inferred using the same heuristic in the `online` case.
        max_card : pb.Cardinality | float | int, optional
            The maximum cardinality for the featurization. If this is omitted, it will be
            inferred using the same heuristic in the `online` case.
        database : Optional[pb.Database], optional
            The database for which the featurization should be built. This is required to build
            the individual column encoders as well as to determine the cardinality range.
        verbose : bool | pb.util.Logger, optional
            Whether to print verbose output during the featurization building process.

        Notes
        -----
        The featurization is tailored to the specific workload. If any additional table, join, or
        filter occurs in the test set that is not present in the workload, the featurization will
        fail.

        We use the same featurization algorithm as standard MSCN. The only thing that changes are
        the parts of the schema that are considered for the one-hot encodings. In particular, we
        use the following logic:

        - All tables that are used in the workload are considered for the table one-hot encoding.
        - All joins that are used in the workload are considered for the join one-hot encoding.
        - All columns that are used in filter predicates of the workload are included in the filter
          column one-hot-encoding.
        - Filter values are min-max scaled to [0, 1] as outlined in the original MSCN paper (and
          including our extensions)
        - Filter operators are one-hot encoded. However, we only include operators that are used for
          filter predicates of the workload.
        - The cardinality is log-scaled similar to the original MSCN paper. We determine the maximum
          cardinality using a heuristic (see below)

        To determine the maximum cardinality for the featurization, we use the following bounding
        heuristic: we determine the three largest tables in the schema and calculate the cardinality
        of their cross product. The minimum cardinality is always assumed to be 0.

        """
        tables: set[pb.TableReference] = set()
        joins: set[pb.qal.BinaryPredicate] = set()
        filter_columns: set[pb.ColumnReference] = set()
        comparison_operators: set[pb.qal.LogicalOperator] = set()

        for query in workload.queries():
            tables.update({tab.drop_alias() for tab in query.tables()})

            for join in query.joins():
                simplified = pb.qal.SimpleJoin(join)
                normalized_join = _normalize_join_key(join, drop_table_aliases=True)
                joins.add(normalized_join)

            for pred in query.filters():
                if not pb.qal.SimpleFilter.can_wrap(pred):
                    warnings.warn(
                        f"Skipping unsupported predicate {pred}",
                        category=FeaturizationWarning,
                        stacklevel=2,
                    )
                    continue
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
        if min_card.is_unknown():
            min_card = pb.Cardinality(0)
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
        workload: Optional[pb.Workload] = None,
        database: Optional[pb.Database] = None,
        verbose: bool | pb.util.Logger = False,
    ) -> MscnFeaturizer:
        """Builsd a featurizer tailored to a specific set of training queries.

        This featurization strategy is the middle-ground between the very specialized
        `infer_from_workload` strategy and the very general `online` strategy.

        Parameters
        ----------
        df : pd.DataFrame | str | Path
            The training samples to build the featurization from. This can either be a pre-loaded
            DataFrame, or a file path to training samples that can Pandas can load.
        query_col : str, optional
            The name of the column in `df` that contains the SQL queries. Defaults to "query".
        cardinality_col : str, optional
            The name of the column in `df` that contains the output cardinality.
            Defaults to "cardinality".
        workload : Optional[pb.Workload], optional
            An optional workload to build the featurization from. See Notes below on how we use this
            information.
        database : Optional[pb.Database], optional
            The database for which the featurization should be built. This is required to build
            the individual column encoders. If the database is not provided, it is loaded from
            the database pool.
        verbose : bool | pb.util.Logger, optional
            Whether to print verbose output during the featurization building process.

        Notes
        -----
        The featurization is tailored to a larger set of query samples and thus more general than
        just the test set. To address smaller discrepancies between the training and test set, users
        can also pass the test workload to this method. Its queries will also be considered in the
        featurization. This ensures that all queries can be featurized.

        We use the same featurization algorithm as standard MSCN. The only thing that changes are
        the parts of the schema that are considered for the one-hot encodings. In particular, we
        use the following logic:

        - All tables that are used in the samples are considered for the table one-hot encoding.
        - All joins that are used in the samples are considered for the join one-hot encoding.
        - All columns that are used in filter predicates of the samples are included in the filter
          column one-hot-encoding.
        - Filter values are min-max scaled to [0, 1] as outlined in the original MSCN paper (and
          including our extensions)
        - Filter operators are one-hot encoded. However, we only include operators that are used for
          filter predicates of the samples.
        - The cardinality is log-scaled similar to the original MSCN paper. We determine the minimum
          and maximum cardinalities based on the samples.
        """

        logger = wrap_logger(verbose)
        df = pd.read_csv(df) if not isinstance(df, pd.DataFrame) else df
        if isinstance(df[query_col].iloc[0], pb.SqlQuery):
            queries = df[query_col]
        else:
            logger("Parsing training queries")
            queries = df[query_col].map(pb.parse_query)

        # If we have both a workload and a set of training samples, we need to be a bit careful:
        # The MSCN featurization is very picky about allowed columns and tables due to the
        # extensive usage of one-hot encodings. Therefore, we need to make sure that test workload
        # and training samples use exactly the same set of columns. If we were to build the
        # featurization over the samples, we would encouter an error as soon as an column that is
        # only in the test set is encoutered and vice-versa.
        # Therefore, we build the featurization over the union of train and test queries.

        if workload is None:
            workload = pb.Workload(
                {i: query for i, query in enumerate(queries, start=1)}
            )
        else:
            sample_queries: dict[str, pb.SqlQuery] = {
                f"train_{i}": query for i, query in enumerate(queries, start=1)
            }
            sample_queries.update(workload)
            workload = pb.Workload(sample_queries)

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
        """Loads a previously built featurization from disk.

        See Also
        --------
        store : inverse method to persist a featurization to disk
        """
        logger = wrap_logger(verbose)

        with open(catalog_path, "r") as f:
            logger("Loading pre-built MSCN featurizer from", catalog_path)
            catalog = json.load(f)

        schema_name = catalog["schema"]

        logger("Parsing tables")
        tables = [pb.parser.load_table_json(t) for t in catalog.get("tables", [])]

        logger("Parsing joins")
        joins = [pb.parser.load_predicate_json(j) for j in catalog.get("joins", [])]

        logger("Parsing filters")
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
            logger("Loading encoder for column", col)

            dtype = encoder_entry["dtype"]
            archive_file = encoder_entry["archive_file"]
            encoder = ColumnEncoder(col, dtype)
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
        workload: Optional[pb.Workload] = None,
        verbose: bool | pb.util.Logger = False,
    ) -> MscnFeaturizer:
        """Integrated inference and storage procedure.

        If the featurization has already been stored to `catalog_path`, it will simply be loaded.
        Otherwise, it will be inferred based on the additional parameters:

        - if both `workload` and `database` are provided, the featurization will use
          `infer_from_workload`
        - if only the `database` is provided, the featurization will use `online`
        """
        log = wrap_logger(verbose)
        catalog_path = Path(catalog_path)
        if catalog_path.exists():
            log("Loading pre-built MSCN featurizer from", catalog_path)
            return MscnFeaturizer.pre_built(catalog_path, verbose=verbose)
        log("MSCN featurizer not found. Building new one.")
        featurizer = (
            MscnFeaturizer.infer_from_workload(
                workload, database=database, verbose=verbose
            )
            if workload
            else MscnFeaturizer.online(database, verbose=verbose)
        )
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
        value_encoders: Mapping[pb.ColumnReference, ColumnEncoder],
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
        """Get the total number of tables supported by the featurization.

        This corresponds to length of the table-specific one-hot vector.
        """
        return len(self._tables)

    @property
    def n_joins(self) -> int:
        """Get the total number of joins supported by the featurization.

        This corresponds to length of the join-specific one-hot vector.
        """
        return len(self._joins)

    @property
    def n_columns(self) -> int:
        """Get the total number of filter columns supported by the featurization.

        This correponds to the number of filter-"subvectors" in the predicate feature vector.
        The final length of each subvector is determined based on the number of columns and the
        number of operators, plus an additional cell for the actual filter value.
        """
        return len(self._columns)

    @property
    def n_operators(self) -> int:
        """Get the total number of filter operators supported by the featurization.

        This corresponds to the length of the operator-specific one-hot vector that is part of the
        predicate feature vector.
        """
        return len(self._operators)

    @property
    def min_card(self) -> pb.Cardinality:
        """Get the minimum cardinality supported by the featurization.

        This is the raw cardinality. A scaled version is also available via the
        `norm_min_card` property.
        """
        return self._min_card

    @property
    def max_card(self) -> pb.Cardinality:
        """Get the maximum cardinality supported by the featurization.

        This is the raw cardinality. A scaled version is also available via the
        `norm_max_card` property.
        """
        return self._max_card

    @property
    def norm_min_card(self) -> float:
        """Get the normalized minimum cardinality supported by the featurization."""
        return np.log1p(self._min_card.value)

    @property
    def norm_max_card(self) -> float:
        """Get the normalized maximum cardinality supported by the featurization."""
        return np.log1p(self._max_card.value)

    def encode_single(self, query: pb.SqlQuery) -> FeaturizedQuery:
        """Encodes a single query into its corresponding feature vectors.

        We use the same featurization strategy as in the original MSCN paper. However, depending
        on the specific tables, joins, and filters that are considered in the featurization, the
        final feature vectors will have different lenghts. See the documentation of the different
        inference strategies for details.
        """
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
        """Encodes multiple queries.

        See Also
        --------
        encode_single
        """
        for query in queries:
            yield self.encode_single(query)

    def encode_tables(self, tables: Collection[pb.TableReference]) -> np.ndarray:
        """Computes the feature matrix for a set of tables.

        The matrix is automatically padded based on the total number of tables.
        """
        if self._drop_table_aliases:
            tables = [t.drop_alias() for t in tables]
        tables_arr = np.asarray([str(t) for t in tables]).reshape(-1, 1)
        enc = self._tables_encoder.transform(tables_arr)
        num_pad = self.n_tables - len(tables)
        return np.pad(enc, {0: (0, num_pad)})

    def encode_tables_batch(
        self, tables_batch: Iterable[Collection[pb.TableReference]]
    ) -> tuple[np.ndarray, Sequence[int]]:
        raw_tables: list[str] = []
        batch_indexes: list[int] = []
        for batch in tables_batch:
            batch_indexes.append(len(batch))
            raw_tables.extend(str(tab) for tab in batch)
        tables_arr = np.asarray(raw_tables).reshape(-1, 1)
        enc = self._tables_encoder.transform(tables_arr)
        return enc, batch_indexes

    def encode_joins(self, joins: Collection[pb.qal.BinaryPredicate]) -> np.ndarray:
        """Computes the feature matrix for a set of joins.

        The matrix is automatically padded based on the total number of joins.
        """
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

    def encode_joins_batch(
        self, joins_batch: Iterable[Collection[pb.qal.BinaryPredicate]]
    ) -> tuple[np.ndarray, Sequence[int]]:
        raw_joins: list[str] = []
        batch_indexes: list[int] = []
        for batch in joins_batch:
            batch_indexes.append(len(batch))
            for join in batch:
                simplified = pb.qal.SimpleJoin(join)
                key1, key2 = simplified.lhs, simplified.rhs

                key1 = _normalize_column(key1, self._drop_table_aliases)
                key2 = _normalize_column(key2, self._drop_table_aliases)
                if key2 < key1:
                    key1, key2 = key2, key1

                normalized_join = pb.qal.as_predicate(
                    key1, pb.qal.LogicalOperator.Equal, key2
                )
                raw_joins.append(str(normalized_join))

        joins_arr = np.asarray(raw_joins).reshape(-1, 1)
        enc = self._joins_encoder.transform(joins_arr)
        return enc, batch_indexes

    def encode_filter_predicates(
        self, predicates: Collection[pb.qal.BinaryPredicate]
    ) -> np.ndarray:
        """Computes the feature matrix for a set of filter predicates.

        The matrix is automatically padded based on the total number of filter columns.
        """
        if not predicates:
            # tensor structure: (column one-hot | operator one-hot | value encoding) x predicates
            n_features = self.n_columns + len(self._operators) + 1
            return np.zeros((self.n_columns, n_features))

        # The predicate featurization logic is quite lengthy. This is because predicate
        # featurization is a very "hot" operation during training and we need to be a bit vary about
        # performance.
        # Specifically, we saw that repeatedly calling each encoder for every single value quickly
        # becomes a bottleneck. Therefore, we try to batch as much of the encoding as possible.
        # However, this leads to the complications you see below.
        #
        # Our algorithm works as follows:
        # First, we iterate over all filter predicates to collect the required information for
        # the encoding, such as the columns or operators.
        # As a complicating detail, the encoding of the filter value depends on the column's data
        # type. Therefore, we need an indirection: we only store the encoder resposible for the
        # specific filter value. In turn this leads to another problem, because different value
        # encoders will have to encode different numbers of filter values.
        # In the end, we settled for the following layout: for each individual encoder, we store
        # the values that will (eventually) need to encode. For each filter predicate, we store
        # the resposible value encoder and the index into this encoder's value list for our filter
        # value.
        # Once we have gathered all featurization data, we invoke the actual encoders.
        # Lastly, we need to stitch the different predicates back together. This results in a
        # second loop where we query the column, operator and value (index) lists and extract the
        # corresponding feature vectors. As part of this step, we also need to resolve the
        # encoder->values indirection to obtain the correct filter value.
        #
        # To make the code easier to follow, we mark the different phases with comments.

        partial_enc = tuple[np.ndarray, np.ndarray, np.ndarray]
        vectors: dict[pb.ColumnReference, partial_enc] = {}

        columns: list[pb.ColumnReference] = []
        column_strings: list[str] = []
        operators: list[str] = []
        value_indexes: list[int] = []
        encoder_batches: dict[pb.ColumnReference, list] = {}

        # Stage 1: gather the data that needs to be encoded.

        for pred in predicates:
            if not pb.qal.SimpleFilter.can_wrap(pred):
                warnings.warn(
                    f"Skipping unsupported predicate {pred}",
                    category=FeaturizationWarning,
                    stacklevel=2,
                )
                continue

            simplified = pb.qal.SimpleFilter(pred)
            col = _normalize_column(simplified.column, self._drop_table_aliases)
            unsupported_ops = [
                pb.qal.LogicalOperator.Between,
                pb.qal.LogicalOperator.In,
            ]
            if simplified.operation in unsupported_ops:
                warnings.warn(
                    f"Skipping unsupported predicate {pred}",
                    category=FeaturizationWarning,
                    stacklevel=2,
                )
                continue

            columns.append(col)
            column_strings.append(str(col))
            operators.append(simplified.operation.value)
            if col in encoder_batches:
                vals = encoder_batches[col]
                val_idx = len(vals)
                vals.append(simplified.value)
                value_indexes.append(val_idx)
            else:
                encoder_batches[col] = [simplified.value]
                value_indexes.append(0)

        # Step 2: invoke the actual encoders

        encoded_columns = self._columns_encoder.transform(
            np.array(column_strings).reshape(-1, 1)
        )
        encoded_operators = self._operator_encoder.transform(
            np.array(operators).reshape(-1, 1)
        )
        encoded_values = {
            col: self._value_encoders[col].encode_batch(np.array(vals).reshape(-1, 1))
            for col, vals in encoder_batches.items()
        }

        # Step 3: stitch the different per-predicate feature vectors back together
        # This is a bit tedious because we need to make sure we keep all indexes locked.
        # Furthermore, we need to accomodate the fact that the same column might have multiple
        # filters. For this case we follow the MSCN strategy of computing the mean of both encodings.

        for i, col in enumerate(columns):
            encoded_col = encoded_columns[i]
            encoded_op = encoded_operators[i]
            value_idx = value_indexes[i]
            encoded_val = encoded_values[col][value_idx]

            existing_vector = vectors.get(col)
            if existing_vector is None:
                vectors[col] = (encoded_col, encoded_op, encoded_val)
                continue

            _, existing_operator, existing_value = existing_vector
            combined_op = np.maximum(existing_operator, encoded_op)
            combined_value = np.concat([existing_value, encoded_val]).mean().reshape(1)
            vectors[col] = (encoded_col, combined_op, combined_value)

        # Step 4: put everything together to form the final filter matrix

        enc = np.stack([np.concat(v) for v in vectors.values()])
        num_pad = self.n_columns - len(vectors)
        return np.pad(enc, {0: (0, num_pad)})

    def build_mask(self, n_set: int, n_max: int) -> np.ndarray:
        """Computes the mask vector to indicate which rows of the feature matrix are set.

        This method assumes that the set rows are always at the top of the matrix.
        """
        present = np.ones(n_set, dtype=np.float32)
        padding = np.zeros(n_max - n_set, dtype=np.float32)
        return np.concatenate([present, padding], axis=0).reshape(-1, 1)

    def store(
        self, catalog_path: Path | str, *, encoder_dir: Optional[Path | str] = None
    ) -> None:
        """Persists the featurization info and column encoders at the specified location.

        To store a featurization, we need to export two kinds of information: general metadata
        (e.g., minimum and maximum cardinality) and the per-column encoders. Encoders will be
        written to HDF files, with one file per table. The general metadata catalog will be exported
        to a JSON file.

        Parameters
        ----------
        catalog_path : Path | str
            The file path to store the featurization catalog. The catalog is a JSON file that
            includes the general metadata about the featurization.
        encoder_dir : Optional[Path | str], optional
            The directory to store the column encoders. If not specified, the encoders will be
            stored in the same directory as the catalog.
        """
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
