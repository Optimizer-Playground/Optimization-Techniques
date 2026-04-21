"""Simple randomized query generator."""

from __future__ import annotations

import collections
import itertools
import random
from collections.abc import Generator, Iterable, Mapping, Sequence
from typing import Any, Literal, Optional

import networkx as nx
import postbound as pb

_NullOps = [pb.qal.LogicalOperator.Is, pb.qal.LogicalOperator.IsNot]


def _like_op(op: pb.qal.LogicalOperator) -> bool:
    return op in [
        pb.qal.LogicalOperator.Like,
        pb.qal.LogicalOperator.NotLike,
        pb.qal.LogicalOperator.ILike,
        pb.qal.LogicalOperator.NotILike,
    ]


def _text_col(dtype: str) -> bool:
    return dtype in ["text", "varchar", "char", "character varying"]


class _ColSpec:
    @staticmethod
    def full(
        column: pb.BoundColumnReference,
        *,
        database: pb.Database,
        operators: Optional[Sequence[pb.qal.LogicalOperator]] = None,
    ) -> _ColSpec:
        dtype = database.schema().datatype(column)
        is_text = _text_col(dtype)
        operators = operators or [
            pb.qal.LogicalOperator.Equal,
            pb.qal.LogicalOperator.Greater,
            pb.qal.LogicalOperator.GreaterEqual,
            pb.qal.LogicalOperator.Less,
            pb.qal.LogicalOperator.LessEqual,
            pb.qal.LogicalOperator.Like,
        ]
        if not is_text:
            operators = [op for op in operators if not _like_op(op)]

        return _ColSpec(
            column=column,
            is_text=is_text,
            allowed_ops=operators,
            value_selection="sample",
            values=[],
        )

    def __init__(
        self,
        column: pb.BoundColumnReference,
        is_text: bool,
        allowed_ops: Iterable[pb.qal.LogicalOperator],
        value_selection: Literal["pick", "range", "sample"],
        values: list,
    ):
        self._column = column
        self._is_text = is_text
        self._allowed_ops = list(allowed_ops)
        self._value_selection = value_selection
        self._values = values

    @property
    def column(self) -> pb.BoundColumnReference:
        return self._column

    @property
    def is_text(self) -> bool:
        return self._is_text

    @property
    def allowed_ops(self) -> Sequence[pb.qal.LogicalOperator]:
        return self._allowed_ops

    @property
    def value_selection(self) -> Literal["pick", "range", "sample"]:
        return self._value_selection

    @property
    def values(self) -> list:
        return self._values

    def add_op(self, op: pb.qal.LogicalOperator) -> None:
        if not self.is_text and _like_op(op):
            return
        if op not in self._allowed_ops:
            self._allowed_ops.append(op)


class _TableSpec:
    @staticmethod
    def full(
        table: pb.TableReference,
        *,
        database: pb.Database,
        operators: Optional[Sequence[pb.qal.LogicalOperator]] = None,
    ) -> _TableSpec:
        filter_cols: Mapping[pb.ColumnReference, _ColSpec] = {}
        tab_info = database.schema()[table]
        for col_info in tab_info.columns:
            if col_info.indexed:
                continue
            filter_cols[col_info.column] = _ColSpec.full(
                col_info.column, database=database, operators=operators
            )
        return _TableSpec(table=table, filter_columns=filter_cols)

    @staticmethod
    def empty(table: pb.TableReference) -> _TableSpec:
        return _TableSpec(table=table, filter_columns={})

    def __init__(
        self,
        table: pb.TableReference,
        filter_columns: Mapping[pb.ColumnReference, _ColSpec],
    ):
        self.table = table
        self.filter_columns = dict(filter_columns)

    def columns(self) -> list[pb.ColumnReference]:
        return list(self.filter_columns.keys())


class _PredicateCollector(pb.qal.PredicateVisitor[None]):
    def __init__(self):
        self.column_filters: Mapping[
            pb.BoundColumnReference, set[pb.qal.LogicalOperator]
        ] = collections.defaultdict(set)
        self.joins: set[tuple[pb.BoundColumnReference, pb.BoundColumnReference]] = set()
        self.filter_weights: collections.Counter[pb.BoundColumnReference] = (
            collections.Counter()
        )

    def visit_binary_predicate(self, predicate: pb.qal.BinaryPredicate) -> None:
        if predicate.is_filter():
            self._visit_binary_filter(predicate)
        elif predicate.is_join():
            self._visit_binary_join(predicate)
        else:
            raise RuntimeError()

    def visit_between_predicate(self, predicate: pb.qal.BetweenPredicate) -> None:
        columns = predicate.columns()
        if len(columns) != 1:
            return
        col = pb.util.simplify(columns)
        if not pb.ColumnReference.assert_bound(col):
            return
        col = col.drop_table_alias()

        # We don't include BETWEEN predicates, because our sampling logic can
        # currently only select single values for the filter columns
        self.column_filters[col].update(
            [
                pb.qal.LogicalOperator.LessEqual,
                pb.qal.LogicalOperator.GreaterEqual,
            ]
        )
        self.filter_weights[col] += 1

    def visit_in_predicate(self, predicate: pb.qal.InPredicate) -> None:
        columns = predicate.columns()
        if len(columns) != 1:
            return
        col = pb.util.simplify(columns)
        if not pb.ColumnReference.assert_bound(col):
            return
        col = col.drop_table_alias()

        # Similar to BETWEEN predicates, we don't include the actual IN predicate,
        # because our sampling logic can currently only select single values for the
        # filter columns. Instead, we treat IN predicates as equality predicates on the column.
        self.column_filters[col].add(pb.qal.LogicalOperator.Equal)
        self.filter_weights[col] += 1

    def visit_unary_predicate(self, predicate: pb.qal.UnaryPredicate) -> None:
        return

    def visit_not_predicate(
        self,
        predicate: pb.qal.CompoundPredicate,
        child_predicate: pb.qal.AbstractPredicate,
    ) -> None:
        return child_predicate.accept_visitor(self)

    def visit_or_predicate(
        self,
        predicate: pb.qal.CompoundPredicate,
        components: Sequence[pb.qal.AbstractPredicate],
    ) -> None:
        for component in components:
            component.accept_visitor(self)

    def visit_and_predicate(
        self,
        predicate: pb.qal.CompoundPredicate,
        components: Sequence[pb.qal.AbstractPredicate],
    ) -> None:
        for component in components:
            component.accept_visitor(self)

    def _visit_binary_filter(self, predicate: pb.qal.BinaryPredicate) -> None:
        if not isinstance(predicate.operation, pb.qal.LogicalOperator):
            return
        columns = predicate.columns()
        if len(columns) != 1:
            return
        col = pb.util.simplify(columns)
        if not pb.ColumnReference.assert_bound(col):
            return

        col = col.drop_table_alias()
        self.column_filters[col].add(predicate.operation)
        self.filter_weights[col] += 1

    def _visit_binary_join(self, predicate: pb.qal.BinaryPredicate) -> None:
        columns = predicate.columns()
        if len(columns) != 2:
            return
        col1, col2 = columns
        if (  #
            not pb.ColumnReference.assert_bound(col1)  #
            or not pb.ColumnReference.assert_bound(col2)
        ):
            return
        if col2 < col1:
            col1, col2 = col2, col1
        col1 = col1.drop_table_alias()
        col2 = col2.drop_table_alias()
        self.joins.add((col1, col2))


class _SampleSpec:
    @staticmethod
    def full(
        database: pb.Database,
        *,
        ignore_tables: Optional[set[pb.TableReference]] = None,
        operators: Optional[Sequence[pb.qal.LogicalOperator]] = None,
    ) -> _SampleSpec:
        schema = database.schema()
        ignore_tables = ignore_tables or set()
        tables = {
            tab: _TableSpec.full(tab, database=database, operators=operators)
            for tab in schema.tables()
            if tab not in ignore_tables and not tab.full_name.startswith("pg_")
        }
        columns = pb.util.flatten(spec.columns() for spec in tables.values())
        filter_weights = {col: 1 for col in columns}

        schema_graph = schema.as_graph()
        join_graph = nx.Graph()
        for _, _, data in schema_graph.edges(data=True):
            fkeys: Sequence[pb.db.ForeignKeyRef] = data["foreign_keys"]
            for fkey in fkeys:
                fk_col, ref_col = fkey.fk_col, fkey.referenced_col
                join_graph.add_edge(
                    fk_col.table,
                    ref_col.table,
                    join_columns={fk_col.table: fk_col, ref_col.table: ref_col},
                )

        min_tables = 1
        max_tables = len(tables)
        min_filters = 0
        max_filters = len(columns)

        return _SampleSpec(
            tables=tables,
            filter_weights=filter_weights,
            joins=join_graph,
            min_tables=min_tables,
            max_tables=max_tables,
            min_filters=min_filters,
            max_filters=max_filters,
        )

    @staticmethod
    def derive_from_workload(
        workload: pb.Workload, *, database: pb.Database
    ) -> _SampleSpec:
        predicate_collector = _PredicateCollector()
        tables: set[pb.TableReference] = set()
        min_tables, max_tables = float("inf"), 0
        min_filters, max_filters = float("inf"), 0
        for query in workload.queries():
            predicate_collector.visit_query_predicates(query)
            tables.update(tab.drop_alias() for tab in query.tables())

            min_tables = min(min_tables, len(query.tables()))
            max_tables = max(max_tables, len(query.tables()))
            n_filters = len(query.filters())
            min_filters = min(min_filters, n_filters)
            max_filters = max(max_filters, n_filters)

        join_graph = nx.Graph()
        for col1, col2 in predicate_collector.joins:
            join_graph.add_edge(
                col1.table,
                col2.table,
                join_columns={col1.table: col1, col2.table: col2},
            )

        col_specs: dict[pb.TableReference, set[_ColSpec]] = collections.defaultdict(set)
        for column, ops in predicate_collector.column_filters.items():
            tab = column.table
            dtype = database.schema().datatype(column)
            col_specs[tab].add(_ColSpec(column, _text_col(dtype), ops, "sample", []))

        table_specs: dict[pb.TableReference, _TableSpec] = {}
        for tab in tables:
            cols = col_specs.get(tab, set())
            if not cols:
                table_specs[tab] = _TableSpec.empty(tab)
                continue

            filter_columns = {col.column: col for col in cols}
            table_specs[tab] = _TableSpec(table=tab, filter_columns=filter_columns)

        return _SampleSpec(
            tables=table_specs,
            filter_weights=predicate_collector.filter_weights,
            joins=join_graph,
            min_tables=min_tables,
            max_tables=max_tables,
            min_filters=min_filters,
            max_filters=max_filters,
        )

    def __init__(
        self,
        tables: Mapping[pb.TableReference, _TableSpec],
        filter_weights: Mapping[pb.ColumnReference, int],
        joins: nx.Graph,
        min_tables: int,
        max_tables: int,
        min_filters: int,
        max_filters: int,
    ) -> None:
        self.tables = dict(tables)
        self.filter_weights = dict(filter_weights)
        self.joins = joins
        self.min_tables = min_tables
        self.max_tables = max_tables
        self.min_filters = min_filters
        self.max_filters = max_filters

    @property
    def n_tables(self) -> int:
        return len(self.tables)

    @property
    def n_filter_columns(self) -> int:
        return sum(len(tab_spec.filter_columns) for tab_spec in self.tables.values())

    def filterable_cols_on(
        self, tables: Sequence[pb.TableReference]
    ) -> Sequence[pb.ColumnReference]:
        return pb.util.flatten(self.tables[tab].filter_columns.keys() for tab in tables)

    def __getitem__(self, key: pb.ColumnReference) -> _ColSpec:
        if not pb.ColumnReference.assert_bound(key):
            raise KeyError(f"Column reference {key} is not bound.")
        table_spec = self.tables.get(key.table)
        if not table_spec:
            raise KeyError(f"Table {key.table} not found in sample spec.")
        col_spec = table_spec.filter_columns.get(key)
        if not col_spec:
            raise KeyError(f"Column {key} is not a filter column.")
        return col_spec


def _draw_tables(spec: _SampleSpec, n: int) -> Sequence[pb.TableReference]:
    tables: list[pb.TableReference] = []
    for tab in pb.util.nx.nx_random_walk(spec.joins):
        tables.append(tab)
        if len(tables) >= n:
            break
    return [] if len(tables) < n else tables


def _draw_filter_cols(
    tables: Sequence[pb.TableReference],
    n: int,
    *,
    spec: _SampleSpec,
) -> Sequence[pb.ColumnReference]:
    cols = spec.filterable_cols_on(tables)
    if not cols:
        return []

    total_weight = 0
    abs_weights: dict[pb.ColumnReference, float] = {}
    for col in cols:
        weight = spec.filter_weights.get(col, 1)
        abs_weights[col] = weight
        total_weight += weight

    rel_weights = {col: weight / total_weight for col, weight in abs_weights.items()}
    return random.choices(cols, k=n, weights=list(rel_weights.values()))


def _draw_filter_value(
    col: _ColSpec,
    operator: pb.qal.LogicalOperator,
    *,
    database: pb.Database,
    retries: int = 3,
) -> tuple[bool, Any]:
    col_name, tab_name = col.column.name, col.column.table.full_name

    # Postgres and DuckDB provide specialized sampling facilities. Use them if possible.
    if isinstance(database, pb.postgres.PostgresInterface):
        # even though the BERNOUILLI sampling method would be "more uniform", we opt for SYSTEM sampling due to its much
        # lower execution time.
        query_template = (
            f"SELECT DISTINCT {col_name} FROM {tab_name} TABLESAMPLE SYSTEM(1)"
        )
    elif isinstance(database, pb.duckdb.DuckDBInterface):
        query_template = (
            f"SELECT DISTINCT {col_name} FROM {tab_name} USING SAMPLE 1 ROWS"
        )
    else:
        query_template = f"SELECT DISTINCT {col_name} FROM {tab_name}"

    result_set = database.execute_query(query_template, cache_enabled=False, raw=True)
    assert result_set is not None
    if not result_set and retries == 0:
        return (False, None)
    elif not result_set:
        return _draw_filter_value(col, operator, database=database, retries=retries - 1)

    candidate_values = [row[0] for row in result_set]
    value = random.choice(candidate_values)

    if _like_op(operator) and col.is_text and value is not None:
        value = f"%{value}%"

    return (True, value)


def _draw_filter_pred(
    col: _ColSpec, *, database: pb.Database
) -> Optional[pb.qal.AbstractPredicate]:
    operator = random.choice(col.allowed_ops)

    # LIKE operators can only be applied to text columns.
    while not col.is_text and _like_op(operator):
        operator = random.choice(col.allowed_ops)

    match col.value_selection:
        case "pick":
            value = random.choice(col.values)
            success = True
        case "range":
            value = random.uniform(min(col.values), max(col.values))
            success = True
        case "sample":
            success, value = _draw_filter_value(col, operator, database=database)

    if not success:
        return None

    # If we have sampled a NULL value, we cannot use an arbitrary operator in our predicate.
    # Instead, we need to select one of the NULL-specific operators (IS or IS NOT).
    if value is None and operator not in _NullOps:
        allowed_null_ops = [op for op in col.allowed_ops if op in _NullOps]
        if not allowed_null_ops:
            return None
        operator = random.choice(allowed_null_ops)

    # Conversely, if we have sampled a non-NULL value, we cannot use NULL-specific operators.
    while value is not None and operator in _NullOps:
        operator = random.choice(col.allowed_ops)

    return pb.qal.as_predicate(col.column, operator, value)


def _generate_join_predicates(
    tables: Sequence[pb.TableReference], *, schema: nx.Graph
) -> Sequence[pb.qal.AbstractPredicate]:
    joins: list[pb.qal.AbstractPredicate] = []
    for tab1, tab2 in itertools.combinations(tables, 2):
        if not schema.has_edge(tab1, tab2):
            continue
        join_info = schema.get_edge_data(tab1, tab2)
        col1 = join_info["join_columns"][tab1]
        col2 = join_info["join_columns"][tab2]
        joins.append(pb.qal.as_predicate(col1, "=", col2))
    return joins


def generate_query(
    target_db: Optional[pb.Database],
    *,
    similar_to: Optional[pb.Workload] = None,
    ignore_tables: Optional[set[pb.TableReference]] = None,
    min_tables: Optional[int] = None,
    max_tables: Optional[int] = None,
    min_filters: Optional[int] = None,
    max_filters: Optional[int] = None,
    count_star: bool = False,
) -> Generator[pb.SqlQuery, None, None]:
    """A simple randomized query generator.

    The sampler operates in one of two modes: If a workload is provided via the `similar_to`, the generated queries are
    made with similar filters as the queries in the workload. Otherwise, all columns are equally likely to be filtered.
    Note that the second case introduces a bias towards tables that have more columns.

    The generator will yield new queries until the user stops requesting them, there is no termination condition.

    Parameters
    ----------
    target_db : Optional[Database]
        The database from which queries should be generated.
        If no workload is provided, the available joins are inferred based on the primary key/foreign key relationships in the
        schema. Likewise, the sampler will use all non-key columns as potential filter columns.
        If the database is not specified, it is inferred from PostBOUND's database pool.
    similar_to : Optional[Workload], optional
        If provided, the generated queries will be made to be similar to the workload queries. Specifically, this means the
        following:
        - Only joins that have been observed in the workload are used
        - Only columns that have been used as filter columns are used as filter columns for the generated queries
        - The probability of a column being used as a filter column is proportional to the number of times it has been used
          as a filter column in the workload queries
    ignore_tables : Optional[set[TableReference]], optional
        An optional set of tables that should never be contained in the generated queries. For Postgres databases, internal
        *pg_XXX* tables are ignored automatically. This parameter is only used if no workload is provided.
    min_tables : Optional[int], optional
        The minimum number of tables that should be contained in each query. Default is 1.
        If a sample workload is provided and this parameter is not set, it is inferred from the samples.
    max_tables : Optional[int], optional
        The maximum number of tables that should be contained in each query. Default is the number of tables in the schema
        graph (minus the ignored tables).
        If a sample workload is provided and this parameter is not set, it is inferred from the samples.
    min_filters : Optional[int], optional
        The minimum number of filter predicates that should be contained in each query. Default is 0.
        If a sample workload is provided and this parameter is not set, it is inferred from the samples.
    max_filters : Optional[int], optional
        The maximum number of filter predicates that should be contained in each query. By default, each column from the
        selected tables can be filtered.
        If a sample workload is provided and this parameter is not set, it is inferred from the samples.
    count_star : bool, optional
        Whether the resulting queries should contain a *COUNT(\\*)* instead of a plain *SELECT \\** clause

    Yields
    ------
    Generator[SqlQuery, None, None]
        A random SQL query

    Examples
    --------
    >>> qgen = generate_query(some_database)
    >>> queries = [next(qgen) for _ in range(5)]
    """

    target_db = target_db or pb.db.current_database()
    spec = (
        _SampleSpec.derive_from_workload(similar_to, database=target_db)
        if similar_to
        else _SampleSpec.full(target_db, ignore_tables=ignore_tables)
    )

    min_tables = min_tables or spec.min_tables
    max_tables = max_tables or spec.max_tables
    min_filters = min_filters or spec.min_filters
    max_filters = max_filters or spec.max_filters

    select_clause = pb.qal.Select.count_star() if count_star else pb.qal.Select.star()

    while True:
        n_tables = random.randint(min_tables, max_tables)
        tables = _draw_tables(spec, n_tables)
        if not tables:
            continue

        max_available_filters = min(max_filters, len(spec.filterable_cols_on(tables)))
        if min_filters < max_available_filters:
            n_filters = random.randint(min_filters, max_available_filters)
        elif min_filters == max_available_filters:
            n_filters = min_filters
        else:
            # The selected tables don't have enough filterable columns to satisfy the minimum filter requirement.
            # We need to redraw tables.
            continue

        filter_cols = _draw_filter_cols(tables, n_filters, spec=spec)
        if n_filters != len(filter_cols):
            continue

        filter_preds = [
            _draw_filter_pred(spec[col], database=target_db) for col in filter_cols
        ]
        if any(pred is None for pred in filter_preds):
            continue

        predicates = []
        if n_tables > 1:
            predicates.extend(_generate_join_predicates(tables, schema=spec.joins))
        if filter_preds:
            predicates.extend(filter_preds)

        if predicates:
            where_clause = pb.qal.Where(pb.qal.CompoundPredicate.create_and(predicates))
        else:
            where_clause = None

        from_clause = pb.qal.ImplicitFromClause.create_for(tables)
        query = pb.qal.ImplicitSqlQuery(
            select_clause=select_clause,
            from_clause=from_clause,
            where_clause=where_clause,
        )
        yield query
