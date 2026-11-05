from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass, field
from enum import Enum
from typing import overload

from postbound import SqlQuery, qal

from ._context import (
    ColumnContext,
    Context,
    SchemaContext,
    SuperTableContext,
    TableContext,
)
from ._schema import DatabaseSchema


class CtxGranularity(Enum):
    SCHEMA = SchemaContext
    SUPERTABLE = SuperTableContext
    TABLE = TableContext
    COLUMN = ColumnContext

    @classmethod
    def ordered(cls):
        return [cls.SCHEMA, cls.SUPERTABLE, cls.TABLE, cls.COLUMN]

    def upwards(self) -> CtxGranularity:
        ordered = self.ordered()
        idx = ordered.index(self)
        return ordered[idx - 1] if idx > 0 else self

    def downwards(self) -> CtxGranularity:
        ordered = self.ordered()
        idx = ordered.index(self)
        return ordered[idx + 1] if idx < len(ordered) - 1 else self


@dataclass
class ContextManager:

    ctx_granularity: CtxGranularity
    queries: Collection[SqlQuery]
    db_schema: DatabaseSchema | dict[str, dict[str, str]]
    fixed_contexts: bool = field(default=False, kw_only=True)
    ctx_cls: type[Context] = field(init=False)

    def __post_init__(self):
        if isinstance(self.db_schema, dict):
            self.db_schema = DatabaseSchema(self.db_schema)
        self.ctx_cls = self.ctx_granularity.value
        self.ctx2q, self.q2ctx = self._build_query_contexts(self.queries)

    def _build_query_contexts(
        self, queries: Collection[SqlQuery]
    ) -> tuple[dict[Context, set[SqlQuery]], dict[SqlQuery, Context]]:
        assert isinstance(self.db_schema, DatabaseSchema)
        ctx2q: dict[Context, set[SqlQuery]] = dict()
        q2ctx: dict[SqlQuery, Context] = dict()
        if self.ctx_granularity == CtxGranularity.SUPERTABLE:
            ctx = SuperTableContext.from_queries(queries, self.db_schema)
            for q in queries:
                q2ctx[q] = ctx
            ctx2q[ctx] = set(queries)
        else:
            for query in queries:
                ctx = self.ctx_cls.from_query(query, self.db_schema)
                q2ctx[query] = ctx
                ctx2q.setdefault(ctx, set()).add(query)
        return ctx2q, q2ctx

    def classify_queries(self, queries: Collection[SqlQuery]) -> tuple[dict[Context, set[SqlQuery]], dict[SqlQuery, Context]]:
        """Group queries, optionally retaining the contexts established at preparation.

        The default preserves the component API's workload-specific contexts for
        generalization experiments. The public optimizer opts into fixed contexts.
        """
        if not self.fixed_contexts:
            return self._build_query_contexts(queries)
        groups: dict[Context, set[SqlQuery]] = {}
        assignments: dict[SqlQuery, Context] = {}
        for query in queries:
            context = self.resolve(query)
            if context is None:
                raise ValueError(f"Query is outside the prepared FastGRES contexts: {query}")
            groups.setdefault(context, set()).add(query)
            assignments[query] = context
        return groups, assignments

    @classmethod
    def from_contexts(cls, granularity: CtxGranularity, contexts: Collection[Context],
                      schema: DatabaseSchema) -> ContextManager:
        """Restore a fixed context registry without requiring the original queries."""
        if not contexts:
            raise ValueError("A prepared context registry must not be empty")
        for context in contexts:
            if not isinstance(context, granularity.value):
                raise ValueError("Context type does not match the context granularity")
            if any(schema.get_datatype(t, c) != dtype for t, c, dtype in context.schema):
                raise ValueError("Context schema does not match the prepared database schema")
        if granularity in (CtxGranularity.SCHEMA, CtxGranularity.SUPERTABLE) and len(contexts) != 1:
            raise ValueError("Schema and supertable modes require exactly one context")
        manager = cls(granularity, [], schema, fixed_contexts=True)
        manager.ctx2q = {context: set() for context in contexts}
        return manager

    def resolve(self, query: SqlQuery) -> Context | None:
        """Match query requirements to a prepared context, independent of SQL literals.

        This registry uses unqualified physical table names, just like the feature
        components. Qualified or virtual relations cannot safely be matched here.
        """
        if not isinstance(query, qal.SelectStatement):
            raise ValueError("FastGRES only supports SELECT statements")
        tables = query.tables()
        if query.subqueries() or any(t.virtual or t.schema or t.catalog for t in tables):
            return None
        columns = query.columns()
        if any(c.table is None for c in columns):
            raise ValueError("FastGRES requires bound columns; parse the query with database column binding")
        table_names = {t.full_name for t in tables}
        required_columns = {(c.table.full_name, c.name) for c in columns if c.table is not None}
        for context in self.ctx2q:
            available_tables = context.schema.all_tables()
            available_columns = context.schema.all_columns()
            if not table_names <= available_tables or not required_columns <= available_columns:
                continue
            if self.ctx_granularity == CtxGranularity.TABLE and table_names != available_tables:
                continue
            if self.ctx_granularity == CtxGranularity.COLUMN and required_columns != available_columns:
                continue
            return context
        return None

    @overload
    def __getitem__(self, item: SqlQuery) -> Context | None: ...

    @overload
    def __getitem__(self, item: Context) -> set[SqlQuery]: ...

    def __getitem__(self, item: SqlQuery | Context) -> Context | set[SqlQuery] | None:
        """
        ContextManager[query] -> Context
        ContextManager[context] -> set[Query]
        """
        if isinstance(item, SqlQuery):
            return self.resolve(item) if self.fixed_contexts else self.q2ctx.get(item, None)
        elif isinstance(item, Context):
            return self.ctx2q.get(item, set())
        else:
            raise KeyError(item)

    def get_query_schema_dict(self, query: SqlQuery) -> dict[str, dict[str, str]]:
        context = self[query]
        if context is None:
            raise ValueError(f"No context registered for query: {query}")
        return context.schema.to_dict()

    def has_context(self, query: SqlQuery) -> bool:
        return self[query] is not None

    def is_empty(self, context: Context) -> bool:
        return not self[context]

    def __repr__(self):
        assert isinstance(self.db_schema, DatabaseSchema)
        tables = len(self.db_schema.all_tables())
        columns = len(self.db_schema.all_columns())
        contexts = len(set(self.ctx2q.keys()))
        return f"C: {self.ctx_granularity.name}, Q: {len(self.queries)}, S: T-{tables} C-{columns}, C: {contexts}"
