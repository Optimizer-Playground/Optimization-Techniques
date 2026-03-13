from __future__ import annotations

from enum import Enum
from typing import Collection, Type
from dataclasses import dataclass, field

from ._context import Context, TableContext, SchemaContext, ColumnContext, SuperTableContext
from ._schema import DatabaseSchema
from postbound import SqlQuery


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
    db_schema: DatabaseSchema | dict
    ctx_cls: Type[Context] = field(init=False)

    def __post_init__(self):
        if isinstance(self.db_schema, dict):
            self.db_schema = DatabaseSchema(self.db_schema)
        self.ctx_cls = self.ctx_granularity.value
        self.ctx2q, self.q2ctx = self._build_query_contexts(self.queries)

    def _build_query_contexts(
        self, queries: Collection[SqlQuery]
    ) -> tuple[dict[Context, set[SqlQuery]], dict[SqlQuery, Context]]:
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
        """Exposes internal builder for classifying test queries."""
        return self._build_query_contexts(queries)

    def __getitem__(self, item) -> Context | set[SqlQuery] | None:
        """
        ContextManager[query] -> Context
        ContextManager[context] -> set[Query]
        """
        if isinstance(item, SqlQuery):
            return self.q2ctx.get(item, None)
        elif isinstance(item, Context):
            return self.ctx2q.get(item, set())
        else:
            raise KeyError(item)

    def get_query_schema_dict(self, query: SqlQuery) -> dict[str, dict[str, str]]:
        return self[query].schema.to_dict()

    def has_context(self, query: SqlQuery) -> bool:
        return self[query] is not None

    def is_empty(self, context: Context) -> bool:
        return not self[context]

    def __repr__(self):
        tables = len(self.db_schema.all_tables())
        columns = len(self.db_schema.all_columns())
        contexts = len(set(self.ctx2q.keys()))
        return f"C: {self.ctx_granularity.name}, Q: {len(self.queries)}, S: T-{tables} C-{columns}, C: {contexts}"
