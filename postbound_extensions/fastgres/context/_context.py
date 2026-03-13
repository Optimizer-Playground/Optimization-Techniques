from __future__ import annotations
from dataclasses import dataclass
from postbound import SqlQuery
from ._schema import DatabaseSchema
from abc import ABC, abstractmethod
from typing import Collection


class Context(ABC):

    @property
    @abstractmethod
    def schema(self) -> DatabaseSchema:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_query(cls, query: SqlQuery, schema: DatabaseSchema) -> Context:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(schema={self.schema})"

    @property
    @abstractmethod
    def type(self) -> str:
        raise NotImplementedError


class ContextFactory:

    @classmethod
    def from_type(cls, ctx_type: str, schema: DatabaseSchema) -> Context:
        match ctx_type:
            case "SchemaContext":
                return SchemaContext(schema)
            case "SuperTableContext":
                return SuperTableContext(schema)
            case "TableContext":
                return TableContext(schema)
            case "ColumnContext":
                return ColumnContext(schema)
            case _:
                raise ValueError(f"Unsupported type {ctx_type}")


@dataclass(frozen=True)
class SchemaContext(Context):
    _schema: DatabaseSchema

    @property
    def schema(self) -> DatabaseSchema:
        return self._schema

    @classmethod
    def from_query(cls, query: SqlQuery, schema: DatabaseSchema) -> SchemaContext:
        return cls(schema)

    @classmethod
    def from_queries(cls, queries: Collection[SqlQuery], schema) -> SchemaContext:
        return cls(schema)

    @property
    def type(self) -> str:
        return "SchemaContext"


@dataclass(frozen=True)
class SuperTableContext(Context):
    _schema: DatabaseSchema

    @property
    def schema(self) -> DatabaseSchema:
        return self._schema

    @classmethod
    def from_query(cls, query: SqlQuery, schema: DatabaseSchema) -> SuperTableContext:
        raise ValueError("SuperTableContext cannot be instantiated from individual queries. Use from_queries instead.")

    @classmethod
    def from_queries(cls, queries: Collection[SqlQuery], schema: DatabaseSchema) -> SuperTableContext:
        merged_schema = DatabaseSchema(dict())
        for query in queries:
            merged_schema += TableContext.from_query(query, schema).schema
        return cls(merged_schema)

    @property
    def type(self) -> str:
        return "SuperTableContext"


@dataclass(frozen=True)
class TableContext(Context):
    _schema: DatabaseSchema

    @property
    def schema(self) -> DatabaseSchema:
        return self._schema

    @classmethod
    def from_query(cls, query: SqlQuery, schema: DatabaseSchema) -> TableContext:
        query_schema = {
            table_reference.full_name: schema[table_reference.full_name]
            for table_reference in query.tables()
        }
        return cls(schema / DatabaseSchema(query_schema))

    @property
    def type(self) -> str:
        return "TableContext"


@dataclass(frozen=True)
class ColumnContext(Context):
    _schema: DatabaseSchema

    @property
    def schema(self) -> DatabaseSchema:
        return self._schema

    @classmethod
    def from_query(cls, query: SqlQuery, schema: DatabaseSchema) -> ColumnContext:
        query_schema = dict()
        for table_reference in query.tables():
            column_references = query.columns_of(table_reference)
            for column_reference in column_references:
                table = table_reference.full_name
                column = column_reference.name
                d_type = schema.get_datatype(table, column)
                query_schema.setdefault(table, dict())[column] = d_type
        return cls(DatabaseSchema(query_schema) / schema)

    @property
    def type(self) -> str:
        return "ColumnContext"
