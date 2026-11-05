
from collections.abc import Iterable
from datetime import date
from decimal import Decimal

import postbound as pb
from postbound import ColumnReference, TableReference
from postbound.qal import BinaryOperator
from psycopg import sql


class DatabaseConnection:
    def __init__(self, pbc: pb.postgres.PostgresDatabase):
        self.pbc = pbc
        self._schema: pb.postgres.PostgresSchema | None = None

    @property
    def fg_schema(self) -> dict[str, dict[str, str]]:
        schema_dict = dict()
        for table_ref in self.schema.tables():
            schema_dict[table_ref.full_name] = dict()
            for column_ref in self.schema.columns(table_ref):
                d_type = self.schema.datatype(column_ref)
                schema_dict[table_ref.full_name][column_ref.name] = d_type
        return schema_dict

    @property
    def schema(self) -> pb.postgres.PostgresSchema:
        if self._schema is None:
            self._schema = self.pbc.schema()
        return self._schema

    @property
    def tables(self) -> set[TableReference]:
        return self.schema.tables()

    def table_str(self) -> set[str]:
        return {t_ref.full_name for t_ref in self.tables}

    def col_and_types_str(self, table: TableReference) -> Iterable[tuple[ColumnReference, str]]:
        for col_ref in self.schema.columns(table):
            d_type = self.schema.datatype(col_ref)
            yield col_ref, d_type

    @staticmethod
    def _table_identifier(table: TableReference) -> sql.Identifier:
        return sql.Identifier(table.schema, table.full_name) if table.schema else sql.Identifier(table.full_name)

    def min_max(self, column: ColumnReference) -> tuple[float | Decimal | date | None, float | Decimal | date | None]:
        if column.table is None:
            raise ValueError(f"Cannot retrieve statistics for an unbound column: {column}")
        query = sql.SQL("SELECT MIN({column}), MAX({column}) FROM {table}").format(
            column=sql.Identifier(column.name), table=self._table_identifier(column.table)
        )
        cursor = self.pbc.cursor()
        cursor.execute(query)
        row = cursor.fetchone()
        if row is None:
            raise ValueError(f"MIN/MAX returned no row for {column}")
        return row[0], row[1]

    def column_count(self, column: ColumnReference) -> list[tuple[str | None, int]]:
        if column.table is None:
            raise ValueError(f"Cannot retrieve statistics for an unbound column: {column}")
        query = sql.SQL("SELECT {column}, COUNT({column}) FROM {table} GROUP BY {column}").format(
            column=sql.Identifier(column.name), table=self._table_identifier(column.table)
        )
        cursor = self.pbc.cursor().execute(query)
        return [(row[0], row[1]) for row in cursor.fetchall()]

    def cardinality(self, table: TableReference) -> int:
        query = sql.SQL("SELECT COUNT(*) FROM {}").format(self._table_identifier(table))
        cursor = self.pbc.cursor().execute(query)
        row = cursor.fetchone()
        if row is None:
            raise ValueError(f"COUNT returned no row for {table}")
        return int(row[0])

    def like_cardinality(self, column: ColumnReference, like_predicate: pb.qal.BinaryPredicate) -> int:
        valid_ops = {BinaryOperator.Like, BinaryOperator.ILike, BinaryOperator.NotLike, BinaryOperator.NotILike}
        op = like_predicate.operator
        if op not in valid_ops:
            raise ValueError(f"like_predicate: {like_predicate} is not a like predicate with operator: {op}")
        if column.table is None:
            raise ValueError(f"Cannot count a LIKE predicate on an unbound column: {column}")
        pb_query = pb.qal.SelectStatement(
            select_clause=pb.qal.Select.count_star(),
            from_clause=pb.qal.From.create_for(column.table),
            where_clause=pb.qal.Where(like_predicate)
        )
        return self.pbc.execute_query(pb_query)

    def filter_by_types(self, types: set[str]) -> Iterable[ColumnReference]:
        for table_ref in self.schema.tables():
            for column_ref in self.schema.columns(table_ref):
                d_type = self.schema.datatype(column_ref)
                if d_type in types:
                    yield column_ref
