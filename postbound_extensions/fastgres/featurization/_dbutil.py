
import postbound as pb
from postbound import ColumnReference, TableReference
from typing import Iterable
import textwrap
from datetime import datetime
from postbound.qal import LogicalOperator


class DatabaseConnection:
    def __init__(self, pbc: pb.postgres.PostgresInterface):
        self.pbc = pbc
        self._schema = None

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
    def schema(self) -> pb.postgres.PostgresSchemaInterface:
        if not self._schema:
            self._schema = self.pbc.schema()
        return self._schema

    @property
    def tables(self) -> set[TableReference]:
        return self._schema.tables()

    def table_str(self) -> set[str]:
        return {t_ref.full_name for t_ref in self.tables}

    def col_and_types_str(self, table: TableReference) -> Iterable[tuple[ColumnReference, str]]:
        for col_ref in self.schema.columns(table):
            d_type = self.schema.datatype(col_ref)
            yield col_ref, d_type

    def min_max(self, column: ColumnReference) -> tuple[int | float | datetime, int | float | datetime]:
        c_n = column.name
        query = textwrap.dedent(f"""
                        SELECT MIN({c_n}), MAX({c_n}) 
                        FROM {column.table.full_name}
                        """)
        self.pbc.cursor().execute(query)
        return self.pbc.cursor().fetchone()

    def column_count(self, column: ColumnReference) -> list[tuple[str, int]]:
        c_n = column.name
        query = textwrap.dedent(f"""
                SELECT {c_n}, COUNT({c_n}) 
                FROM {column.table.full_name}
                GROUP BY {c_n}
                """)
        cursor = self.pbc.cursor().execute(query)
        return cursor.fetchall()

    def cardinality(self, table: TableReference) -> int:
        query = textwrap.dedent(f"""
                        SELECT COUNT(*) 
                        FROM {table.full_name}
                        """)
        cursor = self.pbc.cursor().execute(query)
        return cursor.fetchone()[0]

    def like_cardinality(self, column: ColumnReference, like_predicate: pb.qal.BinaryPredicate) -> int:
        valid_ops = {LogicalOperator.Like, LogicalOperator.ILike, LogicalOperator.NotLike, LogicalOperator.NotILike}
        op = like_predicate.operation
        if op not in valid_ops:
            raise ValueError(f"like_predicate: {like_predicate} is not a like predicate with operator: {op}")
        pb_query = pb.qal.ImplicitSqlQuery(
            select_clause=pb.qal.Select.count_star(),
            from_clause=pb.qal.ImplicitFromClause.create_for(column.table),
            where_clause=pb.qal.Where(like_predicate)
        )
        return self.pbc.execute_query(pb_query)

    def filter_by_types(self, types: set[str]) -> Iterable[ColumnReference]:
        for table_ref in self.schema.tables():
            for column_ref in self.schema.columns(table_ref):
                d_type = self.schema.datatype(column_ref)
                if d_type in types:
                    yield column_ref
