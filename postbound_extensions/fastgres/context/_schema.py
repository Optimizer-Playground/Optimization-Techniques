
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Set, Iterator, Tuple, Optional, TYPE_CHECKING
import postbound as pb

if TYPE_CHECKING:
    from ..featurization import DatabaseConnection

@dataclass(frozen=True)
class DatabaseSchema:
    """Table -> Column -> Datatype"""
    _schema: Dict[str, Dict[str, str]]

    @classmethod
    def from_dbc(cls, dbc: "pb.postgres.PostgresInterface | DatabaseConnection") -> DatabaseSchema:
        from ..featurization import DatabaseConnection
        if isinstance(dbc, pb.postgres.PostgresInterface):
            dbc = DatabaseConnection(dbc)
        return cls(dbc.fg_schema)

    @property
    def schema(self) -> Dict[str, Dict[str, str]]:
        return self._schema

    def all_tables(self) -> Set[str]:
        return set(self._schema.keys())

    def all_columns(self) -> Set[tuple[str, str]]:
        return {(t, c) for t, cols in self._schema.items() for c in cols}

    def columns_for_table(self, table: str) -> Set[str]:
        return set(self._schema.get(table, {}).keys())

    def get_datatype(self, table: str, column: str) -> Optional[str]:
        return self._schema.get(table, {}).get(column, None)

    def __iter__(self) -> Iterator[Tuple[str, str, str]]:
        for table in sorted(self._schema):
            for column in sorted(self._schema[table]):
                yield table, column, self._schema[table][column]

    def __getitem__(self, item):
        return self._schema[item]

    def intersection(self, other: DatabaseSchema) -> DatabaseSchema:
        result = {}
        for table, columns in self._schema.items():
            if table in other._schema:
                common_columns = columns.keys() & other._schema[table].keys()
                if common_columns:
                    result[table] = {col: columns[col] for col in common_columns}
        return DatabaseSchema(result)

    def union(self, other: DatabaseSchema) -> DatabaseSchema:
        result = {}
        # Combine both schemas for each table
        for table in set(self._schema.keys()).union(other._schema.keys()):
            result[table] = {}
            if table in self._schema:
                result[table].update(self._schema[table])
            if table in other._schema:
                result[table].update(other._schema[table])
        return DatabaseSchema(result)

    def difference(self, other: DatabaseSchema) -> DatabaseSchema:
        result = {}
        for table, columns in self._schema.items():
            if table not in other._schema:
                result[table] = columns
            else:
                diff_columns = columns.keys() - other._schema[table].keys()  # difference in column names
                if diff_columns:
                    result[table] = {col: columns[col] for col in diff_columns}
        return DatabaseSchema(result)

    def __add__(self, other: DatabaseSchema) -> DatabaseSchema:
        return self.union(other)

    def __and__(self, other: DatabaseSchema) -> DatabaseSchema:
        return self.intersection(other)

    def __or__(self, other: DatabaseSchema) -> DatabaseSchema:
        return self.union(other)

    def __sub__(self, other: DatabaseSchema) -> DatabaseSchema:
        return self.difference(other)

    def __truediv__(self, other: DatabaseSchema) -> DatabaseSchema:
        return self.intersection(other)

    def __mul__(self, other: DatabaseSchema) -> DatabaseSchema:
        """Symmetric difference"""
        result = {}
        all_tables = set(self._schema.keys()) | set(other._schema.keys())
        for table in all_tables:
            columns1 = set(self._schema.get(table, {}))
            columns2 = set(other._schema.get(table, {}))
            sym_diff = columns1 ^ columns2
            if sym_diff:
                combined = {}
                for col in sym_diff:
                    if col in self._schema.get(table, {}):
                        combined[col] = self._schema[table][col]
                    else:
                        combined[col] = other._schema[table][col]
                result[table] = combined
        return DatabaseSchema(result)

    def __repr__(self):
        return str(self._schema)

    def __hash__(self):
        return hash(tuple(self))

    def to_dict(self) -> dict:
        return self._schema
