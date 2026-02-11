""""""

from __future__ import annotations

import collections
from typing import Literal, Optional

import postbound as pb

Intermediate = frozenset[pb.TableReference]
UpperBound = pb.Cardinality
JoinKey = tuple[pb.ColumnReference, pb.ColumnReference]
JoinPartners = set[JoinKey]


class UESJoinOrdering(pb.JoinOrderOptimization):
    def __init__(
        self,
        *,
        database: Optional[pb.Database] = None,
        estimations: Literal["native", "precise"] = "native",
    ) -> None:
        self._database = database or pb.db.current_database()

        emulate_stats = estimations == "precise"
        self._stats = self._database.statistics()
        self._stats.emulated = emulate_stats

    def optimize_join_order(self, query: pb.SqlQuery) -> pb.JoinTree:
        join_tree = pb.JoinTree()
        expanding_tables, filtering_tables = self._determine_table_types(query)
        upper: dict[Intermediate, UpperBound] = {}
        max_freqs = self._init_max_freqs(query)

        while expanding_tables:
            self._update_upper(
                upper,
                expanding_tables=expanding_tables,
                filter_candidates=filtering_tables,
                max_freqs=max_freqs,
                query=query,
            )

            if not join_tree:
                best_initial = pb.util.argmin(upper)
                join_tree = pb.JoinTree(base_table=pb.util.simplify(best_initial))
                continue

            best_bound = UpperBound.infinite()
            best_candidate: Optional[pb.TableReference] = None
            best_col: Optional[pb.ColumnReference] = None
            best_partner: Optional[pb.ColumnReference] = None
            candidate_joins = self._join_partners(
                expanding_tables, bound_tables=join_tree.tables(), query=query
            )

            for candidate, join_partners in candidate_joins.items():
                join_bounds = {
                    (free_col, bound_col): self._upper_bound(
                        free_col, partner=bound_col, upper=upper, max_freqs=max_freqs
                    )
                    for free_col, bound_col in join_partners
                }
                free_col, bound_col = pb.util.argmin(join_bounds)
                current_bound = join_bounds[(free_col, bound_col)]
                if current_bound < best_bound:
                    best_bound = current_bound
                    best_candidate = candidate
                    best_col = free_col
                    best_partner = bound_col

            assert (
                best_candidate is not None
                and best_col is not None
                and best_partner is not None
            )
            available_pks = self._available_pk_tables(
                best_candidate,
                intermediate=join_tree.tables(),
                candidates=filtering_tables,
                query=query,
            )
            bound_cols = {col for col in max_freqs if col.table in join_tree.tables()}
            if best_bound < self._filter_card(best_candidate, query=query):
                pk_fk_tree = pb.JoinTree(base_table=best_candidate)
                dangling_pks: list[pb.TableReference] = []
                for pk_table in available_pks:
                    if not query.joins_between(pk_fk_tree.tables(), pk_table):
                        dangling_pks.append(pk_table)
                        continue
                    pk_fk_tree = pk_fk_tree.join_with(pk_table)
                join_tree = join_tree.join_with(pk_fk_tree)
                for dangling_pk in dangling_pks:
                    join_tree = join_tree.join_with(dangling_pk)
            else:
                join_tree = join_tree.join_with(best_candidate)
                for pk_table in available_pks:
                    join_tree = join_tree.join_with(pk_table)

            upper[frozenset(join_tree.tables())] = best_bound
            for table in join_tree.tables():
                upper[frozenset([table])] = best_bound

            free_freq = max_freqs[best_col]
            bound_freq = max_freqs[best_partner]
            new_tables = {best_candidate} | set(available_pks)
            new_cols = [
                free_col
                for free_col in max_freqs.keys()
                if free_col.table in new_tables
            ]
            for bound_col in bound_cols:
                max_freqs[bound_col] *= free_freq
            for free_col in new_cols:
                max_freqs[free_col] *= bound_freq

            expanding_tables.remove(best_candidate)
            filtering_tables.difference_update(available_pks)

        return join_tree

    def describe(self) -> pb.util.jsondict:
        return {"name": "UES", "type": "original"}

    def _init_max_freqs(
        self, query: pb.SqlQuery
    ) -> dict[pb.ColumnReference, pb.Cardinality]:
        max_freqs: dict[pb.ColumnReference, pb.Cardinality] = {}
        join_cols = pb.util.set_union(
            join_pred.columns() for join_pred in query.joins()
        )
        for col in join_cols:
            max_freqs[col] = self._max_freq(col)
        return max_freqs

    def _upper_bound(
        self,
        free_col: pb.ColumnReference,
        *,
        partner: pb.ColumnReference,
        upper: dict[Intermediate, UpperBound],
        max_freqs: dict[pb.ColumnReference, pb.Cardinality],
    ) -> UpperBound:
        assert free_col.table, "Unbound candidate"
        assert partner.table, "Unbound partner"
        partner_bound = upper[frozenset([partner.table])]
        candidate_bound = upper[frozenset([free_col.table])]
        partner_freq = max_freqs[partner]
        candidate_freq = max_freqs[free_col]
        return min(partner_bound * candidate_freq, candidate_bound * partner_freq)

    def _determine_table_types(
        self, query: pb.SqlQuery
    ) -> tuple[set[pb.TableReference], set[pb.TableReference]]:
        schema = self._database.schema()
        candidate_joins = query.joins()
        expanding_tables: set[pb.TableReference] = set()

        for join in candidate_joins:
            join_cols = join.columns()
            assert len(join_cols) == 2

            c1, c2 = list(join_cols)
            assert c1.table and c2.table, "Unbound candidate"

            pk_fk = schema.is_primary_key(c1) and schema.has_index(c2)
            inverse_pk_fk = schema.has_index(c1) and schema.is_primary_key(c2)
            if pk_fk or inverse_pk_fk:
                continue

            expanding_tables.update((c1.table, c2.table))

        filtering_tables = query.tables() - expanding_tables
        return expanding_tables, filtering_tables

    def _update_upper(
        self,
        upper: dict[Intermediate, UpperBound],
        *,
        expanding_tables: set[pb.TableReference],
        filter_candidates: set[pb.TableReference],
        max_freqs: dict[pb.ColumnReference, pb.Cardinality],
        query: pb.SqlQuery,
    ) -> None:
        for table in expanding_tables:
            pk_fk_bound: UpperBound = self._filter_card(table, query=query)
            pk_partners = self._pk_partners(
                table, candidates=filter_candidates, query=query
            )
            for pk_table, join_column in pk_partners.items():
                max_freq = max_freqs[join_column]
                pk_card = self._filter_card(pk_table, query=query)
                current_bound = max_freq * pk_card
                pk_fk_bound = min(pk_fk_bound, current_bound)
            upper[frozenset([table])] = pk_fk_bound

    def _pk_partners(
        self,
        table: pb.TableReference,
        *,
        candidates: set[pb.TableReference],
        query: pb.SqlQuery,
    ) -> dict[pb.TableReference, pb.ColumnReference]:
        partner_tables = [
            partner for partner in candidates if query.joins_between(table, partner)
        ]

        partners: dict[pb.TableReference, pb.ColumnReference] = {}
        for partner in partner_tables:
            join_pred = query.joins_between(table, partner)
            assert join_pred is not None
            join_col = join_pred.columns_of(table)
            assert len(join_col) == 1
            partners[partner] = pb.util.simplify(join_col)

        return partners

    def _join_partners(
        self,
        free_tables: set[pb.TableReference],
        *,
        bound_tables: set[pb.TableReference],
        query: pb.SqlQuery,
    ) -> dict[pb.TableReference, JoinPartners]:
        partners: dict[pb.TableReference, JoinPartners] = collections.defaultdict(set)
        for table in free_tables:
            join_predicates = query.joins_between(table, bound_tables)
            if not join_predicates:
                continue

            for join_pred in join_predicates.base_predicates():
                # This only works under the assumption that we either have a simple binary join, or a conjunction of such

                free_key = join_pred.join_partners_of(table)
                assert len(free_key) == 1
                free_key = pb.util.simplify(free_key)
                assert free_key.table, "Unbound candidate"

                bound_key = join_pred.join_partners_of(free_key.table)
                assert len(bound_key) == 1
                bound_key = pb.util.simplify(bound_key)

                partners[table].add((free_key, bound_key))

        return partners

    def _available_pk_tables(
        self,
        table: pb.TableReference,
        *,
        intermediate: Intermediate | set[pb.TableReference],
        candidates: set[pb.TableReference],
        query: pb.SqlQuery,
    ) -> list[pb.TableReference]:
        bound_tables = intermediate.union({table})
        return [
            candidate
            for candidate in candidates
            if query.joins_between(bound_tables, candidate)
        ]

    def _filter_card(
        self, table: pb.TableReference, *, query: pb.SqlQuery
    ) -> pb.Cardinality:
        filter_query = pb.transform.extract_query_fragment(query, table)
        assert filter_query is not None
        return self._database.optimizer().cardinality_estimate(filter_query)

    def _max_freq(self, column: pb.ColumnReference) -> pb.Cardinality:
        assert column.table, "Unbound column"
        mcv = self._database.statistics().most_common_values(column, k=1)[0]
        if mcv:
            return pb.Cardinality(mcv[1])

        # No MCV - assume uniform distribution
        total_card = self._database.statistics().total_rows(column.table)
        distinct_count = self._database.statistics().distinct_values(column)
        assert total_card is not None

        if not distinct_count:
            return pb.Cardinality(1)
        return pb.Cardinality(total_card // distinct_count)


class UESOperators(pb.PhysicalOperatorSelection):
    def __init__(self, *, database: Optional[pb.Database] = None) -> None:
        self._database = database or pb.db.current_database()

    def select_physical_operators(
        self, query: pb.SqlQuery, join_order: Optional[pb.JoinTree] = None
    ) -> pb.PhysicalOperatorAssignment:
        operators = pb.PhysicalOperatorAssignment()
        if self._database.hinting().supports_hint(pb.JoinOperator.NestedLoopJoin):
            operators.set_operator_enabled_globally(
                pb.JoinOperator.NestedLoopJoin, False
            )
        if self._database.hinting().supports_hint(pb.JoinOperator.SortMergeJoin):
            operators.set_operator_enabled_globally(
                pb.JoinOperator.SortMergeJoin, False
            )
        if self._database.hinting().supports_hint(pb.JoinOperator.HashJoin):
            operators.set_operator_enabled_globally(pb.JoinOperator.HashJoin, True)
        return operators

    def describe(self) -> pb.util.jsondict:
        allowed_ops, disabled_ops = [], []

        if self._database.hinting().supports_hint(pb.JoinOperator.NestedLoopJoin):
            disabled_ops.append(pb.JoinOperator.NestedLoopJoin)

        if self._database.hinting().supports_hint(pb.JoinOperator.SortMergeJoin):
            disabled_ops.append(pb.JoinOperator.SortMergeJoin)

        if self._database.hinting().supports_hint(pb.JoinOperator.HashJoin):
            allowed_ops.append(pb.JoinOperator.HashJoin)

        return {
            "name": "UES Operators",
            "allowed_operators": allowed_ops,
            "disabled_operators": disabled_ops,
            "database": self._database.describe(),
        }
