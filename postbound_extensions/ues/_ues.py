"""UES join ordering algorithm and operator selection for PostBOUND

Copyright (C) 2026 Rico Bergmann

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

from __future__ import annotations

import collections
from collections.abc import Iterable
from typing import Literal, Optional

import postbound as pb

Intermediate = frozenset[pb.TableReference]
UpperBound = pb.Cardinality
JoinKey = tuple[pb.ColumnReference, pb.ColumnReference]
JoinPartners = set[JoinKey]


class UesJoinOrdering(pb.JoinOrderOptimization, pb.CardinalityEstimator):
    """UES is a pessimistic join ordering algorithm that combines upper bounds with a greedy enumeration strategy.

    UES does not need any advanced statistics. Instead, it relies on the target database and its statistics catalog to
    obtain cardinality estimates for base tables and maximum frequencies for join columns. One requirement of UES is that
    these estimates are as accurate as possible. Therefore, by default, we emulate perfect statistics via the PostBOUND API.
    This behavior can be controlled via the `estimations` parameter.

    In addition to the join ordering algorithm, we also allow the upper bounds calculated by UES to be used as cardinality
    estimates. However, be aware that UES did not design these bounds as replacements for full cardinality estimates. Instead,
    they are tailored to the needs of the join ordering algorithm. Therefore, using them as estimates outside of the join
    ordering context goes beyond the original design of UES and results might not be particularly meaningful.
    As a sidenote, we implement the cardinality estimation logic by internally calculating the corresponding join order.
    Therefore, estimation time might be surprisingly high.

    Parameters
    ----------
    database : Optional[pb.Database], optional
        The database to obtain all statistics and schema information. If not provided, the database will be inferred from
        PostBOUND's database pool.
    estimations : Literal["native", "precise"], optional
        How the statistics should be obtained. *precise* means that we emulate perfect statistics. Internally, this is achieved
        by issuing actual SQL queries to obtain the true value for base table cardinalities and maximum frequencies. *native*
        means that we use the estimates provided by the database's statistics catalog and cardinality estimator.
        This will improve optimization performance by a lot, but the resulting join order might be worse due to inaccurate
        estimates. Furthermore, be aware that the *native* option goes against the original design of UES and is not
        recommended in an actual benchmarking scenario. The default is *precise*.

    See Also
    --------
    UesOperators : The corresponding operator selection logic for UES.

    References
    ----------
    Axel Hertzschuch et al.: Simplicity Done Right for Join Ordering (CIDR 2021)
    https://vldb.org/cidrdb/papers/2021/cidr2021_paper01.pdf
    """

    def __init__(
        self,
        *,
        database: Optional[pb.Database] = None,
        estimations: Literal["native", "precise"] = "precise",
    ) -> None:
        self._database = database or pb.db.current_database()

        emulate_stats = estimations == "precise"
        self._stats = self._database.statistics()
        self._stats.emulated = emulate_stats

    def optimize_join_order(self, query: pb.SqlQuery) -> pb.JoinTree[pb.Cardinality]:
        join_tree: pb.JoinTree[pb.Cardinality] = pb.JoinTree()
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
                join_tree = join_tree.join_with(pk_fk_tree, annotation=best_bound)
                for dangling_pk in dangling_pks:
                    join_tree = join_tree.join_with(dangling_pk, annotation=best_bound)
            else:
                join_tree = join_tree.join_with(best_candidate, annotation=best_bound)
                for pk_table in available_pks:
                    join_tree = join_tree.join_with(pk_table, annotation=best_bound)

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

    def calculate_estimate(
        self,
        query: pb.SqlQuery,
        intermediate: pb.TableReference | Iterable[pb.TableReference],
    ) -> pb.Cardinality:
        subquery = pb.transform.extract_query_fragment(query, intermediate)
        if subquery is None:
            return pb.Cardinality.unknown()
        join_tree = self.optimize_join_order(subquery)
        return join_tree.annotation

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
        mcv = self._stats.most_common_values(column, k=1)[0]
        if mcv:
            return pb.Cardinality(mcv[1])

        # No MCV - assume uniform distribution
        total_card = self._stats.total_rows(column.table)
        distinct_count = self._stats.distinct_values(column)
        assert total_card is not None

        if not distinct_count:
            return pb.Cardinality(1)
        return pb.Cardinality(total_card // distinct_count)


class UesOperators(pb.PhysicalOperatorSelection):
    """UES-specific selection of physical operators.

    UES employs a very simple operator "selection" that essentially enforces all joins to be executed as hash joins.

    See `UesJoinOrdering` for more details on the design of UES and the role of operator selection in the original paper.

    See Also
    --------
    UesJoinOrdering : The corresponding join ordering logic for UES.
    """

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
