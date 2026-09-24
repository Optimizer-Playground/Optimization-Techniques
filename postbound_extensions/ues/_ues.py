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
from collections.abc import Collection, Iterable, Mapping, Sequence
from typing import Literal, assert_never

import postbound as pb
from postbound.qal import SqlQuery

UesEstimationType = Literal["perfect", "native"]
"""How statistics should be obtained.

*perfect* means that exact statistics are obtained via SQL queries. *native* means that the database's internal
statistics are used.
"""


def _is_unique(column: pb.ColumnReference, *, database: pb.Database) -> bool:
    """Checks, if it can be guaranteed that each value in the column appears exactly once."""
    return database.schema().is_primary_key(column)


def _determine_table_types(
    query: pb.qal.SelectStatement, *, consumed: set[pb.TableReference], schema: pb.db.DatabaseSchema
) -> tuple[list[pb.TableReference], Mapping[pb.TableReference, set[pb.TableReference]]]:
    """Determines expanding and filtering tables in the query.

    Expanding tables are such tables that can be joined with another (expanding) table, the size of the intermediate
    result will increase. In contrast, a filtering table (when joined with its partner), will only reduce the size of
    the intermediate result (or keep it the same), but never increase. Most typically, expanding tables/joins are
    foreign key-foreign key joins, whereas filtering joins are primary key-foreign key joins (where the size of the
    intermediate is capped by the size of the foreign key table).

    If some of the tables have already been integrated into the join tree, they must be provided in `consumed`.
    This is because by joining a filtering table, the filter property is typically lost (after a join, unique columns
    only stay unique, if they are joined with another unique column).

    The result provides a tuple consisting of all expanding tables as the first element, and a map from each table
    to all tables that filter it as a second element.

    The implementation of this function is tightly integrated with the greedy join ordering of UES. For example, it
    assumes that filters are executed as soon as possible. If this assumption is violated, results will be incorrect.
    """

    expanding_tables: list[pb.TableReference] = []
    filter_tables: dict[pb.TableReference, set[pb.TableReference]] = collections.defaultdict(set)

    for join in query.joins():
        simplified = pb.qal.SimpleJoin.attempt_wrap(join)
        if not simplified:
            raise ValueError(f"Query {query} contains non-simple join: {join}")

        lhs_tab, rhs_tab = simplified.lhs.table, simplified.rhs.table
        if lhs_tab is None or rhs_tab is None:
            raise ValueError(f"Query {query} contains join with unbound table: {join}")

        if lhs_tab in consumed and rhs_tab in consumed:
            continue
        elif lhs_tab in consumed:
            # RHS cannot have a unique index, otherwise it would have been a filter and thus already consumed once LHS
            # was consumed
            expanding_tables.append(rhs_tab)
            continue
        elif rhs_tab in consumed:
            # Same reasoning as above, but for LHS
            expanding_tables.append(lhs_tab)
            continue

        lhs_unique = schema.is_primary_key(simplified.lhs)
        rhs_unique = schema.is_primary_key(simplified.rhs)

        if lhs_unique and rhs_unique:
            filter_tables[lhs_tab].add(rhs_tab)
            filter_tables[rhs_tab].add(lhs_tab)
        elif lhs_unique:
            filter_tables[rhs_tab].add(lhs_tab)
        elif rhs_unique:
            filter_tables[lhs_tab].add(rhs_tab)
        else:
            expanding_tables.append(lhs_tab)
            expanding_tables.append(rhs_tab)

    return expanding_tables, filter_tables


def _determine_join_keys(query: pb.qal.SelectStatement) -> Mapping[pb.TableReference, set[pb.ColumnReference]]:
    """For each table in the query, provides the columns that are used in join predicates."""
    result = collections.defaultdict(set)
    for join_pred in query.joins():
        for key1, key2 in join_pred.join_partners():
            if key1.is_bound():
                result[key1.table].add(key1)
            if key2.is_bound():
                result[key2.table].add(key2)
    return result


def _update_novel_freqs(
    max_freqs: Mapping[pb.ColumnReference, int],
    *,
    multiplier: int,
    join_sequence: Sequence[pb.TableReference],
    join_keys: Mapping[pb.TableReference, set[pb.ColumnReference]],
    query: pb.qal.SelectStatement,
) -> Mapping[pb.ColumnReference, int]:
    """Sets the maximum frequencies of all new join keys such that they reflect increases after expanding joins.

    This function is assumed to be called in-between iterations of UES's greedy join search.

    `max_freqs` must contain the frequencies of *all* join keys - those whose base table has already been incorporated
    into the join tree (type 1), those who are still free (type 2), and those who are added to the join tree in the
    current UES iteration (type 3). The update performed by this function affects only type 3 tables.
    **YOU MUST CALL `_updated_bound_freqs` TO UPDATE THE MAX FREQUENCIES OF ALL JOIN COLUMNS THAT ARE ALREADY PART OF
    THE JOIN TREE.** Otherwise, UES will treat some frequencies as too small and favor them instead of other candidates,
    leading to catastrophically bad join orders.

    The `join_sequence` is expected to contain the expanding table as the first element, whereas all following tables
    either filter the expanding table, or filter one of the filter tables. Their order must match the order in the join
    tree. Note that by design of the UES algorithm, none of the filters can be applied to any of the type 1 tables,
    because if the filter is applicable to one of those tables, it would have been applied once the table became type 1.

    The `multiplier` is the frequency of the join key from the bound type 1table (i.e. already part of the join
    tree) that was used to join on the expanding table. The frequencies of all filter key columns are automatically
    inferred based on the expanding table or the filter that they in turn restrict. This information is derived from
    `query`.
    """
    updated_freqs = dict(max_freqs)
    head, *tail = join_sequence
    for join_key in join_keys[head]:
        updated_freqs[join_key] *= multiplier

    consumed = [head]
    for filter_tab in tail:
        join_pred = query.joins_between(consumed, filter_tab)
        assert join_pred is not None

        partner_cols = join_pred.join_partners_of(filter_tab)

        # It is crucial to use the frequency from updated_freqs instead of max_freqs here.
        # Otherwise, we would loose the effect of a join that was performed earlier in the join sequence.
        # This situation can occur if one of the filters of the expanding table is in turn filtered by a later table
        # in the join_sequence.
        multiplier = min(updated_freqs[join_key] for join_key in partner_cols)

        for join_key in join_keys[filter_tab]:
            updated_freqs[join_key] *= multiplier

        consumed.append(filter_tab)

    return updated_freqs


def _update_bound_freqs(
    max_freqs: Mapping[pb.ColumnReference, int], *, bound_tables: set[pb.TableReference], multiplier: int
) -> Mapping[pb.ColumnReference, int]:
    """Sets the maximum frequencies of all existing join keys such that they reflect increases after expanding joins.

    This function is assumed to be called in-between iterations of UES's greedy join search.

    `max_freqs` must contain the frequencies of *all* join keys - those whose base table has already been incorporated
    into the join tree (type 1), those who are still free (type 2), and those who are added to the join tree in the
    current UES iteration (type 3). The update performed by this function affects only type 1 tables.
    **YOU MUST CALL `_update_novel_freqs` TO UPDATE THE MAX FREQUENCIES OF ALL JOIN COLUMNS THAT HAVE JUST BEEN ADDED
    THE JOIN TREE.** Otherwise, UES will treat some frequencies as too small and favor them instead of other candidates,
    leading to catastrophically bad join orders.

    The `bound_tables` are the type 1 tables - they are already part of the join tree. The `multiplier` is the
    frequency of the expanding table's join column that was used to join with the join tree. All frequencies of
    columns that belong to the `bound_tables` will be updated.
    """
    return {
        join_col: freq * multiplier if join_col.table in bound_tables else freq for join_col, freq in max_freqs.items()
    }


class UesJoinOrdering(pb.JoinOrdering, pb.CardinalityEstimator):
    """UES is a pessimistic join ordering algorithm that combines upper bounds with a greedy enumeration strategy.

    UES does not need any advanced statistics. Instead, it relies on the target database and its statistics catalog to
    obtain cardinality estimates for base tables and maximum frequencies for join columns. One requirement of UES is
    that these estimates are as accurate as possible. Therefore, by default, we emulate perfect statistics via the
    PostBOUND API. This behavior can be controlled via the `estimation_type` parameter.

    In addition to the join ordering algorithm, we also allow the upper bounds calculated by UES to be used as
    cardinality estimates. However, be aware that UES did not design these bounds as replacements for full cardinality
    estimates. Instead, they are tailored to the needs of the join ordering algorithm. Therefore, using them as
    estimates outside of the join ordering context goes beyond the original design of UES and results might not be
    particularly meaningful. As a sidenote, we implement the cardinality estimation logic by internally calculating the
    corresponding join order.
    Therefore, estimation time might be surprisingly high.

    Parameters
    ----------
    database : Optional[pb.Database], optional
        The database to obtain all statistics and schema information. If not provided, the database will be inferred
        from PostBOUND's database pool.
    estimations : UesEstimationType, optional
        How the statistics should be obtained. *perfect* means that we emulate perfect statistics. Internally, this is
        achieved by issuing actual SQL queries to obtain the true value for base table cardinalities and maximum
        frequencies. *native* means that we use the estimates provided by the database's statistics catalog and
        cardinality estimator. This will improve optimization performance by a lot, but the resulting join order might
        be worse due to inaccurate estimates. Furthermore, be aware that the *native* option goes against the original
        design of UES and is not recommended in an actual benchmarking scenario. The default is *perfect*.

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
        database: pb.Database | None = None,
        *,
        estimations: UesEstimationType = "perfect",
    ) -> None:
        super().__init__()

        database = database or pb.db.current_database()
        self._database = pb.db.ResultCache.create_cache(database)
        self._estimation_type = estimations
        match estimations:
            case "native":
                self._card = self._database.optimizer().cardinality_estimate
                self._stats = self._database.statistics()
            case "perfect":
                self._card = self._perfect_card_est
                self._stats = pb.db.PreciseStatistics(self._database)
            case _:
                assert_never(estimations)
                raise ValueError(f"Unknown estimation type: {estimations}")

    def optimize_join_order(self, query: pb.SqlQuery) -> pb.JoinTree:
        if not pb.qal.is_select_query(query):
            raise pb.qal.QueryTypeError.expected_select(query)

        query = pb.transform.add_ec_predicates(query)

        expanding_tables, filter_tables = _determine_table_types(query, consumed=set(), schema=self._database.schema())
        base_cards = {tab: self._filter_card(tab, query=query) for tab in query.tables()}
        join_keys = _determine_join_keys(query)
        max_freqs = {join_key: self._max_freq(join_key) for join_key in pb.util.flatten(join_keys.values())}

        # PostBOUND addition:
        # The original paper did not specify how to handle queries without any expanding join.
        # However, there is a straightforward extension of the algorithm to these types of queries.
        # See the documentation on _optimize_star_query() for details.
        if not expanding_tables:
            return self._optimize_star_query(query, filter_tables=filter_tables)

        total_upper: pb.Cardinality
        join_tree = pb.JoinTree.empty()
        while expanding_tables:
            upper = {
                tab: self._filter_bound(
                    tab, query=query, filter_tables=filter_tables[tab], base_cards=base_cards, max_freqs=max_freqs
                )
                for tab in expanding_tables
            }

            if join_tree.is_empty():
                initial_tab = pb.util.argmin(upper)
                total_upper = upper[initial_tab]

                # PostBOUND addition:
                # The algorithm in the original paper did not include filters on the initial table at this point.
                # At the same time, it did not include such tables at all. Therefore, we argue that these have simply
                # been overlooked and include them here - similar to how they are handled in any other part of the
                # algorithm.
                # Note that _greedy_order_filters() is yet another addition/modification. See its documentation for
                # more details.
                initial_filters = self._greedy_order_filters(
                    filter_tables[initial_tab], filter_tables=filter_tables, base_cards=base_cards
                )
                join_sequence = [initial_tab] + list(initial_filters)
                join_tree = self._expand_to_join_tree(join_sequence, query=query)

                # PostBOUND addition:
                # The original algorithm did not remove the initial table from the expanding list.
                # This is an obvious error.
                expanding_tables, filter_tables = _determine_table_types(
                    query, consumed=join_tree.tables(), schema=self._database.schema()
                )
                max_freqs = _update_novel_freqs(
                    max_freqs, multiplier=1, join_sequence=join_sequence, join_keys=join_keys, query=query
                )
                continue

            best_upper = pb.Cardinality.infinite()
            best_candidate: pb.TableReference
            candidate_freq: int
            bound_freq: int

            bound_tables = join_tree.tables()
            for candidate in expanding_tables:
                join_pred = query.joins_between(bound_tables, candidate)
                if join_pred is None:
                    continue

                for bound_col, partner_col in join_pred.join_partners():
                    if bound_col.belongs_to(candidate):
                        # join_partners() has no direction. We must normalize ourselves.
                        bound_col, partner_col = partner_col, bound_col
                    if not partner_col.belongs_to(candidate):
                        # Safety check. In practice, this should never fail because we only consider joins between
                        # a single candidate and a set of bound tables
                        continue

                    current_upper = (
                        min(total_upper / max_freqs[bound_col], upper[candidate] / max_freqs[partner_col])
                        * max_freqs[bound_col]
                        * max_freqs[partner_col]
                    )

                    if current_upper > best_upper:
                        continue

                    best_upper = current_upper
                    best_candidate = candidate
                    candidate_freq = max_freqs[partner_col]
                    bound_freq = max_freqs[bound_col]

            filter_sequence = self._greedy_order_filters(
                filter_tables[best_candidate], filter_tables=filter_tables, base_cards=base_cards
            )

            # PostBOUND addition:
            # The original UES algorithm specified to always put the expanding table/subtree on the inner side of
            # a join. However, the reference implementation generated JOIN ON queries for Postgres. While these
            # queries (in combination with join_collapse_limit) could enforce the correct join order, the assignment
            # of inner/outer relations (and by extension build and probe side for hash joins) were still up to
            # Postgres. In our experiments we have seen that this seemingly subtle difference can have a
            # significant impact on the performance of the query plan - up to making the plan completely infeasible.
            # While we cannot mimic the original behavior 1:1 (PostBOUND has no "use this join order, but assign
            # inner/outer yourself" hint), we can do our best with a simple heuristic: the assignment of inner/outer
            # side matters primarily for hash joins. For nested-loop and merge joins it does not impact the runtime.
            # And for hash joins, there is an obvious rule: the build side should always be the smaller relation.
            # And this decision is based on the native cardinality estimates of the target database anyway. Therefore,
            # we simply ask the target database for its estimates and assign the inner/outer side accordingly.
            # We argue that this is the best compromise between puristic adherence to the original UES algorithm,
            # deviations in the actual implementation, and practicality of the resulting plans.

            if upper[best_candidate] < base_cards[best_candidate]:
                join_sequence = [best_candidate] + list(filter_sequence)
                bushy_tree = self._expand_to_join_tree(join_sequence, query=query)
                join_tree = (
                    bushy_tree.join_with(join_tree)
                    if self._use_as_outer(query, bound=join_tree.tables(), partners=bushy_tree.tables())
                    else join_tree.join_with(bushy_tree)
                )
            else:
                join_tree = (
                    pb.JoinTree.create_scan(best_candidate).join_with(join_tree)
                    if self._use_as_outer(query, bound=join_tree.tables(), partners={best_candidate})
                    else join_tree.join_with(best_candidate)
                )
                for filter_tab in filter_sequence:
                    join_tree = (
                        pb.JoinTree.create_scan(filter_tab).join_with(join_tree)
                        if self._use_as_outer(query, bound=join_tree.tables(), partners={filter_tab})
                        else join_tree.join_with(filter_tab)
                    )

            total_upper = best_upper

            # Below it is crucial to pass the correct multiplier to the updates: novel freqs are updated based on the
            # bound column, bound freqs are updated based on the novel (candidate) column
            max_freqs = _update_novel_freqs(
                max_freqs,
                multiplier=bound_freq,
                join_sequence=[best_candidate] + list(filter_sequence),
                join_keys=join_keys,
                query=query,
            )
            max_freqs = _update_bound_freqs(max_freqs, bound_tables=bound_tables, multiplier=candidate_freq)

            # At the end of each iteration, we must re-build expanding and filtering tables. This is because a single
            # table might act as a filter on multiple expanding tables. If one of these expanding tables was our best
            # candidate, the filter is no longer available for the other candidates.
            expanding_tables, filter_tables = _determine_table_types(
                query, consumed=join_tree.tables(), schema=self._database.schema()
            )

        return join_tree

    def describe(self) -> pb.util.jsondict:
        return {"name": "ues", "estimation_type": self._estimation_type, "database": self._database.describe()}

    def pre_check(self) -> pb.validation.OptimizationPreCheck:
        ues_checks = [pb.validation.EquiJoinPreCheck(), pb.validation.CrossProductPreCheck()]
        return super().pre_check().merge_with(ues_checks)

    def _perfect_card_est(self, query: pb.SqlQuery) -> pb.Cardinality:
        """Calculates the exact cardinality of for a base table (represented by its subquery)."""
        query = pb.transform.as_count_star_query(query)
        card = self._database.execute_query(query)
        return pb.Cardinality.of(card)

    def _use_as_outer(
        self, query: pb.qal.SelectStatement, *, bound: set[pb.TableReference], partners: set[pb.TableReference]
    ) -> bool:
        """Determines, whether `bound` and `partners` should be joined with partners as outer relation.

        This heuristic is a key PostBOUND addition to the original UES algorithm. The heuristic assigns inner/outer
        relations based on the cardinality estimate of each subquery. See the detailed documentation in
        `optimize_join_order` for why this is necessary.
        """
        bound_query = pb.transform.extract_subquery(query, bound)
        bound_card_est = self._database.optimizer().cardinality_estimate(bound_query)

        partner_query = pb.transform.extract_subquery(query, partners)
        partner_card_est = self._database.optimizer().cardinality_estimate(partner_query)

        return bound_card_est < partner_card_est

    def _expand_to_join_tree(self, tables: Sequence[pb.TableReference], query: pb.qal.SelectStatement) -> pb.JoinTree:
        """Transforms the given `tables` into a linear join tree.

        Outer and inner relations will be assigned according to `_use_as_outer`.
        """
        head, *tail = tables
        join_tree = pb.JoinTree.create_scan(head)
        for partner in tail:
            join_tree = (
                pb.JoinTree.create_scan(partner).join_with(join_tree)
                if self._use_as_outer(query, bound=join_tree.tables(), partners={partner})
                else join_tree.join_with(partner)
            )
        return join_tree

    def _optimize_star_query(
        self, query: pb.qal.SelectStatement, *, filter_tables: Mapping[pb.TableReference, set[pb.TableReference]]
    ) -> pb.JoinTree:
        """Greedy join ordering for star queries.

        This function is another key PostBOUND addition to standard UES. The original algorithm was only designed for
        queries with at least one expanding join. Filters were included greedily, i.e. as soon as the filtered
        (expanding) table was added to the join tree. If multiple filters were available, the smallest filter table
        was chosen first. We use the same basic idea for queries without expanding joins (i.e. star queries): we start
        with the smallest table that is not itself a filter for a different table and greedily join it with the smallest
        applicable filter. This process continues, iteratively adding more and more filters that can be joined with one
        of the tables already in the join tree. If multiple candidate joins exist, we prefer filtering joins beyond
        expanding joins (which can appear after certain filters have been executed) and tables with smaller
        cardinalities.
        """

        base_cards = {tab: self._filter_card(tab, query=query) for tab in query.tables()}

        candidates = set(filter_tables.keys())
        for filtering_tables in filter_tables.values():
            candidates -= filtering_tables

        if not candidates:
            raise ValueError("Cyclic filter dependencies detected in query. Cannot determine join order.")

        best_candidate: pb.TableReference
        min_bound = pb.Cardinality.infinite()
        for candidate in candidates:
            current_bound = base_cards[candidate]
            if current_bound < min_bound:
                min_bound = current_bound
                best_candidate = candidate

        join_tree = pb.JoinTree.create_scan(best_candidate)
        free_tables = query.tables() - {best_candidate}

        while free_tables:
            filtering_candidates: list[pb.TableReference] = []
            all_candidates: list[pb.TableReference] = []

            for candidate in free_tables:
                candidate_joins = query.joins_between(join_tree.tables(), candidate)
                if candidate_joins is None:
                    continue

                all_candidates.append(candidate)
                if any(
                    _is_unique(join_col, database=self._database) for join_col in candidate_joins.columns_of(candidate)
                ):
                    filtering_candidates.append(candidate)

            candidates = filtering_candidates if filtering_candidates else all_candidates
            next_table = min(candidates, key=lambda tab: base_cards[tab])
            join_tree = join_tree.join_with(next_table)
            free_tables.remove(next_table)

        return join_tree

    def _filter_card(self, table: pb.TableReference, *, query: pb.qal.SelectStatement) -> pb.Cardinality:
        """Computes the cardinality of a `table` after all matching filters have been applied."""
        subquery = pb.transform.extract_subquery(query, table)
        return self._card(subquery)

    def _max_freq(self, column: pb.ColumnReference) -> int:
        """Computes the maximum value frequency of the `column`."""
        if not pb.ColumnReference.assert_bound(column):
            raise pb.UnboundColumnError(column)

        if _is_unique(column, database=self._database):
            return 1

        mcv = self._stats.most_common_values(column)
        if mcv is None:
            total_freq = self._stats.total_rows(column.table)
            n_distinct = self._stats.num_distinct(column)

            if total_freq is None or n_distinct is None:
                raise ValueError(f"Cannot determine max frequency for column {column} due to missing statistics.")
            return round(int(total_freq) / n_distinct)

        return mcv.frequencies[0]

    def _filter_bound(
        self,
        table: pb.TableReference,
        *,
        query: pb.qal.SelectStatement,
        filter_tables: Iterable[pb.TableReference],
        base_cards: Mapping[pb.TableReference, pb.Cardinality],
        max_freqs: Mapping[pb.ColumnReference, int],
    ) -> pb.Cardinality:
        """Computes the upper bound of a specific `table`, taking predicates and primary-key joins into account."""
        min_bound = base_cards[table]

        for filter_tab in filter_tables:
            filter_card = base_cards[filter_tab]
            join_pred = query.joins_between(table, filter_tab)
            assert join_pred is not None

            for fk_attr in join_pred.join_partners_of(filter_tab):
                max_freq = max_freqs[fk_attr]
                min_bound = min(min_bound, max_freq * filter_card)

        return min_bound

    def _greedy_order_filters(
        self,
        candidates: Iterable[pb.TableReference],
        *,
        filter_tables: Mapping[pb.TableReference, set[pb.TableReference]],
        base_cards: Mapping[pb.TableReference, pb.Cardinality],
    ) -> Sequence[pb.TableReference]:
        """Computes all filters that can be applied to the candidates and orders them greedily by cardinality.

        This function is the final PostBOUND addition to the original UES algorithm. The original algorithm did not
        consider that filter tables can themselves be filtered by additional tables (e.g., in a snowflake schema).
        It either would have needed to include these filters while dealing with the best expanding candidate, or it
        would have needed to add them to the expanding tables afterwards. However, the original UES did neither of
        these things.

        For our implementation, we opt for the former strategy, which is in line with the general "execute filtering
        joins as early as possible": we start with the smallest filter table and add all tables that perform
        "second-level" filtering on this table to the candidates. This process continues iteratively until all
        applicable filter tables have been added.
        """
        candidates = set(candidates)
        if not candidates:
            return []

        ordered: list[pb.TableReference] = []
        while candidates:
            next_tab = min(candidates, key=lambda tab: base_cards[tab])
            ordered.append(next_tab)
            candidates.remove(next_tab)
            candidates |= filter_tables[next_tab]

        return ordered


class UesOperators(pb.OperatorSelection):
    """UES-specific selection of physical operators.

    UES employs a very simple operator "selection" that essentially enforces all joins to be executed as hash joins.

    See `UesJoinOrdering` for more details on the design of UES and the role of operator selection in the original paper.

    See Also
    --------
    UesJoinOrdering : The corresponding join ordering logic for UES.
    """

    def __init__(self) -> None:
        super().__init__()

    def select_physical_operators(
        self, query: SqlQuery, join_order: pb.JoinTree | None
    ) -> pb.PhysicalOperatorAssignment:
        assignment = pb.PhysicalOperatorAssignment()
        assignment.set(pb.JoinOperator.NestedLoopJoin, False)
        assignment.set(pb.JoinOperator.SortMergeJoin, True)
        assignment.set(pb.JoinOperator.HashJoin, True)
        return assignment


class UesOptimizer(pb.OptimizationPipeline):
    """Complete UES optimization pipeline that combines join ordering and operator selection.

    See Also
    --------
    UesJoinOrdering : The join ordering logic for UES.
    UesOperators : The operator selection logic for UES.
    """

    def __init__(
        self,
        target_db: pb.Database | None = None,
        *,
        estimations: UesEstimationType = "perfect",
    ) -> None:
        self._target_db = target_db or pb.db.current_database()
        self._enumerator = UesJoinOrdering(self._target_db, estimations=estimations)
        self._phys_ops = UesOperators()

    def query_execution_plan(self, query: pb.SqlQuery) -> pb.QueryPlan:
        hinted_query = self.optimize_query(query)
        return self._target_db.optimizer().query_plan(hinted_query)

    def optimize_query(self, query: pb.SqlQuery) -> pb.SqlQuery:
        join_order = self._enumerator.optimize_join_order(query)
        phys_ops = self._phys_ops.select_physical_operators(query, join_order)
        return self._target_db.hinting().generate_hints(query, join_order=join_order, physical_operators=phys_ops)

    def stages(self) -> Collection[pb.OptimizationStage]:
        return [self._enumerator, self._phys_ops]

    def target_database(self) -> pb.Database:
        return self._target_db

    def describe(self) -> pb.util.jsondict:
        return {
            "name": "ues",
            "target_database": self._target_db.describe(),
            "estimation_type": self._enumerator._estimation_type,
        }
