# LpBound

- **Original Paper:** _LpBound: Pessimistic Cardinality Estimation using lp-Norms
  of Degree Sequences_
- **Authors:** Haozhe Zhang, Christoph Mayer, Mahmoud Abo Khamis, Dan Olteanu,
  and Dan Suciu
- **Published:** SIGMOD 2025
- **Link:** <https://doi.org/10.1145/3725321>
- **Reference Implementation:** <https://github.com/fdbresearch/LpBound>

LpBound is an upper bound-based cardinality estimator (i.e. a "pessimistic
optimizer") that combines lp-norms with information theory and expresses the
entire optimization process as a linear program.
It supports conditioning the lp-norms on join columns by pre-computing filter predicates.

We implement LpBound completely from scratch based on the original paper. Under
the hood we use NumPy and scipy to accelerate computations.

## High-level Interface

LpBound implements the PostBOUND [`CardinalityEstimator`](https://postbound.readthedocs.io/en/latest/api/core/optimization-pipelines.html#postbound.CardinalityEstimator)
interface. The `LpBoundEstimator` acts as the central entrypoint. It requires
a `LpBoundCatalog` to function correctly. This catalog stores all individual
(conditioned and unconditioned) lp-norms. A catalog can be created either based
on a workload, or inferred completely from the database.
See its documentation for more details.

Due to many similarities between LpBound and SafeBound (one could think of LpBound
as an evolution of SafeBound), we re-use parts of our SafeBound implementation and
structure the module in a very similar manner.

In addition to catalog and estimator, the following functions capture important
concepts of the original paper:

- `LpNorms` models the core data structure.
- `lp_join_graph()` translates an SQL query into the bipartite join graph that
  LpBound uses for its estimation
- `LpBerge` and `LpFlow` implement two different linear programs that are tailored
  to specific query shapes. `is_berge_acyclic` can be used to determine the
  correct program.

Internally, we use SciPy to solve the linear programs. In turn, SciPy bundles the
HiGHS LP solver, which was also used in the reference implementation.

## Storing LpBound Catalogs

Since computing an entire LpBound catalog from scratch involves issuing a large
number of (somewhat complex) SQL queries, it is quite time consuming and we do
not recommend doing it online.
Instead, an existing catalog can be persisted as a JSON object using the `store()`
method.
Afterwards, the `LpBoundCatalog.load()` method allows to re-create the catalog.

The generally recommended pattern is to use `LpBoundCatalog.load_or_build()`.
It checks, whether a catalog has been persisted at the given location. If it has,
it will be loaded. Otherwise, the catalog is created and stored.

## Example

```py
#
# Step 1: generic setup
#
# As usual for PostBOUND, connect to our target database and load the workload
import postbound as pb
import postbound_extensions as pbx

pg_instance = pb.postgres.connect(config_file=".pg-connect-stats.toml")
stats = pb.workloads.stats()


#
# Step 2: build our catalog
#
# We use the same "hyperparameters" (mcv size, histogram depth, etc.)
# as the original papers
lpbound_spec = pbx.lpbound.LpBoundSpec.default()

# If no catalog exists, build a new one fine-tuned for the workload
# otherwise, reload the existing catalog
cat = pbx.lpbound.LpBoundCatalog.load_or_build(
    "models/lpbound/stats", database=pg_instance, workload=stats, spec=lpbound_spec
)


#
# Step 3: create the actual estimator
#
lpbound = pbx.lpbound.LpBoundEstimator(cat)


#
# Step 4: Evaluation
#
# We can already use this optimizer to calculate the bound of different queries:
query = stats["q-10"]
print(lpbound.calculate_estimate(query, query.tables()))

# Or we can run the PostBOUND benchmarking utilities
pb.bench.execute_workload(
    stats,
    on=lpbound,
    query_preparation={"analyze": True, "prewarm": True},
    name="lpbound-test",
    progressive_output="lpbound-benchmark.parquet",
    logger="tqdm"
)
```

## Deviations from the Original Paper

The LpBound implementation contains a faithful adaptation of the following features:

- Calculation of the lp-norms
- Formulation of the LPberge and LPflow linear programs
- Correlated lp-norms for equality predicates based on most-common values
- Correlated lp-norms for range predicates using hierarchies of histograms

To broaden the set of compatible workloads, we added support for half-open range
predicates. See the documentation of `RangeConditionedNorms` for details.

At the same time, we currently do not support the following:

- lp-norms on pre-computed primary key/foreign key joins ("predicate propagation")
- Prefix sharing of degree sequences
- `GROUP BY` queries

Not supporting `GROUP BY` queries is a pragmatic decision since non of the
popular workloads in query optimization currently have such queries.

We do not support pre-computed primary key/foreign key joins (called
"_predicate propagation_" in the paper) with a similar reasoning as in the
SafeBound case: Essentially, this constitutes a materialized view, which is an
orthogonal optimization which is best treated at the schema level.
At the same time, a large portion of LpBounds bound tightness comes from the
fact that it pre-computes those joins. On Stats, we saw a decrease of the median
accuracy from 30.80 to 109.85.

## Authors and License

This implementation of LpBound has been contributed by Rico Bergmann. It is
provided as part of the [Optimizer Collection project](https://github.com/Optimizer-Playground/Optimization-Techniques)
and licensed under the same conditions.

## References

Haozhe Zhang et al.: _LpBound: Pessimistic Cardinality Estimation using lp-Norms
of Degree Sequences_ (SIGMOD 2025); DOI [10.1145/3725321](https://doi.org/10.1145/3725321)

Reference implementation: <https://github.com/fdbresearch/LpBound>
