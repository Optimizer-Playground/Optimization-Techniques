# UES

- **Original Paper:** _Simplicity Done Right for Join Ordering_
- **Authors:** Axel Hertzschuch, Claudio Hartmann, Dirk Habich and Wolfgang Lehner
- **Published:** CIDR 2021
- **Link:** <https://vldb.org/cidrdb/2021/simplicity-done-right-for-join-ordering.html>
- **Reference Implementation:** <https://github.com/axhertz/SimplicityDoneRight>

UES is a pessimistic join ordering algorithm. It computes upper bounds on the intermediate cardinalities
based on the sizes of the input relations and the highest frequency of the join keys.
Based on these bounds, UES applies a simple greedy strategy to determine the join order. Essentially,
the algorithm always picks the join with the lowest upper bound. As an optimization, safe
primary key/foreign key joins can be pushed down into bushy subplans before they are considered for the
greedy join ordering. In addition, UES provides a very simple operator selection that always picks hash
joins.

We provide a faithful implementation of UES based on the original paper. All parts of the algorithm
are implemented from scratch using the PostBOUND API.

## High-level Interface

The complete UES algorithm comes in two parts: the actual join ordering algorithm which is implemented
in the `UesJoinOrdering` class and the `UesOperators` class which provides the operator selection logic.
In addition, the `UesJoinOrdering` class also implements the `CardinalityEstimator` interface, which allows
the internal upper bounds to be used as cardinality estimates. However, be aware that the bounds were not
originally designed to serve as general-purpose cardinality estimates. Using them as such will likely lead
to impractical estiamtes for many queries. The purpose of the estimates was tailored to the needs of the
join ordering algorithm and they are not necessarily suitable for other purposes.

As a key ingredient, the UES join ordering algorithm requires precise cardinality estimates of the (filtered)
base tables. By default, these will be obtained by issuing the appropriate SQL queries to the database. The
same applies to the join key frequencies. As an alternative, the algorithm also allows to use the native
estimates from the target database. Once again be aware that while this mode exists, it goes somewhat against
the spirit of the original paper. See the documentation of the `UesJoinOrdering` class for more details.

## Example

```py
#
# Step 1: generic setup
#
# As usual for PostBOUND, connect to our target database and load the workload
import postbound as pb
import postbound_extensions as pbx

pg_instance = pb.postgres.connect(config_file=".pg-connect-stats.toml")
workload = pb.workloads.stats()

#
# Step 2: create the UES optimization pipeline
#

ues = pbx.ues.UesJoinOrdering(database=pg_instance, estimations="precise")
pipeline = (
    pb.MultiStageOptimizationPipeline(pg_instance)
    .use(ues)
    .use(pbx.ues.UesOperators(database=pg_instance))
    .build()
)


#
# Step 3: Evaluation
#
# We can use the plain UES algorithm to obtain the join order for a single query
query = workload["q-10"]
print(ues.optimize_join_order(query))

# Or we can run the PostBOUND benchmarking utilities
pb.bench.execute_workload(
    workload,
    on=pipeline,
    query_preparation={"analyze": True, "prewarm": True},
    name="ues-test",
    progressive_output="ues-benchmark.parquet",
    logger="tqdm"
)
```

## Deviations from the Original Paper

Our implementation of UES is a faithful adaptation of the original paper. The join ordering algorithm
is a 1:1 mapping of the pseudocode to the appropriate PostBOUND primitives. The only modifications are
extensions to make UES more useful in related scenarios.

Specifically, we added the option to use native estimates instead of using an oracle. In addition, we
also allow the upper bounds to be used as cardinality estimates outside of PostBOUND. Lastly, the
precise semantics of the operator selection logic have not been made entirely clear in the original paper.
It appears that the reference implementation only disabled nested-loop joins but still allowed for merge
joins. In our implementation, we interpret the original paper more strictly and only allow for hash joins.

## Authors and License

This implementation of UES has been contributed by Rico Bergmann. It is provided as part of
the [Optimizer Collection project](https://github.com/Optimizer-Playground/Optimization-Techniques)
and licensed under the same conditions.

## References

Axel Hertzschuch et al.: _Simplicity Done Right for Join Ordering_ (CIDR 2021);
[Link to the paper](https://vldb.org/cidrdb/papers/2021/cidr2021_paper01.pdf)

Reference implementation: <https://github.com/axhertz/SimplicityDoneRight>
