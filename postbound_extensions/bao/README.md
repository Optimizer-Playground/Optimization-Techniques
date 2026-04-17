# BAO

- **Original Paper:** _Bao: Making Learned Query Optimization Practical_
- **Authors:** Ryan Marcus, Parimarjan Negi, Hongzi Mao, Nesime Tatbul,
  Mohammad Alizadeh, and Tim Kraska
- **Published:** SIGMOD 2021
- **Link:** <https://doi.org/10.1145/3448016.3452838>
- **Reference Implementation:** <https://github.com/learnedsystems/BaoForPostgreSQL>

Bao is a learned query optimizer/cost model. It predicts _optimizer hints_
(DBMS settings that modify the optimizer behavior) to generate faster query plans.
These predictions are calculated by a learned cost model which integrated in a
reinforcement learning approach. In short, for each selected optimizer hint, Bao
requests the query plan that the DBMS would use for this hint. Afterwards, the learned
cost model is used to select the cheapest plan among the candidates.

We implement Bao mostly from scratch based on the description in the original paper.
Some parts are reused from the [TreeConvolution project](https://github.com/RyanMarcus/TreeConvolution)
by Ryan Marcus. Likewise, we also take inspiration from the
[reference implementation for PostgreSQL](https://github.com/learnedsystems/BaoForPostgreSQL),
but do not reuse any code. In case of contradicitions between the description in
the paper and the actual code in the reference implementation, we follow the paper
(see [Deviations from the Original Paper](#deviations-from-the-original-paper)
below for details).

## High-level Interface

Bao is a very unviversal idea and can be used in many different contexts within
query optimization, such as selection of entire query plans (which is what the
paper does), selection of physical operators (which is what Bao's hints primarly
modify), or cost estimation (which is what Bao's TCNN essentially does). We try
to support all of these use cases in our PostBOUND implementation.

The `BaoOptimizer` acts as central entrypoint for all optimization scenarios.
It implements multiple interfaces to support different use cases as follows:

- `CompleteOptimizationAlgorithm` to select entire execution plans
- `PhysicalOperatorSelection` to select physical operators only (possibly
  constrained by a join order)
- `CostModel` to estimate the cost of query plans. This is essentially a
  wrapper around the TCNN.

Our Bao optimizer implements the retraining logic outlined in the paper: we
use past query executions as feedback and retrain once enough new queries
have been executed. The retraining parameters must be configured when creating
the optimizer. Afterwards, `add_experience()` is used to store feedback and to
trigger retraining when necessary.

When also add an "offline mode" to learn an initial TCNN configuration with
the `calibrate()` method. This is particularly useful for the learned cost model
approach. Suitable training samples can be generated using the _query-sampler.py_
tool.

In addition to the actual optimizer, the following classes capture important
concepts of the original Bao paper:

- `BaoModel` implements the TCNN-based cost model
- `BaoFeaturizer` handels the transformation of query plans into feature vectors
- `BaoExperience` stores past query executions and keeps track of when retraining
  is necessary. Note that the retraining itself happens inside the `BaoOptimizer`.

## Storing a TCNN

The entire state of a BAO optimizer (TCNN weights, retraining window, and available
hint sets) can be persisted with the `BaoOptimizer.store()` method. Afterwards,
the `pre_trained` factory method can be used to reload the optimizer.

The generally recommended pattern is to use `BaoOptimizer.load_or_build()`.
It checks, whether an archive has been persisted at the given location. If it has,
it will be loaded. Otherwise, a new optimizer will be created, calibrated
and stored.

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
# Step 2: load some calibration queries
# These are used to pre-train the TCNN cost model
#

training_samples = pb.util.read_df("query-samples.parquet")

#
# Step 3: build our optimizer
#

bao = pbx.bao.BaoOptimizer.load_or_build(
    "models/bao/catalog.json",
    database=pg_instance,
    calibration_queries=training_samples,
    retrain=True,
    training_epochs=100,
    experience_window=2000,
    retraining_frequency=20,
    verbose=True,
)


#
# Step 4: Evaluation
#
# We can already use this optimizer to select query plans and hint sets
query = stats["q-10"]
print(bao.select_physical_operators(query, join_order=None))

# Or we can run the PostBOUND benchmarking utilities
# By passing bench_feedback as an exec_callback, we can automatically
# tap into the retraining facilities.
pb.bench.execute_workload(
    stats,
    on=bao,
    query_preparation={"analyze": True, "prewarm": True},
    name="bao-test",
    progressive_output="bao-benchmark.parquet",
    exec_callback=bao.bench_feedback,
    logger="tqdm"
)
```

## Deviations from the Original Paper

While our Bao implementation tries to stay as faithful as possible to the
original paper, there are a few deviations. These are necessary due to two
main reasons:

1. differences in the overall system architecture, and
2. resolving incosistencies

After studying the [reference implementation](https://github.com/learnedsystems/BaoForPostgreSQL)
of Bao, we noticed a few instances were the implementation seems to deviate from
the description in the paper. In this case, we typically favor the
description in the paper. Specifically, this includes the following
aspects of Bao:

The plan featurization uses the full query plan and binarizes nodes with a single
child element. The reference implementation seems to simply drop these nodes and
strip the query plan down to joins and scans. As part of this change, we also
extend the encoding of the physical operators. By default, we use the full set of
operators supported by Postgres. This can be adjusted by customizing the `BaoFeaturizer`.

For the buffer usage statistics, we calculate the cache percentage at the beginning
of the optimization process. Currently, this functionality is limited to and specialized
for Postgres. The cache percentage for sequential scans is inferred directly from
the buffer pool (as in the Bao paper). For index scans and bitmap scans, we try
to accomodate for the fact that Postgres needs to access two kinds of relations:
the actual data pages and the index pages. To include both of the relation types,
we calcualte the cache percentage as the average cache percentage of the index pages
and all data pages.

In addition to these deviations, we changed the following aspects of the Bao system:
We added calibration support to train a TCNN in an offline fashion. Afterwards,
Bao can be used without any feedback as a pure offline model.

One key architectural change was necessary because PostBOUND currently does not
support a tight feedback loop between an optimized plan and its performance. This
is a key component of Bao's reinforcement learning architecure.
Instead, feedback  needs to be handled either explicitly by the user, or as an
`exec_callback` in the benchmarking utilities.
To support this use-case, our Bao implementation offers a `bench_feedback` method
(see example above).

## Authors and License

The implementation of BAO has been contributed by Rico Bergmann.
Some parts (TCNN implementation and some utilities) have been reused and adapted
from the [TreeConvolution project](https://github.com/RyanMarcus/TreeConvolution)
by Ryan Marcus, which is licensed under GPL-3.0 terms.
The implementation is provided as part of the [Optimizer Collection project](https://github.com/Optimizer-Playground/Optimization-Techniques)
and licensed under the terms of the GPL-3.0.

## References

Ryan Marcus et al.: _Bao: Making Learned Query Optimization Practical_
(SIGMOD 2021); DOI [10.1145/3448016.3452838](https://doi.org/10.1145/3448016.3452838).

Bao reference implementation: <https://github.com/learnedsystems/BaoForPostgreSQL>

TreeConvolution project by Ryan Marcus: <https://github.com/RyanMarcus/TreeConvolution>
