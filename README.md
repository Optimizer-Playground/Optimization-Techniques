# Optimizer Collection

This repository contains a curated collection of interesting and novel ideas for
query optimization in relational database systems.
Optimizer prototypes are implemented using the [PostBOUND framework](https://github.com/rbergm/PostBOUND).

Think of the project as a playground or testbed to play around and explore different optimization techniques.
Using PostBOUND's [optimization pipelines](https://postbound.readthedocs.io/en/latest/core/optimization.html),
it is even possible to combine different optimization techniques, e.g. pessimistic
join ordering with learned operator selection.

## Optimizer Prototypes

| Optimizer | Description                                                                    | Original Paper                                                                                                                                                         | Implementation                                                                                                      |
| --------- | ------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| BAO       | A reinforcement learning-based plan selector.                                  | Marcus et al.: _BAO: Making Learned Query Optimization Practical_ (SIGMOD'2021) 📄 [Paper](https://doi.org/10.1145/3448016.3452838)                                    | 📁 [Code](https://github.com/Optimizer-Playground/Optimization-Techniques/blob/main/postbound_extensions/bao)       |
| FASTgres  | A supervised learning-based model for selecting optimizer hints                | Woltmann et al.: _FASTgres: Making Learned Query Optimizer Hinting Effective_ (VLDB'2023) 📄 [Paper](https://doi.org/10.14778/3611479.3611528)                         | 📁 [Code](https://github.com/Optimizer-Playground/Optimization-Techniques/blob/main/postbound_extensions/fastgres)  |
| LpBound   | An upper bound cardinality estimator based on lp-norms and linear programming. | Zhang et al.: _LpBound: Pessimistic Cardinality Estimation using lp-Norms of Degree Sequences_ (SIGMOD'2025) 📄 [Paper](https://doi.org/10.1145/3725321)               | 📁 [Code](https://github.com/Optimizer-Playground/Optimization-Techniques/blob/main/postbound_extensions/lpbound)   |
| MSCN      | A supervised, deep learning-based cardinality estimator.                       | Kipf et al.: _Learned Cardinalities: Estimating Correlated Joins with Deep Learning._ (CIDR'2019) 📄 [Paper](https://vldb.org/cidrdb/papers/2019/p101-kipf-cidr19.pdf) | 📁 [Code](https://github.com/Optimizer-Playground/Optimization-Techniques/blob/main/postbound_extensions/mscn)      |
| Naru      | An unsupervised, deep autoregressive cardinality estimator for base tables.    | Yang et al.: _Deep Unsupervised Cardinality Estimation_ (VLDB'2020) 📄 [Paper](https://doi.org/10.14778/3368289.3368294)                                               | 📁 [Code](https://github.com/Optimizer-Playground/Optimization-Techniques/blob/main/postbound_extensions/naru)      |
| SafeBound | An upper bound cardinality estimator based on most common values statistics.   | Deeds et al.: _SafeBound: A Practical System for Generating Cardinality Bounds_ (SIGMOD'2023) 📄 [Paper](https://doi.org/10.1145/3588907)                              | 📁 [Code](https://github.com/Optimizer-Playground/Optimization-Techniques/blob/main/postbound_extensions/safebound) |
| TONIC     | A learned (but machine learning-free) operator selection algorithm for joins.  | Hertzschuch et al.: _Turbo-Charging SPJ Query Plans with Learned Physical Join Operator Selections._ (VLDB'2022) 📄 [Paper](https://doi.org/10.14778/3551793.3551825)  | 📁 [Code](https://github.com/Optimizer-Playground/Optimization-Techniques/blob/main/postbound_extensions/tonic)     |
| UES       | An upper bound-driven, heuristic join order optimizer.                         | Hertzschuch et al.: _Simplicity Done Right for Join Ordering._ (CIDR'2021) 📄 [Paper](https://vldb.org/cidrdb/2021/simplicity-done-right-for-join-ordering.html)       | 📁 [Code](https://github.com/Optimizer-Playground/Optimization-Techniques/blob/main/postbound_extensions/ues)       |

## Installation

The optimizer collection is available on [PyPI](https://pypi.org/project/optimizer-collection/).
You can install it using pip or [uv](https://github.com/astral-sh/uv), or similar tools:

```bash
pip install postbound-playground
```

## Usage

The different optimizers are available in as submodules of the `postbound_extensions` package.
Their usage is described in the corresponding README files, which can be found in the respective
submodules (see the table above).

For example, to the pessimistic UES join order optimizer can be combined with the TONIC operator selection
algorithm like so:

```python
import postbound as pb
import postbound_extensions as pbx

# Let's evaluate IMDB/JOB on Postgres
target_workload = pb.workloads.job()
target_database = pb.postgres.connect(".pg-connect-imdb.toml")

# Create our individual optimizer components
ues = pbx.ues.UesJoinOrdering(target_database)
tonic = pbx.tonic.TonicOperators.load_or_build(
  "generated/tonic/job",
  database=target_database,
  sample_plans=target_workload.pick_random(13),
  verbose=True
)

# Combine them into a complete optimizer
optimizer = (
  pb.MultiStageOptimizationPipeline(target_database)
  .use(ues)
  .use(tonic)
  .build()
)

# Start the benchmark
pb.bench.execute_workload(
  target_workload,
  on=optimizer,
  query_preparation={"analyze": True, "prewarm": True},
  name="playground-sample",
  progressive_output="playground-sample.parquet",
  logger="tqdm"
)
```

## Contributing

The playground is a community-driven project. We welcome bug fixes and improvements as well as novel optimizer implementations.
In fact, we are particularly happy about the latter. If you recently published a research paper on query optimization and
want to make your implementation available for the broader community, please feel free to open a pull request or contact us!

If you noticed a bug in any part of the code base and do not have the resources to fix it yourself, feel free to open an issue.
The playground is a living project and we will try to fix any issues as soon as possible.

This project is currently being maintained by Rico Bergmann ([@rbergm](https://github.com/rbergm/)).
The current optimizer implementations have been created by the following people:

- BAO: provided by Rico Bergmann ([@rbergm](https://github.com/rbergm/))
- FASTgres: provided by Kira Thiessat ([@KiraThiessat](https://github.com/KiraThiessat))
- LpBound: provided by Rico Bergmann ([@rbergm](https://github.com/rbergm/))
- MSCN: provided by Rico Bergmann ([@rbergm](https://github.com/rbergm/))
- Naru: generated by Claude Opus 5.5 (Anthropic), supervised by Rico Bergmann ([@rbergm](https://github.com/rbergm/))
- SafeBound: provided by Rico Bergmann ([@rbergm](https://github.com/rbergm/))
- TONIC: provided by Rico Bergmann ([@rbergm](https://github.com/rbergm/))
- UES: provided by Rico Bergmann ([@rbergm](https://github.com/rbergm/))

## Additional Utilities

In addition to the optimizer implementations, the playground also ships a number of utilities that deal with the "meta aspects"
of query optimization, such as generating query workloads and gathering training data.

Currently, we provide the following utilities:

- the `postbound_extensions.sampler` module contains functions to generate SQL queries and execution plans
- the `postbound_extensions.ceb` module implements the [CEB workload sampler](https://www.vldb.org/pvldb/vol14/p2019-negi.pdf)
- the `postbound_extensions.preprocessing` module contains utilities to featurize different data types into a numeric format
  and to cast values as written in SQL queries to their corresponding database types.
- the `postbound_extensions.util` module gathers all kinds of utilities that did not really fit into any specific module.
  Best check out the exported symbols.

A detailed documentation is currently being written and will be available soon. In the meantime, it is probably best to
just take a look at the exported symbols of each module. The members usually have a pretty extensive docstring that explains
their purpose and usage, as well as related functions.
