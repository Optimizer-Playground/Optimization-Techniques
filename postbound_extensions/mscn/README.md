# MSCN

- **Original Paper:** _Learned Cardinalities: Estimating Correlated Joins with
  Deep Learning_
- **Authors:** Andreas Kipf, Thomas Kipf, Bernhard Radke, Viktor Leis, Peter Boncz,
  Alfons Kemper
- **Published:** CIDR 2019
- **Link:** <https://vldb.org/cidrdb/2019/learned-cardinalities-estimating-correlated-joins-with-deep-learning.html>
- **Reference Implementation:** <https://github.com/andreaskipf/learnedcardinalities>

MSCN is a learned cardinality estimator. It uses a neural network to predict the
cardinality of an input SQL query. Queries are featurized based on their tables,
joins, and filter predicates. For training, MSCN learns from a large corpus of
(query, cardinality) pairs.

We implement MSCN based on the original paper and the reference implementation by
Andreas Kipf. Some parts of the source code have been reused, while others have
been implemented from scratch (see [Authors and License](#authors-and-license) below).

## High-level Interface

MSCN implements the PostBOUND [`CardinalityEstimator`](https://postbound.readthedocs.io/en/latest/prototyping.html#cardinality-estimation)
interface. The `MscnEstimator` acts as the central entrypoint. It requires an
`MscnFeaturizer` and a `SetConv` model to function properly. The featurizer takes
care of transforming the SQL queries into appropriate vectors for the set convolution
prediction model. Since the featurization contains a number of one-hot encodings,
it is recommended to build the featurizer based on the training samples as well
as the test workload. This ensures that all columns and tables that are relevant
for the model are included in the featurization. See the documentation of
`MscnFeaturizer` for more details.
Once the featurization has been created, the estimator can be trained on a set of
(query, cardinality) pairs using its `train()` method. Afterwards, it is ready for
estimation.

Training samples can be obtained by using the `query-sampler.py` tool that is
shipped as part of the Optimization Techniques project. See its documentation for
details.

## Storing Trained Models

Since the training process can be quite time consuming, it is recommended to export
the resulting model once training is completed. The `store()` method of the estimator
is responsible for persisting the model state along with all required featurization
info to disk. Afterwards, calling `MscnEstimator.load()` allows to re-create the
estimator.

The generally recommended pattern is to use `MscnEstimator.load_or_build()`. It
checks, whether the model has already
been persisted at the given location. If it has, it is simply re-loaded. Otherwise,
a new featurizer is created and the model is trained from scratch. This model also
takes care of storing the model after training, so any consecutive call to `load_or_build()`
will be very fast.

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
# Step 2: load our training samples
# 
samples = pb.util.read_df("query-samples.parquet")

#
# Step 3: create the estimator
# 
mscn = pbx.mscn.MscnEstimator.load_or_build(
    f"models/mscn/{workload.name}-catalog.json",
    samples=samples,
    workload=workload,
    database=pg_instance,
    verbose=True
)


#
# Step 4: Evaluation
#
# We can already use this optimizer to calculate the bound of different queries:
query = workload["q-10"]
print(mscn.calculate_estimate(query, query.tables()))

# Or we can run the PostBOUND benchmarking utilities
pb.bench.execute_workload(
    workload,
    on=mscn,
    query_preparation={"analyze": True, "prewarm": True},
    name="safebound-test",
    progressive_output="mscn-benchmark.parquet",
    logger="tqdm"
)
```

## Deviations from the Original Paper

One of the goals of the MSCN implementation using PostBOUND is to enable benchmarks
across a wider range of datasets and workloads. To achieve this goal, we had to
generalize MSCN in a few ways. These affect the featurization of the SQL queries,
the MSCN model itself is left unchanged. Specifically, we perform the following
generalizations/changes:

- We do not using the few-hot encoding scheme for the filter operators. Instead,
  we use a **pure one-hot encoding for operators**. This allows us to easily incorporate
  additional predicate types such as _LIKE_.
- We add support for **date and timestamp column types** to the featurization. Internally,
  these are converted to UNIX timestamps which allows to keep the normal integer
  featurization.
- We add support for **string column types** to the featurization. They are featurized
  with an ordinal encoding: we store all column values in a (ordered) dictionary.
  At query time, we lookup the insertion position of the filter value in this dictionary.
  Afterwards, we normalize the index similar to the integer featurization. We
  argue that this kind of featurization is closest in spirit to the featurization
  strategy of the original paper.
- We (currently) **do not support bitmap samples** to refine the encoding of base
  tables.
  This does not have a deep conceptual reason, but is a pragmatic decision to
  reduce the required implementation effort.

In addition to the workload-based featurization used in the original paper, we
also provide an _online_ mode which does not leak any information from the training
samples or the test workload to build its features.
While this is the most realistic setting, it will typically provide the worst
results. This is because the featurization has to consider the entire schema and
the entire set of operators, which leads to the largest (and by consequence sparsest)
vectors and the largest model. Nevertheless we argue that one should generally opt
for this mode to become the most realistic impression of the model performance.

## Authors and License

The implementation of the PostBOUND adapter and the featurization logic have been
implemented by Rico Bergmann.
The multi-set convolution model as well as some utilities have been re-used from
the reference implementation by Andreas Kipf, which is licensed under MIT terms.
We license this implementation under the same terms and make it available as part
of the [Optimizer Collection project](https://github.com/Optimizer-Playground/Optimization-Techniques).
The project as a whole is licensed under GPL-3.0 terms.

## References

Andreas Kipf et al.: _Learned Cardinalities: Estimating Correlated Joins with Deep
Learning_ (CIDR 2019); [Link to the paper](https://vldb.org/cidrdb/papers/2019/p101-kipf-cidr19.pdf)

Reference implementation: <https://github.com/andreaskipf/learnedcardinalities>
