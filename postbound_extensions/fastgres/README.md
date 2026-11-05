# FASTgres with FASTlabel

This module brings learned optimizer hinting to PostgreSQL through
[PostBOUND](https://github.com/rbergm/PostBOUND). It combines **FASTgres** for
predicting useful hint sets with **FASTlabel** for producing the training labels.
PostgreSQL uses those hints to choose and execute a query plan.

**This is an adapted and modernized implementation of both approaches**, developed
for the Optimizer Playground. It retains their central ideas while updating the
interfaces, query handling, and persistence for PostBOUND.

## How it works

A hint set is a combination of PostgreSQL planner switches, such as allowing or
discouraging hash joins, nested-loop joins, or index scans. Different queries can
benefit from different combinations.

FASTlabel searches promising combinations and measures their execution times. The
best observed combination becomes the query's training label. FASTgres then learns
to predict these labels from query features, using gradient boosting models for
groups of related queries called *contexts*.

The workflow has four steps:

1. **Label a workload.** Run the adapted FASTlabel search, or reuse existing labels.
2. **Prepare query features.** Collect database statistics and encode query filters.
3. **Train FASTgres.** Group queries into contexts and learn a hint predictor for
   each context.
4. **Optimize queries.** Predict a hint set and let PostgreSQL plan the query using
   those settings.

## This adaptation

The implementation brings the two approaches into a shared PostBOUND workflow:

- Query parsing and database interaction use the PostBOUND 0.23 APIs.
- Encoding statistics and CSV labels can be reused; trained models can be saved
  and loaded for later prediction.
- Versioned hint catalogs cover PostgreSQL 12 through 19, including their declared
  JIT defaults.

The current PostBOUND operator-selection adapter supports the **six core scan and
join hints**. The additional hints in the versioned catalogs require further
integration with that adapter.

## Where to start

[`FastgresOptimizer`](_optimizer.py) is the public entry point. It defaults to one
supertable model over the tables in the preparation workload. Training subsets and
new queries use that same context and feature layout. A query outside the trained
contexts emits `FastgresContextWarning` and receives the PostgreSQL defaults
declared by the selected hint library.

```python
from postbound_extensions.fastgres import FastgresOptimizer

# pg, encoding_workload, training_workload, and query are PostBOUND objects.
optimizer = FastgresOptimizer(database=pg)
optimizer.prepare(encoding_workload, encodings="postbound_extensions/fastgres/generated/imdb")
optimizer.train(training_workload, labels="postbound_extensions/fastgres/generated/imdb/labels/label.csv")
hints = optimizer.predict(query)  # HintSet: inspect dict(hints) or hints.int_rep
plan = optimizer.optimize_query(query)

catalog = optimizer.store("models/fastgres")  # New or empty directory
restored = FastgresOptimizer.pre_trained(catalog)  # Prediction needs no label CSV or database
```

Preparation collects statistics only when cached encodings are omitted. Labeling
is a separate, explicit call to `optimizer.label(workload)`; its returned DataFrame
can be passed to `train`. `load_or_build` loads an existing bundle or prepares,
trains, and stores one using supplied labels. Bundles include the encodings,
contexts, models, and ordered hint definitions.

The individual stages are available in [labeling](labeling/),
[featurization](featurization/), [context](context/), and [model](model/).

## References

- **FASTgres:** Woltmann, L.; Thiessat, J.; Hartmann, C.; Habich, D.; Lehner, W.
  *FASTgres: Making Learned Query Optimizer Hinting Effective.*
  Proceedings of the VLDB Endowment, 16(11), 3310-3322, 2023.
  [Paper](https://doi.org/10.14778/3611479.3611528) |
  [Original implementation](https://github.com/db-tu-dresden/FASTgres-PVLDBv16)
- **FASTlabel:** Thiessat, K.; Habich, D.; Lehner, W.
  *FASTlabel: Making Supervised Query Optimizer Hinting Practical.*
  BTW 2025, 241-264.
  [Paper](https://doi.org/10.18420/BTW2025-11) |
  [Original implementation](https://github.com/db-tu-dresden/BTW25-FASTLabel)

When using this adaptation in research, please cite both underlying approaches
and identify the version of this repository used in your experiments.

## Attribution and license

This adaptation is contributed by Kira Thiessat as part of the
[Optimizer Playground](https://github.com/Optimizer-Playground/Optimization-Techniques).
See the repository [license](../../LICENSE) for licensing terms.
