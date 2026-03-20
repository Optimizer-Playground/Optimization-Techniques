# SafeBound

- **Original Paper:** _SafeBound: A Practical System for Generating Cardinality Bounds_
- **Authors:** Kyle Deeds, Dan Suciu, Magdalena Balazinska
- **Published:** SIGMOD 2023
- **Link:** https://doi.org/10.1145/3588907

SafeBound is an upper bound-based cardinality estimator (i.e. a "pessimistic optimizer") that uses
degree sequences on join columns to calculate upper bounds on the size of intermediate results.
To keep estimation and storage overhead manageable, SafeBound compresses degree sequences to short
piecewise constant functions. It supports conditioning the degree sequences on join columns by
pre-computing filter predicates.

We implement SafeBound completely from scratch based on the original paper. Under the hood we use
numpy to accelerate computations.


## High-level Interface

SafeBound implements the PostBOUND [`CardinalityEstimator`](https://postbound.readthedocs.io/en/latest/api/core/optimization-pipelines.html#postbound.CardinalityEstimator)
interface. The `SafeBoundEstimator` acts as the central entrypoint. It requires a
`SafeBoundCatalog` to function correctly. This catalog stores all individual (conditioned and
unconditioned) piecewise constant functions. A catalog can be created either based on a workload,
or inferred completely from the database. See its documentation for more details.

In addition to catalog and estimator, the following functions capture important concepts of the
original paper:

- `DegreeSequence` and `PiecewiseConstantFunction` model the core data structures.
- `decompose_acyclic()` implements the query decomposition logic (see below).
- `valid_compress()` implements the compression algorithm from the paper to transform a degree
  sequence into a piecewise constant function.
- `fdsb()` computes the upper bound for a query, given its decomposition as well as the relevant
  PCFs (typically retrieved from the catalog).


## Storing SafeBound Catalogs

Since computing an entire SafeBound catalog from scratch involves issuing a large number of
(somewhat complex) SQL queries, it is quite time consuming and we do not recommend doing it online.
Instead, an existing catalog can be persisted as a JSON object using the `store()` method.
Afterwards, the `SafeBoundCatalog.load()` method allows to re-create the catalog.

The generally recommended pattern is to use `SafeBoundCatalog.load_or_build()`. It checks, whether
a catalog has been persisted at the given location. If it has, it will be loaded. Otherwise, the
catalog is created and stored.


## Deviations from the Original Paper

The SafeBound implementation contains a faithful adaptation of the following features:

- Compression of degree sequences to piecewise constant functions
- Computation of the upper bound using alpha steps and beta steps
- Correlated degree sequences for equality predicates based on most-common values
- Correlated degree sequences for range predicates using hierarchies of histograms
- Correlated degree sequences for LIKE predicates based on most-common values

To get a running implementation of SafeBound, we developed our own query decomposition algorithm,
since the original paper did not contain any information on how to do that. See the documentation
of `decompose_acyclic` for a detailed rundown of the algorithm.

At the same time, we currently do not support the following:

- Variable MCV sizes and histogram depths
- Optimizations (Section 4 of the paper), including
  - Degree sequences on pre-computed primary key/foreign key joins
  - Group compression of PCFs
  - Bloom filters on MCV lists

While group compression and bloom filtering "only" improve the resource consumption of SafeBound,
pre-computed primary key/foreign key joins also increase the accuracy. We do not implement the first
two because minimizing the resource consumption is currently not a primary goal of our
implementation. Materializing primary key/foreign key joins is not implemented because this
essentially constitutes a materialized view, which is an orthogonal optimization which is best
treated at the schema level.


## Authors and License

This implementation of SafeBound has been contributed by Rico Bergmann. It is provided as part of
the [Optimizer Collection project](https://github.com/Optimizer-Playground/Optimization-Techniques)
and licensed under the same conditions.


## References

Kyle Deeds et al.: _SafeBound: A Practical System for Generating Cardinality Bounds_ (SIGMOD 2023);
DOI [10.1145/3588907](https://doi.org/10.1145/3588907)
