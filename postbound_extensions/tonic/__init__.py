"""TONIC is a learned selection algorithm for join operators.

For a specific (query, join order) pair TONIC chooses the optimal join operators based on previous experience. At the core of
TONIC is a plan synopsis which compactly encodes the different query plans and their performance. This synopsis is a structural
representation that does not need any machine learning.

TONIC relies on an external component to provide the join order. It plays particularly well with UES and circumvents one of its
central limitations. We provide a faithful implementation of TONIC based on the original paper. All parts of the algorithm are
implemented from scratch using the PostBOUND API.

High-level Interface
--------------------

The `TonicOperatorSelection` class acts as the central entry point for the TONIC algorithm. It implements the
`PhysicalOperatorSelection` interface and should be used for most purposes. To train the underlying plan synopsis, call
`integrate_cost` with past query plans. Similarly, `simulate_feedback` can be used to execute training queries and integrate
their observed cost into the model.
Lastly, `explore_costs` is suitable for situations where you want to integrate all possible operator assignments for a specific
join order of a query.
Training queries can be obtained by using the `query-generator.py` tool that is shipped as part of the Optimization Techniques
project. See its documentation for details.

In addition to the selection algorithm, `QepSynopsis` stores the past experience. It is constructed of individual `QepsNode`s,
which represent joins in the different query plans. Under normal conditions, users of do not need to interact with these
classes directly.

Storing a Synopsis
------------------

Once a synopsis has received enough feedback, it can be persisted to disk using the `store()` method on the operator selection
instance. This allows to re-load the entire optimizer state at a later point using `TonicOperatorSelection.load_model()`.

The generally recommended pattern is to use `TonicOperatorSelection.load_or_build()`. It checks, whether a synopsis has already
been persisted at the given location. If it has, it is simply re-loaded. Otherwise, a new synopsis is created. It can
optionally be pre-trained using a set of sample queries or sample plans.

Example
-------

.. code-block:: python

    #
    # Step 1: generic setup
    #
    # As usual for PostBOUND, connect to our target database and load the workload
    import postbound as pb
    import postbound_extensions as pbx

    pg_instance = pb.postgres.connect(config_file=".pg-connect-stats.toml")
    workload = pb.workloads.stats()

    #
    # Step 2: load training samples for the TONIC synopsis
    #
    raw_samples = pb.util.read_df("tonic-samples.parquet")
    tonic_samples = raw_samples.to_dict("list")

    #
    # Step 3: create a UES/TONIC optimization pipeline
    #

    ues = pbx.ues.UesJoinOrdering(database=pg_instance, estimations="precise")
    tonic = pbx.tonic.TonicOperatorSelection.load_or_build(
        "models/tonic/stats",
        filter_aware=True,
        gamma=0.8,
        sample_plans=tonic_samples,
        database=pg_instance
    )
    pipeline = (
        pb.MultiStageOptimizationPipeline(pg_instance)
        .use(ues)
        .use(tonic)
        .build()
    )


    #
    # Step 4: Evaluation
    #
    # We can use the plain TONIC algorithm to obtain the join operators for a single query
    query = workload["q-10"]
    join_order = ues.optimize_join_order(query)
    print(tonic.select_physical_operators(query, join_order))

    # Or we can run the PostBOUND benchmarking utilities
    pb.bench.execute_workload(
        workload,
        on=pipeline,
        query_preparation={"analyze": True, "prewarm": True},
        name="tonic-test",
        progressive_output="tonic-benchmark.parquet",
        logger="tqdm"
    )

Deviations from the Original Paper
----------------------------------

Our implementation of TONIC is a faithful adaptation of the original paper. We are not aware of any deviations.
The only part of TONIC that we currently do not support is flexibility in the choice of target metric.
The QEP always uses the estimated cost from the query plan. However, PostBOUND provides the means to update
these cost estimates to arbitrary values. Therefore, this is not a fundamental limitation of our implementation,
just a shift of responsibility to PostBOUND.

Authors
-------

This implementation of UES has been contributed by Rico Bergmann. It is provided as part of
the `Optimizer Collection project <https://github.com/Optimizer-Playground/Optimization-Techniques>`__.

References
----------

.. Axel Hertzschuch et al.: Turbo-Charging SPJ Query Plans with Learned Physical Join Operator Selections (PVLDB 2022);
   https://doi.org/10.14778/3551793.3551825
.. Reference implementation: https://github.com/axhertz/TONIC

License
-------

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

from ._tonic import (
    QepsIdentifier,
    QepsNode,
    QepSynopsis,
    TonicOperatorSelection,
    make_qeps,
)

__all__ = [
    "QepsIdentifier",
    "QepsNode",
    "QepSynopsis",
    "make_qeps",
    "TonicOperatorSelection",
]
