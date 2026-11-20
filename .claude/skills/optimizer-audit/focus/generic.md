# Focus: generic optimizer component

For optimizers that do not fit one of the other families (e.g. join ordering heuristics, operator
selection). When drafting the scope file, propose a new family checklist if the optimizer is likely
to be audited again.

## Checklist

- **Output definition:** what exactly does the optimizer produce (join order, operator per join,
  hints, estimates), and which output differences matter most? Put this in the scope's
  "critical outcome".
- **Inputs:** statistics, estimates or learned state it consumes; their source, units and
  freshness; consistency between the moment they are built and the moment they are used.
- **Algorithm:** step-by-step comparison with the paper's pseudocode and the reference's code path;
  every formula checked on a worked example.
- **Learned or fitted components:** training data, transforms and their inverses, hyperparameters
  against the paper.
- **Entry points and defaults:** factory methods, default parameters, behaviour without prior
  state (cold start).
- **Edge cases:** empty inputs, cyclic queries, self-joins, NULLs, unsupported predicates, timeouts.
- **Persistence:** store → load → identical output.

## Recommended measurements

- The optimizer's output on the target workload, as shipped and with each fix, against the
  reference on the same data.
- A metric of output agreement (e.g. same plan / same operator per query) plus a metric of output
  quality (runtime, q-error), with seeds or repetitions to establish the noise floor.
