import argparse
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Literal, get_args

import pandas as pd
import postbound as pb
from tqdm import tqdm

import postbound_extensions as pbx

SupportedWorkloads = Literal["job", "job-light", "job-complex", "stats", "stack", "custom"]
SupportedEstimators = Literal["true-cards", "safebound", "mscn", "custom"]


def collect_subqueries(workload: pb.Workload, *, estimator: pb.CardinalityEstimator, verbose: bool) -> set[pb.SqlQuery]:
    subqueries: set[pb.SqlQuery] = set()
    query_iter = tqdm(workload.queries()) if verbose else workload.queries()
    for query in query_iter:
        for intermediate in estimator.generate_intermediates(query):
            subquery = pb.transform.extract_subquery(query, intermediate)
            subqueries.add(subquery)
    return subqueries


def estimate_cardinalities(
    queries: Iterable[pb.SqlQuery], *, estimator: pb.CardinalityEstimator, verbose: bool
) -> Mapping[pb.SqlQuery, pb.Cardinality]:
    query_iter = tqdm(queries) if verbose else queries
    cardinalities: dict[pb.SqlQuery, pb.Cardinality] = {}

    for query in query_iter:
        try:
            card = estimator.calculate_estimate(query, query.tables())
        except Exception as e:
            if isinstance(e, InterruptedError) or str(e) == "Query interrupted":
                # DuckDB does not raise an interrupted error but instead a generic exception with interrupted
                # in the text. What is even weirder, is that execution usually continues with the next query instead
                # of just cancelling the entire workload.
                #
                # We fix the first issue by checking the exception text.
                # The second issue we can actually use to our advantage: by just breaking the loop, we can stop the
                # execution and still export the cardinalities we have collected so far.
                break
            card = pb.Cardinality.unknown()

        cardinalities[query] = card

    return cardinalities


def main() -> None:
    parser = argparse.ArgumentParser(description="Stores cardinalities offline for later re-use.")
    parser.add_argument("--estimator", "-e", choices=get_args(SupportedEstimators), default="true-cards")
    parser.add_argument("--workload", "-w", choices=get_args(SupportedWorkloads), default="")
    parser.add_argument("--workload-path", type=Path, required=False, default=None)
    parser.add_argument("--estimator-path", type=Path, required=False, default=None)
    parser.add_argument("--duckdb", type=Path)
    parser.add_argument("--timeout", type=float, required=False, default=None)
    parser.add_argument("--output", "-o", type=Path, required=True)
    parser.add_argument("--continue", action="store_true", dest="filter_existing")
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()
    logger = pb.util.standard_logger(args.verbose)
    output: Path = args.output

    logger("Connecting to database")
    duck_instance = pb.duckdb.connect(args.duckdb)

    logger("Loading workload")
    match args.workload:
        case "job":
            workload = pb.workloads.job()
        case "job-light":
            workload = pb.workloads.job_light()
        case "job-complex":
            workload = pb.workloads.job_complex()
        case "stats":
            workload = pb.workloads.stats()
        case "stack":
            workload = pb.workloads.stack()
        case "custom":
            workload = pb.workloads.read_workload(args.workload_path)
        case _:
            parser.error(f"Unsupported workload: {args.workload}")

    logger("Creating estimator")
    match args.estimator:
        case "true-cards":
            estimator = pb.opt.PerfectCardinalities(duck_instance, timeout=args.timeout)
        case "safebound":
            cat = pbx.safebound.SafeBoundCatalog.load(args.estimator_path, database=duck_instance, verbose=args.verbose)
            estimator = pbx.safebound.SafeBoundEstimator(cat)
        case "mscn":
            estimator = pbx.mscn.MscnEstimator.pre_trained(
                args.estimator_path, database=duck_instance, verbose=args.verbose
            )
        case "custom":
            estimator = pbx.meta.load_cardinality_estimator(args.estimator_path)
        case _:
            parser.error(f"Unsupported estimator: {args.estimator}")

    logger("Collecting all intermediates to estimate")
    subqueries = collect_subqueries(workload, estimator=estimator, verbose=args.verbose)

    if args.filter_existing and output.is_file():
        logger("Reading existing estimates")
        df = pb.util.read_df(output)
        existing = df[~df["cardinality"].isna()]
        existing_subqueries = {pb.parse_query(query) for query in existing["query"]}
        subqueries -= existing_subqueries
    else:
        df = pd.DataFrame({"query": [], "cardinality": []})

    logger("Found", len(subqueries), "intermediates to estimate")

    logger("Estimating cardinalities")
    cardinalities = estimate_cardinalities(subqueries, estimator=estimator, verbose=args.verbose)

    logger("Exporting results")
    queries: list[str] = []
    cards: list[float] = []
    for subquery, card in cardinalities.items():
        queries.append(str(subquery))
        cards.append(float(card))

    df = pd.concat([df, pd.DataFrame({"query": queries, "cardinality": cards})], ignore_index=True)
    pb.util.write_df(df, path=args.output)


if __name__ == "__main__":
    main()
