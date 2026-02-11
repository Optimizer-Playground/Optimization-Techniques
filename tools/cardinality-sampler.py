import argparse
import atexit
import functools
import signal
from pathlib import Path

import pandas as pd
import postbound as pb

from postbound_extensions import sampler

description = """
Generate random queries and determine their output cardinalities.

The sampler creates queries by selecting a random subset of connected tables from the database schema.
For this subset, filter columns, filter predicates, and filter values are selected at random.
The connectivity is inferred based on the primary key/foreign key relationships in the database schema.

Each sampled query is executed and its query plan (EXPLAIN ANALYZE) and output cardinality are determined.
Results are streamed to a CSV file.

Currently, the sampler is only implemented for PostgreSQL.
"""


def signal_handler(signum, frame, *, sampling_ctl: sampler.CardinalitySampler):
    sampling_ctl.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=description, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--n-queries",
        "-n",
        type=int,
        default=1000,
        help="Number of queries to generate. If --infinite is given, this is interpreted as the batch size. "
        "Default: 1000",
    )
    parser.add_argument(
        "--infinite", action="store_true", help="Generate samples until interrupted."
    )
    parser.add_argument(
        "--min-tables",
        type=int,
        default=2,
        help="Generate queries that join at least this many base tables. Default: 2",
    )
    parser.add_argument(
        "--max-tables",
        type=int,
        default=5,
        help="Generate queries that join at least this many base tables. Default: 5",
    )
    parser.add_argument(
        "--min-filters",
        type=int,
        default=1,
        help="Generate queries with at least this many filter predicates. Filter columns are selected randomly with replacement. "
        "Default: 1",
    )
    parser.add_argument(
        "--max-filters",
        type=int,
        default=5,
        help="Generate queries with at most this many filter predicates. Default: 5",
    )
    parser.add_argument(
        "--jobs",
        "-j",
        type=int,
        default=1,
        help="Generate samples in parallel with --jobs many workers. Note that this affects the runtime measurements of the execution plans. "
        "Default: 1 (sequential execution)",
    )
    parser.add_argument(
        "--pg-connect",
        "-c",
        type=str,
        default="",
        help="Configuration file that provides the Postgres connection parameters. "
        "If this parameter is omitted, the connection parameters must be provided via --connect-string.",
    )
    parser.add_argument(
        "--connect-string",
        type=str,
        default="",
        help="Raw Postgres connection string. "
        "If this parameter is omitted, the connection parameters must be provided via --pg-connect.",
    )
    parser.add_argument(
        "--timeout",
        "-t",
        type=int,
        help="Cancel queries that run for longer than --timeout (full) seconds. By default, queries run until completion.",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Print log messages."
    )
    parser.add_argument("out", type=Path, help="Output file to store the results.")

    args = parser.parse_args()

    out_file: Path = args.out
    logger = pb.util.standard_logger(args.verbose)
    n_workers = args.jobs
    if args.connect_string:
        pg_instance = pb.postgres.connect(connect_string=args.connect_string)
    elif args.pg_connect:
        pg_instance = pb.postgres.connect(config_file=args.pg_connect)
    else:
        parser.error("Either --pg-connect or --connect-string must be given.")

    logger("Initializing query generator")
    query_gen = sampler.generate_query(
        pg_instance,
        count_star=False,
        min_tables=args.min_tables,
        max_tables=args.max_tables,
        min_filters=args.min_filters,
        max_filters=args.max_filters,
    )

    logger("Initializing sampler with", n_workers, "workers")
    timeout = args.timeout * 1000 if args.timeout else None
    sampler_ctl = sampler.CardinalitySampler(
        query_gen,
        connect_string=pg_instance.connect_string,
        timeout_ms=timeout,
        n_workers=n_workers,
        verbose=args.verbose,
    )

    out_file.parent.mkdir(parents=True, exist_ok=True)
    if not out_file.is_file():
        df = pd.DataFrame([], columns=sampler.CardinalitySample.csv_cols())  # type: ignore[call-arg]
        df.to_csv(out_file, mode="w", index=False, header=True)

    ctl_c_handler = functools.partial(signal_handler, sampling_ctl=sampler_ctl)
    signal.signal(signal.SIGINT, ctl_c_handler)
    signal.signal(signal.SIGKILL, ctl_c_handler)
    atexit.register(sampler_ctl.shutdown)

    if args.infinite:
        logger("Starting initial sampling run")
    else:
        logger("Starting sampling run")

    sampler_ctl.sample(args.n_queries, stream_to=out_file)
    while args.infinite:
        logger("Starting next sampling run")
        sampler_ctl.sample(args.n_queries, stream_to=out_file)


if __name__ == "__main__":
    main()
