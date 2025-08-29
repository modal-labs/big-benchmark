import logging
from pathlib import Path

import modal
from stopwatch.benchmark import GuideLLM
from stopwatch.constants import DB_PATH, GUIDELLM_VERSION, HOURS, LLMServerType
from stopwatch.resources import app, db_volume, results_volume

from .web import export_results

DATASETTE_PATH = "/datasette"
MAX_CONCURRENT_BENCHMARKS = 45
MAX_CONSTANT_RATES = 10
MIN_QPS_STEP_SIZE = 0.5
RESULTS_PATH = "/results"
TIMEOUT = 24 * HOURS  # 1 day

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


benchmark_suite_image = (
    modal.Image.debian_slim(python_version="3.13")
    .apt_install("git")
    .uv_pip_install(
        "fastapi[standard]",
        "pandas",
        "SQLAlchemy",
        "git+https://github.com/modal-labs/stopwatch.git#a38f9ec",
    )
)

with benchmark_suite_image.imports():
    import asyncio
    import json
    import traceback
    import uuid
    import warnings
    from typing import Any

    from stopwatch.constants import RateType

    from .db import Benchmark, benchmark_class_factory, create_all, engine, session


def find_function_call(
    call_graph: list[modal.call_graph.InputInfo],
    function_call_id: str,
) -> modal.call_graph.InputInfo | None:
    """
    Return the input to a function call with a specific ID out of all function call
    inputs in a call graph.

    :param: call_graph: The call graph to search.
    :param: function_call_id: The ID of the function call to find.
    :return: The input to the function call with the given ID, or None if no such
        function call is found.
    """

    for fc_input in call_graph:
        if fc_input.function_call_id == function_call_id:
            return fc_input

        if found_input := find_function_call(fc_input.children, function_call_id):
            return found_input

    return None


async def run_benchmark(
    config: dict[str, Any],
    server_url: str,
    server_id: str,
    semaphore: asyncio.Semaphore,
) -> None:
    """
    Run a benchmark or set of benchmarks, wait for the result(s), and save the
    result(s) to the database.

    :param: config: The benchmark configuration.
    :param: server_url: The URL of the LLM server to use.
    :param: server_id: The ID of the LLM server to use.
    :param: semaphore: A semaphore to limit the number of benchmarks that can run
        concurrently.
    """

    async with semaphore:
        run_benchmark_kwargs = {
            "endpoint": f"{server_url}/v1",
            "model": config["model"],
            "rate_type": RateType.sweep.value,
            "data": config["data"],
            "client_config": config["client_config"],
            "server_id": server_id,
        }

        logger.info("Starting benchmarks with kwargs %s", run_benchmark_kwargs)

        # Run the benchmark
        guidellm = GuideLLM.with_options(region=config.get("client_region"))
        fc = guidellm().run_benchmark.spawn(**run_benchmark_kwargs)

        try:
            results = await fc.get.aio()
        except modal.exception.RemoteError as e:
            # Happens when the function call is interrupted manually
            warnings.warn(
                f"WARNING: Function call result could not be retrieved: {e}",
                stacklevel=2,
            )
            return
        except modal.exception.FunctionTimeoutError:
            warnings.warn("WARNING: Benchmark timed out", stacklevel=2)
            return
        except Exception:  # noqa: BLE001
            warnings.warn(
                "WARNING: Unexpected error when running benchmark",
                stacklevel=2,
            )
            traceback.print_exc()
            return

        logger.info("Saving results for %s", fc.object_id)

        with (Path(RESULTS_PATH) / f"{fc.object_id}.json").open("w") as f:
            # The full results are saved to disk since they are too big to fit in the
            # database (~20MB per benchmark run)
            json.dump(results, f)

        for result in results:
            benchmark_record = Benchmark(
                **config,
                function_call_id=fc.object_id,
                rate_type=result["args"]["strategy"]["type_"],
                rate=result["args"].get("rate"),
            )
            session.add(benchmark_record)
            benchmark_record.save_results(result)

        session.commit()
        db_volume.commit()


async def run_benchmarks_in_parallel(
    benchmark_configs: list[tuple[dict[str, Any], str, str]],
) -> None:
    """
    Given a list of benchmark configurations, identify which benchmarks need to be run,
    run them in parallel, and wait until all benchmarks have completed.

    :param: benchmark_configs: The benchmark configurations to run.
    """

    semaphore = asyncio.Semaphore(MAX_CONCURRENT_BENCHMARKS)
    tasks = []

    for config, server_url, server_id in benchmark_configs:
        successful_benchmark_count = (
            session.query(Benchmark)
            .filter_by(
                **{k: v for k, v in config.items() if k != "group_id"},
            )
            .count()
        )

        if successful_benchmark_count > 0:
            # This benchmark has already completed successfully
            continue

        task = asyncio.create_task(
            run_benchmark(config, server_url, server_id, semaphore),
        )
        tasks.append(task)

        # Yield control to the event loop to allow the task to be scheduled
        await asyncio.sleep(0.01)

    if len(tasks) == 0:
        return

    await asyncio.gather(*tasks)


@app.function(
    image=benchmark_suite_image,
    volumes={
        DB_PATH: db_volume,
        RESULTS_PATH: results_volume,
    },
    cpu=2,
    memory=1 * 1024,
    max_containers=1,
    scaledown_window=2,
    timeout=TIMEOUT,
)
@modal.concurrent(max_inputs=1)
async def run_benchmark_suite(
    benchmarks: list[tuple[dict[str, Any], str, str]],
    suite_id: str,
) -> None:
    """
    Run a suite of benchmarks.

    :param: benchmarks: A list of benchmarks to run. Each item in this list is a tuple
        with three items: the benchmark configuration, the URL of the LLM server to
        use, and the ID of the LLM server to use.
    :param: suite_id: The ID of the benchmark suite.
    """

    logger.info(
        "Running %d configs with benchmark suite id %s",
        len(benchmarks),
        suite_id,
    )

    db_volume.reload()

    create_all()

    # Validate benchmarks
    logger.info("Validating benchmarks...")

    for benchmark_config, _, _ in benchmarks:
        for key in ["llm_server_type", "model", "data", "gpu"]:
            if benchmark_config.get(key) is None:
                msg = f"Benchmark {benchmark_config} has no {key}"
                raise Exception(msg)

        if "llm_server_config" not in benchmark_config:
            benchmark_config["llm_server_config"] = {}

        if "client_config" not in benchmark_config:
            benchmark_config["client_config"] = {}

        benchmark_config["group_id"] = uuid.uuid4().hex[:8]
        benchmark_config["version_metadata"] = {
            "guidellm": GUIDELLM_VERSION,
            benchmark_config["llm_server_type"]: benchmark_config[
                "llm_server_config"
            ].get(
                "version",
                LLMServerType(benchmark_config["llm_server_type"]).get_version(),
            ),
        }

    # Run benchmarks
    logger.info("Running benchmarks...")
    await run_benchmarks_in_parallel(benchmarks)

    # Save results from this benchmark suite to their own table
    SuiteBenchmark = benchmark_class_factory(  # noqa: N806
        table_name=suite_id.replace("-", "_"),
    )

    SuiteBenchmark.__table__.drop(engine, checkfirst=True)
    SuiteBenchmark.__table__.create(engine)

    non_pk_columns = [
        k
        for k in Benchmark.__table__.columns.keys()  # noqa: SIM118
        if k not in Benchmark.__table__.primary_key.columns.keys()  # noqa: SIM118
    ]

    for config, _, _ in benchmarks:
        benchmark_records = (
            session.query(Benchmark)
            .filter_by(**{k: v for k, v in config.items() if k != "group_id"})
            .all()
        )

        for benchmark_record in benchmark_records:
            session.add(
                SuiteBenchmark(
                    **{c: getattr(benchmark_record, c) for c in non_pk_columns},
                ),
            )

    session.commit()
    db_volume.commit()

    # Export results in frontend format
    logger.info("Exporting results to frontend format...")
    export_results.local(SuiteBenchmark)
