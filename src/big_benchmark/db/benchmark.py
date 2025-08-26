import itertools
from typing import Any

from sqlalchemy import JSON, Column, DateTime, Float, Integer, String
from sqlalchemy.sql import func

from .base import Base


def benchmark_class_factory(table_name: str = "benchmarks") -> type:
    """
    Create a benchmark class that can be used by SQLAlchemy to store benchmark results.

    :param: table_name: The name of this benchmark's database table.
    :return: A benchmark class.
    """

    class Benchmark(Base):
        __tablename__ = table_name

        # Metadata
        id = Column(Integer, primary_key=True)
        created_at = Column(DateTime, default=func.now())
        function_call_id = Column(String, nullable=False)
        group_id = Column(String, nullable=False)

        # Parameters
        llm_server_type = Column(String, nullable=False)
        llm_server_config = Column(JSON, nullable=False)
        client_config = Column(JSON, nullable=False)
        model = Column(String, nullable=False)
        rate_type = Column(String, nullable=False)
        rate = Column(Float)
        data = Column(String, nullable=False)
        gpu = Column(String, nullable=False)
        server_region = Column(String)
        client_region = Column(String)
        version_metadata = Column(JSON)

        # Data
        output_tokens = Column(Integer)
        output_tokens_variance = Column(Integer)
        prompt_tokens = Column(Integer)
        prompt_tokens_variance = Column(Integer)

        # Results
        start_time = Column(Float)
        end_time = Column(Float)
        duration = Column(Float)
        queue_duration = Column(Float)
        cold_start_duration = Column(Float)
        completed_request_count = Column(Integer)
        completed_request_rate = Column(Float)

        itl_mean = Column(Float)
        itl_p50 = Column(Float)
        itl_p90 = Column(Float)
        itl_p95 = Column(Float)
        itl_p99 = Column(Float)
        ttft_mean = Column(Float)
        ttft_p50 = Column(Float)
        ttft_p90 = Column(Float)
        ttft_p95 = Column(Float)
        ttft_p99 = Column(Float)
        ttlt_mean = Column(Float)
        ttlt_p50 = Column(Float)
        ttlt_p90 = Column(Float)
        ttlt_p95 = Column(Float)
        ttlt_p99 = Column(Float)

        def get_config(self) -> dict[str, Any]:
            """
            Get the input parameters of this benchmark.

            :return: A dictionary of the benchmark's configuration.
            """

            return {
                "llm_server_type": self.llm_server_type,
                "llm_server_config": self.llm_server_config,
                "client_config": self.client_config,
                "model": self.model,
                "rate_type": self.rate_type,
                "rate": self.rate,
                "data": self.data,
                "gpu": self.gpu,
                "server_region": self.server_region,
                "client_region": self.client_region,
                "version_metadata": self.version_metadata,
            }

        def save_results(self, results: dict[str, Any]) -> None:
            requests = results["requests"]["successful"]

            if len(requests) > 0:
                self.start_time = requests[0]["start_time"]
                self.end_time = requests[-1]["end_time"]
            else:
                self.start_time = results["run_stats"]["start_time"]
                self.end_time = results["run_stats"]["end_time"]

            self.duration = self.end_time - self.start_time
            self.cold_start_duration = results["cold_start_duration"]
            self.queue_duration = results["queue_duration"]
            self.completed_request_count = len(requests)
            self.completed_request_rate = self.completed_request_count / self.duration

            data_config = {
                k: int(v)
                for param in self.data.split(",")
                for k, v in [param.split("=")]
            }

            self.prompt_tokens = data_config.get("prompt_tokens", 0)
            self.prompt_tokens_variance = data_config.get("prompt_tokens_variance", 0)
            self.output_tokens = data_config.get("output_tokens", 0)
            self.output_tokens_variance = data_config.get("output_tokens_variance", 0)

            for (db_key, result_key), (
                db_statistic_key,
                statistic_key,
            ) in itertools.product(
                zip(
                    ["itl", "ttft", "ttlt"],
                    [
                        "inter_token_latency_ms",
                        "time_to_first_token_ms",
                        "request_latency",
                    ],
                    strict=False,
                ),
                zip(
                    ["mean", "p50", "p90", "p95", "p99"],
                    ["mean", "median", "p90", "p95", "p99"],
                    strict=False,
                ),
            ):
                if statistic_key.startswith("p"):
                    value = results["metrics"][result_key]["successful"]["percentiles"][
                        statistic_key
                    ]
                else:
                    value = results["metrics"][result_key]["successful"]["mean"]

                if result_key.endswith("_ms"):
                    value /= 1000

                setattr(self, f"{db_key}_{db_statistic_key}", value)

    return Benchmark


Benchmark = benchmark_class_factory()
