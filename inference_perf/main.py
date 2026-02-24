# Copyright 2025 The Kubernetes Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from argparse import ArgumentParser
from inference_perf.analysis.analyze import analyze_reports
from typing import List, Optional
from inference_perf.client.modelserver.tgi_client import TGImodelServerClient
from inference_perf.loadgen import LoadGenerator, AgenticLoadGenerator
from inference_perf.config import (
    DataGenType,
    LoadType,
    MetricsClientType,
    ModelServerType,
    ReportConfig,
    StandardLoadStage,
    ConcurrentLoadStage,
    read_config,
)
from inference_perf.datagen import (
    DataGenerator,
    MockDataGenerator,
    HFShareGPTDataGenerator,
    SyntheticDataGenerator,
    RandomDataGenerator,
    SharedPrefixDataGenerator,
    CNNDailyMailDataGenerator,
    InfinityInstructDataGenerator,
    BillsumConversationsDataGenerator,
    Tau2BenchDataGenerator,
    LangfuseDataGenerator,
    OpenTelemetryDataGenerator,
    AgenticSyntheticDataGenerator,
    AgenticCsvDataGenerator,
)
from inference_perf.client.modelserver import (
    ModelServerClient,
    vLLMModelServerClient,
    SGlangModelServerClient,
    MockModelServerClient,
)
from inference_perf.client.metricsclient.base import MetricsClient, PerfRuntimeParameters
from inference_perf.client.metricsclient.prometheus_client import PrometheusMetricsClient, GoogleManagedPrometheusMetricsClient
from inference_perf.client.filestorage import (
    StorageClient,
    GoogleCloudStorageClient,
    LocalStorageClient,
    SimpleStorageServiceClient,
)
from inference_perf.client.requestdatacollector import (
    RequestDataCollector,
    LocalRequestDataCollector,
    MultiprocessRequestDataCollector,
)
from inference_perf.circuit_breaker import init_circuit_breakers
from inference_perf.reportgen import ReportGenerator
from inference_perf.utils import CustomTokenizer, ReportFile
from inference_perf.logger import setup_logging
import asyncio
import time


class InferencePerfRunner:
    def __init__(
        self,
        client: ModelServerClient,
        loadgen: LoadGenerator,
        reportgen: ReportGenerator,
        storage_clients: List[StorageClient],
    ) -> None:
        self.client = client
        self.loadgen = loadgen
        self.reportgen = reportgen
        self.storage_clients = storage_clients

    def run(self) -> None:
        async def _run() -> None:
            # Start the collector, which will gather metrics from the model_server_client
            async with self.reportgen.get_metrics_collector().start():
                # Generate load that is sent to inference endpoint
                await self.loadgen.run(self.client)

        asyncio.run(_run())

    def generate_reports(self, report_config: ReportConfig, runtime_parameters: PerfRuntimeParameters) -> List[ReportFile]:
        return asyncio.run(self.reportgen.generate_reports(report_config=report_config, runtime_parameters=runtime_parameters))

    def save_reports(self, reports: List[ReportFile]) -> None:
        for storage_client in self.storage_clients:
            storage_client.save_report(reports)

    def stop(self) -> None:
        asyncio.run(self.loadgen.stop())


def main_cli() -> None:
    # Parse command line arguments
    parser = ArgumentParser()
    parser.add_argument("-c", "--config_file", help="Config File", required=False)
    parser.add_argument("-a", "--analyze", nargs="*", help="Path to a report directory to analyze.", required=False)
    parser.add_argument(
        "--log-level", help="Logging level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    )
    args = parser.parse_args()

    setup_logging(args.log_level)

    if args.analyze and len(args.analyze) > 0:
        analyze_reports(args.analyze)
        return

    if not args.config_file:
        parser.error("argument -c/--config_file is required when not using --analyze")

    config = read_config(args.config_file)

    # Set stage rates to high values if using concurrent load type
    if config.load.type == LoadType.CONCURRENT:
        # The validation is now handled by Pydantic in the config classes
        # Convert ConcurrentLoadStage to have rate/duration for load generation
        for stage in config.load.stages:
            if isinstance(stage, ConcurrentLoadStage):
                # Generate all of the requests in the span of a second (enqueuing everything) to saturate workers
                stage.duration = 1
                stage.rate = stage.num_requests
        # Set to 0 to show that worker_max_concurrency was not relevant in concurrent load type
        config.load.worker_max_concurrency = 0
    # Note: StandardLoadStage validation is automatically handled by Pydantic

    # Define Circuit Breakers
    if config.circuit_breakers:
        init_circuit_breakers(config.circuit_breakers)

    # Define Metrics Client
    metrics_client: Optional[MetricsClient] = None
    if config.metrics:
        if config.metrics.type == MetricsClientType.PROMETHEUS and config.metrics.prometheus:
            if config.metrics.prometheus.google_managed:
                metrics_client = GoogleManagedPrometheusMetricsClient(config.metrics.prometheus)
            else:
                metrics_client = PrometheusMetricsClient(config=config.metrics.prometheus)

    # Define Storage Clients
    storage_clients: List[StorageClient] = []
    if config.storage:
        if config.storage.local_storage:
            storage_clients.append(LocalStorageClient(config=config.storage.local_storage))
        if config.storage.google_cloud_storage:
            storage_clients.append(GoogleCloudStorageClient(config=config.storage.google_cloud_storage))
        if config.storage.simple_storage_service:
            storage_clients.append(SimpleStorageServiceClient(config=config.storage.simple_storage_service))

    # Define Report Generator
    collector: RequestDataCollector
    if config.load.num_workers > 0:
        collector = MultiprocessRequestDataCollector()
    else:
        collector = LocalRequestDataCollector()
    reportgen = ReportGenerator(metrics_client, collector, config=config)

    # Create tokenizer based on tokenizer config
    tokenizer: Optional[CustomTokenizer] = None
    if config.tokenizer and config.tokenizer.pretrained_model_name_or_path:
        try:
            tokenizer = CustomTokenizer(config.tokenizer)
        except Exception as e:
            raise Exception("Tokenizer initialization failed") from e

    # Define Model Server Client
    model_server_client: ModelServerClient
    if config.server:
        if config.server.type == ModelServerType.VLLM:
            model_server_client = vLLMModelServerClient(
                reportgen.get_metrics_collector(),
                api_config=config.api,
                uri=config.server.base_url,
                model_name=config.server.model_name,
                tokenizer_config=config.tokenizer,
                ignore_eos=config.server.ignore_eos,
                max_tcp_connections=config.load.worker_max_tcp_connections,
                additional_filters=config.metrics.prometheus.filters if config.metrics and config.metrics.prometheus else [],
                api_key=config.server.api_key,
                timeout=config.load.request_timeout,
                cert_path=config.server.cert_path,
                key_path=config.server.key_path,
                lora_config=config.load.lora_traffic_split,
            )
            # vllm_client supports inferring the tokenizer
            tokenizer = model_server_client.tokenizer
        if config.server.type == ModelServerType.SGLANG:
            model_server_client = SGlangModelServerClient(
                reportgen.get_metrics_collector(),
                api_config=config.api,
                uri=config.server.base_url,
                model_name=config.server.model_name,
                tokenizer_config=config.tokenizer,
                ignore_eos=config.server.ignore_eos,
                max_tcp_connections=config.load.worker_max_tcp_connections,
                additional_filters=config.metrics.prometheus.filters if config.metrics and config.metrics.prometheus else [],
                api_key=config.server.api_key,
                timeout=config.load.request_timeout,
                lora_config=config.load.lora_traffic_split,
            )
            # sglang_client supports inferring the tokenizer
            tokenizer = model_server_client.tokenizer
        if config.server.type == ModelServerType.TGI:
            model_server_client = TGImodelServerClient(
                reportgen.get_metrics_collector(),
                api_config=config.api,
                uri=config.server.base_url,
                model_name=config.server.model_name,
                tokenizer_config=config.tokenizer,
                ignore_eos=config.server.ignore_eos,
                max_tcp_connections=config.load.worker_max_tcp_connections,
                additional_filters=config.metrics.prometheus.filters if config.metrics and config.metrics.prometheus else [],
                api_key=config.server.api_key,
                timeout=config.load.request_timeout,
                lora_config=config.load.lora_traffic_split,
            )
            # tgi_client supports inferring the tokenizer
            tokenizer = model_server_client.tokenizer
        if config.server.type == ModelServerType.MOCK:
            model_server_client = MockModelServerClient(
                reportgen.get_metrics_collector(),
                api_config=config.api,
                timeout=config.load.request_timeout,
                ignore_eos=config.server.ignore_eos,
                max_completion_tokens=config.server.max_completion_tokens,
                debug_log_enabled=config.server.debug_log_enabled,
                debug_log_file=config.server.debug_log_file,
            )
            tokenizer = model_server_client.tokenizer
    else:
        raise Exception("model server client config missing")

    # Check load exists so datagen can derive total_count from the
    # stage configurations.
    if config.load is None:
        raise Exception("load config missing")

    # Check if this is an agentic load type
    is_agentic = config.load.type in (
        LoadType.AGENTIC,
        LoadType.AGENTIC_CONCURRENT,
        LoadType.AGENTIC_TRACE_REPLAY
    )

    # For agentic loads, check session_arrival.stages; for others, check load.stages
    if is_agentic:
        if config.load.session_arrival is None or len(config.load.session_arrival.stages) == 0:
            if config.load.agentic_sweep is None:
                raise Exception("For agentic loads: session_arrival stages must be configured, or agentic_sweep must be configured")
    else:
        if len(config.load.stages) == 0 and config.load.sweep is None:
            raise Exception("Load stages must be configured, or sweep must be configured")

    # Define DataGenerator
    datagen: DataGenerator
    if config.data:
        # Common checks for generators that require a tokenizer / distribution
        if config.data.type in set(
            {
                DataGenType.ShareGPT,
                DataGenType.Synthetic,
                DataGenType.Random,
                DataGenType.CNNDailyMail,
                DataGenType.InfinityInstruct,
                DataGenType.BillsumConversations,
            }
        ):
            if tokenizer is None:
                raise Exception(
                    f"{config.data.type.value} data generator requires a configured tokenizer. "
                    "Please ensure a valid tokenizer is configured in the 'tokenizer' section of your config file."
                )

        if config.data.type in [DataGenType.Synthetic, DataGenType.Random]:
            if config.data.trace is None:
                if config.data.input_distribution is None:
                    raise Exception(
                        f"{config.data.type.value} data generator requires 'input_distribution' to be configured if no trace config is provided"
                    )
                if config.data.output_distribution is None:
                    raise Exception(
                        f"{config.data.type.value} data generator requires 'output_distribution' to be configured if no trace config is provided"
                    )

                # Calculate total count based on stage type
                max_requests = 0
                for stage in config.load.stages:
                    if isinstance(stage, StandardLoadStage):
                        max_requests = max(max_requests, int(stage.rate * stage.duration))
                    elif isinstance(stage, ConcurrentLoadStage):
                        max_requests = max(max_requests, stage.num_requests)
                total_count = max_requests + 1
                if config.data.input_distribution.total_count is None:
                    config.data.input_distribution.total_count = total_count
                if config.data.output_distribution.total_count is None:
                    config.data.output_distribution.total_count = total_count

        if config.data.type == DataGenType.SharedPrefix and config.data.shared_prefix is None:
            raise Exception(f"{config.data.type.value} data generator requires 'shared_prefix' to be configured")

        if config.data.type == DataGenType.ShareGPT:
            datagen = HFShareGPTDataGenerator(config.api, config.data, tokenizer)
        elif config.data.type == DataGenType.CNNDailyMail:
            datagen = CNNDailyMailDataGenerator(config.api, config.data, tokenizer)
        elif config.data.type == DataGenType.Synthetic:
            datagen = SyntheticDataGenerator(config.api, config.data, tokenizer)
        elif config.data.type == DataGenType.Random:
            datagen = RandomDataGenerator(config.api, config.data, tokenizer)
        elif config.data.type == DataGenType.SharedPrefix:
            datagen = SharedPrefixDataGenerator(config.api, config.data, tokenizer)
        elif config.data.type == DataGenType.InfinityInstruct:
            datagen = InfinityInstructDataGenerator(config.api, config.data, tokenizer)
        elif config.data.type == DataGenType.BillsumConversations:
            datagen = BillsumConversationsDataGenerator(config.api, config.data, tokenizer)
        elif config.data.type == DataGenType.Tau2Bench:
            datagen = Tau2BenchDataGenerator(config.api, config.data, tokenizer)
        elif config.data.type == DataGenType.Langfuse:
            datagen = LangfuseDataGenerator(config.api, config.data, tokenizer)
        elif config.data.type == DataGenType.OpenTelemetry:
            datagen = OpenTelemetryDataGenerator(config.api, config.data, tokenizer)
        elif config.data.type == DataGenType.AgenticSynthetic:
            if config.data.agentic_synthetic is None:
                raise Exception("AgenticSynthetic data generator requires 'agentic_synthetic' to be configured")
            datagen = AgenticSyntheticDataGenerator(config.api, config.data, tokenizer)
        elif config.data.type == DataGenType.AgenticCsv:
            if config.data.agentic_csv is None:
                raise Exception("AgenticCsv data generator requires 'agentic_csv' to be configured")
            datagen = AgenticCsvDataGenerator(config.api, config.data, tokenizer)
        else:
            datagen = MockDataGenerator(config.api, config.data, tokenizer)
    else:
        raise Exception("data config missing")

    # Define LoadGenerator
    if isinstance(metrics_client, PrometheusMetricsClient) and config.report.prometheus and config.report.prometheus.per_stage:
        config.load.interval = max(config.load.interval, metrics_client.scrape_interval)

    # Check if using agentic load type
    is_agentic_load = config.load.type in (
        LoadType.AGENTIC,
        LoadType.AGENTIC_CONCURRENT,
        LoadType.AGENTIC_TRACE_REPLAY,
    )

    if is_agentic_load:
        # Agentic load requires session-based data generator
        if not hasattr(datagen, 'get_sessions'):
            raise Exception(
                f"Agentic load types require a data generator that provides sessions. "
                f"Use data type 'agentic_synthetic', 'agentic_csv', 'otel', or 'langfuse'."
            )

        # Get sessions from the data generator
        sessions = datagen.get_sessions()
        if not sessions:
            raise Exception("No sessions available from data generator for agentic load")

        # Log sweep configuration if present
        if config.load.agentic_sweep:
            import logging
            logger = logging.getLogger(__name__)
            logger.info(
                f"Agentic sweep configured: "
                f"num_sessions={config.load.agentic_sweep.num_sessions}, "
                f"probe_range=[{config.load.agentic_sweep.min_probe_rate:.1f}, "
                f"{config.load.agentic_sweep.max_probe_rate:.1f}], "
                f"num_stages={config.load.agentic_sweep.num_stages}"
            )

        # Create a factory function for model server clients
        def client_factory() -> ModelServerClient:
            return model_server_client

        # Use AgenticLoadGenerator for session-based workloads
        agentic_loadgen = AgenticLoadGenerator(
            sessions=sessions,
            load_config=config.load,
            api_config=config.api,
            client_factory=client_factory,
            data_generator=datagen,
        )
        loadgen = LoadGenerator(datagen, config.load)  # For stage_runtime_info compatibility
        loadgen_to_use = agentic_loadgen
    else:
        loadgen = LoadGenerator(datagen, config.load)
        loadgen_to_use = loadgen

    # Setup Perf Test Runner
    perfrunner = InferencePerfRunner(model_server_client, loadgen, reportgen, storage_clients)

    start_time = time.time()

    # Run Perf Test
    agentic_session_results = None
    agentic_stage_results = None

    if is_agentic_load:
        # Run agentic load generator
        async def _run_agentic() -> None:
            async with reportgen.get_metrics_collector().start():
                await agentic_loadgen.run(model_server_client)
        asyncio.run(_run_agentic())

        # Capture session results for reporting
        agentic_stage_results = agentic_loadgen.get_stage_results()
        agentic_session_results = agentic_loadgen.get_all_session_results()
    else:
        perfrunner.run()

    end_time = time.time()
    duration = end_time - start_time  # Calculate the duration of the test

    # Generate Reports after the tests
    # For agentic loads, use agentic_loadgen.stage_runtime_info; for regular loads, use loadgen.stage_runtime_info
    stage_runtime_info = agentic_loadgen.stage_runtime_info if is_agentic_load else loadgen.stage_runtime_info

    reports = perfrunner.generate_reports(
        report_config=config.report,
        runtime_parameters=PerfRuntimeParameters(
            start_time=start_time,
            duration=duration,
            model_server_metrics=model_server_client.get_prometheus_metric_metadata(),
            stages=stage_runtime_info,
        ),
    )

    # Generate agentic session reports if we have session results
    if is_agentic_load and agentic_session_results:
        session_reports = reportgen.generate_session_reports(
            session_results=agentic_session_results,
            stage_results=agentic_stage_results,
            report_config=config.report,
        )
        reports.extend(session_reports)

        # Generate sweep/saturation report if sweep was run
        saturation_result = agentic_loadgen.get_saturation_result()
        if saturation_result:
            sweep_reports = reportgen.generate_sweep_report(saturation_result)
            reports.extend(sweep_reports)

    # Save Reports
    perfrunner.save_reports(reports=reports)

    perfrunner.stop()


if __name__ == "__main__":
    main_cli()
