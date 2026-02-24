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

from inference_perf.client.requestdatacollector import RequestDataCollector
from typing import List, Optional
from inference_perf.config import APIConfig, APIType
from inference_perf.apis import InferenceAPIData, InferenceInfo, RequestLifecycleMetric, ErrorResponseInfo
from .base import ModelServerClient, ModelServerPrometheusMetric, PrometheusMetricMetadata
import asyncio
import time
import logging
import json
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class MockModelServerClient(ModelServerClient):
    def __init__(
        self,
        metrics_collector: RequestDataCollector,
        api_config: APIConfig,
        timeout: Optional[float] = None,
        mock_latency: float = 1,
        ignore_eos: bool = True,
        max_completion_tokens: int = 30,
        debug_log_enabled: bool = False,
        debug_log_file: str = "inference_requests_debug.json",
    ) -> None:
        super().__init__(api_config, timeout)
        self.metrics_collector = metrics_collector
        self.mock_latency = mock_latency
        self.ignore_eos = ignore_eos
        self.max_completion_tokens = max_completion_tokens
        self.tokenizer = None
        self.debug_log_enabled = debug_log_enabled
        self.debug_log_file = Path(debug_log_file) if debug_log_enabled else None
        self.request_count = 0

        # Initialize debug log file if enabled
        if self.debug_log_enabled:
            # Create parent directory if it doesn't exist
            self.debug_log_file.parent.mkdir(parents=True, exist_ok=True)
            # Initialize log file with empty array
            with open(self.debug_log_file, "w") as f:
                json.dump([], f)
            logger.info(f"Debug logging enabled. Logging requests to: {self.debug_log_file}")

    def _log_request_to_file(
        self,
        payload: dict,
        stage_id: int,
        scheduled_time: float,
        effective_model_name: str,
        lora_adapter: Optional[str],
        program_id: Optional[int],
        turn_index: Optional[int],
    ) -> None:
        """Log request details to JSON debug file."""
        try:
            request_log = {
                "request_id": self.request_count,
                "timestamp": datetime.now().isoformat(),
                "stage_id": stage_id,
                "scheduled_time": scheduled_time,
                "lora_adapter": lora_adapter,
                "model_name": effective_model_name,
                "program_id": program_id,
                "turn_index": turn_index,
                "payload": payload,
            }

            self.request_count += 1

            # Read existing logs, append new one, and write back
            with open(self.debug_log_file, "r") as f:
                logs = json.load(f)
            logs.append(request_log)
            with open(self.debug_log_file, "w") as f:
                json.dump(logs, f, indent=2)

            logger.debug(f"Logged request {self.request_count} to {self.debug_log_file}")

        except Exception as e:
            logger.error(f"Failed to log request to JSON file: {e}")

    async def process_request(
        self, data: InferenceAPIData, stage_id: int, scheduled_time: float, lora_adapter: Optional[str] = None
    ) -> None:
        start = time.perf_counter()
        logger.debug("Processing mock request for stage %d", stage_id)
        effective_model_name = lora_adapter if lora_adapter else "mock_model"

        # Log request to JSON file if debug logging is enabled
        if self.debug_log_enabled:
            payload = await data.to_payload(
                effective_model_name,
                self.max_completion_tokens,
                self.ignore_eos,
                self.api_config.streaming,
            )
            self._log_request_to_file(
                payload=payload,
                stage_id=stage_id,
                scheduled_time=scheduled_time,
                effective_model_name=effective_model_name,
                lora_adapter=lora_adapter,
                program_id=data.program_id,
                turn_index=data.turn_index,
            )

        try:
            if self.timeout and self.timeout < self.mock_latency:
                await asyncio.sleep(self.timeout)
                raise asyncio.exceptions.TimeoutError()
            else:
                if self.mock_latency > 0:
                    await asyncio.sleep(self.mock_latency)
                self.metrics_collector.record_metric(
                    RequestLifecycleMetric(
                        stage_id=stage_id,
                        request_data=str(
                            await data.to_payload(
                                effective_model_name,
                                self.max_completion_tokens,
                                self.ignore_eos,
                                self.api_config.streaming,
                            )
                        ),
                        info=InferenceInfo(
                            input_tokens=0,
                            output_tokens=0,
                            lora_adapter=lora_adapter,
                        ),
                        error=None,
                        start_time=start,
                        end_time=time.perf_counter(),
                        scheduled_time=scheduled_time,
                        program_id=data.program_id,
                        turn_index=data.turn_index,
                    )
                )
        except asyncio.exceptions.TimeoutError as e:
            logger.debug("Request timedout after %f seconds", self.timeout)
            self.metrics_collector.record_metric(
                RequestLifecycleMetric(
                    stage_id=stage_id,
                    request_data=str(
                        await data.to_payload(
                            effective_model_name,
                            self.max_completion_tokens,
                            self.ignore_eos,
                            self.api_config.streaming,
                        )
                    ),
                    info=InferenceInfo(
                        input_tokens=0,
                        output_tokens=0,
                        lora_adapter=lora_adapter,
                    ),
                    error=ErrorResponseInfo(
                        error_msg=str(e),
                        error_type=type(e).__name__,
                    ),
                    start_time=start,
                    end_time=time.perf_counter(),
                    scheduled_time=scheduled_time,
                    program_id=data.program_id,
                    turn_index=data.turn_index,
                )
            )

    def get_supported_apis(self) -> List[APIType]:
        return [APIType.Completion, APIType.Chat]

    def get_prometheus_metric_metadata(self) -> PrometheusMetricMetadata:
        mock_prometheus_metric = ModelServerPrometheusMetric(
            name="mock_metric",
            op="mean",
            type="counter",
            filters=[],
        )
        return PrometheusMetricMetadata(
            # Throughput
            prompt_tokens_per_second=mock_prometheus_metric,
            output_tokens_per_second=mock_prometheus_metric,
            requests_per_second=mock_prometheus_metric,
            # Latency
            avg_request_latency=mock_prometheus_metric,
            median_request_latency=mock_prometheus_metric,
            p90_request_latency=mock_prometheus_metric,
            p99_request_latency=mock_prometheus_metric,
            # Request
            total_requests=mock_prometheus_metric,
            avg_prompt_tokens=mock_prometheus_metric,
            avg_output_tokens=mock_prometheus_metric,
            avg_queue_length=mock_prometheus_metric,
            # Others
            avg_time_to_first_token=None,
            median_time_to_first_token=None,
            p90_time_to_first_token=None,
            p99_time_to_first_token=None,
            avg_time_per_output_token=None,
            median_time_per_output_token=None,
            p90_time_per_output_token=None,
            p99_time_per_output_token=None,
            avg_inter_token_latency=None,
            median_inter_token_latency=None,
            p90_inter_token_latency=None,
            p99_inter_token_latency=None,
            avg_kv_cache_usage=None,
            median_kv_cache_usage=None,
            p90_kv_cache_usage=None,
            p99_kv_cache_usage=None,
            num_preemptions_total=None,
            num_requests_swapped=None,
            prefix_cache_hits=None,
            prefix_cache_queries=None,
            avg_num_requests_running=None,
            avg_request_queue_time=None,
            median_request_queue_time=None,
            p90_request_queue_time=None,
            p99_request_queue_time=None,
            avg_request_inference_time=None,
            median_request_inference_time=None,
            p90_request_inference_time=None,
            p99_request_inference_time=None,
            avg_request_prefill_time=None,
            median_request_prefill_time=None,
            p90_request_prefill_time=None,
            p99_request_prefill_time=None,
            avg_request_decode_time=None,
            median_request_decode_time=None,
            p90_request_decode_time=None,
            p99_request_decode_time=None,
            avg_request_prompt_tokens=None,
            median_request_prompt_tokens=None,
            p90_request_prompt_tokens=None,
            p99_request_prompt_tokens=None,
            avg_request_generation_tokens=None,
            median_request_generation_tokens=None,
            p90_request_generation_tokens=None,
            p99_request_generation_tokens=None,
            avg_request_max_num_generation_tokens=None,
            median_request_max_num_generation_tokens=None,
            p90_request_max_num_generation_tokens=None,
            p99_request_max_num_generation_tokens=None,
            avg_request_params_n=None,
            median_request_params_n=None,
            p90_request_params_n=None,
            p99_request_params_n=None,
            avg_request_params_max_tokens=None,
            median_request_params_max_tokens=None,
            p90_request_params_max_tokens=None,
            p99_request_params_max_tokens=None,
            request_success_count=None,
            avg_iteration_tokens=None,
            median_iteration_tokens=None,
            p90_iteration_tokens=None,
            p99_iteration_tokens=None,
            prompt_tokens_cached=None,
            prompt_tokens_recomputed=None,
            external_prefix_cache_hits=None,
            external_prefix_cache_queries=None,
            mm_cache_hits=None,
            mm_cache_queries=None,
            corrupted_requests=None,
            avg_request_prefill_kv_computed_tokens=None,
            median_request_prefill_kv_computed_tokens=None,
            p90_request_prefill_kv_computed_tokens=None,
            p99_request_prefill_kv_computed_tokens=None,
            avg_kv_block_idle_before_evict=None,
            median_kv_block_idle_before_evict=None,
            p90_kv_block_idle_before_evict=None,
            p99_kv_block_idle_before_evict=None,
            avg_kv_block_lifetime=None,
            median_kv_block_lifetime=None,
            p90_kv_block_lifetime=None,
            p99_kv_block_lifetime=None,
            avg_kv_block_reuse_gap=None,
            median_kv_block_reuse_gap=None,
            p90_kv_block_reuse_gap=None,
            p99_kv_block_reuse_gap=None,
        )
