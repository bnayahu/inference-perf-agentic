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

"""System-level metrics for agentic workload benchmarking.

These metrics capture the aggregate performance characteristics of the
system under agentic workloads, including throughput, concurrency patterns,
and resource utilization.
"""

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

from inference_perf.loadgen.agentic_load_generator import StageResult
from inference_perf.loadgen.session_runner import SessionResult
from .session_metrics import SessionMetrics


@dataclass
class SystemMetrics:
    """System-level metrics for agentic workloads.

    Captures the overall system behavior during a benchmark run,
    including throughput, concurrency patterns, and aggregate latencies.

    Attributes:
        active_sessions_timeseries: Time series of (timestamp, active_count).
        effective_request_rate_timeseries: Time series of (timestamp, requests/sec).
        session_throughput: Sessions completed per second.
        inter_turn_idle_ratio: Average ratio of idle time between turns.
        context_growth_rate: Average context growth in tokens per turn.
        session_arrival_rate: Measured session arrival rate.
        session_completion_rate: Measured session completion rate.
        avg_session_latency_ms: Average session latency.
        p50_session_latency_ms: Median session latency.
        p90_session_latency_ms: 90th percentile session latency.
        p99_session_latency_ms: 99th percentile session latency.
        avg_turns_per_session: Average number of turns completed.
        total_sessions: Total number of sessions executed.
        total_turns: Total number of turns executed.
        total_input_tokens: Total input tokens processed.
        total_output_tokens: Total output tokens generated.
    """
    active_sessions_timeseries: List[Tuple[float, int]] = field(default_factory=list)
    effective_request_rate_timeseries: List[Tuple[float, float]] = field(default_factory=list)
    session_throughput: float = 0.0
    inter_turn_idle_ratio: float = 0.0
    context_growth_rate: float = 0.0
    session_arrival_rate: float = 0.0
    session_completion_rate: float = 0.0
    avg_session_latency_ms: float = 0.0
    p50_session_latency_ms: float = 0.0
    p90_session_latency_ms: float = 0.0
    p99_session_latency_ms: float = 0.0
    avg_turns_per_session: float = 0.0
    total_sessions: int = 0
    total_turns: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0

    @classmethod
    def from_stage_results(
        cls,
        stage_results: List[StageResult],
    ) -> "SystemMetrics":
        """Create SystemMetrics from stage results.

        Args:
            stage_results: List of StageResult from load generation.

        Returns:
            SystemMetrics with computed values.
        """
        metrics = cls()

        if not stage_results:
            return metrics

        # Collect all session results
        all_session_results: List[SessionResult] = []
        for stage in stage_results:
            all_session_results.extend(stage.session_results)

        if not all_session_results:
            return metrics

        # Basic counts
        metrics.total_sessions = len(all_session_results)
        metrics.total_turns = sum(sr.turns_completed for sr in all_session_results)
        metrics.total_input_tokens = sum(sr.session_input_tokens for sr in all_session_results)
        metrics.total_output_tokens = sum(sr.session_output_tokens for sr in all_session_results)

        # Session latencies
        latencies = [sr.session_latency_ms for sr in all_session_results]
        if latencies:
            metrics.avg_session_latency_ms = float(np.mean(latencies))
            metrics.p50_session_latency_ms = float(np.percentile(latencies, 50))
            metrics.p90_session_latency_ms = float(np.percentile(latencies, 90))
            metrics.p99_session_latency_ms = float(np.percentile(latencies, 99))

        # Turns per session
        turns_per_session = [sr.turns_completed for sr in all_session_results]
        if turns_per_session:
            metrics.avg_turns_per_session = float(np.mean(turns_per_session))

        # Inter-turn idle ratio (pause time / total time)
        duty_cycles = [sr.inference_duty_cycle for sr in all_session_results]
        if duty_cycles:
            avg_duty_cycle = float(np.mean(duty_cycles))
            metrics.inter_turn_idle_ratio = 1.0 - avg_duty_cycle

        # Throughput calculations
        if stage_results:
            total_duration = sum(
                stage.end_time - stage.start_time
                for stage in stage_results
            )
            if total_duration > 0:
                metrics.session_throughput = metrics.total_sessions / total_duration
                metrics.session_completion_rate = metrics.session_throughput

                # Arrival rate (sessions started per second)
                # Approximation: same as completion rate in steady state
                metrics.session_arrival_rate = metrics.session_throughput

        # Context growth rate (average increase in input tokens per turn)
        context_growths = []
        for sr in all_session_results:
            if sr.turns_completed > 1:
                # Calculate growth from first to last turn
                first_tokens = sr.turn_results[0].input_tokens if sr.turn_results else 0
                last_tokens = sr.turn_results[-1].input_tokens if sr.turn_results else 0
                turns = sr.turns_completed - 1
                if turns > 0:
                    context_growths.append((last_tokens - first_tokens) / turns)
        if context_growths:
            metrics.context_growth_rate = float(np.mean(context_growths))

        # Aggregate time series data from all stages
        cumulative_time = 0.0
        for stage in stage_results:
            stage_duration = stage.end_time - stage.start_time
            # Adjust timestamps to be cumulative across stages
            if hasattr(stage, 'active_sessions_timeseries'):
                for ts, value in stage.active_sessions_timeseries:
                    metrics.active_sessions_timeseries.append((cumulative_time + ts, value))
            if hasattr(stage, 'request_rate_timeseries'):
                for ts, value in stage.request_rate_timeseries:
                    metrics.effective_request_rate_timeseries.append((cumulative_time + ts, value))
            cumulative_time += stage_duration

        return metrics

    @classmethod
    def from_session_metrics_list(
        cls,
        session_metrics: List[SessionMetrics],
        total_duration: float,
    ) -> "SystemMetrics":
        """Create SystemMetrics from SessionMetrics list.

        Args:
            session_metrics: List of SessionMetrics.
            total_duration: Total benchmark duration in seconds.

        Returns:
            SystemMetrics with computed values.
        """
        metrics = cls()

        if not session_metrics:
            return metrics

        # Basic counts
        metrics.total_sessions = len(session_metrics)
        metrics.total_turns = sum(sm.turns_completed for sm in session_metrics)
        metrics.total_input_tokens = sum(sm.session_input_tokens for sm in session_metrics)
        metrics.total_output_tokens = sum(sm.session_output_tokens for sm in session_metrics)

        # Session latencies
        latencies = [sm.session_latency_ms for sm in session_metrics]
        if latencies:
            metrics.avg_session_latency_ms = float(np.mean(latencies))
            metrics.p50_session_latency_ms = float(np.percentile(latencies, 50))
            metrics.p90_session_latency_ms = float(np.percentile(latencies, 90))
            metrics.p99_session_latency_ms = float(np.percentile(latencies, 99))

        # Turns per session
        turns_per_session = [sm.turns_completed for sm in session_metrics]
        if turns_per_session:
            metrics.avg_turns_per_session = float(np.mean(turns_per_session))

        # Inter-turn idle ratio
        duty_cycles = [sm.inference_duty_cycle for sm in session_metrics]
        if duty_cycles:
            avg_duty_cycle = float(np.mean(duty_cycles))
            metrics.inter_turn_idle_ratio = 1.0 - avg_duty_cycle

        # Throughput
        if total_duration > 0:
            metrics.session_throughput = metrics.total_sessions / total_duration
            metrics.session_completion_rate = metrics.session_throughput
            metrics.session_arrival_rate = metrics.session_throughput

        # Context growth rate
        context_growths = []
        for sm in session_metrics:
            if sm.turns_completed > 1:
                input_tokens = list(sm.input_tokens_by_turn.values())
                if len(input_tokens) >= 2:
                    first_tokens = input_tokens[0]
                    last_tokens = input_tokens[-1]
                    turns = sm.turns_completed - 1
                    if turns > 0:
                        context_growths.append((last_tokens - first_tokens) / turns)
        if context_growths:
            metrics.context_growth_rate = float(np.mean(context_growths))

        return metrics

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "session_throughput": self.session_throughput,
            "inter_turn_idle_ratio": self.inter_turn_idle_ratio,
            "context_growth_rate": self.context_growth_rate,
            "session_arrival_rate": self.session_arrival_rate,
            "session_completion_rate": self.session_completion_rate,
            "avg_session_latency_ms": self.avg_session_latency_ms,
            "p50_session_latency_ms": self.p50_session_latency_ms,
            "p90_session_latency_ms": self.p90_session_latency_ms,
            "p99_session_latency_ms": self.p99_session_latency_ms,
            "avg_turns_per_session": self.avg_turns_per_session,
            "total_sessions": self.total_sessions,
            "total_turns": self.total_turns,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
        }

    def to_summary_dict(self) -> dict:
        """Convert to a summary dictionary for reporting."""
        return {
            "throughput": {
                "sessions_per_second": self.session_throughput,
                "total_sessions": self.total_sessions,
                "total_turns": self.total_turns,
            },
            "latency": {
                "avg_session_ms": self.avg_session_latency_ms,
                "p50_session_ms": self.p50_session_latency_ms,
                "p90_session_ms": self.p90_session_latency_ms,
                "p99_session_ms": self.p99_session_latency_ms,
            },
            "session_characteristics": {
                "avg_turns_per_session": self.avg_turns_per_session,
                "inter_turn_idle_ratio": self.inter_turn_idle_ratio,
                "context_growth_rate": self.context_growth_rate,
            },
            "tokens": {
                "total_input_tokens": self.total_input_tokens,
                "total_output_tokens": self.total_output_tokens,
            },
        }
