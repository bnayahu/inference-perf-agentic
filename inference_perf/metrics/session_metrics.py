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

"""Session-level metrics for agentic workload benchmarking.

These metrics capture the performance characteristics of individual
agentic sessions, including latency breakdown, token counts, and
turn-level details.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from inference_perf.loadgen.session_runner import SessionResult
from inference_perf.models import Session


@dataclass
class SessionMetrics:
    """Comprehensive metrics for a single agentic session.

    Attributes:
        session_id: Unique identifier for the session.
        session_latency_ms: Wall-clock duration including pauses.
        session_inference_time_ms: Sum of LLM call durations.
        session_pause_time_ms: Time spent in delays between turns.
        inference_duty_cycle: Ratio of inference time to total latency.
        session_speedup_ratio: Original trace duration / measured duration.
        turns_completed: Number of successfully completed turns.
        turns_total: Total number of turns in the session.
        session_input_tokens: Total input tokens across all turns.
        session_output_tokens: Total output tokens across all turns.
        peak_context_length: Maximum input tokens in any single turn.
        ttft_by_turn: Time to first token for each turn position.
        input_tokens_by_turn: Input tokens for each turn position.
        output_tokens_by_turn: Output tokens for each turn position.
        latency_by_turn: Latency (ms) for each turn position.
    """
    session_id: str
    session_latency_ms: float = 0.0
    session_inference_time_ms: float = 0.0
    session_pause_time_ms: float = 0.0
    inference_duty_cycle: float = 0.0
    session_speedup_ratio: Optional[float] = None
    turns_completed: int = 0
    turns_total: int = 0
    session_input_tokens: int = 0
    session_output_tokens: int = 0
    peak_context_length: int = 0
    ttft_by_turn: Dict[int, float] = field(default_factory=dict)
    input_tokens_by_turn: Dict[int, int] = field(default_factory=dict)
    output_tokens_by_turn: Dict[int, int] = field(default_factory=dict)
    latency_by_turn: Dict[int, float] = field(default_factory=dict)

    @classmethod
    def from_session_result(
        cls,
        result: SessionResult,
        original_session: Optional[Session] = None,
    ) -> "SessionMetrics":
        """Create SessionMetrics from a SessionResult.

        Args:
            result: The SessionResult from execution.
            original_session: Optional original Session for speedup calculation.

        Returns:
            SessionMetrics with computed values.
        """
        metrics = cls(session_id=result.session_id)

        # Basic timing metrics
        metrics.session_latency_ms = result.session_latency_ms
        metrics.session_inference_time_ms = result.session_inference_time_ms
        metrics.session_pause_time_ms = result.session_pause_time_ms
        metrics.inference_duty_cycle = result.inference_duty_cycle

        # Turn counts
        metrics.turns_completed = result.turns_completed
        metrics.turns_total = len(result.turn_results)

        # Token counts
        metrics.session_input_tokens = result.session_input_tokens
        metrics.session_output_tokens = result.session_output_tokens
        metrics.peak_context_length = result.peak_context_length

        # Per-turn metrics
        for turn_result in result.turn_results:
            turn_idx = turn_result.turn_index
            metrics.input_tokens_by_turn[turn_idx] = turn_result.input_tokens
            metrics.output_tokens_by_turn[turn_idx] = turn_result.output_tokens
            metrics.latency_by_turn[turn_idx] = (
                turn_result.end_time - turn_result.start_time
            ) * 1000

            if turn_result.ttft_ms is not None:
                metrics.ttft_by_turn[turn_idx] = turn_result.ttft_ms

        # Speedup ratio (if original session data available)
        if original_session and original_session.original_session_latency_ms > 0:
            metrics.session_speedup_ratio = (
                original_session.original_session_latency_ms /
                metrics.session_latency_ms
            )

        return metrics

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "session_id": self.session_id,
            "session_latency_ms": self.session_latency_ms,
            "session_inference_time_ms": self.session_inference_time_ms,
            "session_pause_time_ms": self.session_pause_time_ms,
            "inference_duty_cycle": self.inference_duty_cycle,
            "session_speedup_ratio": self.session_speedup_ratio,
            "turns_completed": self.turns_completed,
            "turns_total": self.turns_total,
            "session_input_tokens": self.session_input_tokens,
            "session_output_tokens": self.session_output_tokens,
            "peak_context_length": self.peak_context_length,
            "ttft_by_turn": self.ttft_by_turn,
            "input_tokens_by_turn": self.input_tokens_by_turn,
            "output_tokens_by_turn": self.output_tokens_by_turn,
            "latency_by_turn": self.latency_by_turn,
        }


@dataclass
class TurnPositionMetrics:
    """Aggregated metrics for a specific turn position across sessions.

    Useful for analyzing how performance varies by turn number
    (e.g., TTFT typically increases with turn number due to context growth).

    Attributes:
        turn_index: The turn position (0-indexed).
        count: Number of sessions that reached this turn.
        avg_ttft_ms: Average time to first token.
        p50_ttft_ms: Median time to first token.
        p90_ttft_ms: 90th percentile TTFT.
        p99_ttft_ms: 99th percentile TTFT.
        avg_latency_ms: Average turn latency.
        avg_input_tokens: Average input tokens at this position.
        avg_output_tokens: Average output tokens at this position.
    """
    turn_index: int
    count: int = 0
    avg_ttft_ms: float = 0.0
    p50_ttft_ms: float = 0.0
    p90_ttft_ms: float = 0.0
    p99_ttft_ms: float = 0.0
    avg_latency_ms: float = 0.0
    avg_input_tokens: float = 0.0
    avg_output_tokens: float = 0.0

    @classmethod
    def from_session_metrics_list(
        cls,
        turn_index: int,
        metrics_list: List[SessionMetrics],
    ) -> "TurnPositionMetrics":
        """Create aggregated metrics for a turn position.

        Args:
            turn_index: The turn position to aggregate.
            metrics_list: List of SessionMetrics to aggregate from.

        Returns:
            TurnPositionMetrics with aggregated values.
        """
        import numpy as np

        result = cls(turn_index=turn_index)

        # Collect values from sessions that have this turn
        ttft_values = []
        latency_values = []
        input_token_values = []
        output_token_values = []

        for metrics in metrics_list:
            if turn_index in metrics.latency_by_turn:
                result.count += 1
                latency_values.append(metrics.latency_by_turn[turn_index])

                if turn_index in metrics.ttft_by_turn:
                    ttft_values.append(metrics.ttft_by_turn[turn_index])

                if turn_index in metrics.input_tokens_by_turn:
                    input_token_values.append(metrics.input_tokens_by_turn[turn_index])

                if turn_index in metrics.output_tokens_by_turn:
                    output_token_values.append(metrics.output_tokens_by_turn[turn_index])

        if result.count == 0:
            return result

        # Compute averages
        if latency_values:
            result.avg_latency_ms = float(np.mean(latency_values))

        if ttft_values:
            result.avg_ttft_ms = float(np.mean(ttft_values))
            result.p50_ttft_ms = float(np.percentile(ttft_values, 50))
            result.p90_ttft_ms = float(np.percentile(ttft_values, 90))
            result.p99_ttft_ms = float(np.percentile(ttft_values, 99))

        if input_token_values:
            result.avg_input_tokens = float(np.mean(input_token_values))

        if output_token_values:
            result.avg_output_tokens = float(np.mean(output_token_values))

        return result

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "turn_index": self.turn_index,
            "count": self.count,
            "avg_ttft_ms": self.avg_ttft_ms,
            "p50_ttft_ms": self.p50_ttft_ms,
            "p90_ttft_ms": self.p90_ttft_ms,
            "p99_ttft_ms": self.p99_ttft_ms,
            "avg_latency_ms": self.avg_latency_ms,
            "avg_input_tokens": self.avg_input_tokens,
            "avg_output_tokens": self.avg_output_tokens,
        }
