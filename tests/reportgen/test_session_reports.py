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

"""Tests for session report generation."""

import pytest

from inference_perf.loadgen.session_runner import SessionResult, TurnResult
from inference_perf.loadgen.agentic_load_generator import StageResult
from inference_perf.reportgen.base import summarize_sessions
from inference_perf.metrics import SessionMetrics, TurnPositionMetrics, SystemMetrics


def create_mock_turn_result(
    session_id: str,
    turn_index: int,
    input_tokens: int = 500,
    output_tokens: int = 100,
    start_time: float = 0.0,
    end_time: float = 1.0,
    ttft_ms: float = 50.0,
) -> TurnResult:
    """Create a mock TurnResult for testing."""
    return TurnResult(
        session_id=session_id,
        turn_index=turn_index,
        scheduled_time=start_time - 0.1,  # Scheduled slightly before start
        start_time=start_time,
        end_time=end_time,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        ttft_ms=ttft_ms,
        success=True,
    )


def create_mock_session_result(
    session_id: str,
    num_turns: int = 3,
) -> SessionResult:
    """Create a mock SessionResult for testing."""
    turn_results = []
    for i in range(num_turns):
        tr = create_mock_turn_result(
            session_id=session_id,
            turn_index=i,
            input_tokens=500 + i * 100,
            output_tokens=100,
            start_time=float(i),
            end_time=float(i) + 0.8,
        )
        turn_results.append(tr)

    return SessionResult(
        session_id=session_id,
        turn_results=turn_results,
        start_time=0.0,
        end_time=float(num_turns),  # End time based on number of turns
    )


def create_mock_stage_result(
    stage_id: int,
    num_sessions: int = 5,
    start_time: float = 0.0,
    end_time: float = 10.0,
) -> StageResult:
    """Create a mock StageResult for testing."""
    session_results = [
        create_mock_session_result(f"session_{i}", num_turns=3 + i % 3)
        for i in range(num_sessions)
    ]

    return StageResult(
        stage_id=stage_id,
        start_time=start_time,
        end_time=end_time,
        session_results=session_results,
        active_sessions_timeseries=[(1.0, 3), (2.0, 5), (3.0, 4)],
        request_rate_timeseries=[(1.0, 0.5), (2.0, 1.0), (3.0, 1.2)],
    )


class TestSummarizeSessions:
    """Tests for summarize_sessions function."""

    def test_summarize_sessions_basic(self):
        """Test basic session summarization."""
        session_results = [
            create_mock_session_result(f"session_{i}")
            for i in range(10)
        ]

        summary = summarize_sessions(session_results, [50, 90, 99])

        assert summary["count"] == 10
        assert "session_latency_ms" in summary
        assert "session_inference_time_ms" in summary
        assert "inference_duty_cycle" in summary
        assert "throughput" in summary
        assert summary["throughput"]["total_sessions"] == 10

    def test_summarize_sessions_empty(self):
        """Test summarization with empty list."""
        summary = summarize_sessions([], [50, 90, 99])
        assert summary == {}

    def test_summarize_sessions_percentiles(self):
        """Test that percentiles are computed correctly."""
        session_results = [
            create_mock_session_result(f"session_{i}", num_turns=i + 1)
            for i in range(10)
        ]

        summary = summarize_sessions(session_results, [50, 90, 99])

        # Check that percentile keys exist
        assert "median" in summary["session_latency_ms"]
        assert "p90" in summary["session_latency_ms"]
        assert "p99" in summary["session_latency_ms"]


class TestSessionMetrics:
    """Tests for SessionMetrics class."""

    def test_from_session_result(self):
        """Test creating SessionMetrics from SessionResult."""
        session_result = create_mock_session_result("test_session", num_turns=5)
        metrics = SessionMetrics.from_session_result(session_result)

        assert metrics.session_id == "test_session"
        assert metrics.turns_completed == 5
        assert metrics.session_latency_ms > 0  # Computed from turn results
        assert metrics.session_inference_time_ms > 0  # Computed from turn results
        assert len(metrics.latency_by_turn) == 5

    def test_to_dict(self):
        """Test serialization to dictionary."""
        session_result = create_mock_session_result("test_session")
        metrics = SessionMetrics.from_session_result(session_result)
        data = metrics.to_dict()

        assert data["session_id"] == "test_session"
        assert "session_latency_ms" in data
        assert "ttft_by_turn" in data
        assert "latency_by_turn" in data


class TestTurnPositionMetrics:
    """Tests for TurnPositionMetrics class."""

    def test_from_session_metrics_list(self):
        """Test creating TurnPositionMetrics from list of SessionMetrics."""
        session_results = [
            create_mock_session_result(f"session_{i}", num_turns=5)
            for i in range(10)
        ]
        session_metrics = [
            SessionMetrics.from_session_result(sr)
            for sr in session_results
        ]

        # Get metrics for turn 0
        tpm = TurnPositionMetrics.from_session_metrics_list(0, session_metrics)

        assert tpm.turn_index == 0
        assert tpm.count == 10  # All sessions have turn 0
        assert tpm.avg_latency_ms > 0

    def test_turn_position_with_varying_turns(self):
        """Test turn position metrics when sessions have different numbers of turns."""
        session_results = [
            create_mock_session_result(f"session_{i}", num_turns=i + 2)
            for i in range(5)
        ]
        session_metrics = [
            SessionMetrics.from_session_result(sr)
            for sr in session_results
        ]

        # Turn 0 should have all sessions
        tpm_0 = TurnPositionMetrics.from_session_metrics_list(0, session_metrics)
        assert tpm_0.count == 5

        # Turn 5 should have fewer sessions (only those with 6 turns)
        tpm_5 = TurnPositionMetrics.from_session_metrics_list(5, session_metrics)
        assert tpm_5.count == 1  # Only session_4 has 6 turns


class TestSystemMetrics:
    """Tests for SystemMetrics class."""

    def test_from_stage_results(self):
        """Test creating SystemMetrics from stage results."""
        stage_results = [
            create_mock_stage_result(0, num_sessions=5),
            create_mock_stage_result(1, num_sessions=5),
        ]

        metrics = SystemMetrics.from_stage_results(stage_results)

        assert metrics.total_sessions == 10
        assert metrics.total_turns > 0
        assert metrics.session_throughput > 0

    def test_timeseries_aggregation(self):
        """Test that time series data is aggregated correctly."""
        stage_results = [
            create_mock_stage_result(0, num_sessions=5, start_time=0.0, end_time=10.0),
            create_mock_stage_result(1, num_sessions=5, start_time=10.0, end_time=20.0),
        ]

        metrics = SystemMetrics.from_stage_results(stage_results)

        # Should have time series data from both stages
        assert len(metrics.active_sessions_timeseries) > 0
        assert len(metrics.effective_request_rate_timeseries) > 0

    def test_to_summary_dict(self):
        """Test summary dictionary generation."""
        stage_results = [create_mock_stage_result(0, num_sessions=5)]
        metrics = SystemMetrics.from_stage_results(stage_results)

        summary = metrics.to_summary_dict()

        assert "throughput" in summary
        assert "latency" in summary
        assert "session_characteristics" in summary
        assert "tokens" in summary


class TestStageResultTimeseries:
    """Tests for StageResult time series fields."""

    def test_stage_result_has_timeseries_fields(self):
        """Test that StageResult has time series fields."""
        stage_result = create_mock_stage_result(0, num_sessions=5)

        assert hasattr(stage_result, 'active_sessions_timeseries')
        assert hasattr(stage_result, 'request_rate_timeseries')
        assert len(stage_result.active_sessions_timeseries) > 0
        assert len(stage_result.request_rate_timeseries) > 0

    def test_timeseries_format(self):
        """Test that time series data has correct format."""
        stage_result = create_mock_stage_result(0, num_sessions=5)

        # Each entry should be (timestamp, value) tuple
        for ts, val in stage_result.active_sessions_timeseries:
            assert isinstance(ts, (int, float))
            assert isinstance(val, (int, float))
