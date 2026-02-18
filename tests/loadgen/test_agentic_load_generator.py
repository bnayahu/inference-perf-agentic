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

"""Tests for AgenticLoadGenerator."""

import pytest
from unittest.mock import MagicMock, AsyncMock

from inference_perf.config import (
    LoadConfig,
    LoadType,
    APIConfig,
    SessionArrivalConfig,
    SessionArrivalType,
    SessionStageConfig,
    AgenticDelayConfig,
)
from inference_perf.loadgen.agentic_load_generator import AgenticLoadGenerator, StageResult
from inference_perf.models import Session, Turn, FinishReason


def create_mock_session(session_id: str, num_turns: int = 3) -> Session:
    """Create a mock Session for testing."""
    turns = [
        Turn(
            session_id=session_id,
            turn_index=i,
            input_tokens=500 + i * 100,
            output_tokens=100,
            finish_reason=FinishReason.STOP if i == num_turns - 1 else FinishReason.TOOL_CALLS,
        )
        for i in range(num_turns)
    ]
    return Session(session_id=session_id, turns=turns)


def create_load_config(
    load_type: LoadType = LoadType.AGENTIC,
    num_workers: int = 4,
    worker_affinity: bool = True,
) -> LoadConfig:
    """Create a LoadConfig for testing."""
    return LoadConfig(
        type=load_type,
        num_workers=num_workers,
        worker_affinity=worker_affinity,
        session_arrival=SessionArrivalConfig(
            type=SessionArrivalType.CONSTANT,
            stages=[SessionStageConfig(rate=1.0, duration=10, total_sessions=5)],
        ),
        agentic=AgenticDelayConfig(),
    )


class TestWorkerAffinity:
    """Tests for worker affinity feature."""

    def test_worker_affinity_consistent_hashing(self):
        """Test that the same session always maps to the same worker."""
        sessions = [create_mock_session(f"session_{i}") for i in range(10)]
        load_config = create_load_config(worker_affinity=True, num_workers=4)
        api_config = APIConfig()

        generator = AgenticLoadGenerator(
            sessions=sessions,
            load_config=load_config,
            api_config=api_config,
            client_factory=MagicMock,
        )

        # Same session should always map to same worker
        session = sessions[0]
        worker1 = generator._get_worker_for_session(session)
        worker2 = generator._get_worker_for_session(session)
        worker3 = generator._get_worker_for_session(session)

        assert worker1 == worker2 == worker3

    def test_worker_affinity_distribution(self):
        """Test that sessions are distributed across workers."""
        sessions = [create_mock_session(f"session_{i}") for i in range(100)]
        load_config = create_load_config(worker_affinity=True, num_workers=4)
        api_config = APIConfig()

        generator = AgenticLoadGenerator(
            sessions=sessions,
            load_config=load_config,
            api_config=api_config,
            client_factory=MagicMock,
        )

        # Count sessions per worker
        worker_counts = {i: 0 for i in range(4)}
        for session in sessions:
            worker_id = generator._get_worker_for_session(session)
            worker_counts[worker_id] += 1

        # All workers should have at least some sessions
        for worker_id, count in worker_counts.items():
            assert count > 0, f"Worker {worker_id} has no sessions"

    def test_worker_affinity_disabled(self):
        """Test behavior when worker affinity is disabled."""
        sessions = [create_mock_session(f"session_{i}") for i in range(10)]
        load_config = create_load_config(worker_affinity=False, num_workers=4)
        api_config = APIConfig()

        generator = AgenticLoadGenerator(
            sessions=sessions,
            load_config=load_config,
            api_config=api_config,
            client_factory=MagicMock,
        )

        # Worker should be based on session index modulo
        # Without affinity, it uses round-robin based on _session_index
        worker_id = generator._get_worker_for_session(sessions[0])
        assert 0 <= worker_id < 4

    def test_worker_affinity_single_worker(self):
        """Test worker affinity with single worker."""
        sessions = [create_mock_session(f"session_{i}") for i in range(10)]
        load_config = create_load_config(worker_affinity=True, num_workers=1)
        api_config = APIConfig()

        generator = AgenticLoadGenerator(
            sessions=sessions,
            load_config=load_config,
            api_config=api_config,
            client_factory=MagicMock,
        )

        # All sessions should map to worker 0
        for session in sessions:
            assert generator._get_worker_for_session(session) == 0


class TestArrivalTimes:
    """Tests for arrival time generation."""

    def test_constant_arrivals(self):
        """Test constant arrival time generation."""
        sessions = [create_mock_session(f"session_{i}") for i in range(10)]
        load_config = create_load_config()
        api_config = APIConfig()

        generator = AgenticLoadGenerator(
            sessions=sessions,
            load_config=load_config,
            api_config=api_config,
            client_factory=MagicMock,
        )

        arrival_times = generator._generate_arrival_times(10, rate=2.0, poisson=False)

        # Constant arrivals should have uniform spacing
        assert len(arrival_times) == 10
        expected_interval = 0.5  # 1/rate
        for i in range(1, len(arrival_times)):
            assert abs(arrival_times[i] - arrival_times[i-1] - expected_interval) < 0.001

    def test_poisson_arrivals(self):
        """Test Poisson arrival time generation."""
        sessions = [create_mock_session(f"session_{i}") for i in range(100)]
        load_config = create_load_config()
        api_config = APIConfig()

        generator = AgenticLoadGenerator(
            sessions=sessions,
            load_config=load_config,
            api_config=api_config,
            client_factory=MagicMock,
        )

        arrival_times = generator._generate_arrival_times(100, rate=10.0, poisson=True)

        # Poisson arrivals should have exponential inter-arrival times
        assert len(arrival_times) == 100
        # All times should be increasing
        for i in range(1, len(arrival_times)):
            assert arrival_times[i] > arrival_times[i-1]


class TestStageResultProperties:
    """Tests for StageResult properties."""

    def test_stage_result_properties(self):
        """Test StageResult computed properties."""
        from inference_perf.loadgen.session_runner import SessionResult, TurnResult

        turn_result = TurnResult(
            session_id="test",
            turn_index=0,
            scheduled_time=0.0,
            start_time=0.0,
            end_time=1.0,
            input_tokens=500,
            output_tokens=100,
            success=True,
        )

        session_result = SessionResult(
            session_id="test",
            turn_results=[turn_result],
            start_time=0.0,
            end_time=1.0,
        )

        stage_result = StageResult(
            stage_id=0,
            start_time=0.0,
            end_time=10.0,
            session_results=[session_result],
            active_sessions_timeseries=[(1.0, 1)],
            request_rate_timeseries=[(1.0, 0.1)],
        )

        assert stage_result.total_sessions == 1
        assert stage_result.completed_sessions == 1
        assert stage_result.session_throughput == 0.1  # 1 session / 10 seconds

    def test_stage_result_timeseries_fields(self):
        """Test that StageResult has timeseries fields."""
        stage_result = StageResult(
            stage_id=0,
            start_time=0.0,
            end_time=10.0,
            session_results=[],
            active_sessions_timeseries=[(1.0, 5), (2.0, 10)],
            request_rate_timeseries=[(1.0, 0.5), (2.0, 1.0)],
        )

        assert len(stage_result.active_sessions_timeseries) == 2
        assert len(stage_result.request_rate_timeseries) == 2
        assert stage_result.active_sessions_timeseries[0] == (1.0, 5)
        assert stage_result.request_rate_timeseries[1] == (2.0, 1.0)
