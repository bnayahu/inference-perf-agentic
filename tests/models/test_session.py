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

"""Tests for Session and SessionSummary models."""

import pytest
from inference_perf.models import Session, SessionSummary, Turn, ToolCall, FinishReason


class TestSession:
    """Tests for Session model."""

    def test_create_empty_session(self) -> None:
        """Test creating an empty session."""
        session = Session(session_id="s1")
        assert session.session_id == "s1"
        assert session.turns == []
        assert session.num_turns == 0

    def test_session_with_turns(self) -> None:
        """Test session with multiple turns."""
        turns = [
            Turn(session_id="s1", turn_index=0, input_tokens=100, output_tokens=50),
            Turn(session_id="s1", turn_index=1, input_tokens=200, output_tokens=75),
            Turn(session_id="s1", turn_index=2, input_tokens=350, output_tokens=100),
        ]
        session = Session(session_id="s1", turns=turns)
        assert session.num_turns == 3

    def test_session_total_input_tokens(self) -> None:
        """Test calculation of total input tokens."""
        turns = [
            Turn(session_id="s1", turn_index=0, input_tokens=100, output_tokens=50),
            Turn(session_id="s1", turn_index=1, input_tokens=200, output_tokens=75),
        ]
        session = Session(session_id="s1", turns=turns)
        assert session.total_input_tokens == 300

    def test_session_total_output_tokens(self) -> None:
        """Test calculation of total output tokens."""
        turns = [
            Turn(session_id="s1", turn_index=0, input_tokens=100, output_tokens=50),
            Turn(session_id="s1", turn_index=1, input_tokens=200, output_tokens=75),
        ]
        session = Session(session_id="s1", turns=turns)
        assert session.total_output_tokens == 125

    def test_session_max_context_length(self) -> None:
        """Test calculation of maximum context length."""
        turns = [
            Turn(session_id="s1", turn_index=0, input_tokens=100, output_tokens=50),
            Turn(session_id="s1", turn_index=1, input_tokens=500, output_tokens=75),
            Turn(session_id="s1", turn_index=2, input_tokens=350, output_tokens=100),
        ]
        session = Session(session_id="s1", turns=turns)
        assert session.max_context_length == 500

    def test_session_num_tool_calls(self) -> None:
        """Test calculation of total tool calls."""
        tc1 = ToolCall(name="t1", duration_ms=100, result_tokens=10)
        tc2 = ToolCall(name="t2", duration_ms=200, result_tokens=20)
        turns = [
            Turn(session_id="s1", turn_index=0, input_tokens=100, output_tokens=50, tool_calls=[tc1]),
            Turn(session_id="s1", turn_index=1, input_tokens=200, output_tokens=75, tool_calls=[tc2]),
        ]
        session = Session(session_id="s1", turns=turns)
        assert session.num_tool_calls == 2

    def test_session_total_tool_pause_ms(self) -> None:
        """Test calculation of total tool pause time."""
        tc1 = ToolCall(name="t1", duration_ms=100, result_tokens=10)
        tc2 = ToolCall(name="t2", duration_ms=200, result_tokens=20)
        turns = [
            Turn(session_id="s1", turn_index=0, input_tokens=100, output_tokens=50, tool_calls=[tc1, tc2]),
        ]
        session = Session(session_id="s1", turns=turns)
        assert session.total_tool_pause_ms == 300

    def test_session_total_llm_latency_ms(self) -> None:
        """Test calculation of total LLM latency."""
        turns = [
            Turn(session_id="s1", turn_index=0, input_tokens=100, output_tokens=50, llm_latency_ms=500),
            Turn(session_id="s1", turn_index=1, input_tokens=200, output_tokens=75, llm_latency_ms=700),
        ]
        session = Session(session_id="s1", turns=turns)
        assert session.total_llm_latency_ms == 1200

    def test_session_is_complete(self) -> None:
        """Test session completion check."""
        turn1 = Turn(
            session_id="s1", turn_index=0, input_tokens=100, output_tokens=50,
            finish_reason=FinishReason.TOOL_CALLS
        )
        turn2 = Turn(
            session_id="s1", turn_index=1, input_tokens=200, output_tokens=75,
            finish_reason=FinishReason.STOP
        )

        session_incomplete = Session(session_id="s1", turns=[turn1])
        assert session_incomplete.is_complete() is False

        session_complete = Session(session_id="s1", turns=[turn1, turn2])
        assert session_complete.is_complete() is True

    def test_session_add_turn(self) -> None:
        """Test adding turns to session."""
        session = Session(session_id="s1")

        turn0 = Turn(session_id="s1", turn_index=0, input_tokens=100, output_tokens=50)
        session.add_turn(turn0)
        assert session.num_turns == 1

        turn1 = Turn(session_id="s1", turn_index=1, input_tokens=200, output_tokens=75)
        session.add_turn(turn1)
        assert session.num_turns == 2

    def test_session_add_turn_wrong_session_id(self) -> None:
        """Test adding turn with wrong session ID."""
        session = Session(session_id="s1")
        turn = Turn(session_id="s2", turn_index=0, input_tokens=100, output_tokens=50)

        with pytest.raises(ValueError, match="doesn't match"):
            session.add_turn(turn)

    def test_session_add_turn_wrong_index(self) -> None:
        """Test adding turn with wrong index."""
        session = Session(session_id="s1")
        turn = Turn(session_id="s1", turn_index=1, input_tokens=100, output_tokens=50)

        with pytest.raises(ValueError, match="doesn't match expected"):
            session.add_turn(turn)

    def test_session_get_turn(self) -> None:
        """Test getting turn by index."""
        turns = [
            Turn(session_id="s1", turn_index=0, input_tokens=100, output_tokens=50),
            Turn(session_id="s1", turn_index=1, input_tokens=200, output_tokens=75),
        ]
        session = Session(session_id="s1", turns=turns)

        assert session.get_turn(0) is not None
        assert session.get_turn(0).input_tokens == 100
        assert session.get_turn(1) is not None
        assert session.get_turn(1).input_tokens == 200
        assert session.get_turn(2) is None

    def test_session_to_dict(self) -> None:
        """Test serialization to dictionary."""
        turn = Turn(session_id="s1", turn_index=0, input_tokens=100, output_tokens=50)
        session = Session(
            session_id="s1",
            turns=[turn],
            original_start_time_ms=1000000,
            metadata={"adapter_name": "lora1"},
        )
        d = session.to_dict()
        assert d["session_id"] == "s1"
        assert len(d["turns"]) == 1
        assert d["original_start_time_ms"] == 1000000
        assert d["metadata"]["adapter_name"] == "lora1"
        assert d["num_turns"] == 1

    def test_session_from_dict(self) -> None:
        """Test deserialization from dictionary."""
        data = {
            "session_id": "s1",
            "turns": [
                {"session_id": "s1", "turn_index": 0, "input_tokens": 100, "output_tokens": 50}
            ],
            "original_start_time_ms": 1000000,
        }
        session = Session.from_dict(data)
        assert session.session_id == "s1"
        assert session.num_turns == 1
        assert session.original_start_time_ms == 1000000

    def test_session_to_csv_rows(self) -> None:
        """Test CSV rows conversion."""
        turns = [
            Turn(session_id="s1", turn_index=0, input_tokens=100, output_tokens=50),
            Turn(session_id="s1", turn_index=1, input_tokens=200, output_tokens=75),
        ]
        session = Session(session_id="s1", turns=turns)
        rows = session.to_csv_rows()
        assert len(rows) == 2
        assert rows[0]["session_id"] == "s1"
        assert rows[0]["turn_index"] == 0
        assert rows[1]["turn_index"] == 1

    def test_session_context_growth_rate(self) -> None:
        """Test context growth rate calculation."""
        turns = [
            Turn(session_id="s1", turn_index=0, input_tokens=100, output_tokens=50, new_context_tokens=0),
            Turn(session_id="s1", turn_index=1, input_tokens=200, output_tokens=75, new_context_tokens=100),
            Turn(session_id="s1", turn_index=2, input_tokens=350, output_tokens=100, new_context_tokens=150),
        ]
        session = Session(session_id="s1", turns=turns)
        # Average growth: (100 + 150) / 2 = 125
        assert session.context_growth_rate == 125.0


class TestSessionSummary:
    """Tests for SessionSummary model."""

    def test_create_summary_from_session(self) -> None:
        """Test creating summary from session."""
        tc = ToolCall(name="t1", duration_ms=100, result_tokens=10)
        turns = [
            Turn(session_id="s1", turn_index=0, input_tokens=100, output_tokens=50, llm_latency_ms=500, tool_calls=[tc]),
            Turn(session_id="s1", turn_index=1, input_tokens=200, output_tokens=75, llm_latency_ms=700),
        ]
        session = Session(session_id="s1", turns=turns)

        summary = SessionSummary.from_session(session)
        assert summary.session_id == "s1"
        assert summary.num_turns == 2
        assert summary.num_tool_calls == 1
        assert summary.total_input_tokens == 300
        assert summary.total_output_tokens == 125
        assert summary.max_context_length == 200
        assert summary.total_tool_pause_ms == 100
        assert summary.total_llm_latency_ms == 1200

    def test_summary_to_dict(self) -> None:
        """Test summary serialization."""
        summary = SessionSummary(
            session_id="s1",
            num_turns=3,
            num_tool_calls=2,
            total_input_tokens=500,
            total_output_tokens=250,
        )
        d = summary.to_dict()
        assert d["session_id"] == "s1"
        assert d["num_turns"] == 3
        assert d["num_tool_calls"] == 2
        assert d["total_input_tokens"] == 500
        assert d["total_output_tokens"] == 250
