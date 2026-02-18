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

"""Tests for Turn and ToolCall models."""

import pytest
from inference_perf.models import Turn, ToolCall, FinishReason


class TestToolCall:
    """Tests for ToolCall model."""

    def test_create_tool_call(self) -> None:
        """Test basic tool call creation."""
        tc = ToolCall(
            name="get_weather",
            duration_ms=100,
            result_tokens=50,
        )
        assert tc.name == "get_weather"
        assert tc.duration_ms == 100
        assert tc.result_tokens == 50

    def test_tool_call_defaults(self) -> None:
        """Test tool call default values."""
        tc = ToolCall(name="test_tool")
        assert tc.duration_ms == 0
        assert tc.result_tokens == 0
        assert tc.arguments is None
        assert tc.tool_call_id is None

    def test_tool_call_to_dict(self) -> None:
        """Test serialization to dictionary."""
        tc = ToolCall(
            name="search",
            duration_ms=200,
            result_tokens=100,
            arguments='{"query": "test"}',
            tool_call_id="tc_123",
        )
        d = tc.to_dict()
        assert d["name"] == "search"
        assert d["duration_ms"] == 200
        assert d["result_tokens"] == 100
        assert d["arguments"] == '{"query": "test"}'
        assert d["tool_call_id"] == "tc_123"

    def test_tool_call_from_dict(self) -> None:
        """Test deserialization from dictionary."""
        data = {
            "name": "calculate",
            "duration_ms": 50,
            "result_tokens": 25,
        }
        tc = ToolCall.from_dict(data)
        assert tc.name == "calculate"
        assert tc.duration_ms == 50
        assert tc.result_tokens == 25


class TestTurn:
    """Tests for Turn model."""

    def test_create_turn(self) -> None:
        """Test basic turn creation."""
        turn = Turn(
            session_id="session_1",
            turn_index=0,
            input_tokens=100,
            output_tokens=50,
        )
        assert turn.session_id == "session_1"
        assert turn.turn_index == 0
        assert turn.input_tokens == 100
        assert turn.output_tokens == 50

    def test_turn_defaults(self) -> None:
        """Test turn default values."""
        turn = Turn(
            session_id="s1",
            turn_index=0,
            input_tokens=100,
            output_tokens=50,
        )
        assert turn.new_context_tokens == 0
        assert turn.finish_reason == FinishReason.STOP
        assert turn.tool_calls == []
        assert turn.llm_latency_ms is None
        assert turn.ttft_ms is None
        assert turn.timestamp_ms is None

    def test_turn_with_tool_calls(self) -> None:
        """Test turn with tool calls."""
        tc = ToolCall(name="tool1", duration_ms=100, result_tokens=50)
        turn = Turn(
            session_id="s1",
            turn_index=0,
            input_tokens=100,
            output_tokens=50,
            finish_reason=FinishReason.TOOL_CALLS,
            tool_calls=[tc],
        )
        assert turn.has_tool_calls is True
        assert len(turn.tool_calls) == 1
        assert turn.finish_reason == FinishReason.TOOL_CALLS

    def test_turn_total_tool_duration(self) -> None:
        """Test calculation of total tool duration."""
        tcs = [
            ToolCall(name="t1", duration_ms=100, result_tokens=10),
            ToolCall(name="t2", duration_ms=200, result_tokens=20),
        ]
        turn = Turn(
            session_id="s1",
            turn_index=0,
            input_tokens=100,
            output_tokens=50,
            tool_calls=tcs,
        )
        assert turn.total_tool_duration_ms == 300

    def test_turn_total_tool_result_tokens(self) -> None:
        """Test calculation of total tool result tokens."""
        tcs = [
            ToolCall(name="t1", duration_ms=100, result_tokens=10),
            ToolCall(name="t2", duration_ms=200, result_tokens=20),
        ]
        turn = Turn(
            session_id="s1",
            turn_index=0,
            input_tokens=100,
            output_tokens=50,
            tool_calls=tcs,
        )
        assert turn.total_tool_result_tokens == 30

    def test_turn_to_dict(self) -> None:
        """Test serialization to dictionary."""
        turn = Turn(
            session_id="s1",
            turn_index=1,
            input_tokens=200,
            output_tokens=100,
            new_context_tokens=100,
            finish_reason=FinishReason.STOP,
            llm_latency_ms=500,
        )
        d = turn.to_dict()
        assert d["session_id"] == "s1"
        assert d["turn_index"] == 1
        assert d["input_tokens"] == 200
        assert d["output_tokens"] == 100
        assert d["new_context_tokens"] == 100
        assert d["finish_reason"] == "stop"
        assert d["llm_latency_ms"] == 500

    def test_turn_from_dict(self) -> None:
        """Test deserialization from dictionary."""
        data = {
            "session_id": "s1",
            "turn_index": 0,
            "input_tokens": 100,
            "output_tokens": 50,
            "finish_reason": "tool_calls",
            "tool_calls": [
                {"name": "t1", "duration_ms": 100, "result_tokens": 10}
            ],
        }
        turn = Turn.from_dict(data)
        assert turn.session_id == "s1"
        assert turn.turn_index == 0
        assert turn.finish_reason == FinishReason.TOOL_CALLS
        assert len(turn.tool_calls) == 1
        assert turn.tool_calls[0].name == "t1"

    def test_turn_to_csv_row(self) -> None:
        """Test CSV row conversion."""
        turn = Turn(
            session_id="s1",
            turn_index=0,
            input_tokens=100,
            output_tokens=50,
            finish_reason=FinishReason.STOP,
            tool_calls=[ToolCall(name="t1", duration_ms=100, result_tokens=25)],
            llm_latency_ms=500,
        )
        row = turn.to_csv_row()
        assert row["session_id"] == "s1"
        assert row["turn_index"] == 0
        assert row["input_tokens"] == 100
        assert row["output_tokens"] == 50
        assert row["finish_reason"] == "stop"
        assert row["num_tool_calls"] == 1
        assert row["tool_duration_ms"] == 100
        assert row["tool_result_tokens"] == 25
        assert row["llm_latency_ms"] == 500


class TestFinishReason:
    """Tests for FinishReason enum."""

    def test_finish_reason_values(self) -> None:
        """Test finish reason enum values."""
        assert FinishReason.STOP.value == "stop"
        assert FinishReason.TOOL_CALLS.value == "tool_calls"
        assert FinishReason.LENGTH.value == "length"
        assert FinishReason.CONTENT_FILTER.value == "content_filter"
        assert FinishReason.UNKNOWN.value == "unknown"
