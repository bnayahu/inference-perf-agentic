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

"""Agentic CSV data generator.

Loads agentic sessions from CSV files containing turn-level data.
"""

import csv
import logging
import random
import string
from pathlib import Path
from typing import Dict, Generator, List, Optional

from inference_perf.apis import (
    InferenceAPIData,
    LazyLoadInferenceAPIData,
    ChatCompletionAPIData,
    ChatMessage,
    CompletionAPIData,
)
from inference_perf.config import (
    APIConfig,
    APIType,
    DataConfig,
    AgenticCsvConfig,
)
from inference_perf.utils.custom_tokenizer import CustomTokenizer
from inference_perf.models import Session, Turn, ToolCall, FinishReason
from .base import DataGenerator, LazyLoadDataMixin

logger = logging.getLogger(__name__)


# Expected CSV columns
CSV_COLUMNS = [
    "session_id",
    "turn_index",
    "input_tokens",
    "output_tokens",
    "finish_reason",
    "num_tool_calls",
    "tool_duration_ms",
    "tool_result_tokens",
    "llm_latency_ms",
]

# Optional CSV columns
OPTIONAL_COLUMNS = [
    "ttft_ms",
    "timestamp_ms",
]


class AgenticCsvDataGenerator(DataGenerator, LazyLoadDataMixin):
    """Load agentic sessions from CSV file.

    CSV Format:
    session_id,turn_index,input_tokens,output_tokens,finish_reason,
    num_tool_calls,tool_duration_ms,tool_result_tokens,llm_latency_ms

    Optional columns:
    ttft_ms,timestamp_ms
    """

    def __init__(
        self,
        api_config: APIConfig,
        config: DataConfig,
        tokenizer: Optional[CustomTokenizer],
    ) -> None:
        super().__init__(api_config, config, tokenizer)

        if config.agentic_csv is None:
            raise ValueError("agentic_csv configuration is required")

        self.csv_config = config.agentic_csv
        self.sessions: List[Session] = []

        # Load sessions from CSV
        self._load_csv()

        if not self.sessions:
            raise ValueError(f"No valid sessions found in CSV file: {self.csv_config.path}")

        logger.info(
            f"Loaded {len(self.sessions)} sessions from CSV "
            f"with average {sum(s.num_turns for s in self.sessions) / len(self.sessions):.1f} turns"
        )

    def _load_csv(self) -> None:
        """Load and parse the CSV file into Session objects."""
        csv_path = Path(self.csv_config.path)

        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        # Read CSV and group by session_id
        session_data: Dict[str, List[dict]] = {}

        with open(csv_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)

            # Validate required columns
            if reader.fieldnames is None:
                raise ValueError("CSV file has no header row")

            missing_columns = set(CSV_COLUMNS) - set(reader.fieldnames)
            if missing_columns:
                raise ValueError(f"CSV missing required columns: {missing_columns}")

            for row in reader:
                session_id = row['session_id']
                if session_id not in session_data:
                    session_data[session_id] = []
                session_data[session_id].append(row)

        # Convert to Session objects
        for session_id, rows in session_data.items():
            try:
                session = self._build_session(session_id, rows)
                self.sessions.append(session)
            except Exception as e:
                logger.warning(f"Failed to parse session {session_id}: {e}")

    def _build_session(self, session_id: str, rows: List[dict]) -> Session:
        """Build a Session object from CSV rows."""
        # Sort by turn_index
        rows.sort(key=lambda r: int(r['turn_index']))

        turns: List[Turn] = []
        prev_input_tokens = 0

        for row in rows:
            turn = self._build_turn(session_id, row, prev_input_tokens)
            turns.append(turn)
            prev_input_tokens = turn.input_tokens

        # Get earliest timestamp for session start time
        original_start_time_ms = None
        if turns and turns[0].timestamp_ms is not None:
            original_start_time_ms = turns[0].timestamp_ms

        return Session(
            session_id=session_id,
            turns=turns,
            original_start_time_ms=original_start_time_ms,
        )

    def _build_turn(
        self,
        session_id: str,
        row: dict,
        prev_input_tokens: int,
    ) -> Turn:
        """Build a Turn object from a CSV row."""
        turn_index = int(row['turn_index'])
        input_tokens = int(row['input_tokens'])
        output_tokens = int(row['output_tokens'])

        # Calculate new context tokens
        new_context_tokens = max(0, input_tokens - prev_input_tokens) if turn_index > 0 else 0

        # Parse finish reason
        finish_reason_str = row.get('finish_reason', 'stop')
        try:
            finish_reason = FinishReason(finish_reason_str)
        except ValueError:
            finish_reason = FinishReason.UNKNOWN

        # Build tool calls
        tool_calls = self._build_tool_calls(row)

        # Optional fields
        llm_latency_ms = self._parse_optional_int(row.get('llm_latency_ms'))
        ttft_ms = self._parse_optional_int(row.get('ttft_ms'))
        timestamp_ms = self._parse_optional_int(row.get('timestamp_ms'))

        return Turn(
            session_id=session_id,
            turn_index=turn_index,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            new_context_tokens=new_context_tokens,
            finish_reason=finish_reason,
            tool_calls=tool_calls,
            llm_latency_ms=llm_latency_ms,
            ttft_ms=ttft_ms,
            timestamp_ms=timestamp_ms,
        )

    def _build_tool_calls(self, row: dict) -> List[ToolCall]:
        """Build tool call list from CSV row."""
        num_tool_calls = int(row.get('num_tool_calls', 0))
        if num_tool_calls == 0:
            return []

        total_duration = int(row.get('tool_duration_ms', 0))
        total_result_tokens = int(row.get('tool_result_tokens', 0))

        # Distribute evenly across tool calls
        tool_calls = []
        for i in range(num_tool_calls):
            tool_calls.append(ToolCall(
                name=f"tool_{i}",
                duration_ms=total_duration // num_tool_calls,
                result_tokens=total_result_tokens // num_tool_calls,
            ))

        return tool_calls

    def _parse_optional_int(self, value: Optional[str]) -> Optional[int]:
        """Parse an optional integer value."""
        if value is None or value == '':
            return None
        try:
            return int(value)
        except (ValueError, TypeError):
            return None

    def _generate_content(self, token_count: int) -> str:
        """Generate synthetic content for the specified token count."""
        char_count = max(1, token_count * 4)
        return ''.join(random.choices(string.ascii_letters + string.digits + ' ', k=char_count))

    def get_sessions(self) -> List[Session]:
        """Get all loaded sessions."""
        return self.sessions

    def get_supported_apis(self) -> List[APIType]:
        return [APIType.Chat, APIType.Completion]

    def is_io_distribution_supported(self) -> bool:
        return False

    def is_shared_prefix_supported(self) -> bool:
        return False

    def is_prefered_worker_requested(self) -> bool:
        return True

    def load_lazy_data(self, data: LazyLoadInferenceAPIData) -> InferenceAPIData:
        """Load actual inference data for a lazy reference."""
        session_idx = data.data_index % len(self.sessions)
        session = self.sessions[session_idx]

        # Determine which turn within the session
        turn_idx = (data.data_index // len(self.sessions)) % session.num_turns
        turn = session.turns[turn_idx]

        if self.api_config.type == APIType.Chat:
            messages = self._build_messages_for_turn(session, turn_idx)
            return ChatCompletionAPIData(
                messages=messages,
                program_id=session.session_id,
                turn_index=turn_idx,
            )
        else:
            prompt = self._build_prompt_for_turn(session, turn_idx)
            return CompletionAPIData(
                prompt=prompt,
                max_tokens=turn.output_tokens,
                program_id=session.session_id,
                turn_index=turn_idx,
            )

    def _build_messages_for_turn(
        self,
        session: Session,
        target_turn: int,
    ) -> List[ChatMessage]:
        """Build message history up to and including target turn."""
        messages: List[ChatMessage] = []

        for i in range(target_turn + 1):
            turn = session.turns[i]

            # User message
            user_content = self._generate_content(turn.new_context_tokens or 100)
            messages.append(ChatMessage(role="user", content=user_content))

            # Assistant response (for previous turns)
            if i < target_turn:
                assistant_content = self._generate_content(turn.output_tokens)
                messages.append(ChatMessage(role="assistant", content=assistant_content))

                # Tool responses
                for tc in turn.tool_calls:
                    tool_content = self._generate_content(tc.result_tokens)
                    messages.append(ChatMessage(
                        role="tool",
                        content=tool_content,
                        id=tc.tool_call_id,
                    ))

        return messages

    def _build_prompt_for_turn(
        self,
        session: Session,
        target_turn: int,
    ) -> str:
        """Build prompt string up to target turn."""
        parts = []

        for i in range(target_turn + 1):
            turn = session.turns[i]

            user_content = self._generate_content(turn.new_context_tokens or 100)
            parts.append(f"User: {user_content}")

            if i < target_turn:
                assistant_content = self._generate_content(turn.output_tokens)
                parts.append(f"Assistant: {assistant_content}")

        return "\n".join(parts)

    def get_data(self) -> Generator[InferenceAPIData, None, None]:
        """Generate inference requests from sessions."""
        if not self.sessions:
            return

        i = 0
        while True:
            session_idx = i % len(self.sessions)
            yield LazyLoadInferenceAPIData(
                data_index=i,
                prefered_worker_id=session_idx,
            )
            i += 1
