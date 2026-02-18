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

"""Agentic synthetic data generator.

Generates synthetic agentic sessions based on configurable distributions
for turns, tokens, and tool calls.
"""

import logging
import random
import string
from typing import Generator, List, Optional
from uuid import uuid4

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
    AgenticSyntheticConfig,
    DistributionParams,
)
from inference_perf.utils.custom_tokenizer import CustomTokenizer
from inference_perf.models import Session, Turn, ToolCall, FinishReason
from .base import DataGenerator, LazyLoadDataMixin

logger = logging.getLogger(__name__)


class AgenticSyntheticDataGenerator(DataGenerator, LazyLoadDataMixin):
    """Generate synthetic agentic sessions from distributions.

    Creates sessions with configurable:
    - Number of turns per session
    - Tool call probability and count
    - Input/output token distributions
    - Context growth patterns
    """

    def __init__(
        self,
        api_config: APIConfig,
        config: DataConfig,
        tokenizer: Optional[CustomTokenizer],
    ) -> None:
        super().__init__(api_config, config, tokenizer)

        if config.agentic_synthetic is None:
            raise ValueError("agentic_synthetic configuration is required")

        self.agentic_config = config.agentic_synthetic
        self.sessions: List[Session] = []

        # Generate sessions
        self._generate_sessions()

        logger.info(
            f"Generated {len(self.sessions)} synthetic sessions "
            f"with average {sum(s.num_turns for s in self.sessions) / len(self.sessions):.1f} turns"
        )

    def _generate_sessions(self) -> None:
        """Generate all synthetic sessions."""
        for _ in range(self.agentic_config.num_sessions):
            session = self._generate_session()
            self.sessions.append(session)

    def _generate_session(self) -> Session:
        """Generate a single synthetic session."""
        session_id = str(uuid4())

        # Sample number of turns
        num_turns = int(self._sample_distribution(
            self.agentic_config.turns_per_session
        ))
        num_turns = max(1, num_turns)

        turns: List[Turn] = []
        context_tokens = int(self._sample_distribution(
            self.agentic_config.input_tokens_turn_0
        ))
        context_tokens = max(10, context_tokens)

        # Add system prompt tokens to initial context
        context_tokens += self.agentic_config.system_prompt_tokens

        for turn_idx in range(num_turns):
            turn = self._generate_turn(session_id, turn_idx, context_tokens)
            turns.append(turn)

            # Update context for next turn
            context_tokens = self._compute_next_context(turn, context_tokens)

        return Session(
            session_id=session_id,
            turns=turns,
        )

    def _generate_turn(
        self,
        session_id: str,
        turn_index: int,
        context_tokens: int,
    ) -> Turn:
        """Generate a single turn."""
        # Sample output tokens
        output_tokens = int(self._sample_distribution(
            self.agentic_config.output_tokens_per_turn
        ))
        output_tokens = max(1, output_tokens)

        # Determine finish reason
        has_tool_calls = random.random() < self.agentic_config.tool_call_probability
        finish_reason = FinishReason.TOOL_CALLS if has_tool_calls else FinishReason.STOP

        # Generate tool calls if applicable
        tool_calls: List[ToolCall] = []
        if has_tool_calls:
            num_tool_calls = int(self._sample_distribution(
                self.agentic_config.tool_calls_per_turn
            ))
            num_tool_calls = max(1, num_tool_calls)

            for tc_idx in range(num_tool_calls):
                tool_call = self._generate_tool_call(tc_idx)
                tool_calls.append(tool_call)

        # Calculate new context tokens
        new_context_tokens = 0
        if turn_index > 0:
            # Context growth from previous turn's output + tool results
            new_context_tokens = output_tokens + sum(tc.result_tokens for tc in tool_calls)

        return Turn(
            session_id=session_id,
            turn_index=turn_index,
            input_tokens=context_tokens,
            output_tokens=output_tokens,
            new_context_tokens=new_context_tokens,
            finish_reason=finish_reason,
            tool_calls=tool_calls,
        )

    def _generate_tool_call(self, index: int) -> ToolCall:
        """Generate a synthetic tool call."""
        # Sample result tokens
        result_tokens = int(self._sample_distribution(
            self.agentic_config.tool_result_tokens
        ))
        result_tokens = max(1, result_tokens)

        # Sample duration (using result tokens as rough proxy)
        # Assume ~100ms per 100 tokens of processing
        duration_ms = max(10, result_tokens)

        return ToolCall(
            name=f"tool_{index}",
            duration_ms=duration_ms,
            result_tokens=result_tokens,
            tool_call_id=f"tc_{uuid4().hex[:8]}",
        )

    def _compute_next_context(self, turn: Turn, current_context: int) -> int:
        """Compute context size for next turn."""
        # Add output tokens and tool results to context
        new_tokens = turn.output_tokens + sum(tc.result_tokens for tc in turn.tool_calls)
        return current_context + new_tokens

    def _sample_distribution(self, params: DistributionParams) -> float:
        """Sample a value from the configured distribution."""
        if params.type == "normal":
            value = random.gauss(params.mean, params.std_dev)
            return max(params.min, min(params.max, value))
        elif params.type == "uniform":
            return random.uniform(params.min, params.max)
        elif params.type == "constant":
            return params.value or params.mean
        else:
            return params.mean

    def _generate_content(self, token_count: int) -> str:
        """Generate synthetic content for the specified token count.

        Args:
            token_count: Approximate number of tokens to generate.

        Returns:
            String of synthetic content.
        """
        strategy = self.agentic_config.content_strategy

        if strategy == "random":
            # Random alphanumeric content (~4 chars per token)
            char_count = token_count * 4
            return ''.join(random.choices(string.ascii_letters + string.digits + ' ', k=char_count))

        elif strategy == "template":
            # Repeated template content
            template = "This is a synthetic message for benchmarking purposes. "
            repeats = max(1, token_count // 10)
            return template * repeats

        else:  # synthetic
            # More realistic synthetic text
            words = [
                "the", "be", "to", "of", "and", "a", "in", "that", "have", "it",
                "for", "not", "on", "with", "he", "as", "you", "do", "at", "this",
                "but", "his", "by", "from", "they", "we", "say", "her", "she", "or",
                "an", "will", "my", "one", "all", "would", "there", "their", "what",
            ]
            word_count = max(1, token_count * 3 // 4)  # ~0.75 words per token
            return ' '.join(random.choices(words, k=word_count))

    def get_sessions(self) -> List[Session]:
        """Get all generated sessions."""
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
            # Build messages for chat completion
            messages = self._build_messages_for_turn(session, turn_idx)
            return ChatCompletionAPIData(
                messages=messages,
                program_id=session.session_id,
                turn_index=turn_idx,
            )
        else:
            # Build prompt for completion
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

        # Add system message if configured
        if self.agentic_config.system_prompt_tokens > 0:
            system_content = self._generate_content(
                self.agentic_config.system_prompt_tokens
            )
            messages.append(ChatMessage(role="system", content=system_content))

        # Add previous turns' messages
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

        # Add system prompt
        if self.agentic_config.system_prompt_tokens > 0:
            system_content = self._generate_content(
                self.agentic_config.system_prompt_tokens
            )
            parts.append(f"System: {system_content}")

        # Add previous turns
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
            # Assign preferred worker based on session
            session_idx = i % len(self.sessions)
            yield LazyLoadInferenceAPIData(
                data_index=i,
                prefered_worker_id=session_idx,
            )
            i += 1
