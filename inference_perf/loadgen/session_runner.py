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

"""Session runner for agentic workload execution.

The SessionRunner manages the execution of a single agentic session,
handling context accumulation, inter-turn delays, and result collection.
"""

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any, List, Optional

from inference_perf.apis import ChatCompletionAPIData, ChatMessage, InferenceAPIData
from inference_perf.client.modelserver import ModelServerClient
from inference_perf.config import AgenticDelayConfig, DelayConfig, DelayType, APIConfig
from inference_perf.models import Session, Turn, FinishReason

logger = logging.getLogger(__name__)


@dataclass
class TurnResult:
    """Result of a single turn execution."""
    session_id: str
    turn_index: int
    scheduled_time: float
    start_time: float
    end_time: float
    input_tokens: int
    output_tokens: int
    ttft_ms: Optional[float] = None
    success: bool = True
    error_message: Optional[str] = None


@dataclass
class SessionResult:
    """Result of a complete session execution."""
    session_id: str
    turn_results: List[TurnResult] = field(default_factory=list)
    start_time: float = 0.0
    end_time: float = 0.0

    @property
    def session_latency_ms(self) -> float:
        """Total wall-clock time including pauses."""
        return (self.end_time - self.start_time) * 1000

    @property
    def session_inference_time_ms(self) -> float:
        """Sum of LLM call durations (excludes pauses)."""
        return sum(
            (tr.end_time - tr.start_time) * 1000
            for tr in self.turn_results
        )

    @property
    def session_pause_time_ms(self) -> float:
        """Total pause time between turns."""
        return self.session_latency_ms - self.session_inference_time_ms

    @property
    def inference_duty_cycle(self) -> float:
        """Ratio of inference time to total latency."""
        if self.session_latency_ms == 0:
            return 0.0
        return self.session_inference_time_ms / self.session_latency_ms

    @property
    def turns_completed(self) -> int:
        """Number of successfully completed turns."""
        return sum(1 for tr in self.turn_results if tr.success)

    @property
    def session_input_tokens(self) -> int:
        """Total input tokens across all turns."""
        return sum(tr.input_tokens for tr in self.turn_results)

    @property
    def session_output_tokens(self) -> int:
        """Total output tokens across all turns."""
        return sum(tr.output_tokens for tr in self.turn_results)

    @property
    def peak_context_length(self) -> int:
        """Maximum input tokens in any single turn."""
        if not self.turn_results:
            return 0
        return max(tr.input_tokens for tr in self.turn_results)


class SessionRunner:
    """Manages execution of a single agentic session.

    The SessionRunner iterates through turns in a session, building
    requests with accumulated context, applying inter-turn delays,
    and collecting results.
    """

    def __init__(
        self,
        session: Session,
        client: ModelServerClient,
        api_config: APIConfig,
        delay_config: Optional[AgenticDelayConfig] = None,
        stage_id: int = 0,
        lora_adapter: Optional[str] = None,
        data_generator: Optional[Any] = None,
    ):
        """Initialize the session runner.

        Args:
            session: The Session object containing turns to execute.
            client: Model server client for LLM calls.
            api_config: API configuration for requests.
            delay_config: Configuration for inter-turn delays.
            stage_id: Load stage identifier.
            lora_adapter: Optional LoRA adapter name.
            data_generator: Optional data generator for loading lazy data.
        """
        self.session = session
        self.client = client
        self.api_config = api_config
        self.delay_config = delay_config or AgenticDelayConfig()
        self.stage_id = stage_id
        self.lora_adapter = lora_adapter
        self.data_generator = data_generator

        # Context accumulation
        self.context: List[ChatMessage] = []
        self.results: List[TurnResult] = []

    async def run(self) -> SessionResult:
        """Execute all turns in the session.

        Returns:
            SessionResult containing results for all turns.
        """
        session_result = SessionResult(session_id=self.session.session_id)
        session_result.start_time = time.perf_counter()

        for turn in self.session.turns:
            try:
                # Build and send request
                turn_result = await self._execute_turn(turn)
                self.results.append(turn_result)
                session_result.turn_results.append(turn_result)

                if not turn_result.success:
                    logger.warning(
                        f"Session {self.session.session_id} turn {turn.turn_index} failed: "
                        f"{turn_result.error_message}"
                    )
                    break

                # Apply inter-turn delay if not the last turn
                if turn.turn_index < len(self.session.turns) - 1:
                    await self._apply_inter_turn_delay(turn)

            except Exception as e:
                logger.error(
                    f"Session {self.session.session_id} turn {turn.turn_index} exception: {e}"
                )
                error_result = TurnResult(
                    session_id=self.session.session_id,
                    turn_index=turn.turn_index,
                    scheduled_time=time.perf_counter(),
                    start_time=time.perf_counter(),
                    end_time=time.perf_counter(),
                    input_tokens=turn.input_tokens,
                    output_tokens=0,
                    success=False,
                    error_message=str(e),
                )
                session_result.turn_results.append(error_result)
                break

        session_result.end_time = time.perf_counter()

        logger.debug(
            f"Session {self.session.session_id} completed: "
            f"{session_result.turns_completed}/{len(self.session.turns)} turns, "
            f"{session_result.session_latency_ms:.0f}ms total"
        )

        return session_result

    async def _execute_turn(self, turn: Turn) -> TurnResult:
        """Execute a single turn.

        Args:
            turn: The Turn object to execute.

        Returns:
            TurnResult with execution metrics.
        """
        scheduled_time = time.perf_counter()

        # Build request with accumulated context
        request = self._build_request(turn)

        start_time = time.perf_counter()

        try:
            # Process the request through the client
            await self.client.process_request(
                request,
                self.stage_id,
                scheduled_time,
                self.lora_adapter
            )

            end_time = time.perf_counter()

            # Update context with the response (for next turn)
            self._update_context(turn)

            return TurnResult(
                session_id=self.session.session_id,
                turn_index=turn.turn_index,
                scheduled_time=scheduled_time,
                start_time=start_time,
                end_time=end_time,
                input_tokens=turn.input_tokens,
                output_tokens=turn.output_tokens,
                success=True,
            )

        except Exception as e:
            end_time = time.perf_counter()
            return TurnResult(
                session_id=self.session.session_id,
                turn_index=turn.turn_index,
                scheduled_time=scheduled_time,
                start_time=start_time,
                end_time=end_time,
                input_tokens=turn.input_tokens,
                output_tokens=0,
                success=False,
                error_message=str(e),
            )

    def _build_request(self, turn: Turn) -> InferenceAPIData:
        """Build an inference request for a turn.

        Args:
            turn: The Turn object containing request parameters.

        Returns:
            InferenceAPIData ready to send to the model server.
        """
        # If data generator is available and supports lazy loading, use it
        if self.data_generator and hasattr(self.data_generator, 'load_lazy_data'):
            # Create a LazyLoadInferenceAPIData reference
            from inference_perf.apis import LazyLoadInferenceAPIData

            # Find the session index
            session_idx = next(
                (i for i, s in enumerate(self.data_generator.sessions) if s.session_id == self.session.session_id),
                0
            )

            # Calculate data_index: session_idx + turn_idx * num_sessions
            data_index = session_idx + turn.turn_index * len(self.data_generator.sessions)

            lazy_data = LazyLoadInferenceAPIData(
                data_index=data_index,
                prefered_worker_id=session_idx,
            )

            # Load the actual data with proper messages including system prompt
            return self.data_generator.load_lazy_data(lazy_data)

        # Fallback: build a ChatCompletionAPIData with placeholder messages
        messages: List[ChatMessage] = []

        # Add accumulated context
        messages.extend(self.context)

        # Add a user message for this turn
        messages.append(ChatMessage(
            role="user",
            content=f"Turn {turn.turn_index} input"
        ))

        return ChatCompletionAPIData(
            messages=messages,
            program_id=self.session.session_id,
            turn_index=turn.turn_index,
        )

    def _update_context(self, turn: Turn) -> None:
        """Update context with turn results.

        Args:
            turn: The completed Turn.
        """
        # Add assistant response to context
        self.context.append(ChatMessage(
            role="assistant",
            content=f"Turn {turn.turn_index} response"
        ))

        # If there were tool calls, add tool messages
        if turn.has_tool_calls:
            for tool_call in turn.tool_calls:
                # Add tool call message
                self.context.append(ChatMessage(
                    role="tool",
                    content=f"Tool result for {tool_call.name}",
                    id=tool_call.tool_call_id,
                ))

    async def _apply_inter_turn_delay(self, turn: Turn) -> None:
        """Apply delay between turns.

        Args:
            turn: The completed Turn (determines which delay to apply).
        """
        if turn.finish_reason == FinishReason.TOOL_CALLS:
            delay_ms = self._compute_delay(
                self.delay_config.tool_call_delay,
                turn
            )
        else:
            delay_ms = self._compute_delay(
                self.delay_config.user_think_delay,
                turn
            )

        if delay_ms > 0:
            await asyncio.sleep(delay_ms / 1000)

    def _compute_delay(self, config: DelayConfig, turn: Turn) -> float:
        """Compute delay in milliseconds based on configuration.

        Args:
            config: Delay configuration.
            turn: The current turn (for replay mode).

        Returns:
            Delay in milliseconds.
        """
        if config.type == DelayType.ZERO:
            return 0.0

        elif config.type == DelayType.FIXED:
            return float(config.fixed_ms or 0)

        elif config.type == DelayType.REPLAY:
            # Use the original tool duration from the trace
            return float(turn.total_tool_duration_ms)

        elif config.type == DelayType.DISTRIBUTION:
            if config.distribution is None:
                return 0.0

            dist = config.distribution
            if dist.type == "normal":
                value = random.gauss(dist.mean, dist.std_dev)
                return max(dist.min, min(dist.max, value))
            elif dist.type == "uniform":
                return random.uniform(dist.min, dist.max)
            elif dist.type == "constant":
                return dist.value or 0.0
            else:
                return 0.0

        return 0.0
