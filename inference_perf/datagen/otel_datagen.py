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

"""OpenTelemetry data generator for inference-perf."""

import logging
import random
from datetime import datetime, timedelta
from typing import Generator, List, Optional, Tuple

from inference_perf.apis import (
    InferenceAPIData,
    LazyLoadInferenceAPIData,
    ChatCompletionAPIData,
    ChatMessage,
    CompletionAPIData,
)
from inference_perf.apis.user_session import LocalUserSession, UserSessionCompletionAPIData
from inference_perf.config import APIConfig, APIType, DataConfig, OTelBackendType
from inference_perf.utils.custom_tokenizer import CustomTokenizer
from inference_perf.client.otel import JaegerClient, OTelBackendClient, OTelTrace, OTelSpan
from inference_perf.utils.otel_extractor import GenAIMessageExtractor, ToolResponseExtractor
from inference_perf.models import Session, Turn, ToolCall, FinishReason
from .base import DataGenerator, LazyLoadDataMixin

logger = logging.getLogger(__name__)


class OpenTelemetryDataGenerator(DataGenerator, LazyLoadDataMixin):
    """
    Data generator for OpenTelemetry traces.

    Fetches traces from OTel-compatible backends and extracts LLM conversations
    using GenAI semantic conventions. Supports multi-turn conversations with
    tool calls.
    """

    def __init__(
        self,
        api_config: APIConfig,
        config: DataConfig,
        tokenizer: Optional[CustomTokenizer]
    ) -> None:
        super().__init__(api_config, config, tokenizer)

        if config.otel is None:
            raise ValueError("OpenTelemetry configuration is required")

        self.otel_config = config.otel

        # Initialize backend client
        self.client = self._initialize_client()

        # Initialize extractors
        self.message_extractor = GenAIMessageExtractor()
        self.tool_extractor = ToolResponseExtractor()

        # Storage for conversations
        self.conversations: List[List[ChatMessage]] = []
        self.user_sessions: List[LocalUserSession] = []
        self.conversation_metadata: List[tuple[str, int]] = []
        self.enable_multi_turn_chat = self.otel_config.enable_multi_turn_chat

        # Storage for agentic sessions
        self.sessions: List[Session] = []

        # Fetch and process traces
        try:
            self._fetch_and_process_traces()
        except Exception as e:
            logger.error(f"Failed to fetch traces: {e}")
            raise ValueError(f"Failed to initialize OpenTelemetry data generator: {e}")

        if len(self.conversations) == 0:
            raise ValueError("No valid conversations found in OpenTelemetry traces")

        logger.info(f"Loaded {len(self.conversations)} conversations from OpenTelemetry traces")

    def _initialize_client(self) -> OTelBackendClient:
        """Initialize the appropriate backend client."""
        backend = self.otel_config.backend

        if backend == OTelBackendType.JAEGER:
            return JaegerClient(
                endpoint=self.otel_config.endpoint,
                auth=self.otel_config.auth
            )
        else:
            raise ValueError(f"Unsupported OpenTelemetry backend: {backend}")

    def _fetch_and_process_traces(self) -> None:
        """Fetch traces from backend and process them into conversations."""
        logger.info("Fetching traces from OpenTelemetry backend...")

        # Parse time range
        start_time, end_time = self._parse_time_range()

        # Parse tags
        tags_dict = self._parse_tags()

        # Query traces
        try:
            traces = self.client.query_traces(
                service_name=self.otel_config.service_name,
                operation_name=self.otel_config.operation_name,
                tags=tags_dict,
                start_time=start_time,
                end_time=end_time,
                limit=self.otel_config.limit,
                min_duration_ms=self.otel_config.min_duration_ms,
                max_duration_ms=self.otel_config.max_duration_ms,
            )
            logger.info(f"Fetched {len(traces)} traces from backend")

        except Exception as e:
            logger.error(f"Failed to fetch traces: {e}")
            raise

        # Process each trace
        for trace in traces:
            try:
                # Extract agentic session (new)
                session = self._extract_session_from_trace(trace)
                if session is not None:
                    self.sessions.append(session)

                # Extract conversations (existing behavior)
                conversations = self._extract_conversations_from_trace(trace)
                for conversation in conversations:
                    # Filter by minimum turns
                    if len(conversation) >= self.otel_config.min_turns:
                        if self.enable_multi_turn_chat:
                            self._expand_multi_turn_conversation(conversation, trace)
                        else:
                            self.conversations.append(conversation)
                            self.conversation_metadata.append((trace.trace_id, 0))
            except Exception as e:
                logger.warning(f"Failed to process trace {trace.trace_id}: {e}")
                continue

        # Shuffle conversations for randomness (single-turn mode only)
        if not self.enable_multi_turn_chat:
            combined = list(zip(self.conversations, self.conversation_metadata))
            random.shuffle(combined)
            self.conversations, self.conversation_metadata = zip(*combined) if combined else ([], [])
            self.conversations = list(self.conversations)
            self.conversation_metadata = list(self.conversation_metadata)

        logger.info(f"Extracted {len(self.conversations)} conversations and {len(self.sessions)} sessions from {len(traces)} traces")

    def _parse_time_range(self) -> tuple[Optional[datetime], Optional[datetime]]:
        """Parse time range from config."""
        start_time = self.otel_config.start_time
        end_time = self.otel_config.end_time

        # Handle lookback
        if self.otel_config.lookback and not start_time:
            lookback = self.otel_config.lookback
            # Parse lookback string (e.g., "24h", "7d", "30m")
            if lookback.endswith('h'):
                hours = int(lookback[:-1])
                start_time = datetime.utcnow() - timedelta(hours=hours)
            elif lookback.endswith('d'):
                days = int(lookback[:-1])
                start_time = datetime.utcnow() - timedelta(days=days)
            elif lookback.endswith('m'):
                minutes = int(lookback[:-1])
                start_time = datetime.utcnow() - timedelta(minutes=minutes)
            else:
                logger.warning(f"Unknown lookback format: {lookback}, using default 24h")
                start_time = datetime.utcnow() - timedelta(hours=24)

        if not end_time and start_time:
            end_time = datetime.utcnow()

        return start_time, end_time

    def _parse_tags(self) -> Optional[dict]:
        """Parse tags from config into dict."""
        if not self.otel_config.tags:
            return None

        tags_dict = {}
        for tag in self.otel_config.tags:
            if '=' in tag:
                key, value = tag.split('=', 1)
                tags_dict[key.strip()] = value.strip()

        return tags_dict if tags_dict else None

    def _extract_session_from_trace(self, trace: OTelTrace) -> Optional[Session]:
        """Extract an agentic Session from an OTel trace.

        Maps OTel spans to Turn objects, extracting:
        - Token counts from gen_ai.usage attributes
        - Tool calls and their durations from child spans
        - Timestamps for trace replay
        """
        # Find all LLM spans (spans with gen_ai.system attribute)
        llm_spans = [
            span for span in trace.spans
            if 'gen_ai.system' in span.attributes
        ]

        if not llm_spans:
            return None

        # Sort spans by start time to get turn order
        llm_spans.sort(key=lambda s: s.start_time)

        # Filter by minimum turns
        if len(llm_spans) < self.otel_config.min_turns:
            return None

        turns: List[Turn] = []
        prev_input_tokens = 0

        for turn_idx, span in enumerate(llm_spans):
            # Extract token usage
            input_tokens, output_tokens = self.message_extractor.extract_usage(span)

            # Calculate new context tokens
            new_context_tokens = max(0, input_tokens - prev_input_tokens) if turn_idx > 0 else 0
            prev_input_tokens = input_tokens

            # Determine finish reason
            finish_reason = self._get_finish_reason(span)

            # Extract tool calls
            tool_calls = self._extract_tool_calls_from_span(span, trace.spans)

            # Extract timing information
            llm_latency_ms = int(span.duration_ms)
            ttft_ms = self._extract_ttft_from_span(span)
            timestamp_ms = int(span.start_time.timestamp() * 1000)

            turn = Turn(
                session_id=trace.trace_id,
                turn_index=turn_idx,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                new_context_tokens=new_context_tokens,
                finish_reason=finish_reason,
                tool_calls=tool_calls,
                llm_latency_ms=llm_latency_ms,
                ttft_ms=ttft_ms,
                timestamp_ms=timestamp_ms,
            )
            turns.append(turn)

        if not turns:
            return None

        # Create session
        session = Session(
            session_id=trace.trace_id,
            turns=turns,
            original_start_time_ms=int(trace.start_time.timestamp() * 1000),
        )

        return session

    def _get_finish_reason(self, span: OTelSpan) -> FinishReason:
        """Extract finish reason from span attributes."""
        # Check completion attributes for finish reason
        for key, value in span.attributes.items():
            if 'finish_reason' in key:
                if value == 'tool_calls':
                    return FinishReason.TOOL_CALLS
                elif value == 'stop':
                    return FinishReason.STOP
                elif value == 'length':
                    return FinishReason.LENGTH
                elif value == 'content_filter':
                    return FinishReason.CONTENT_FILTER

        # Default to stop
        return FinishReason.STOP

    def _extract_tool_calls_from_span(
        self,
        llm_span: OTelSpan,
        all_spans: List[OTelSpan]
    ) -> List[ToolCall]:
        """Extract tool calls and their durations from span and child spans."""
        tool_calls: List[ToolCall] = []

        # Find tool call attributes in the LLM span
        tc_indices = self.message_extractor._get_attribute_indices(
            llm_span.attributes, 'gen_ai.completion.0.tool_calls'
        )

        # Find child spans that represent tool executions
        child_spans = [s for s in all_spans if s.parent_span_id == llm_span.span_id]

        for tc_idx in tc_indices:
            prefix = f'gen_ai.completion.0.tool_calls.{tc_idx}'
            tool_call_id = llm_span.attributes.get(f'{prefix}.id')
            function_name = llm_span.attributes.get(f'{prefix}.function.name')
            function_args = llm_span.attributes.get(f'{prefix}.function.arguments')

            if function_name:
                # Find matching tool execution span for duration
                duration_ms = 0
                result_tokens = 0

                tool_span = self._find_tool_execution_span(
                    child_spans, function_name, tool_call_id
                )

                if tool_span:
                    duration_ms = int(tool_span.duration_ms)
                    # Try to estimate result tokens from tool result
                    result_tokens = self._estimate_tool_result_tokens(tool_span)

                tool_calls.append(ToolCall(
                    name=function_name,
                    duration_ms=duration_ms,
                    result_tokens=result_tokens,
                    arguments=function_args,
                    tool_call_id=tool_call_id,
                ))

        return tool_calls

    def _find_tool_execution_span(
        self,
        child_spans: List[OTelSpan],
        tool_name: str,
        tool_call_id: Optional[str]
    ) -> Optional[OTelSpan]:
        """Find the span corresponding to a tool execution."""
        for span in child_spans:
            # Check operation name
            if tool_name.lower() in span.operation_name.lower():
                return span

            # Check various attribute patterns
            if span.attributes.get('tool.name') == tool_name:
                return span

            if tool_call_id and span.attributes.get('tool.call.id') == tool_call_id:
                return span

            if span.attributes.get('function.name') == tool_name:
                return span

        return None

    def _estimate_tool_result_tokens(self, tool_span: OTelSpan) -> int:
        """Estimate the number of tokens in a tool result."""
        # Try to get result from span
        result = (
            tool_span.attributes.get('tool.result') or
            tool_span.attributes.get('function.result') or
            tool_span.attributes.get('output') or
            tool_span.attributes.get('response') or
            tool_span.attributes.get('result')
        )

        if result is not None:
            result_str = str(result)
            # Rough estimation: ~4 characters per token
            return len(result_str) // 4

        return 0

    def _extract_ttft_from_span(self, span: OTelSpan) -> Optional[int]:
        """Extract time to first token from span events or attributes."""
        # Check for TTFT in attributes
        ttft = span.attributes.get('gen_ai.ttft_ms')
        if ttft is not None:
            try:
                return int(ttft)
            except (TypeError, ValueError):
                pass

        # Check span events for first token event
        for event in span.events:
            if 'first_token' in event.get('name', '').lower():
                # Calculate TTFT from event timestamp relative to span start
                event_time = event.get('timestamp')
                if event_time:
                    try:
                        if isinstance(event_time, datetime):
                            ttft_ms = int((event_time - span.start_time).total_seconds() * 1000)
                            return max(0, ttft_ms)
                    except (TypeError, ValueError):
                        pass

        return None

    def get_sessions(self) -> List[Session]:
        """Get extracted sessions for agentic workload generation.

        Returns:
            List of Session objects extracted from OTel traces.
        """
        return self.sessions

    def _extract_conversations_from_trace(self, trace: OTelTrace) -> List[List[ChatMessage]]:
        """Extract conversations from a trace."""
        conversations = []

        # Find all LLM spans (spans with gen_ai.system attribute)
        llm_spans = [
            span for span in trace.spans
            if 'gen_ai.system' in span.attributes
        ]

        if not llm_spans:
            return conversations

        # Sort spans by start time
        llm_spans.sort(key=lambda s: s.start_time)

        # Build conversation from LLM spans
        conversation: List[ChatMessage] = []

        for span in llm_spans:
            # Extract messages from this span
            messages = self.message_extractor.extract_messages_from_span(span)

            # If this is the first span and we should include system prompts
            if not conversation and self.otel_config.include_system_prompts:
                # Add system message if present in prompt messages
                system_msgs = [m for m in messages if m.role == 'system']
                if system_msgs:
                    conversation.extend(system_msgs)
                    # Remove system messages from the list
                    messages = [m for m in messages if m.role != 'system']

            # Add non-system messages
            for msg in messages:
                if msg.role != 'system':
                    conversation.append(msg)

            # If this span has tool calls, try to find tool responses
            if self.otel_config.extract_tool_calls:
                # Check if the last message has tool calls
                if conversation and conversation[-1].tool_calls:
                    tool_responses = self.tool_extractor.extract_tool_responses(
                        conversation[-1].tool_calls,
                        trace.spans,
                        span.span_id
                    )
                    conversation.extend(tool_responses)

            # Extract tool definitions if present (for first LLM span)
            if span == llm_spans[0]:
                tools = self.message_extractor.extract_tools(span)
                if tools and conversation and conversation[0].role == 'system':
                    # Add tools to system message
                    conversation[0].tools = tools

        if conversation:
            conversations.append(conversation)

        return conversations

    def _expand_multi_turn_conversation(
        self,
        conversation: List[ChatMessage],
        trace: OTelTrace
    ) -> None:
        """
        Expand a conversation into multiple instances for multi-turn chat.

        Similar to tau2_bench and langfuse implementations.
        """
        # Find all user message indices
        user_message_indices = [
            idx for idx, msg in enumerate(conversation)
            if msg.role == "user"
        ]

        program_id = trace.trace_id

        for turn_idx, user_msg_idx in enumerate(user_message_indices):
            # Create a conversation instance up to and including this user message
            incremental_conversation = conversation[:user_msg_idx + 1]
            self.conversations.append(incremental_conversation)
            self.conversation_metadata.append((program_id, turn_idx))

            # Create a user session for this conversation instance
            initial_context = ""
            if conversation and conversation[0].role == "system" and conversation[0].content:
                initial_context = conversation[0].content

            self.user_sessions.append(
                LocalUserSession(
                    user_session_id=f"otel_session_{program_id}_turn_{turn_idx}",
                    context=initial_context
                )
            )

    def get_supported_apis(self) -> List[APIType]:
        return [APIType.Chat, APIType.Completion]

    def is_io_distribution_supported(self) -> bool:
        return False

    def is_shared_prefix_supported(self) -> bool:
        return True

    def is_prefered_worker_requested(self) -> bool:
        return True if self.enable_multi_turn_chat else False

    def load_lazy_data(self, data: LazyLoadInferenceAPIData) -> InferenceAPIData:
        """Load the actual conversation data for lazy-loaded requests."""
        i = data.data_index % len(self.conversations)
        conversation = self.conversations[i]
        program_id, turn_index = self.conversation_metadata[i]

        if self.api_config.type == APIType.Chat:
            return ChatCompletionAPIData(
                messages=conversation,
                program_id=program_id,
                turn_index=turn_index
            )
        elif self.api_config.type == APIType.Completion:
            if self.enable_multi_turn_chat:
                # Multi-turn: use user session to maintain context
                user_id = data.data_index % len(self.user_sessions)
                round_num = data.data_index // len(self.user_sessions)

                # Get the last user message
                user_messages = [msg for msg in conversation if msg.role == "user"]
                if user_messages:
                    prompt = user_messages[-1].content or ""
                else:
                    prompt = conversation[0].content if conversation else ""

                return UserSessionCompletionAPIData(
                    prompt=prompt,
                    max_tokens=150,
                    user_session=self.user_sessions[user_id],
                    target_round=round_num,
                    program_id=program_id,
                    turn_index=turn_index,
                )
            else:
                # Single-turn: concatenate all messages
                prompt = self._conversation_to_prompt(conversation)
                return CompletionAPIData(
                    prompt=prompt,
                    max_tokens=150,
                    program_id=program_id,
                    turn_index=turn_index,
                )
        else:
            raise ValueError(f"Unsupported API type: {self.api_config.type}")

    def _conversation_to_prompt(self, conversation: List[ChatMessage]) -> str:
        """Convert a conversation to a single prompt string for completion API."""
        prompt_parts = []
        for msg in conversation:
            content = msg.content or ""
            if msg.role == "system":
                prompt_parts.append(f"System: {content}")
            elif msg.role == "user":
                prompt_parts.append(f"User: {content}")
            elif msg.role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
            elif msg.role == "tool":
                prompt_parts.append(f"Tool: {content}")
        return "\n".join(prompt_parts)

    def get_data(self) -> Generator[InferenceAPIData, None, None]:
        """Generate inference requests from the loaded conversations."""
        if not self.conversations:
            return

        i = 0
        while True:
            prefered_worker_id = i % len(self.conversations) if self.enable_multi_turn_chat else -1
            yield LazyLoadInferenceAPIData(data_index=i, prefered_worker_id=prefered_worker_id)
            i += 1
