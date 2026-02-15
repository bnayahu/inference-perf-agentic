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
from typing import Generator, List, Optional

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
from inference_perf.client.otel import JaegerClient, OTelBackendClient, OTelTrace
from inference_perf.utils.otel_extractor import GenAIMessageExtractor, ToolResponseExtractor
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

        logger.info(f"Extracted {len(self.conversations)} conversations from {len(traces)} traces")

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
