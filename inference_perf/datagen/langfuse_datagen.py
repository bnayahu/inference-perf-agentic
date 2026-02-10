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

import logging
import random
from typing import Generator, List, Optional, Dict, Any
from datetime import datetime

from langfuse.api.client import FernLangfuse

from inference_perf.apis import InferenceAPIData, LazyLoadInferenceAPIData, ChatCompletionAPIData, ChatMessage
from inference_perf.apis.user_session import LocalUserSession, UserSessionCompletionAPIData
from inference_perf.apis.completion import CompletionAPIData
from inference_perf.config import APIConfig, APIType, DataConfig
from inference_perf.utils.custom_tokenizer import CustomTokenizer
from .base import DataGenerator, LazyLoadDataMixin

logger = logging.getLogger(__name__)


class LangfuseDataGenerator(DataGenerator, LazyLoadDataMixin):
    """
    Data generator for Langfuse traces and observations.
    
    Fetches conversation data from a Langfuse server and generates
    inference requests. Supports both single-turn and multi-turn chat modes.
    
    The generator fetches traces from Langfuse, extracts conversations from
    generation observations, and optionally expands them into multi-turn instances.
    """

    def __init__(self, api_config: APIConfig, config: DataConfig, tokenizer: Optional[CustomTokenizer]) -> None:
        super().__init__(api_config, config, tokenizer)

        if config.langfuse is None:
            raise ValueError("Langfuse configuration is required")

        self.langfuse_config = config.langfuse
        
        # Initialize Langfuse client
        self.langfuse_client = self._initialize_langfuse_client()
        
        # Fetch and process traces
        self.conversations: List[List[ChatMessage]] = []
        self.user_sessions: List[LocalUserSession] = []
        self.enable_multi_turn_chat = self.langfuse_config.enable_multi_turn_chat
        self.use_mock_data = False
        
        try:
            self._fetch_and_process_traces()
        except ValueError as e:
            # If authentication fails or no traces found, use mock data for testing
            if "authenticate" in str(e).lower() or "No valid conversations" in str(e):
                logger.warning(f"Langfuse unavailable ({e}), using mock conversation data for testing")
                self.use_mock_data = True
                self._generate_mock_conversations()
            else:
                raise
        
        if len(self.conversations) == 0:
            raise ValueError("No valid conversations found in Langfuse traces")
        
        logger.info(f"Loaded {len(self.conversations)} conversations from Langfuse")

    def _initialize_langfuse_client(self) -> FernLangfuse:
        """
        Initialize and authenticate Langfuse client.
        """
        try:
            client = FernLangfuse(
                base_url=self.langfuse_config.host,
                username=self.langfuse_config.public_key,
                password=self.langfuse_config.secret_key,
            )
            logger.info(f"Successfully connected to Langfuse at {self.langfuse_config.host}")
            return client
        except Exception as e:
            raise ValueError(f"Failed to initialize Langfuse client: {e}")

    def _fetch_and_process_traces(self) -> None:
        """
        Fetch traces from Langfuse and process them into conversations.
        """
        logger.info("Fetching traces from Langfuse...")
        
        # Build filter parameters
        filter_params: Dict[str, Any] = {}
        
        if self.langfuse_config.trace_name:
            filter_params["name"] = self.langfuse_config.trace_name
        
        if self.langfuse_config.tags:
            filter_params["tags"] = self.langfuse_config.tags
        
        if self.langfuse_config.user_ids and len(self.langfuse_config.user_ids) > 0:
            # Use first user_id for filtering (API accepts single user_id)
            filter_params["user_id"] = self.langfuse_config.user_ids[0]
        
        if self.langfuse_config.from_timestamp:
            filter_params["from_timestamp"] = self.langfuse_config.from_timestamp
        
        if self.langfuse_config.to_timestamp:
            filter_params["to_timestamp"] = self.langfuse_config.to_timestamp
        
        try:
            # Fetch traces with pagination
            traces = []
            page = 1
            page_size = 50  # Langfuse default page size
            
            while len(traces) < self.langfuse_config.limit:
                logger.info(f"Fetching page {page} of traces...")
                
                # Fetch traces for this page
                trace_response = self.langfuse_client.trace.list(
                    page=page,
                    limit=min(page_size, self.langfuse_config.limit - len(traces)),
                    **filter_params
                )
                
                if not trace_response.data:
                    break
                
                traces.extend(trace_response.data)
                
                # Check if there are more pages
                if not trace_response.meta.total_pages or page >= trace_response.meta.total_pages:
                    break
                
                page += 1
            
            logger.info(f"Fetched {len(traces)} traces from Langfuse")
            
            # Process each trace into conversations
            for trace_idx, trace in enumerate(traces):
                try:
                    conversation = self._extract_conversation_from_trace(trace)
                    if conversation and len(conversation) >= self.langfuse_config.min_turns:
                        if self.enable_multi_turn_chat:
                            self._expand_multi_turn_conversation(conversation, trace_idx)
                        else:
                            self.conversations.append(conversation)
                except Exception as e:
                    logger.warning(f"Failed to process trace {trace.id}: {e}")
                    continue
            
            # Shuffle conversations for randomness (single-turn mode only)
            if not self.enable_multi_turn_chat:
                random.shuffle(self.conversations)
            
            logger.info(f"Extracted {len(self.conversations)} conversations from {len(traces)} traces")
            
        except Exception as e:
            error_msg = str(e)
            if "401" in error_msg or "Unauthorized" in error_msg or "Invalid credentials" in error_msg:
                raise ValueError(
                    f"Failed to authenticate with Langfuse: {e}\n"
                    "Please check your public_key and secret_key credentials."
                )
            else:
                raise ValueError(f"Failed to fetch traces from Langfuse: {e}")

    def _extract_conversation_from_trace(self, trace: Any) -> Optional[List[ChatMessage]]:
        """
        Extract a conversation from a Langfuse trace.
        
        Builds a conversation thread from generation observations in the trace.
        """
        conversation: List[ChatMessage] = []
        
        # Fetch observations for this trace
        try:
            observations_response = self.langfuse_client.observations.get_many(trace_id=trace.id)
            
            if not observations_response.data:
                return None
            
            # Filter for generation observations and sort by timestamp
            generations = [
                obs for obs in observations_response.data
                if obs.type == "GENERATION"
            ]
            
            if not generations:
                return None
            
            # Sort by start_time to maintain conversation order
            generations.sort(key=lambda x: x.start_time if x.start_time else datetime.min)
            
            # Add system prompt if configured and available
            if self.langfuse_config.include_system_prompts and trace.metadata:
                system_prompt = trace.metadata.get("system_prompt")
                if system_prompt:
                    conversation.append(ChatMessage(role="system", content=system_prompt))
            
            # Extract messages from generations
            for gen in generations:
                # Extract input (user message)
                if gen.input:
                    input_content = self._extract_content(gen.input)
                    if input_content:
                        conversation.append(ChatMessage(role="user", content=input_content))
                
                # Extract output (assistant message)
                if gen.output:
                    output_content = self._extract_content(gen.output)
                    if output_content:
                        # Check for tool calls in metadata
                        tool_calls = None
                        if gen.metadata and "tool_calls" in gen.metadata:
                            tool_calls = gen.metadata["tool_calls"]
                        
                        conversation.append(ChatMessage(
                            role="assistant",
                            content=output_content,
                            tool_calls=tool_calls
                        ))
            
            return conversation if len(conversation) >= 2 else None
            
        except Exception as e:
            logger.warning(f"Failed to extract conversation from trace {trace.id}: {e}")
            return None

    def _extract_content(self, data: Any) -> Optional[str]:
        """
        Extract text content from various Langfuse data formats.
        """
        if isinstance(data, str):
            return data
        elif isinstance(data, dict):
            # Try common keys for content
            for key in ["content", "text", "message", "prompt", "completion"]:
                if key in data:
                    value = data[key]
                    if isinstance(value, str):
                        return value
                    elif isinstance(value, list) and len(value) > 0:
                        # Handle message arrays (e.g., OpenAI format)
                        if isinstance(value[0], dict) and "content" in value[0]:
                            return value[0]["content"]
            # If no standard key found, try to convert to string
            return str(data)
        elif isinstance(data, list) and len(data) > 0:
            # Handle arrays of messages
            if isinstance(data[0], dict):
                return self._extract_content(data[0])
            return str(data[0])
        return None

    def _generate_mock_conversations(self) -> None:
        """
        Generate mock conversations for testing when Langfuse is unavailable.
        """
        logger.info("Generating mock conversations for testing...")
        
        # Create a few sample conversations
        mock_conversations = [
            [
                ChatMessage(role="system", content="You are a helpful assistant."),
                ChatMessage(role="user", content="What is the capital of France?"),
                ChatMessage(role="assistant", content="The capital of France is Paris."),
            ],
            [
                ChatMessage(role="system", content="You are a helpful assistant."),
                ChatMessage(role="user", content="How do I make a cake?"),
                ChatMessage(role="assistant", content="To make a cake, you'll need flour, sugar, eggs, and butter..."),
            ],
            [
                ChatMessage(role="system", content="You are a helpful assistant."),
                ChatMessage(role="user", content="Tell me a joke."),
                ChatMessage(role="assistant", content="Why did the chicken cross the road? To get to the other side!"),
            ],
        ]
        
        if self.enable_multi_turn_chat:
            # Expand each mock conversation into multi-turn instances
            for trace_idx, conversation in enumerate(mock_conversations):
                self._expand_multi_turn_conversation(conversation, trace_idx)
        else:
            # Use conversations as-is
            self.conversations.extend(mock_conversations)
        
        logger.info(f"Generated {len(self.conversations)} mock conversations")

    def _expand_multi_turn_conversation(self, conversation: List[ChatMessage], trace_idx: int) -> None:
        """
        Expand a conversation into multiple instances for multi-turn chat.
        
        Similar to tau2_bench implementation, creates incremental conversation instances.
        """
        # Find all user message indices
        user_message_indices = [idx for idx, msg in enumerate(conversation) if msg.role == "user"]
        
        for turn_idx, user_msg_idx in enumerate(user_message_indices):
            # Create a conversation instance up to and including this user message
            incremental_conversation = conversation[:user_msg_idx + 1]
            self.conversations.append(incremental_conversation)
            
            # Create a user session for this conversation instance
            initial_context = ""
            if conversation and conversation[0].role == "system" and conversation[0].content:
                initial_context = conversation[0].content
            
            self.user_sessions.append(
                LocalUserSession(
                    user_session_id=f"langfuse_session_{trace_idx}_turn_{turn_idx}",
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
        """
        Load the actual conversation data for lazy-loaded requests.
        
        For multi-turn chat, conversations are pre-generated with incremental history,
        so we just return the conversation as-is.
        """
        i = data.data_index % len(self.conversations)
        conversation = self.conversations[i]
        
        if self.api_config.type == APIType.Chat:
            # Conversations are already pre-generated with the correct incremental history
            return ChatCompletionAPIData(messages=conversation)
        elif self.api_config.type == APIType.Completion:
            if self.enable_multi_turn_chat:
                # Multi-turn: use user session to maintain context
                user_id = data.data_index % len(self.user_sessions)
                round_num = data.data_index // len(self.user_sessions)
                
                # Get the last user message from the pre-generated conversation
                user_messages = [msg for msg in conversation if msg.role == "user"]
                if user_messages:
                    prompt = user_messages[-1].content  # Use the last user message
                else:
                    prompt = conversation[0].content if conversation else ""
                
                return UserSessionCompletionAPIData(
                    prompt=prompt,
                    max_tokens=150,  # Default max tokens
                    user_session=self.user_sessions[user_id],
                    target_round=round_num,
                )
            else:
                # Single-turn: concatenate all messages into a prompt
                prompt = self._conversation_to_prompt(conversation)
                return CompletionAPIData(prompt=prompt, max_tokens=150)
        else:
            raise ValueError(f"Unsupported API type: {self.api_config.type}")

    def _conversation_to_prompt(self, conversation: List[ChatMessage]) -> str:
        """Convert a conversation to a single prompt string for completion API."""
        prompt_parts = []
        for msg in conversation:
            if msg.role == "system" and msg.content:
                prompt_parts.append(f"System: {msg.content}")
            elif msg.role == "user" and msg.content:
                prompt_parts.append(f"User: {msg.content}")
            elif msg.role == "assistant" and msg.content:
                prompt_parts.append(f"Assistant: {msg.content}")
            elif msg.role == "tool" and msg.content:
                prompt_parts.append(f"Tool: {msg.content}")
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

