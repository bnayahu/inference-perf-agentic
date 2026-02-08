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

import json
import logging
import os
import random
import urllib.request
import urllib.error
from typing import Generator, List, Optional
from urllib.parse import urlparse

from inference_perf.apis import InferenceAPIData, LazyLoadInferenceAPIData, ChatCompletionAPIData, ChatMessage
from inference_perf.apis.user_session import LocalUserSession, UserSessionCompletionAPIData
from inference_perf.apis.completion import CompletionAPIData
from inference_perf.config import APIConfig, APIType, DataConfig
from inference_perf.utils.custom_tokenizer import CustomTokenizer
from .base import DataGenerator, LazyLoadDataMixin

logger = logging.getLogger(__name__)


class Tau2BenchDataGenerator(DataGenerator, LazyLoadDataMixin):
    """
    Data generator for tau2-bench simulation files.
    
    Loads conversation data from tau2-bench simulation JSON files and generates
    inference requests. Supports both single-turn and multi-turn chat modes.
    
    The simulation files contain conversations between users and agents with
    multiple turns of dialogue that can be used for benchmarking.
    """

    def __init__(self, api_config: APIConfig, config: DataConfig, tokenizer: Optional[CustomTokenizer]) -> None:
        super().__init__(api_config, config, tokenizer)

        if config.path is None:
            raise ValueError("Path or URL to tau2-bench simulation file is required")

        # Load simulation data (supports both local files and URLs)
        self.simulation_data = self._load_simulation_file(config.path)
        
        # Extract conversations from simulations
        self.conversations: List[List[ChatMessage]] = []
        self.user_sessions: List[LocalUserSession] = []
        self.enable_multi_turn_chat = config.tau2_bench is not None and config.tau2_bench.enable_multi_turn_chat
        
        self._extract_conversations()
        
        if len(self.conversations) == 0:
            raise ValueError(f"No valid conversations found in {config.path}")
        
        logger.info(f"Loaded {len(self.conversations)} conversations from tau2-bench simulation file")

    def _load_simulation_file(self, path: str) -> dict:
        """
        Load and parse the tau2-bench simulation JSON file.
        Supports both local file paths and URLs (http/https).
        """
        try:
            # Check if path is a URL
            parsed = urlparse(path)
            is_url = parsed.scheme in ('http', 'https')
            
            if is_url:
                logger.info(f"Loading simulation data from URL: {path}")
                # Handle GitHub blob URLs by converting to raw content URLs
                if 'github.com' in parsed.netloc and '/blob/' in path:
                    path = path.replace('github.com', 'raw.githubusercontent.com').replace('/blob/', '/')
                    logger.info(f"Converted GitHub URL to raw content URL: {path}")
                
                with urllib.request.urlopen(path) as response:
                    data = json.loads(response.read().decode('utf-8'))
            else:
                # Local file path
                if not os.path.exists(path):
                    raise ValueError(f"Local file does not exist: {path}")
                logger.info(f"Loading simulation data from local file: {path}")
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            
            return data
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON from {path}: {e}")
        except urllib.error.URLError as e:
            raise ValueError(f"Failed to download file from URL {path}: {e}")
        except Exception as e:
            raise ValueError(f"Failed to load simulation file {path}: {e}")

    def _extract_conversations(self) -> None:
        """
        Extract conversations from simulation data.
        
        Each simulation contains a series of messages between user and assistant.
        We extract these as separate conversations that can be used for benchmarking.
        """
        simulations = self.simulation_data.get("simulations", [])
        
        for sim_idx, simulation in enumerate(simulations):
            messages = simulation.get("messages", [])
            
            if len(messages) < 2:
                continue
            
            conversation: List[ChatMessage] = []
            
            for msg in messages:
                role = msg.get("role")
                content = msg.get("content")
                
                # Skip tool messages and messages without content
                if role == "tool" or content is None:
                    continue
                
                # Map roles to standard chat format
                if role in ["user", "assistant"]:
                    conversation.append(ChatMessage(role=role, content=content))
            
            if len(conversation) >= 2:
                self.conversations.append(conversation)
                
                # For multi-turn chat, create user sessions
                if self.enable_multi_turn_chat:
                    # Use first message as context (system prompt)
                    initial_context = conversation[0].content if conversation else ""
                    self.user_sessions.append(
                        LocalUserSession(
                            user_session_id=f"tau2_session_{sim_idx}",
                            context=initial_context
                        )
                    )
        
        # Shuffle conversations for randomness
        if self.enable_multi_turn_chat:
            # Shuffle both lists together to maintain correspondence
            combined = list(zip(self.conversations, self.user_sessions))
            random.shuffle(combined)
            if combined:
                conversations_tuple, sessions_tuple = zip(*combined)
                self.conversations = list(conversations_tuple)
                self.user_sessions = list(sessions_tuple)
        else:
            random.shuffle(self.conversations)

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
        
        if self.api_config.type == APIType.Chat:
            return ChatCompletionAPIData(messages=conversation)
        elif self.api_config.type == APIType.Completion:
            if self.enable_multi_turn_chat:
                # Multi-turn: use user session to maintain context
                user_id = data.data_index % len(self.user_sessions)
                round_num = data.data_index // len(self.user_sessions)
                
                # Get the user's message (skip first if it's the context)
                user_messages = [msg for msg in conversation if msg.role == "user"]
                if user_messages:
                    prompt = user_messages[0].content
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
            if msg.role == "user":
                prompt_parts.append(f"User: {msg.content}")
            elif msg.role == "assistant":
                prompt_parts.append(f"Assistant: {msg.content}")
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

