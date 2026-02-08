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

# Copied from https://github.com/sierra-research/tau2-bench/blob/37199f36924c8896f5e048360691f8476cd89ba1/src/tau2/agent/llm_agent.py
AGENT_INSTRUCTION = """
You are a customer service agent that helps the user according to the <policy> provided below.
In each turn you can either:
- Send a message to the user.
- Make a tool call.
You cannot do both at the same time.

Try to be helpful and always follow the policy. Always make sure you generate valid JSON only.
""".strip()

SYSTEM_PROMPT = """
<instructions>
{agent_instruction}
</instructions>
<policy>
{domain_policy}
</policy>
""".strip()

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
        
        # Resolve the tau2 domain and get the corresponding system prompt
        self.tau2_domain = self.simulation_data.get("info",{}).get("environment_info", {}).get("domain_name", "")
        self.tau2_system_prompt = self._get_system_prompt(self.tau2_domain)
        self.tau2_domain_tools = self._get_domain_tools(self.tau2_domain)

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

    def _get_system_prompt(self, domain_name: str) -> str:
        """
        Get the Tau2-bench system prompt for the domain. 
        """
        policy_file = f"https://raw.githubusercontent.com/sierra-research/tau2-bench/main/web/leaderboard/public/task-data/domains/{domain_name}/policy.md"
        try:
            with urllib.request.urlopen(policy_file) as response:
                domain_policy = response.read().decode('utf-8')
        except Exception as e:
            logger.error(f"Failed to load policy file for domain '{domain_name}': {e}")
            domain_policy = ""

        system_prompt = SYSTEM_PROMPT.format(agent_instruction=AGENT_INSTRUCTION, domain_policy=domain_policy)
        return system_prompt
    
    def _get_domain_tools(self, domain_name: str) -> List:
        """
        Get the tool definitions for the domain. 
        """
        tools_file = f"https://raw.githubusercontent.com/sierra-research/tau2-bench/main/web/leaderboard/public/task-data/tools-data.json"
        try:
            with urllib.request.urlopen(tools_file) as response:
                tools_data = json.loads(response.read().decode('utf-8'))
                tools_list = tools_data.get(domain_name,{}).get("tools", [])
                return tools_list
        except Exception as e:
            logger.error(f"Failed to load tools file for domain '{domain_name}': {e}")
            return []
        
    def _extract_conversations(self) -> None:
        """
        Extract conversations from simulation data.
        
        Each simulation contains a series of messages between user and assistant.
        We extract these as separate conversations that can be used for benchmarking.
        
        When enable_multi_turn_chat is True, we create multiple conversation instances
        from each simulation, where each instance includes the conversation history up to
        and including a specific user message.
        """
        simulations = self.simulation_data.get("simulations", [])
        
        for sim_idx, simulation in enumerate(simulations):
            messages = simulation.get("messages", [])
            
            if len(messages) < 2:
                continue
            
            conversation: List[ChatMessage] = []
            
            # system prompt goes on top
            conversation.append(ChatMessage(role="system", content=self.tau2_system_prompt, tools=self.tau2_domain_tools))

            for msg in messages:
                role = msg.get("role")
                content = msg.get("content")
                tool_calls = msg.get("tool_calls")
                msg_id = msg.get("id")
                
                # Include user, assistant, and tool messages
                if role == "user":
                    conversation.append(ChatMessage(role=role, content=content))
                elif role == "assistant":
                    # Assistant messages may have content, tool_calls, or both
                    conversation.append(ChatMessage(
                        role=role,
                        content=content,
                        tool_calls=tool_calls
                    ))
                elif role == "tool":
                    # Tool messages have an id and content
                    conversation.append(ChatMessage(
                        role=role,
                        content=content,
                        id=msg_id
                    ))
            
            if len(conversation) >= 2:
                if self.enable_multi_turn_chat:
                    # Create multiple conversation instances, one for each user message
                    # Each instance includes all messages up to and including that user message
                    user_message_indices = [idx for idx, msg in enumerate(conversation) if msg.role == "user"]
                    
                    for turn_idx, user_msg_idx in enumerate(user_message_indices):
                        # Create a conversation instance up to and including this user message
                        incremental_conversation = conversation[:user_msg_idx + 1]
                        self.conversations.append(incremental_conversation)
                        
                        # Create a user session for this conversation instance
                        # Use first message as context (system prompt)
                        initial_context = conversation[0].content if conversation and conversation[0].content is not None else ""
                        self.user_sessions.append(
                            LocalUserSession(
                                user_session_id=f"tau2_session_{sim_idx}_turn_{turn_idx}",
                                context=initial_context
                            )
                        )
                else:
                    # Single-turn: just add the full conversation
                    self.conversations.append(conversation)
        
        # Shuffle conversations for randomness (single-turn mode only)
        if not self.enable_multi_turn_chat:
            random.shuffle(self.conversations)

        logger.info(f"Extracted {len(self.conversations)} conversations from {len(simulations)} simulations")
                
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
            if msg.role == "user" and msg.content:
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

