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

"""Message extraction from OpenTelemetry spans using GenAI semantic conventions."""

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from inference_perf.apis import ChatMessage
from inference_perf.client.otel import OTelSpan

logger = logging.getLogger(__name__)


class GenAIMessageExtractor:
    """Extract LLM messages from OTel spans using GenAI semantic conventions."""

    def extract_messages_from_span(self, span: OTelSpan) -> List[ChatMessage]:
        """
        Extract messages from a span using gen_ai.* attributes.

        GenAI semantic conventions v1.28.0+:
        - gen_ai.prompt.{index}.role
        - gen_ai.prompt.{index}.content
        - gen_ai.completion.{index}.role
        - gen_ai.completion.{index}.content
        - gen_ai.completion.{index}.tool_calls.{index}.*
        """
        messages = []

        # Extract prompts (input messages)
        prompt_messages = self._extract_prompts(span)
        messages.extend(prompt_messages)

        # Extract completions (output messages)
        completion_messages = self._extract_completions(span)
        messages.extend(completion_messages)

        return messages

    def _extract_prompts(self, span: OTelSpan) -> List[ChatMessage]:
        """Extract prompt messages from gen_ai.prompt.* attributes."""
        messages = []
        prompt_indices = self._get_attribute_indices(span.attributes, 'gen_ai.prompt')

        for idx in prompt_indices:
            role = span.attributes.get(f'gen_ai.prompt.{idx}.role')
            content = span.attributes.get(f'gen_ai.prompt.{idx}.content')
            tool_call_id = span.attributes.get(f'gen_ai.prompt.{idx}.tool_call_id')

            if role:
                # For tool role messages, include the id
                if role == 'tool' and tool_call_id:
                    messages.append(ChatMessage(
                        role=role,
                        content=content or "",
                        id=tool_call_id
                    ))
                else:
                    messages.append(ChatMessage(
                        role=role,
                        content=content or ""
                    ))

        return messages

    def _extract_completions(self, span: OTelSpan) -> List[ChatMessage]:
        """Extract completion messages from gen_ai.completion.* attributes."""
        messages = []
        completion_indices = self._get_attribute_indices(span.attributes, 'gen_ai.completion')

        for idx in completion_indices:
            role = span.attributes.get(f'gen_ai.completion.{idx}.role', 'assistant')
            content = span.attributes.get(f'gen_ai.completion.{idx}.content')
            finish_reason = span.attributes.get(f'gen_ai.completion.{idx}.finish_reason')

            # Extract tool calls if present
            tool_calls = None
            if finish_reason == 'tool_calls':
                tool_calls = self._extract_tool_calls(span, idx)

            # Only add message if there's content or tool calls
            if content or tool_calls:
                messages.append(ChatMessage(
                    role=role,
                    content=content,
                    tool_calls=tool_calls
                ))

        return messages

    def _extract_tool_calls(self, span: OTelSpan, completion_idx: int) -> Optional[List[dict]]:
        """Extract tool calls from completion attributes."""
        tool_call_indices = self._get_attribute_indices(
            span.attributes,
            f'gen_ai.completion.{completion_idx}.tool_calls'
        )

        if not tool_call_indices:
            return None

        tool_calls = []
        for tc_idx in tool_call_indices:
            prefix = f'gen_ai.completion.{completion_idx}.tool_calls.{tc_idx}'

            tool_call_id = span.attributes.get(f'{prefix}.id')
            tool_type = span.attributes.get(f'{prefix}.type', 'function')
            function_name = span.attributes.get(f'{prefix}.function.name')
            function_args = span.attributes.get(f'{prefix}.function.arguments')

            if tool_call_id and function_name:
                tool_call = {
                    'id': tool_call_id,
                    'type': tool_type,
                    'function': {
                        'name': function_name,
                        'arguments': function_args or '{}'
                    }
                }
                tool_calls.append(tool_call)

        return tool_calls if tool_calls else None

    def _get_attribute_indices(self, attributes: Dict[str, Any], prefix: str) -> List[int]:
        """Get all numeric indices for a given attribute prefix."""
        indices = set()
        # Pattern to match: prefix.{number}.anything
        pattern = re.compile(rf'{re.escape(prefix)}\.(\d+)\.')

        for key in attributes.keys():
            match = pattern.match(key)
            if match:
                indices.add(int(match.group(1)))

        return sorted(indices)

    def extract_usage(self, span: OTelSpan) -> Tuple[int, int]:
        """Extract token usage from span."""
        input_tokens = span.attributes.get('gen_ai.usage.input_tokens', 0)
        output_tokens = span.attributes.get('gen_ai.usage.output_tokens', 0)

        # Handle various attribute naming conventions
        if not input_tokens:
            input_tokens = span.attributes.get('gen_ai.usage.prompt_tokens', 0)
        if not output_tokens:
            output_tokens = span.attributes.get('gen_ai.usage.completion_tokens', 0)

        try:
            return int(input_tokens), int(output_tokens)
        except (TypeError, ValueError):
            return 0, 0

    def extract_tools(self, span: OTelSpan) -> Optional[List[dict]]:
        """Extract tool definitions from span attributes."""
        # Look for tool definitions in the request
        # GenAI conventions may include tool definitions as:
        # gen_ai.request.tools (JSON array)
        tools_json = span.attributes.get('gen_ai.request.tools')

        if tools_json:
            try:
                if isinstance(tools_json, str):
                    return json.loads(tools_json)
                elif isinstance(tools_json, list):
                    return tools_json
            except (json.JSONDecodeError, TypeError) as e:
                logger.warning(f"Failed to parse tool definitions: {e}")

        return None


class ToolResponseExtractor:
    """Extract tool responses from child spans."""

    def extract_tool_responses(
        self,
        tool_calls: List[dict],
        all_spans: List[OTelSpan],
        parent_span_id: str
    ) -> List[ChatMessage]:
        """
        Match tool calls with their response spans.

        Looks for child spans of the LLM span that represent tool executions.
        """
        tool_responses = []

        # Find child spans of the parent
        child_spans = [s for s in all_spans if s.parent_span_id == parent_span_id]

        for tool_call in tool_calls:
            tool_name = tool_call['function']['name']
            tool_id = tool_call['id']

            # Find matching tool execution span
            tool_span = self._find_tool_span(child_spans, tool_name, tool_id)

            if tool_span:
                # Extract tool result from span
                result = self._extract_tool_result(tool_span)

                tool_responses.append(ChatMessage(
                    role='tool',
                    content=result,
                    id=tool_id
                ))
            else:
                # If no tool span found, create empty response
                logger.debug(f"No tool span found for {tool_name} (id: {tool_id})")
                tool_responses.append(ChatMessage(
                    role='tool',
                    content="",
                    id=tool_id
                ))

        return tool_responses

    def _find_tool_span(
        self,
        spans: List[OTelSpan],
        tool_name: str,
        tool_id: str
    ) -> Optional[OTelSpan]:
        """Find the span corresponding to a tool execution."""
        for span in spans:
            # Check if span operation matches tool name
            if tool_name.lower() in span.operation_name.lower():
                return span

            # Check attributes for tool identification
            if span.attributes.get('tool.name') == tool_name:
                return span

            if span.attributes.get('tool.call.id') == tool_id:
                return span

            if span.attributes.get('function.name') == tool_name:
                return span

        return None

    def _extract_tool_result(self, span: OTelSpan) -> str:
        """Extract tool execution result from span."""
        # Check various attribute patterns for tool results
        result = (
            span.attributes.get('tool.result') or
            span.attributes.get('function.result') or
            span.attributes.get('output') or
            span.attributes.get('response') or
            span.attributes.get('result')
        )

        if result is not None:
            if isinstance(result, (dict, list)):
                return json.dumps(result)
            return str(result)

        # If no explicit result attribute, check span events
        for event in span.events:
            if event.get('name') in ['result', 'output', 'response']:
                attrs = event.get('attributes', {})
                if 'value' in attrs:
                    return str(attrs['value'])

        # Last resort: use empty string
        return ""
