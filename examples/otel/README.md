# OpenTelemetry Data Generator Examples

This directory contains example configurations and test files for the OpenTelemetry (OTel) data generator.

## Overview

The OpenTelemetry data generator enables replay-based benchmarking from production systems instrumented with OpenTelemetry tracing. It fetches traces from OTel-compatible backends (like Jaeger, Tempo) and extracts LLM conversations using [GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/).

## Quick Start

### 1. Test with Mock Data

Run the test script to verify the generator works:

```bash
python examples/otel/test_otel_generator.py
```

This starts a mock Jaeger server and loads the sample trace.

### 2. Run with Mock Server

```bash
# Uses the sample trace with mock inference server
inference-perf -c examples/otel/config-jaeger.yml
```

### 3. Connect to Real Jaeger

Update `config-jaeger.yml` with your Jaeger endpoint:

```yaml
data:
  type: otel
  otel:
    backend: jaeger
    endpoint: "http://your-jaeger:16686"
    service_name: "your-service"
    lookback: "24h"
```

Then run:

```bash
inference-perf -c examples/otel/config-jaeger.yml
```

## Configuration

### Required Fields

```yaml
data:
  type: otel
  otel:
    backend: jaeger  # Currently supported: jaeger
    endpoint: "http://localhost:16686"  # Backend API URL
```

### Filtering Options

```yaml
data:
  type: otel
  otel:
    # Service and operation filters
    service_name: "langchain-agent"  # Filter by service
    operation_name: "chat.completions"  # Filter by operation (optional)

    # Tag-based filtering (GenAI semantic conventions)
    tags:
      - "gen_ai.system=openai"  # Only OpenAI traces
      - "gen_ai.request.model=gpt-4"  # Only GPT-4
      - "environment=production"  # Custom tags

    # Time range
    lookback: "24h"  # Relative (24h, 7d, 30m)
    # OR explicit range:
    # start_time: "2024-01-01T00:00:00Z"
    # end_time: "2024-01-02T00:00:00Z"

    # Duration filters
    min_duration_ms: 100  # Minimum trace duration
    max_duration_ms: 30000  # Maximum trace duration

    # Limit
    limit: 1000  # Maximum traces to fetch
```

### Conversation Extraction

```yaml
data:
  type: otel
  otel:
    enable_multi_turn_chat: true  # Expand into incremental turns
    include_system_prompts: true  # Include system prompts
    extract_tool_calls: true  # Extract tool calls from spans
    min_turns: 2  # Minimum conversation turns
```

### Authentication

```yaml
data:
  type: otel
  otel:
    auth:
      type: basic  # basic | bearer | api_key
      username: "${JAEGER_USERNAME}"
      password: "${JAEGER_PASSWORD}"

    # OR for bearer token:
    # auth:
    #   type: bearer
    #   bearer_token: "${JAEGER_TOKEN}"

    # OR for API key:
    # auth:
    #   type: api_key
    #   api_key: "${API_KEY}"
    #   api_key_header: "X-API-Key"
```

## GenAI Semantic Conventions

The generator expects traces instrumented with [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/) v1.28.0+.

### Required Attributes

For LLM spans:
- `gen_ai.system` - LLM provider (e.g., "openai", "anthropic")
- `gen_ai.prompt.{index}.role` - Message role
- `gen_ai.prompt.{index}.content` - Message content
- `gen_ai.completion.{index}.role` - Response role
- `gen_ai.completion.{index}.content` - Response content

### Tool Calls

For tool-calling spans:
- `gen_ai.completion.{index}.finish_reason` = "tool_calls"
- `gen_ai.completion.{index}.tool_calls.{index}.id` - Tool call ID
- `gen_ai.completion.{index}.tool_calls.{index}.function.name` - Function name
- `gen_ai.completion.{index}.tool_calls.{index}.function.arguments` - Arguments JSON

Tool execution spans:
- `tool.name` - Tool function name
- `tool.call.id` - Matching call ID
- `tool.result` - Tool result

### Token Usage

- `gen_ai.usage.input_tokens` - Input tokens
- `gen_ai.usage.output_tokens` - Output tokens

## Sample Trace Format

See `sample-trace.json` for an example of a properly formatted Jaeger trace with:
- 2-turn tool-calling conversation
- GenAI semantic conventions
- Tool execution span
- Single-turn conversation

## Features

### Multi-Turn Expansion

When `enable_multi_turn_chat: true`, conversations are expanded into incremental turns:

```
Turn 0: [system, user1]
Turn 1: [system, user1, assistant1, tool_result1, user2]
Turn 2: [system, user1, assistant1, tool_result1, user2, assistant2, user3]
```

This simulates how real agents send growing context with each turn.

### Tool Call Extraction

When `extract_tool_calls: true`:
1. Extracts tool calls from assistant messages
2. Finds matching tool execution child spans
3. Reconstructs tool responses
4. Builds complete conversation with tool interactions

### Program-Level Tracking

- `program_id` = trace_id (groups requests by trace)
- `turn_index` = 0, 1, 2, ... (tracks turn number)
- Compatible with program-level reporting

## Backends

### Supported

✅ **Jaeger** - Full support via HTTP API
- Endpoint: `http://jaeger:16686`
- Query API: `/api/traces`

### Planned

🚧 **Grafana Tempo** - TraceQL support
🚧 **Zipkin** - Zipkin API support
🚧 **Honeycomb** - HoneyQL support
🚧 **Custom OTLP** - Direct OTLP endpoint

## Use Cases

### 1. Production Replay

Replay real production traffic to measure performance:

```yaml
otel:
  service_name: "prod-agent"
  tags:
    - "environment=production"
  lookback: "7d"
  limit: 10000
```

### 2. Framework Comparison

Compare performance across different LLM frameworks:

```yaml
# LangChain traces
otel:
  service_name: "langchain-app"
  tags:
    - "framework=langchain"

# vs LlamaIndex traces
otel:
  service_name: "llamaindex-app"
  tags:
    - "framework=llamaindex"
```

### 3. Tool-Calling Analysis

Analyze tool-calling patterns:

```yaml
otel:
  tags:
    - "gen_ai.completion.0.finish_reason=tool_calls"
  extract_tool_calls: true
  enable_multi_turn_chat: true
```

### 4. Model Comparison

Compare different models on same workload:

```yaml
# GPT-4 traces
otel:
  tags:
    - "gen_ai.request.model=gpt-4"

# vs GPT-3.5 traces
otel:
  tags:
    - "gen_ai.request.model=gpt-3.5-turbo"
```

## Troubleshooting

### No conversations found

**Cause**: Traces don't have GenAI semantic conventions

**Solution**: Check your instrumentation emits `gen_ai.*` attributes

```bash
# Verify trace attributes in Jaeger UI
# Should see: gen_ai.system, gen_ai.prompt.*, gen_ai.completion.*
```

### Authentication failed

**Cause**: Invalid credentials

**Solution**: Use environment variables

```bash
export JAEGER_USERNAME="user"
export JAEGER_PASSWORD="pass"
inference-perf -c config-jaeger.yml
```

### Connection refused

**Cause**: Jaeger endpoint not accessible

**Solution**: Verify Jaeger is running and accessible

```bash
curl http://localhost:16686/api/services
```

### Empty tool responses

**Cause**: Tool execution spans not found

**Solution**: Check span parent-child relationships

```yaml
# Enable debug logging
otel:
  extract_tool_calls: true  # Should match tool calls with child spans
```

## Example Output

With program-level reporting enabled:

```json
{
  "aggregate": {
    "num_programs": 2,
    "program_completion_time": {
      "mean": 2.5,
      "p50": 2.3,
      "p99": 4.2
    },
    "num_turns": {
      "mean": 2.0,
      "p50": 2.0
    }
  },
  "per_program": {
    "abc123def456": {
      "program_completion_time": 2.7,
      "num_turns": 2,
      "per_turn_ttft": [null, null],
      "per_turn_request_latency": [1.5, 1.2],
      "total_input_tokens": 130,
      "total_output_tokens": 27
    },
    "xyz789abc012": {
      "program_completion_time": 0.8,
      "num_turns": 1,
      "per_turn_ttft": [null],
      "per_turn_request_latency": [0.8],
      "total_input_tokens": 25,
      "total_output_tokens": 20
    }
  }
}
```

## References

- [OpenTelemetry](https://opentelemetry.io/)
- [GenAI Semantic Conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/)
- [Jaeger](https://www.jaegertracing.io/)
- [OTel Data Generator Design Plan](../../OTEL_DATAGEN_PLAN.md)
