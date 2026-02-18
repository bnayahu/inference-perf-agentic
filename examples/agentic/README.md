# Agentic Workload Benchmarking Examples

This directory contains example configurations for benchmarking agentic workloads with inference-perf.

## Overview

Agentic workloads differ from traditional single-request benchmarks in that they involve:
- **Sessions**: A complete agent execution with multiple LLM calls
- **Turns**: Individual LLM calls within a session
- **Tool Calls**: Function/tool invocations between turns
- **Context Growth**: Input tokens typically grow with each turn

## Configuration Files

### Data Sources

| File | Description |
|------|-------------|
| `config-agentic-synthetic.yml` | Generate synthetic sessions from distributions |
| `config-agentic-csv.yml` | Load sessions from a CSV file |
| `config-agentic-otel.yml` | Extract sessions from OpenTelemetry traces |

### Load Patterns

| File | Description |
|------|-------------|
| `config-agentic-rate-based.yml` | Rate-based session arrivals (sessions/sec) |
| `config-agentic-concurrent.yml` | Fixed concurrency (N active sessions) |
| `config-agentic-trace-replay.yml` | Replay sessions at original timestamps |

### Delay Modes

| File | Description |
|------|-------------|
| `config-agentic-replay-delays.yml` | Replay original tool execution times |
| `config-agentic-fixed-delays.yml` | Fixed delays between turns |
| `config-agentic-distribution-delays.yml` | Distribution-based random delays |
| `config-agentic-zero-delays.yml` | No delays (stress test mode) |

### Advanced

| File | Description |
|------|-------------|
| `config-agentic-lora.yml` | Multi-adapter LoRA testing |

## Usage

```bash
# Run with synthetic data and rate-based load
inference-perf -c examples/agentic/config-agentic-synthetic.yml

# Run with CSV data and concurrent sessions
inference-perf -c examples/agentic/config-agentic-csv.yml

# Run with OTel trace replay
inference-perf -c examples/agentic/config-agentic-trace-replay.yml
```

## Key Configuration Options

### Data Configuration

```yaml
data:
  type: agentic_synthetic  # or agentic_csv, otel
  agentic_synthetic:
    num_sessions: 100
    turns_per_session:
      type: normal
      mean: 5
      std_dev: 2
      min: 1
      max: 20
    tool_call_probability: 0.5
```

### Load Configuration

```yaml
load:
  type: agentic  # or agentic_concurrent, agentic_trace_replay
  worker_affinity: true  # Pin sessions to workers for better cache utilization
  session_arrival:
    type: constant  # or poisson, trace
    stages:
      - rate: 2.0
        duration: 60
  agentic:
    tool_call_delay:
      type: replay  # or fixed, distribution, zero
    user_think_delay:
      type: zero
```

### Report Configuration

```yaml
report:
  request_lifecycle:
    summary: true
    per_session: true         # Detailed per-session metrics
    per_turn_position: true   # Metrics aggregated by turn number
    session_summary: true     # Summary statistics for all sessions
    system_summary: true      # System-level agentic metrics
    timeseries: true          # Time series data (active sessions over time)
```

## CSV Format

When using `agentic_csv` data type, the CSV should have these columns:

```csv
session_id,turn_index,input_tokens,output_tokens,finish_reason,num_tool_calls,tool_duration_ms,tool_result_tokens,llm_latency_ms
session_001,0,500,150,tool_calls,2,200,100,800
session_001,1,750,100,stop,0,0,0,600
```

## Metrics

Agentic workloads produce additional metrics:

### Session-Level Metrics
- **Session Latency**: Total wall-clock time including pauses
- **Session Inference Time**: Sum of LLM call durations
- **Session Pause Time**: Time spent waiting between turns
- **Inference Duty Cycle**: Ratio of inference time to total latency
- **Turns Completed**: Number of turns successfully executed
- **Peak Context Length**: Maximum input tokens in any turn

### System-Level Metrics
- **Session Throughput**: Sessions completed per second
- **Context Growth Rate**: Average input token increase per turn
- **Inter-Turn Idle Ratio**: Average ratio of idle time between turns
- **Total Sessions/Turns/Tokens**: Aggregate counts

### Per-Turn-Position Metrics
- **TTFT by Position**: Time to first token aggregated by turn number
- **Latency by Position**: Turn latency aggregated by turn number
- **Token Counts by Position**: Input/output tokens by turn number

### Time Series Data
- **Active Sessions**: Number of concurrent sessions over time
- **Request Rate**: Session completion rate over time

## Session-Based vs Request-Based Approaches

This directory contains **session-based** configurations that use the new agentic load types (`agentic`, `agentic_concurrent`, `agentic_trace_replay`). These maintain session state across turns and simulate realistic agent execution patterns.

For **request-based** benchmarking with multi-turn data (where each turn is sent as an independent request), see these alternative configs:

| File | Description |
|------|-------------|
| `../vllm/config-langfuse.yml` | Langfuse traces expanded to individual requests |
| `../vllm/config-tau2-bench.yml` | Tau2 benchmark data as individual requests |
| `../otel/config-jaeger.yml` | OTel traces expanded to individual requests |

### When to use each approach

**Session-based (this folder)**:
- Testing session-level throughput and latency
- Simulating realistic agent execution with context accumulation
- Benchmarking with inter-turn delays (tool execution time)
- Measuring session-level metrics (duty cycle, speedup ratio)

**Request-based (../vllm, ../otel)**:
- Testing raw request throughput at various context sizes
- Stress testing without session overhead
- Comparing performance across different context lengths from production data
