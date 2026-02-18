# Agentic Workload Benchmarking Guide

This guide provides comprehensive documentation for benchmarking LLM inference systems under agentic workloads using inference-perf.

## Table of Contents

1. [Introduction](#introduction)
2. [Key Concepts](#key-concepts)
3. [Quick Start](#quick-start)
4. [Data Sources](#data-sources)
   - [Synthetic Data Generation](#synthetic-data-generation)
   - [CSV Trace Files](#csv-trace-files)
   - [OpenTelemetry Traces](#opentelemetry-traces)
5. [Load Patterns](#load-patterns)
   - [Rate-Based Arrivals](#rate-based-arrivals)
   - [Concurrent Sessions](#concurrent-sessions)
   - [Trace Replay](#trace-replay)
6. [Inter-Turn Delays](#inter-turn-delays)
7. [Advanced Features](#advanced-features)
   - [Sweep Mode](#sweep-mode)
   - [LoRA Multi-Adapter Testing](#lora-multi-adapter-testing)
8. [Metrics and Reporting](#metrics-and-reporting)
9. [Tutorials](#tutorials)
10. [Best Practices](#best-practices)
11. [Troubleshooting](#troubleshooting)

---

## Introduction

### What Are Agentic Workloads?

Agentic workloads represent a fundamentally different pattern from traditional single-request LLM benchmarking. In an agentic workload, the unit of work is a **session** - a multi-turn conversation where an AI agent iteratively calls the LLM, executes tools, and builds up context over multiple turns.

A typical agent loop looks like:

```
[LLM call 1] → [tool exec: 400ms] → [LLM call 2] → [tool exec: 200ms] → [LLM call 3]
```

This creates unique characteristics:

- **Intra-session requests are serial**: Each turn depends on the previous turn's response
- **Inter-session requests are independent**: Many users/agents can hit the system concurrently
- **Context grows per turn**: Prefill cost increases monotonically within a session
- **Tool pauses create gaps**: Dead time from the inference system's perspective

Traditional load models (flat QPS, fixed concurrency) cannot express these patterns. Agentic benchmarking in inference-perf addresses this by making the **session** the unit of arrival rather than individual requests.

### Why Benchmark Agentic Workloads?

1. **Realistic capacity planning**: Understand how many concurrent agent sessions your system can support
2. **Prefix caching evaluation**: Measure KV-cache effectiveness across turns
3. **End-to-end latency**: Track session completion time, not just individual request latency
4. **Context growth impact**: Understand how growing context affects performance

---

## Key Concepts

### Terminology

| Term | Definition |
|------|------------|
| **Session** | One complete agent execution - a multi-turn agent loop. Contains one or more turns. |
| **Turn** | One LLM inference call within a session. Each turn has input tokens and output tokens. |
| **Tool Call** | A function/API invocation triggered by the model (finish_reason=tool_calls). |
| **Tool Pause** | Wall-clock duration of tool execution. Not improvable by the inference stack. |
| **Session Latency** | Wall-clock time from first turn's request to last turn's final token (includes pauses). |
| **Session Inference Time** | Sum of LLM call durations within a session (excludes pauses). |
| **Context Growth** | Increase in input token count from turn N to turn N+1. |

### Session Structure

A session is a sequence of turns with growing context:

```
Session (session_id = "abc")
├── Turn 0: LLM call  [input: 312 tok, output: 28 tok, finish: tool_calls]
│   └── Tool exec     [name: get_weather, duration: 400ms, result: 52 tok]
├── Turn 1: LLM call  [input: 392 tok, output: 94 tok, finish: stop]
│   (no tool call - session ends)
└── Total: 2 turns, 1 tool call, ~1.6s inference, ~0.4s tool pause
```

### Concurrency Model

Concurrency operates at two levels:

| Level | Definition | Controlled By |
|-------|------------|---------------|
| **Session Concurrency** | Number of active sessions at any point | `session_arrival.rate` or `active_sessions` |
| **Request Concurrency** | Number of in-flight LLM requests | Emergent (always ≤ session concurrency) |

Request concurrency is always less than or equal to session concurrency because turns within a session are serial. During a tool pause, a session is active but not issuing a request.

---

## Quick Start

### Prerequisites

1. A running LLM inference server (vLLM, llm-d, etc.)
2. inference-perf installed
3. Network access to the inference server

### Minimal Example

Create a file `agentic-test.yml`:

```yaml
api:
  type: chat
  streaming: true

server:
  type: vllm
  base_url: http://localhost:8000
  model_name: meta-llama/Llama-3.1-8B-Instruct

data:
  type: agentic_synthetic
  agentic_synthetic:
    num_sessions: 50
    turns_per_session:
      type: normal
      mean: 3
      std_dev: 1
      min: 1
      max: 6
    tool_call_probability: 0.5
    input_tokens_turn_0:
      type: normal
      mean: 400
      std_dev: 100
      min: 100
      max: 1000
    output_tokens_per_turn:
      type: normal
      mean: 100
      std_dev: 30
      min: 20
      max: 200

load:
  type: agentic
  session_arrival:
    type: constant
    stages:
      - rate: 1.0
        duration: 60

report:
  request_lifecycle:
    summary: true
    per_session: true
```

Run the benchmark:

```bash
inference-perf -c agentic-test.yml
```

---

## Data Sources

inference-perf supports three data sources for agentic workloads:

### Synthetic Data Generation

Generate sessions from statistical distributions. Best for controlled experimentation.

**Example: `config-agentic-synthetic.yml`**

```yaml
data:
  type: agentic_synthetic
  agentic_synthetic:
    # Number of synthetic sessions to generate
    num_sessions: 100

    # Distribution for turns per session
    turns_per_session:
      type: normal
      mean: 5
      std_dev: 2
      min: 1
      max: 15

    # Probability that a turn ends with tool calls
    tool_call_probability: 0.6

    # Distribution for number of tool calls per turn
    tool_calls_per_turn:
      type: normal
      mean: 2
      std_dev: 1
      min: 1
      max: 5

    # Distribution for input tokens in first turn
    input_tokens_turn_0:
      type: normal
      mean: 500
      std_dev: 150
      min: 100
      max: 2000

    # Distribution for output tokens per turn
    output_tokens_per_turn:
      type: normal
      mean: 150
      std_dev: 50
      min: 20
      max: 500

    # Distribution for tool result tokens
    tool_result_tokens:
      type: normal
      mean: 100
      std_dev: 40
      min: 10
      max: 400

    # Fixed system prompt tokens (added to initial context)
    system_prompt_tokens: 200

    # Content generation strategy: random, synthetic, template
    content_strategy: random
```

**Distribution Types:**

| Type | Parameters | Description |
|------|------------|-------------|
| `normal` | `mean`, `std_dev`, `min`, `max` | Gaussian distribution clamped to [min, max] |
| `uniform` | `min`, `max` | Uniform distribution between min and max |

### CSV Trace Files

Load sessions from a CSV file. Best for replaying production traces or custom datasets.

**Example: `config-agentic-csv.yml`**

```yaml
data:
  type: agentic_csv
  agentic_csv:
    path: examples/agentic/data/sessions.csv
```

**CSV Format:**

```csv
session_id,turn_index,input_tokens,output_tokens,finish_reason,num_tool_calls,tool_duration_ms,tool_result_tokens,llm_latency_ms,ttft_ms,timestamp_ms
session_001,0,500,120,tool_calls,2,350,180,850,45,1708000000000
session_001,1,800,150,tool_calls,1,200,100,720,52,1708000001500
session_001,2,1050,100,stop,0,0,0,580,48,1708000002500
```

**Column Definitions:**

| Column | Required | Description |
|--------|----------|-------------|
| `session_id` | Yes | Groups turns into sessions |
| `turn_index` | Yes | 0-based position within session |
| `input_tokens` | Yes | Total input tokens for this LLM call |
| `output_tokens` | Yes | Tokens generated by model |
| `finish_reason` | Yes | `stop` or `tool_calls` |
| `num_tool_calls` | No | Number of tool calls in this turn |
| `tool_duration_ms` | No | Total tool execution time (for replay mode) |
| `tool_result_tokens` | No | Tokens added from tool results |
| `llm_latency_ms` | No | Original LLM call duration |
| `ttft_ms` | No | Original time to first token |
| `timestamp_ms` | No | Original timestamp (for trace replay) |

### OpenTelemetry Traces

Extract sessions from OTel traces collected from a Jaeger or Tempo backend.

**Example: `config-agentic-otel.yml`**

```yaml
data:
  type: otel
  otel:
    # Backend configuration
    backend: jaeger
    endpoint: http://localhost:16686/api

    # Optional authentication
    # auth:
    #   type: bearer
    #   bearer_token: ${JAEGER_TOKEN}

    # Trace filtering
    service_name: my-agent-service
    operation_name: llm_call

    # Time range
    lookback: 24h
    # Or use explicit timestamps:
    # start_time: "2024-02-15T00:00:00Z"
    # end_time: "2024-02-16T00:00:00Z"

    # Trace selection
    limit: 500
    min_duration_ms: 100

    # Conversation extraction
    enable_multi_turn_chat: true
    include_system_prompts: true
    extract_tool_calls: true
    min_turns: 2  # Only include sessions with 2+ turns
```

---

## Load Patterns

### Rate-Based Arrivals

Sessions arrive at a controlled rate (sessions per second). Use `type: agentic`.

**Example: `config-agentic-rate-based.yml`**

```yaml
load:
  type: agentic
  num_workers: 8
  worker_max_concurrency: 100

  session_arrival:
    type: constant  # or 'poisson' for bursty arrivals
    stages:
      # Warm-up stage
      - rate: 0.5       # 0.5 sessions/sec
        duration: 30    # 30 seconds

      # Ramp-up stage
      - rate: 1.0       # 1 session/sec
        duration: 60    # 1 minute

      # Peak load stage
      - rate: 2.0       # 2 sessions/sec
        duration: 120   # 2 minutes

      # Cool-down stage
      - rate: 0.5
        duration: 30
```

**Arrival Types:**

| Type | Description |
|------|-------------|
| `constant` | Sessions arrive at exact intervals (1/rate seconds apart) |
| `poisson` | Sessions arrive following a Poisson process (more realistic, bursty) |

### Concurrent Sessions

Maintain a fixed number of concurrent active sessions. Use `type: agentic_concurrent`.

**Example: `config-agentic-concurrent.yml`**

```yaml
load:
  type: agentic_concurrent
  num_workers: 8
  worker_max_concurrency: 100

  session_arrival:
    type: constant
    stages:
      # Low concurrency stage
      - active_sessions: 5    # Maintain 5 concurrent sessions
        total_sessions: 50    # Run 50 sessions total

      # Medium concurrency stage
      - active_sessions: 10
        total_sessions: 100

      # High concurrency stage
      - active_sessions: 20
        total_sessions: 200
```

This mode answers the question: "Can my system sustain N concurrent agent sessions?"

### Trace Replay

Replay sessions at their original timestamps from the source data.

**Example: `config-agentic-trace-replay.yml`**

```yaml
load:
  type: agentic_trace_replay
  num_workers: 8
  worker_max_concurrency: 100

  session_arrival:
    type: trace  # Use original timestamps

    # Time scale for replay:
    # - 1.0 = real-time (original speed)
    # - 2.0 = 2x speed (sessions arrive twice as fast)
    # - 0.5 = half speed
    time_scale: 1.0

    stages:
      - total_sessions: 100  # Replay first 100 sessions
```

**Note:** The CSV data must include `timestamp_ms` column for trace replay.

---

## Inter-Turn Delays

Tool calls and user think time create pauses between turns. Configure how these delays are handled.

### Delay Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| `zero` | No delay between turns | Stress testing, finding throughput ceiling |
| `fixed` | Constant delay | Controlled benchmarking |
| `distribution` | Random delay from distribution | Simulating realistic variance |
| `replay` | Use original delays from data | Production fidelity |

### Zero Delays (Stress Test)

**Example: `config-agentic-zero-delays.yml`**

```yaml
load:
  agentic:
    tool_call_delay:
      type: zero
    user_think_delay:
      type: zero
```

Use this to find the maximum throughput of your inference system without tool execution overhead.

### Fixed Delays

**Example: `config-agentic-fixed-delays.yml`**

```yaml
load:
  agentic:
    # Fixed 200ms delay after tool calls
    tool_call_delay:
      type: fixed
      fixed_ms: 200

    # Fixed 500ms delay for user think time
    user_think_delay:
      type: fixed
      fixed_ms: 500
```

### Distribution-Based Delays

**Example: `config-agentic-distribution-delays.yml`**

```yaml
load:
  agentic:
    # Tool call delays from normal distribution
    tool_call_delay:
      type: distribution
      distribution:
        type: normal
        mean: 150    # Average 150ms
        std_dev: 50
        min: 50
        max: 500

    # User think time from uniform distribution
    user_think_delay:
      type: distribution
      distribution:
        type: uniform
        min: 100
        max: 1000
```

### Replay Original Delays

**Example: `config-agentic-replay-delays.yml`**

```yaml
load:
  agentic:
    # Use tool_duration_ms from CSV or OTel traces
    tool_call_delay:
      type: replay
    user_think_delay:
      type: zero
```

---

## Advanced Features

### Sweep Mode

Automatically find the saturation point of your inference system by ramping session arrival rates.

**Example: `config-agentic-sweep.yml`**

```yaml
load:
  type: agentic

  # Sweep configuration
  agentic_sweep:
    type: linear               # or 'geometric'
    num_sessions: 100          # Sessions per probe stage
    timeout: 120               # Probe timeout in seconds
    num_stages: 5              # Number of stages to generate
    stage_duration: 180        # Duration per generated stage
    saturation_metric: session_inference_time_p95
    degradation_threshold: 0.2 # 20% increase triggers saturation
    min_probe_rate: 1.0        # Minimum sessions/sec to probe
    max_probe_rate: 20.0       # Maximum sessions/sec to probe
    num_probes: 5              # Number of probe rate levels

  agentic:
    tool_call_delay:
      type: zero
    user_think_delay:
      type: zero
```

The sweep process:
1. Runs probe stages at increasing session arrival rates
2. Monitors `session_inference_time_p95` for degradation
3. Generates optimal stages up to the saturation point

### LoRA Multi-Adapter Testing

Test multiple LoRA adapters with traffic split across them. Each session uses one adapter for all its turns.

**Example: `config-agentic-lora.yml`**

```yaml
load:
  type: agentic

  # Traffic split across LoRA adapters
  lora_traffic_split:
    - name: customer_support_lora
      split: 0.4   # 40% of sessions
    - name: code_assistant_lora
      split: 0.35  # 35% of sessions
    - name: data_analyst_lora
      split: 0.25  # 25% of sessions

  session_arrival:
    type: constant
    stages:
      - rate: 2.0
        duration: 180

report:
  request_lifecycle:
    per_adapter: true         # Report metrics per LoRA adapter
    per_adapter_stage: true   # Report per adapter per stage
```

---

## Metrics and Reporting

### Report Configuration

```yaml
report:
  request_lifecycle:
    summary: true             # High-level summary
    per_stage: true           # Breakdown by load stage
    per_session: true         # Session-level metrics
    per_turn_position: true   # Metrics by turn index
    session_summary: true     # Session summary statistics
    system_summary: true      # System-level agentic metrics
    timeseries: true          # Time series data
```

### Per-Request Metrics

Standard metrics collected for each LLM call:

| Metric | Definition | Unit |
|--------|------------|------|
| TTFT | Time to first token | ms |
| TPOT | Time per output token | ms/token |
| ITL | Inter-token latency distribution | ms |
| E2E Latency | Request sent → last token received | ms |

### Per-Session Metrics

Aggregated across all turns within a session:

| Metric | Definition | Unit |
|--------|------------|------|
| Session Latency | Wall-clock time including pauses | ms |
| Session Inference Time | Sum of LLM call durations (excludes pauses) | ms |
| Session Pause Time | Session Latency - Session Inference Time | ms |
| Inference Duty Cycle | Session Inference Time / Session Latency | ratio [0,1] |
| Turns Completed | Number of turns in the session | count |
| Session Input Tokens | Sum of input_tokens across all turns | tokens |
| Session Output Tokens | Sum of output_tokens across all turns | tokens |
| Peak Context Length | Maximum input_tokens in any turn | tokens |

### System-Level Metrics

| Metric | Definition | Unit |
|--------|------------|------|
| Active Sessions | Concurrent active sessions over time | count (time series) |
| Session Throughput | Completed sessions per second | sessions/sec |
| Context Growth Rate | Average input token increase per turn | tokens/turn |
| Inter-Turn Idle Ratio | Fraction of time in tool/user pauses | ratio [0,1] |

### Per-Turn-Position Metrics

Metrics grouped by turn index to reveal caching effects:

| Metric | Definition |
|--------|------------|
| TTFT by Position | Time to first token for turn 0, turn 1, etc. |
| Latency by Position | Turn latency aggregated by turn number |
| Token Counts by Position | Input/output tokens by turn number |

Turn 0 has no cache benefit. Turns 1+ may benefit from prefix caching. A flat TTFT across turn positions indicates the cache is not being utilized.

---

## Tutorials

### Tutorial 1: Basic Synthetic Workload

**Goal:** Run a simple agentic benchmark with synthetic data.

1. Create `tutorial-basic.yml`:

```yaml
api:
  type: chat
  streaming: true

server:
  type: vllm
  base_url: http://localhost:8000
  model_name: meta-llama/Llama-3.1-8B-Instruct

data:
  type: agentic_synthetic
  agentic_synthetic:
    num_sessions: 50
    turns_per_session:
      type: normal
      mean: 4
      std_dev: 1.5
      min: 1
      max: 10
    tool_call_probability: 0.5
    tool_calls_per_turn:
      type: uniform
      min: 1
      max: 3
    input_tokens_turn_0:
      type: normal
      mean: 400
      std_dev: 100
      min: 100
      max: 1500
    output_tokens_per_turn:
      type: normal
      mean: 120
      std_dev: 40
      min: 20
      max: 300
    tool_result_tokens:
      type: uniform
      min: 50
      max: 200
    system_prompt_tokens: 150

load:
  type: agentic
  session_arrival:
    type: constant
    stages:
      - rate: 1.0
        duration: 60

  agentic:
    tool_call_delay:
      type: fixed
      fixed_ms: 100
    user_think_delay:
      type: zero

report:
  request_lifecycle:
    summary: true
    per_session: true
    per_turn_position: true

storage:
  local_storage:
    path: reports/tutorial-basic
```

2. Run the benchmark:

```bash
inference-perf -c tutorial-basic.yml
```

3. Review the results in `reports/tutorial-basic/`.

### Tutorial 2: Stress Testing with Zero Delays

**Goal:** Find the maximum throughput by eliminating tool delays.

1. Use `config-agentic-zero-delays.yml` as a starting point
2. Key configuration:

```yaml
load:
  type: agentic
  num_workers: 8
  worker_max_concurrency: 200  # High concurrency

  session_arrival:
    type: poisson  # Bursty arrivals
    stages:
      - rate: 5.0       # 5 sessions per second
        duration: 120

  agentic:
    tool_call_delay:
      type: zero    # No delays!
    user_think_delay:
      type: zero
```

3. Compare results with realistic delays to understand how much latency is inference vs. tool execution.

### Tutorial 3: Capacity Testing with Fixed Concurrency

**Goal:** Determine if your system can handle N concurrent agent sessions.

1. Use `config-agentic-concurrent.yml`:

```yaml
load:
  type: agentic_concurrent
  session_arrival:
    type: constant
    stages:
      # Start low
      - active_sessions: 5
        total_sessions: 50

      # Increase concurrency
      - active_sessions: 10
        total_sessions: 100

      # Push limits
      - active_sessions: 20
        total_sessions: 200

      # Find breaking point
      - active_sessions: 50
        total_sessions: 500
```

2. Monitor session inference time degradation as concurrency increases.

### Tutorial 4: Replaying Production Traces

**Goal:** Benchmark using real production traffic patterns.

1. Export your production traces to CSV format:

```csv
session_id,turn_index,input_tokens,output_tokens,finish_reason,num_tool_calls,tool_duration_ms,tool_result_tokens,llm_latency_ms,timestamp_ms
session_001,0,500,120,tool_calls,2,350,180,850,1708000000000
session_001,1,800,150,stop,0,0,0,720,1708000001500
```

2. Use `config-agentic-trace-replay.yml`:

```yaml
data:
  type: agentic_csv
  agentic_csv:
    path: your-production-traces.csv

load:
  type: agentic_trace_replay
  session_arrival:
    type: trace
    time_scale: 1.0  # Real-time replay

  agentic:
    tool_call_delay:
      type: replay  # Use original tool durations
    user_think_delay:
      type: replay
```

3. Use `time_scale: 2.0` to replay at 2x speed for stress testing.

### Tutorial 5: Evaluating Prefix Caching

**Goal:** Measure KV-cache effectiveness across turns.

1. Use synthetic data with a fixed system prompt:

```yaml
data:
  type: agentic_synthetic
  agentic_synthetic:
    num_sessions: 100
    turns_per_session:
      type: normal
      mean: 6        # More turns = more cache opportunity
      std_dev: 2
      min: 3
      max: 10
    system_prompt_tokens: 1024  # Large shared prefix
    content_strategy: template  # Preserves prefix caching opportunity
```

2. Enable per-turn-position metrics:

```yaml
report:
  request_lifecycle:
    per_turn_position: true
```

3. Analyze TTFT by turn position:
   - Turn 0 TTFT: No cache benefit (cold start)
   - Turn 1+ TTFT: Should decrease if caching is effective

### Tutorial 6: Multi-Stage Load Profile

**Goal:** Run a realistic benchmark with warm-up, peak, and cool-down phases.

Use `config-agentic-rate-based.yml`:

```yaml
load:
  type: agentic
  session_arrival:
    type: constant
    stages:
      # Warm-up: Let the system stabilize
      - rate: 0.5
        duration: 30

      # Ramp-up: Gradually increase load
      - rate: 1.0
        duration: 60

      # Peak: Sustained high load
      - rate: 2.0
        duration: 120

      # Cool-down: Graceful wind-down
      - rate: 0.5
        duration: 30
```

---

## Best Practices

### 1. Start with Synthetic Data

Begin with synthetic data generation to understand baseline performance before replaying production traces. This allows controlled experimentation with specific characteristics.

### 2. Use Zero Delays First

Run with `tool_call_delay: zero` first to establish the pure inference throughput ceiling. Then add realistic delays to understand the impact of tool execution time.

### 3. Match Production Characteristics

When benchmarking for capacity planning, configure synthetic data to match your production workload:
- Average turns per session
- Context sizes (input tokens)
- Output token distributions
- Tool call frequency

### 4. Monitor Session Metrics, Not Just Request Metrics

Session-level metrics (session latency, inference duty cycle) better represent user experience than per-request metrics alone.

### 5. Use Per-Turn-Position Metrics for Cache Analysis

If TTFT is flat across turn positions, investigate why prefix caching isn't helping:
- Check if the inference server has caching enabled
- Verify session affinity is routing turns to the same worker
- Ensure system prompts are shared

### 6. Size Your Workers Appropriately

```yaml
load:
  num_workers: 8                  # Usually CPU count
  worker_max_concurrency: 100     # Max active sessions per worker
```

Total capacity = `num_workers × worker_max_concurrency`

### 7. Use Poisson Arrivals for Realism

`type: poisson` provides more realistic bursty traffic patterns than `type: constant`.

---

## Troubleshooting

### Sessions Completing Too Slowly

**Symptoms:** Low session throughput, high session latency

**Possible Causes:**
1. Tool delays too high - try `type: zero` to isolate
2. Server overloaded - reduce `session_arrival.rate`
3. Context too large - check `input_tokens_turn_0` distribution

### High TTFT Variance

**Symptoms:** Large spread in time-to-first-token

**Possible Causes:**
1. Request queuing at server - reduce concurrency
2. Mixed context sizes - check input token distribution
3. Prefix caching not working - enable `per_turn_position` metrics

### Worker Capacity Exhausted

**Symptoms:** Warning messages about worker capacity

**Solution:**
```yaml
load:
  worker_max_concurrency: 200  # Increase from default 100
```

### Missing Timestamp Data for Trace Replay

**Symptoms:** Error when using `type: agentic_trace_replay`

**Solution:** Ensure your CSV includes `timestamp_ms` column with Unix epoch milliseconds.

### OTel Connection Failures

**Symptoms:** Cannot fetch traces from Jaeger/Tempo

**Checklist:**
1. Verify `endpoint` URL is correct
2. Check authentication if required
3. Ensure `service_name` matches your instrumented service
4. Verify `lookback` covers the time range with traces

---

## Configuration Reference

### Full Data Section

```yaml
data:
  type: agentic_synthetic | agentic_csv | otel

  # For agentic_synthetic
  agentic_synthetic:
    num_sessions: 100
    turns_per_session:
      type: normal | uniform
      mean: 5
      std_dev: 2
      min: 1
      max: 20
    tool_call_probability: 0.6
    tool_calls_per_turn:
      type: normal | uniform
      ...
    input_tokens_turn_0:
      type: normal | uniform
      ...
    output_tokens_per_turn:
      type: normal | uniform
      ...
    tool_result_tokens:
      type: normal | uniform
      ...
    system_prompt_tokens: 200
    content_strategy: random | synthetic | template

  # For agentic_csv
  agentic_csv:
    path: ./traces/sessions.csv

  # For otel
  otel:
    backend: jaeger | tempo
    endpoint: http://localhost:16686/api
    service_name: my-agent
    lookback: 24h
    limit: 500
    enable_multi_turn_chat: true
    extract_tool_calls: true
    min_turns: 2
```

### Full Load Section

```yaml
load:
  type: agentic | agentic_concurrent | agentic_trace_replay

  num_workers: 8
  worker_max_concurrency: 100
  worker_max_tcp_connections: 2500
  request_timeout: 60.0
  worker_affinity: true

  session_arrival:
    type: constant | poisson | trace
    time_scale: 1.0  # For trace type only
    stages:
      # For agentic (rate-based)
      - rate: 2.0
        duration: 120

      # For agentic_concurrent
      - active_sessions: 50
        total_sessions: 500

      # For agentic_trace_replay
      - total_sessions: 100

  agentic:
    tool_call_delay:
      type: zero | fixed | distribution | replay
      fixed_ms: 200  # For fixed type
      distribution:  # For distribution type
        type: normal | uniform
        mean: 150
        std_dev: 50
        min: 50
        max: 500
    user_think_delay:
      type: zero | fixed | distribution | replay
      ...

  # Optional: sweep mode
  agentic_sweep:
    type: linear | geometric
    num_sessions: 100
    timeout: 120
    num_stages: 5
    stage_duration: 180
    saturation_metric: session_inference_time_p95
    degradation_threshold: 0.2

  # Optional: LoRA traffic split
  lora_traffic_split:
    - name: adapter_1
      split: 0.5
    - name: adapter_2
      split: 0.5
```

### Full Report Section

```yaml
report:
  request_lifecycle:
    summary: true
    per_stage: true
    per_request: false
    per_session: true
    per_turn_position: true
    session_summary: true
    system_summary: true
    timeseries: true
    per_adapter: false
    per_adapter_stage: false
    per_program: false
```
