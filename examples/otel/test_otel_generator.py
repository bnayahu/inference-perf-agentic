#!/usr/bin/env python3
"""
Test script for OpenTelemetry data generator.

This script creates a mock Jaeger server and tests the OTel generator.
"""

import json
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler
from threading import Thread
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from inference_perf.config import (
    APIConfig,
    APIType,
    DataConfig,
    DataGenType,
    OpenTelemetryConfig,
    OTelBackendType,
)
from inference_perf.datagen.otel_datagen import OpenTelemetryDataGenerator


class MockJaegerHandler(BaseHTTPRequestHandler):
    """Mock Jaeger HTTP API handler."""

    def do_GET(self):
        """Handle GET requests."""
        if self.path.startswith('/api/traces'):
            # Load sample trace data
            sample_trace_path = Path(__file__).parent / "sample-trace.json"
            with open(sample_trace_path, 'r') as f:
                data = json.load(f)

            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(data).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        """Suppress log messages."""
        pass


def start_mock_server(port=16686):
    """Start mock Jaeger server in background thread."""
    server = HTTPServer(('localhost', port), MockJaegerHandler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def test_otel_generator():
    """Test the OpenTelemetry data generator."""
    print("Starting mock Jaeger server...")
    server = start_mock_server()

    print("Initializing OpenTelemetry data generator...")
    api_config = APIConfig(type=APIType.Chat)

    otel_config = OpenTelemetryConfig(
        backend=OTelBackendType.JAEGER,
        endpoint="http://localhost:16686",
        service_name="langchain-agent",
        limit=10,
        enable_multi_turn_chat=True,
        extract_tool_calls=True,
    )

    data_config = DataConfig(
        type=DataGenType.OpenTelemetry,
        otel=otel_config
    )

    try:
        generator = OpenTelemetryDataGenerator(api_config, data_config, None)

        print(f"\n✅ Successfully loaded {len(generator.conversations)} conversations")
        print(f"   - Multi-turn mode: {generator.enable_multi_turn_chat}")
        print(f"   - Total conversation instances: {len(generator.conversations)}")

        # Print conversation details
        for i, (conv, metadata) in enumerate(zip(generator.conversations, generator.conversation_metadata)):
            program_id, turn_index = metadata
            print(f"\n   Conversation {i}:")
            print(f"     - Program ID: {program_id}")
            print(f"     - Turn Index: {turn_index}")
            print(f"     - Messages: {len(conv)}")
            for msg in conv:
                role = msg.role
                content = (msg.content[:50] + "...") if msg.content and len(msg.content) > 50 else msg.content
                if msg.tool_calls:
                    print(f"       - [{role}] {content} (+ {len(msg.tool_calls)} tool calls)")
                else:
                    print(f"       - [{role}] {content}")

        # Test lazy loading
        print("\n   Testing lazy data loading...")
        lazy_data = next(generator.get_data())
        actual_data = generator.load_lazy_data(lazy_data)
        print(f"   ✅ Lazy loading works: {type(actual_data).__name__}")

        print("\n✅ All tests passed!")
        return True

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        server.shutdown()


if __name__ == "__main__":
    success = test_otel_generator()
    sys.exit(0 if success else 1)
