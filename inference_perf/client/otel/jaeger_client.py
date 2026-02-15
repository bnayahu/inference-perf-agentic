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

"""Jaeger backend client for OpenTelemetry traces."""

import logging
import requests
from datetime import datetime
from typing import Any, Dict, List, Optional

from inference_perf.config import OTelAuthConfig, OTelAuthType
from .base import OTelBackendClient, OTelSpan, OTelTrace

logger = logging.getLogger(__name__)


class JaegerClient(OTelBackendClient):
    """Client for Jaeger backend using HTTP API."""

    def __init__(
        self,
        endpoint: str,
        auth: Optional[OTelAuthConfig] = None
    ):
        self.endpoint = endpoint.rstrip('/')
        self.auth = auth
        self.session = requests.Session()

        # Configure authentication
        if auth:
            if auth.type == OTelAuthType.BASIC and auth.username and auth.password:
                self.session.auth = (auth.username, auth.password)
            elif auth.type == OTelAuthType.BEARER and auth.bearer_token:
                self.session.headers['Authorization'] = f'Bearer {auth.bearer_token}'
            elif auth.type == OTelAuthType.API_KEY and auth.api_key and auth.api_key_header:
                self.session.headers[auth.api_key_header] = auth.api_key

    def query_traces(
        self,
        service_name: Optional[str] = None,
        operation_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100,
        min_duration_ms: Optional[int] = None,
        max_duration_ms: Optional[int] = None,
    ) -> List[OTelTrace]:
        """
        Query traces from Jaeger using the HTTP API.

        Jaeger API: GET /api/traces?service={service}&start={start}&end={end}
        """
        params: Dict[str, Any] = {
            'limit': limit,
        }

        if service_name:
            params['service'] = service_name

        if operation_name:
            params['operation'] = operation_name

        if start_time:
            # Jaeger expects microseconds
            params['start'] = int(start_time.timestamp() * 1_000_000)

        if end_time:
            params['end'] = int(end_time.timestamp() * 1_000_000)

        if min_duration_ms:
            params['minDuration'] = f'{min_duration_ms}ms'

        if max_duration_ms:
            params['maxDuration'] = f'{max_duration_ms}ms'

        if tags:
            # Jaeger tags format: key:value key2:value2
            params['tags'] = ' '.join(f'{k}:{v}' for k, v in tags.items())

        try:
            logger.info(f"Querying Jaeger at {self.endpoint}/api/traces with params: {params}")
            response = self.session.get(
                f'{self.endpoint}/api/traces',
                params=params,
                timeout=30
            )
            response.raise_for_status()

            data = response.json()
            traces = self._parse_jaeger_traces(data)
            logger.info(f"Fetched {len(traces)} traces from Jaeger")
            return traces

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to query Jaeger: {e}")
            raise ValueError(f"Failed to query Jaeger backend: {e}")

    def get_trace(self, trace_id: str) -> Optional[OTelTrace]:
        """Get a specific trace by ID."""
        try:
            response = self.session.get(
                f'{self.endpoint}/api/traces/{trace_id}',
                timeout=10
            )

            if response.status_code == 404:
                return None

            response.raise_for_status()
            data = response.json()

            traces = self._parse_jaeger_traces(data)
            return traces[0] if traces else None

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to get trace {trace_id} from Jaeger: {e}")
            return None

    def _parse_jaeger_traces(self, data: Dict[str, Any]) -> List[OTelTrace]:
        """Parse Jaeger JSON response into normalized OTelTrace objects."""
        traces = []

        for trace_data in data.get('data', []):
            trace_id = trace_data.get('traceID')
            if not trace_id:
                continue

            spans = []

            for span_data in trace_data.get('spans', []):
                # Parse span attributes from tags
                attributes = {}
                for tag in span_data.get('tags', []):
                    key = tag.get('key')
                    value = tag.get('value')
                    if key and value is not None:
                        attributes[key] = value

                # Parse span events from logs
                events = []
                for log in span_data.get('logs', []):
                    event_attrs = {}
                    for field in log.get('fields', []):
                        event_attrs[field.get('key')] = field.get('value')

                    events.append({
                        'timestamp': datetime.fromtimestamp(log.get('timestamp', 0) / 1_000_000),
                        'attributes': event_attrs
                    })

                # Get parent span ID from references
                parent_span_id = None
                for ref in span_data.get('references', []):
                    if ref.get('refType') == 'CHILD_OF':
                        parent_span_id = ref.get('spanID')
                        break

                # Calculate times
                start_time_us = span_data.get('startTime', 0)
                duration_us = span_data.get('duration', 0)
                start_time = datetime.fromtimestamp(start_time_us / 1_000_000)
                end_time = datetime.fromtimestamp((start_time_us + duration_us) / 1_000_000)

                span = OTelSpan(
                    trace_id=trace_id,
                    span_id=span_data.get('spanID', ''),
                    parent_span_id=parent_span_id,
                    operation_name=span_data.get('operationName', ''),
                    start_time=start_time,
                    end_time=end_time,
                    duration_ms=duration_us / 1000,
                    attributes=attributes,
                    events=events,
                    status="ERROR" if attributes.get('error') else "OK"
                )
                spans.append(span)

            if spans:
                trace = OTelTrace(
                    trace_id=trace_id,
                    spans=spans,
                    start_time=min(s.start_time for s in spans),
                    end_time=max(s.end_time for s in spans),
                    duration_ms=(max(s.end_time for s in spans) - min(s.start_time for s in spans)).total_seconds() * 1000
                )
                traces.append(trace)

        return traces
