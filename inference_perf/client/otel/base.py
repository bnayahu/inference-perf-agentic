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

"""Base classes for OpenTelemetry backend clients."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class OTelSpan:
    """Normalized span representation."""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    operation_name: str
    start_time: datetime
    end_time: datetime
    duration_ms: float
    attributes: Dict[str, Any]
    events: List[Dict[str, Any]]
    status: str  # "OK" | "ERROR"


@dataclass
class OTelTrace:
    """Normalized trace representation."""
    trace_id: str
    spans: List[OTelSpan]
    start_time: datetime
    end_time: datetime
    duration_ms: float


class OTelBackendClient(ABC):
    """Abstract base class for OpenTelemetry backend clients."""

    @abstractmethod
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
        """Query traces from the backend."""
        pass

    @abstractmethod
    def get_trace(self, trace_id: str) -> Optional[OTelTrace]:
        """Get a specific trace by ID."""
        pass
