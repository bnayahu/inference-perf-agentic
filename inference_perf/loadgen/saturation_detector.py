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

"""Saturation detection for agentic workloads.

This module provides tools for detecting the saturation point of an
inference system under agentic workloads by monitoring session inference
time degradation as session arrival rate increases.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from inference_perf.config import AgenticSweepConfig, StageGenType

logger = logging.getLogger(__name__)


@dataclass
class ProbeResult:
    """Result of a single probe stage at a specific session arrival rate.

    Attributes:
        rate: Session arrival rate (sessions/sec) for this probe.
        session_inference_times_ms: List of session inference times in milliseconds.
        completed_sessions: Number of sessions that completed successfully.
        total_sessions: Total number of sessions started.
    """
    rate: float
    session_inference_times_ms: List[float] = field(default_factory=list)
    completed_sessions: int = 0
    total_sessions: int = 0

    @property
    def p50(self) -> float:
        """50th percentile (median) session inference time."""
        if not self.session_inference_times_ms:
            return 0.0
        return float(np.percentile(self.session_inference_times_ms, 50))

    @property
    def p90(self) -> float:
        """90th percentile session inference time."""
        if not self.session_inference_times_ms:
            return 0.0
        return float(np.percentile(self.session_inference_times_ms, 90))

    @property
    def p95(self) -> float:
        """95th percentile session inference time."""
        if not self.session_inference_times_ms:
            return 0.0
        return float(np.percentile(self.session_inference_times_ms, 95))

    @property
    def p99(self) -> float:
        """99th percentile session inference time."""
        if not self.session_inference_times_ms:
            return 0.0
        return float(np.percentile(self.session_inference_times_ms, 99))

    @property
    def mean(self) -> float:
        """Mean session inference time."""
        if not self.session_inference_times_ms:
            return 0.0
        return float(np.mean(self.session_inference_times_ms))

    def to_dict(self) -> dict:
        """Convert to dictionary for reporting."""
        return {
            "rate": self.rate,
            "completed_sessions": self.completed_sessions,
            "total_sessions": self.total_sessions,
            "mean_ms": self.mean,
            "p50_ms": self.p50,
            "p90_ms": self.p90,
            "p95_ms": self.p95,
            "p99_ms": self.p99,
        }


@dataclass
class SaturationResult:
    """Result of saturation detection.

    Attributes:
        saturation_rate: Detected saturation rate (sessions/sec).
        probe_results: List of probe results for each tested rate.
        baseline_p95: Baseline p95 session inference time at lowest rate.
        saturation_p95: p95 session inference time at saturation point.
        degradation_detected: Whether degradation was detected before max rate.
    """
    saturation_rate: float
    probe_results: List[ProbeResult]
    baseline_p95: float
    saturation_p95: float
    degradation_detected: bool

    def to_dict(self) -> dict:
        """Convert to dictionary for reporting."""
        return {
            "saturation_rate": self.saturation_rate,
            "baseline_p95_ms": self.baseline_p95,
            "saturation_p95_ms": self.saturation_p95,
            "degradation_detected": self.degradation_detected,
            "probe_results": [pr.to_dict() for pr in self.probe_results],
        }


class SaturationDetector:
    """Detects saturation point for agentic workloads.

    The detector runs multiple probe stages at increasing session arrival
    rates and monitors session_inference_time_p95 for degradation. When
    the p95 latency increases beyond the configured threshold, saturation
    is detected.

    Example:
        config = AgenticSweepConfig(
            min_probe_rate=1.0,
            max_probe_rate=20.0,
            num_probes=5,
            degradation_threshold=0.2,
        )
        detector = SaturationDetector(config)

        # Add probe results as they complete
        detector.add_probe_result(rate=1.0, inference_times=[100, 110, 120])
        detector.add_probe_result(rate=5.0, inference_times=[105, 115, 130])
        detector.add_probe_result(rate=10.0, inference_times=[150, 180, 200])

        # Detect saturation
        result = detector.detect_saturation()
        print(f"Saturation at {result.saturation_rate} sessions/sec")
    """

    def __init__(self, config: AgenticSweepConfig):
        """Initialize the saturation detector.

        Args:
            config: Agentic sweep configuration with probe parameters.
        """
        self.config = config
        self.probe_results: List[ProbeResult] = []

    def get_probe_rates(self) -> List[float]:
        """Generate probe rates to test.

        Returns:
            List of session arrival rates to probe, from min to max.
        """
        if self.config.probe_rates:
            return sorted(self.config.probe_rates)

        # Generate rates from min to max
        if self.config.type == StageGenType.LINEAR:
            rates = np.linspace(
                self.config.min_probe_rate,
                self.config.max_probe_rate,
                self.config.num_probes
            )
        else:  # GEOMETRIC
            rates = np.geomspace(
                self.config.min_probe_rate,
                self.config.max_probe_rate,
                self.config.num_probes
            )

        return [float(r) for r in rates]

    def add_probe_result(
        self,
        rate: float,
        inference_times_ms: List[float],
        completed_sessions: int,
        total_sessions: int,
    ) -> None:
        """Add results from a probe stage.

        Args:
            rate: Session arrival rate tested.
            inference_times_ms: List of session inference times in milliseconds.
            completed_sessions: Number of sessions completed successfully.
            total_sessions: Total sessions started in probe.
        """
        probe = ProbeResult(
            rate=rate,
            session_inference_times_ms=inference_times_ms,
            completed_sessions=completed_sessions,
            total_sessions=total_sessions,
        )
        self.probe_results.append(probe)

        logger.info(
            f"Probe at rate={rate:.1f}: {completed_sessions}/{total_sessions} sessions, "
            f"p95={probe.p95:.0f}ms, mean={probe.mean:.0f}ms"
        )

    def detect_saturation(self) -> SaturationResult:
        """Analyze probe results to detect saturation point.

        The saturation point is where the p95 session inference time
        increases beyond the configured degradation threshold relative
        to the baseline (lowest rate tested).

        Returns:
            SaturationResult with detected saturation rate and metrics.

        Raises:
            ValueError: If no probe results have been added.
        """
        if not self.probe_results:
            raise ValueError("No probe results to analyze")

        # Sort by rate
        sorted_probes = sorted(self.probe_results, key=lambda p: p.rate)

        # Get baseline p95 from lowest rate with valid data
        baseline_p95 = 0.0
        for probe in sorted_probes:
            if probe.p95 > 0:
                baseline_p95 = probe.p95
                break

        if baseline_p95 <= 0:
            logger.warning("No valid baseline p95 found, using max rate as saturation")
            return SaturationResult(
                saturation_rate=sorted_probes[-1].rate,
                probe_results=self.probe_results,
                baseline_p95=0.0,
                saturation_p95=sorted_probes[-1].p95,
                degradation_detected=False,
            )

        logger.info(f"Baseline session_inference_time_p95: {baseline_p95:.0f}ms")

        # Find saturation point: where p95 degrades by threshold
        threshold_multiplier = 1.0 + self.config.degradation_threshold
        saturation_rate = sorted_probes[-1].rate  # Default to max if no degradation
        saturation_p95 = sorted_probes[-1].p95
        degradation_detected = False

        for probe in sorted_probes[1:]:
            if probe.p95 > baseline_p95 * threshold_multiplier:
                # Found degradation point
                saturation_rate = probe.rate
                saturation_p95 = probe.p95
                degradation_detected = True
                degradation_pct = (probe.p95 / baseline_p95 - 1) * 100

                logger.info(
                    f"Saturation detected at rate={probe.rate:.1f}: "
                    f"p95={probe.p95:.0f}ms ({degradation_pct:.0f}% increase from baseline)"
                )
                break

        if not degradation_detected:
            logger.warning(
                f"No saturation detected up to rate={sorted_probes[-1].rate:.1f}, "
                f"using max rate as saturation point"
            )

        return SaturationResult(
            saturation_rate=saturation_rate,
            probe_results=self.probe_results,
            baseline_p95=baseline_p95,
            saturation_p95=saturation_p95,
            degradation_detected=degradation_detected,
        )

    def generate_stages(
        self,
        saturation_rate: float,
    ) -> List[Tuple[float, int]]:
        """Generate session arrival rate stages based on detected saturation.

        Args:
            saturation_rate: Detected saturation rate.

        Returns:
            List of (rate, duration) tuples for each stage.
        """
        num_stages = self.config.num_stages
        stage_duration = self.config.stage_duration

        # Generate rates from low to saturation
        if self.config.type == StageGenType.LINEAR:
            rates = np.linspace(
                self.config.min_probe_rate,
                saturation_rate,
                num_stages
            )
        else:  # GEOMETRIC
            rates = np.geomspace(
                self.config.min_probe_rate,
                saturation_rate,
                num_stages
            )

        stages = [(float(rate), stage_duration) for rate in rates]

        logger.info(
            f"Generated {len(stages)} stages with rates: "
            f"{[f'{r:.1f}' for r, _ in stages]}"
        )

        return stages

    def reset(self) -> None:
        """Clear all probe results for a fresh detection run."""
        self.probe_results.clear()
