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

"""Tests for SaturationDetector."""

import pytest

from inference_perf.config import AgenticSweepConfig, StageGenType
from inference_perf.loadgen.saturation_detector import (
    SaturationDetector,
    ProbeResult,
    SaturationResult,
)


class TestProbeResult:
    """Tests for ProbeResult dataclass."""

    def test_percentile_calculations(self):
        """Test percentile calculations with valid data."""
        probe = ProbeResult(
            rate=5.0,
            session_inference_times_ms=[100, 110, 120, 130, 200],
            completed_sessions=5,
            total_sessions=5,
        )

        assert probe.mean == pytest.approx(132.0, rel=0.01)
        assert probe.p50 == pytest.approx(120.0, rel=0.1)
        assert probe.p95 == pytest.approx(200.0, rel=0.1)

    def test_empty_results(self):
        """Test handling of empty results."""
        probe = ProbeResult(
            rate=5.0,
            session_inference_times_ms=[],
            completed_sessions=0,
            total_sessions=5,
        )

        assert probe.p50 == 0.0
        assert probe.p95 == 0.0
        assert probe.mean == 0.0

    def test_to_dict(self):
        """Test serialization to dictionary."""
        probe = ProbeResult(
            rate=5.0,
            session_inference_times_ms=[100, 150, 200],
            completed_sessions=3,
            total_sessions=5,
        )

        data = probe.to_dict()

        assert data["rate"] == 5.0
        assert data["completed_sessions"] == 3
        assert data["total_sessions"] == 5
        assert "p95_ms" in data
        assert "mean_ms" in data


class TestSaturationDetector:
    """Tests for SaturationDetector."""

    def test_get_probe_rates_linear(self):
        """Test linear probe rate generation."""
        config = AgenticSweepConfig(
            type=StageGenType.LINEAR,
            min_probe_rate=1.0,
            max_probe_rate=10.0,
            num_probes=5,
        )
        detector = SaturationDetector(config)
        rates = detector.get_probe_rates()

        assert len(rates) == 5
        assert rates[0] == pytest.approx(1.0, rel=0.01)
        assert rates[-1] == pytest.approx(10.0, rel=0.01)
        # Linear spacing
        assert rates[1] == pytest.approx(3.25, rel=0.01)
        assert rates[2] == pytest.approx(5.5, rel=0.01)

    def test_get_probe_rates_geometric(self):
        """Test geometric probe rate generation."""
        config = AgenticSweepConfig(
            type=StageGenType.GEOM,
            min_probe_rate=1.0,
            max_probe_rate=100.0,
            num_probes=3,
        )
        detector = SaturationDetector(config)
        rates = detector.get_probe_rates()

        assert len(rates) == 3
        assert rates[0] == pytest.approx(1.0, rel=0.01)
        assert rates[1] == pytest.approx(10.0, rel=0.01)
        assert rates[2] == pytest.approx(100.0, rel=0.01)

    def test_get_probe_rates_explicit(self):
        """Test explicit probe rate configuration."""
        config = AgenticSweepConfig(
            probe_rates=[1.0, 3.0, 5.0, 8.0],
        )
        detector = SaturationDetector(config)
        rates = detector.get_probe_rates()

        assert rates == [1.0, 3.0, 5.0, 8.0]

    def test_add_probe_result(self):
        """Test adding probe results."""
        config = AgenticSweepConfig()
        detector = SaturationDetector(config)

        detector.add_probe_result(
            rate=5.0,
            inference_times_ms=[100, 110, 120],
            completed_sessions=3,
            total_sessions=3,
        )

        assert len(detector.probe_results) == 1
        assert detector.probe_results[0].rate == 5.0
        assert detector.probe_results[0].completed_sessions == 3

    def test_detect_saturation_with_clear_degradation(self):
        """Test saturation detection when degradation is clear."""
        config = AgenticSweepConfig(degradation_threshold=0.2)  # 20%
        detector = SaturationDetector(config)

        # Add probe results with clear degradation at rate=10
        # Rate 1.0: baseline p95 = 100ms
        detector.add_probe_result(
            rate=1.0,
            inference_times_ms=[90, 95, 100, 100, 100],
            completed_sessions=5,
            total_sessions=5,
        )
        # Rate 5.0: p95 = 110ms (10% increase - below threshold)
        detector.add_probe_result(
            rate=5.0,
            inference_times_ms=[100, 105, 108, 110, 110],
            completed_sessions=5,
            total_sessions=5,
        )
        # Rate 10.0: p95 = 150ms (50% increase - above threshold)
        detector.add_probe_result(
            rate=10.0,
            inference_times_ms=[120, 130, 140, 145, 150],
            completed_sessions=5,
            total_sessions=5,
        )

        result = detector.detect_saturation()

        assert result.degradation_detected
        assert result.saturation_rate == 10.0
        assert result.baseline_p95 == pytest.approx(100.0, rel=0.1)
        assert result.saturation_p95 == pytest.approx(150.0, rel=0.1)

    def test_detect_saturation_no_degradation(self):
        """Test saturation detection when no degradation occurs."""
        config = AgenticSweepConfig(degradation_threshold=0.5)  # 50%
        detector = SaturationDetector(config)

        # All rates show similar latency (no degradation)
        for rate in [1.0, 5.0, 10.0]:
            detector.add_probe_result(
                rate=rate,
                inference_times_ms=[100, 105, 108, 110, 115],
                completed_sessions=5,
                total_sessions=5,
            )

        result = detector.detect_saturation()

        assert not result.degradation_detected
        assert result.saturation_rate == 10.0  # Falls back to max rate

    def test_detect_saturation_empty_results(self):
        """Test error handling with no probe results."""
        config = AgenticSweepConfig()
        detector = SaturationDetector(config)

        with pytest.raises(ValueError, match="No probe results"):
            detector.detect_saturation()

    def test_generate_stages_linear(self):
        """Test linear stage generation."""
        config = AgenticSweepConfig(
            type=StageGenType.LINEAR,
            num_stages=5,
            stage_duration=60,
            min_probe_rate=1.0,
        )
        detector = SaturationDetector(config)

        stages = detector.generate_stages(saturation_rate=10.0)

        assert len(stages) == 5
        # First stage
        assert stages[0][0] == pytest.approx(1.0, rel=0.1)
        assert stages[0][1] == 60
        # Last stage
        assert stages[-1][0] == pytest.approx(10.0, rel=0.1)
        assert stages[-1][1] == 60

    def test_generate_stages_geometric(self):
        """Test geometric stage generation."""
        config = AgenticSweepConfig(
            type=StageGenType.GEOM,
            num_stages=3,
            stage_duration=120,
            min_probe_rate=1.0,
        )
        detector = SaturationDetector(config)

        stages = detector.generate_stages(saturation_rate=100.0)

        assert len(stages) == 3
        assert stages[0][0] == pytest.approx(1.0, rel=0.1)
        assert stages[1][0] == pytest.approx(10.0, rel=0.1)
        assert stages[2][0] == pytest.approx(100.0, rel=0.1)
        assert all(s[1] == 120 for s in stages)

    def test_reset(self):
        """Test clearing probe results."""
        config = AgenticSweepConfig()
        detector = SaturationDetector(config)

        detector.add_probe_result(
            rate=5.0,
            inference_times_ms=[100],
            completed_sessions=1,
            total_sessions=1,
        )
        assert len(detector.probe_results) == 1

        detector.reset()
        assert len(detector.probe_results) == 0

    def test_saturation_result_to_dict(self):
        """Test SaturationResult serialization."""
        probe = ProbeResult(
            rate=5.0,
            session_inference_times_ms=[100, 150],
            completed_sessions=2,
            total_sessions=2,
        )

        result = SaturationResult(
            saturation_rate=10.0,
            probe_results=[probe],
            baseline_p95=100.0,
            saturation_p95=150.0,
            degradation_detected=True,
        )

        data = result.to_dict()

        assert data["saturation_rate"] == 10.0
        assert data["baseline_p95_ms"] == 100.0
        assert data["saturation_p95_ms"] == 150.0
        assert data["degradation_detected"] is True
        assert len(data["probe_results"]) == 1
