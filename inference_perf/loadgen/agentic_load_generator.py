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

"""Agentic load generator for session-based workloads.

The AgenticLoadGenerator manages the scheduling and execution of agentic
sessions, supporting various arrival patterns (constant, poisson, trace replay)
and concurrency models.
"""

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from types import FrameType
from typing import Callable, List, Optional
import signal

import numpy as np
from tqdm import tqdm

from inference_perf.client.modelserver import ModelServerClient
from inference_perf.config import (
    APIConfig,
    LoadConfig,
    LoadType,
    SessionArrivalType,
    SessionArrivalConfig,
    SessionStageConfig,
    AgenticDelayConfig,
    AgenticSweepConfig,
    MultiLoRAConfig,
)
from inference_perf.models import Session

from .session_runner import SessionRunner, SessionResult
from .saturation_detector import SaturationDetector, SaturationResult

logger = logging.getLogger(__name__)


@dataclass
class StageResult:
    """Result of a load stage execution."""
    stage_id: int
    start_time: float
    end_time: float
    session_results: List[SessionResult] = field(default_factory=list)
    # Time series data
    active_sessions_timeseries: List[tuple] = field(default_factory=list)
    request_rate_timeseries: List[tuple] = field(default_factory=list)

    @property
    def total_sessions(self) -> int:
        return len(self.session_results)

    @property
    def completed_sessions(self) -> int:
        return sum(1 for sr in self.session_results if sr.turns_completed > 0)

    @property
    def avg_session_latency_ms(self) -> float:
        if not self.session_results:
            return 0.0
        return sum(sr.session_latency_ms for sr in self.session_results) / len(self.session_results)

    @property
    def session_throughput(self) -> float:
        """Sessions completed per second."""
        duration = self.end_time - self.start_time
        if duration <= 0:
            return 0.0
        return self.completed_sessions / duration


class AgenticLoadGenerator:
    """Schedules session arrivals and manages session execution.

    Supports three load modes:
    - AGENTIC: Rate-based session arrivals (constant or Poisson)
    - AGENTIC_CONCURRENT: Maintains fixed number of concurrent sessions
    - AGENTIC_TRACE_REPLAY: Replays sessions at original timestamps
    """

    def __init__(
        self,
        sessions: List[Session],
        load_config: LoadConfig,
        api_config: APIConfig,
        client_factory: Callable[[], ModelServerClient],
    ):
        """Initialize the agentic load generator.

        Args:
            sessions: List of Session objects to execute.
            load_config: Load configuration with arrival patterns.
            api_config: API configuration for requests.
            client_factory: Factory function to create model server clients.
        """
        self.sessions = sessions
        self.load_config = load_config
        self.api_config = api_config
        self.client_factory = client_factory

        # Session arrival config
        self.arrival_config = load_config.session_arrival
        self.delay_config = load_config.agentic or AgenticDelayConfig()

        # LoRA configuration
        self.lora_adapters: Optional[List[str]] = None
        self.lora_weights: Optional[List[float]] = None
        if load_config.lora_traffic_split is not None:
            self.lora_adapters = [config.name for config in load_config.lora_traffic_split]
            self.lora_weights = [config.split for config in load_config.lora_traffic_split]

        # Results storage
        self.stage_results: List[StageResult] = []

        # Sweep configuration
        self.sweep_config = load_config.agentic_sweep
        self.saturation_result: Optional[SaturationResult] = None

        # Interrupt handling
        self.interrupt_sig = False
        signal.signal(signal.SIGINT, self._sigint_handler)

        # Session index for round-robin
        self._session_index = 0

    def _sigint_handler(self, _signum: int, _frame: Optional[FrameType]) -> None:
        """Handle SIGINT by setting interrupt flag."""
        self.interrupt_sig = True

    def _get_next_session(self) -> Session:
        """Get the next session in round-robin order."""
        session = self.sessions[self._session_index % len(self.sessions)]
        self._session_index += 1
        return session

    def _get_worker_for_session(self, session: Session) -> int:
        """Determine which worker should handle this session.

        Uses consistent hashing based on session_id to ensure all turns
        of a session go to the same worker for better cache utilization.

        Args:
            session: The session to assign.

        Returns:
            Worker index (0-based).
        """
        if not self.load_config.worker_affinity:
            # Round-robin when affinity is disabled
            return self._session_index % max(1, self.load_config.num_workers)

        # Consistent hashing based on session_id
        return hash(session.session_id) % max(1, self.load_config.num_workers)

    def _get_lora_adapter(self) -> Optional[str]:
        """Get a randomly selected LoRA adapter based on configured weights."""
        if self.lora_adapters is not None and self.lora_weights is not None:
            return str(np.random.choice(self.lora_adapters, p=self.lora_weights))
        return None

    async def run(self, client: ModelServerClient) -> None:
        """Run all configured load stages.

        If sweep_config is set, runs preprocessing first to automatically
        detect saturation and generate optimal stages.

        Args:
            client: Model server client for LLM calls.
        """
        # Run preprocessing if sweep is configured
        if self.sweep_config:
            await self.preprocess(client)

        if not self.arrival_config or not self.arrival_config.stages:
            logger.warning("No stages configured, running single default stage")
            stages = [SessionStageConfig(
                rate=1.0,
                duration=60,
                total_sessions=len(self.sessions)
            )]
        else:
            stages = self.arrival_config.stages

        for stage_id, stage_config in enumerate(stages):
            if self.interrupt_sig:
                logger.info("Interrupted, stopping load generation")
                break

            logger.info(f"Starting stage {stage_id}")
            result = await self.run_stage(stage_id, stage_config, client)
            self.stage_results.append(result)
            logger.info(
                f"Stage {stage_id} completed: {result.completed_sessions}/{result.total_sessions} sessions, "
                f"avg latency {result.avg_session_latency_ms:.0f}ms"
            )

    async def run_stage(
        self,
        stage_id: int,
        stage_config: SessionStageConfig,
        client: ModelServerClient,
    ) -> StageResult:
        """Run a single load stage.

        Args:
            stage_id: Identifier for this stage.
            stage_config: Configuration for this stage.
            client: Model server client.

        Returns:
            StageResult with metrics from this stage.
        """
        if self.load_config.type == LoadType.AGENTIC:
            return await self._run_rate_based(stage_id, stage_config, client)
        elif self.load_config.type == LoadType.AGENTIC_CONCURRENT:
            return await self._run_concurrent(stage_id, stage_config, client)
        elif self.load_config.type == LoadType.AGENTIC_TRACE_REPLAY:
            return await self._run_trace_replay(stage_id, stage_config, client)
        else:
            raise ValueError(f"Unsupported load type: {self.load_config.type}")

    async def _run_rate_based(
        self,
        stage_id: int,
        stage_config: SessionStageConfig,
        client: ModelServerClient,
    ) -> StageResult:
        """Run a rate-based load stage (constant or Poisson arrivals).

        Args:
            stage_id: Stage identifier.
            stage_config: Configuration with rate and duration.
            client: Model server client.

        Returns:
            StageResult with session metrics.
        """
        result = StageResult(
            stage_id=stage_id,
            start_time=time.perf_counter(),
            end_time=0.0,
        )

        rate = stage_config.rate or 1.0
        duration = stage_config.duration or 60
        total_sessions = stage_config.total_sessions or int(rate * duration)

        # Determine arrival pattern
        is_poisson = (
            self.arrival_config and
            self.arrival_config.type == SessionArrivalType.POISSON
        )

        # Generate session start times
        start_times = self._generate_arrival_times(
            total_sessions, rate, is_poisson
        )

        # Track running sessions
        running_sessions: List[asyncio.Task] = []

        stage_start = time.perf_counter()

        # Time series data collection
        timeseries_data: List[tuple] = []
        request_rate_data: List[tuple] = []
        sampler_running = True
        last_completed = 0

        async def sampler() -> None:
            """Sample active sessions and completion rate periodically."""
            nonlocal last_completed
            while sampler_running:
                elapsed = time.perf_counter() - stage_start
                active = sum(1 for t in running_sessions if not t.done())
                timeseries_data.append((elapsed, active))

                # Calculate completion rate
                completed_now = sum(1 for t in running_sessions if t.done())
                if elapsed > 0:
                    rate_now = completed_now / elapsed
                    request_rate_data.append((elapsed, rate_now))
                last_completed = completed_now

                await asyncio.sleep(0.5)

        sampler_task = asyncio.create_task(sampler())

        with tqdm(total=total_sessions, desc=f"Stage {stage_id} sessions") as pbar:
            for i, arrival_time in enumerate(start_times):
                if self.interrupt_sig:
                    break

                # Wait until arrival time
                current_time = time.perf_counter() - stage_start
                if arrival_time > current_time:
                    await asyncio.sleep(arrival_time - current_time)

                # Start the session
                session = self._get_next_session()
                task = asyncio.create_task(
                    self._run_session(session, stage_id, client)
                )
                running_sessions.append(task)

                # Update progress
                completed = sum(1 for t in running_sessions if t.done())
                pbar.n = completed
                pbar.refresh()

        # Stop sampler
        sampler_running = False
        sampler_task.cancel()
        try:
            await sampler_task
        except asyncio.CancelledError:
            pass

        # Wait for all sessions to complete
        session_results = await asyncio.gather(*running_sessions, return_exceptions=True)

        # Collect results
        for res in session_results:
            if isinstance(res, SessionResult):
                result.session_results.append(res)
            elif isinstance(res, Exception):
                logger.error(f"Session failed with exception: {res}")

        result.end_time = time.perf_counter()
        result.active_sessions_timeseries = timeseries_data
        result.request_rate_timeseries = request_rate_data
        return result

    async def _run_concurrent(
        self,
        stage_id: int,
        stage_config: SessionStageConfig,
        client: ModelServerClient,
    ) -> StageResult:
        """Run with fixed concurrency (maintain N active sessions).

        Args:
            stage_id: Stage identifier.
            stage_config: Configuration with active_sessions and total_sessions.
            client: Model server client.

        Returns:
            StageResult with session metrics.
        """
        result = StageResult(
            stage_id=stage_id,
            start_time=time.perf_counter(),
            end_time=0.0,
        )

        active_sessions = stage_config.active_sessions or 10
        total_sessions = stage_config.total_sessions or len(self.sessions)

        # Semaphore to limit concurrency
        semaphore = asyncio.Semaphore(active_sessions)
        sessions_started = 0

        async def run_with_semaphore(session: Session) -> SessionResult:
            async with semaphore:
                return await self._run_session(session, stage_id, client)

        tasks: List[asyncio.Task] = []
        stage_start = time.perf_counter()

        # Time series data collection
        timeseries_data: List[tuple] = []
        request_rate_data: List[tuple] = []
        sampler_running = True

        async def sampler() -> None:
            """Sample active sessions and completion rate periodically."""
            while sampler_running:
                elapsed = time.perf_counter() - stage_start
                active = sum(1 for t in tasks if not t.done())
                timeseries_data.append((elapsed, active))

                # Calculate completion rate
                completed_now = sum(1 for t in tasks if t.done())
                if elapsed > 0:
                    rate_now = completed_now / elapsed
                    request_rate_data.append((elapsed, rate_now))

                await asyncio.sleep(0.5)

        sampler_task = asyncio.create_task(sampler())

        with tqdm(total=total_sessions, desc=f"Stage {stage_id} sessions") as pbar:
            while sessions_started < total_sessions and not self.interrupt_sig:
                session = self._get_next_session()
                task = asyncio.create_task(run_with_semaphore(session))
                tasks.append(task)
                sessions_started += 1

                # Update progress periodically
                completed = sum(1 for t in tasks if t.done())
                pbar.n = completed
                pbar.refresh()
                await asyncio.sleep(0)

        # Stop sampler
        sampler_running = False
        sampler_task.cancel()
        try:
            await sampler_task
        except asyncio.CancelledError:
            pass

        # Wait for all sessions to complete
        session_results = await asyncio.gather(*tasks, return_exceptions=True)

        for res in session_results:
            if isinstance(res, SessionResult):
                result.session_results.append(res)
            elif isinstance(res, Exception):
                logger.error(f"Session failed with exception: {res}")

        result.end_time = time.perf_counter()
        result.active_sessions_timeseries = timeseries_data
        result.request_rate_timeseries = request_rate_data
        return result

    async def _run_trace_replay(
        self,
        stage_id: int,
        stage_config: SessionStageConfig,
        client: ModelServerClient,
    ) -> StageResult:
        """Replay sessions at original timestamps with time scaling.

        Args:
            stage_id: Stage identifier.
            stage_config: Configuration (time_scale from arrival_config).
            client: Model server client.

        Returns:
            StageResult with session metrics.
        """
        result = StageResult(
            stage_id=stage_id,
            start_time=time.perf_counter(),
            end_time=0.0,
        )

        time_scale = self.arrival_config.time_scale if self.arrival_config else 1.0
        total_sessions = stage_config.total_sessions or len(self.sessions)

        # Sort sessions by original start time
        sorted_sessions = sorted(
            self.sessions[:total_sessions],
            key=lambda s: s.original_start_time_ms or 0
        )

        if not sorted_sessions:
            result.end_time = time.perf_counter()
            return result

        # Calculate relative start times
        first_time = sorted_sessions[0].original_start_time_ms or 0
        scheduled_starts = [
            ((s.original_start_time_ms or 0) - first_time) / 1000 / time_scale
            for s in sorted_sessions
        ]

        running_sessions: List[asyncio.Task] = []
        stage_start = time.perf_counter()

        # Time series data collection
        timeseries_data: List[tuple] = []
        request_rate_data: List[tuple] = []
        sampler_running = True

        async def sampler() -> None:
            """Sample active sessions and completion rate periodically."""
            while sampler_running:
                elapsed = time.perf_counter() - stage_start
                active = sum(1 for t in running_sessions if not t.done())
                timeseries_data.append((elapsed, active))

                # Calculate completion rate
                completed_now = sum(1 for t in running_sessions if t.done())
                if elapsed > 0:
                    rate_now = completed_now / elapsed
                    request_rate_data.append((elapsed, rate_now))

                await asyncio.sleep(0.5)

        sampler_task = asyncio.create_task(sampler())

        with tqdm(total=len(sorted_sessions), desc=f"Stage {stage_id} sessions") as pbar:
            for session, scheduled_start in zip(sorted_sessions, scheduled_starts):
                if self.interrupt_sig:
                    break

                # Wait until scheduled time
                current_time = time.perf_counter() - stage_start
                if scheduled_start > current_time:
                    await asyncio.sleep(scheduled_start - current_time)

                # Start the session
                task = asyncio.create_task(
                    self._run_session(session, stage_id, client)
                )
                running_sessions.append(task)

                # Update progress
                completed = sum(1 for t in running_sessions if t.done())
                pbar.n = completed
                pbar.refresh()

        # Stop sampler
        sampler_running = False
        sampler_task.cancel()
        try:
            await sampler_task
        except asyncio.CancelledError:
            pass

        # Wait for all sessions to complete
        session_results = await asyncio.gather(*running_sessions, return_exceptions=True)

        for res in session_results:
            if isinstance(res, SessionResult):
                result.session_results.append(res)
            elif isinstance(res, Exception):
                logger.error(f"Session failed with exception: {res}")

        result.end_time = time.perf_counter()
        result.active_sessions_timeseries = timeseries_data
        result.request_rate_timeseries = request_rate_data
        return result

    async def _run_session(
        self,
        session: Session,
        stage_id: int,
        client: ModelServerClient,
    ) -> SessionResult:
        """Execute a single session.

        Args:
            session: The Session to execute.
            stage_id: Current stage identifier.
            client: Model server client.

        Returns:
            SessionResult with execution metrics.
        """
        lora_adapter = self._get_lora_adapter()

        runner = SessionRunner(
            session=session,
            client=client,
            api_config=self.api_config,
            delay_config=self.delay_config,
            stage_id=stage_id,
            lora_adapter=lora_adapter,
        )

        return await runner.run()

    def _generate_arrival_times(
        self,
        count: int,
        rate: float,
        poisson: bool = False,
    ) -> List[float]:
        """Generate session arrival times.

        Args:
            count: Number of sessions.
            rate: Sessions per second.
            poisson: If True, use Poisson arrivals; otherwise constant.

        Returns:
            List of arrival times in seconds from start.
        """
        if poisson:
            # Poisson arrivals (exponential inter-arrival times)
            inter_arrivals = np.random.exponential(1.0 / rate, count)
            return list(np.cumsum(inter_arrivals))
        else:
            # Constant arrivals
            interval = 1.0 / rate
            return [i * interval for i in range(count)]

    def get_stage_results(self) -> List[StageResult]:
        """Get results from all completed stages."""
        return self.stage_results

    def get_all_session_results(self) -> List[SessionResult]:
        """Get all session results across all stages."""
        all_results = []
        for stage_result in self.stage_results:
            all_results.extend(stage_result.session_results)
        return all_results

    def get_saturation_result(self) -> Optional[SaturationResult]:
        """Get the saturation detection result if sweep was run."""
        return self.saturation_result

    async def preprocess(self, client: ModelServerClient) -> None:
        """Run preprocessing to detect saturation point.

        Executes probe stages at increasing session arrival rates
        and generates optimal stages based on detected saturation.

        Args:
            client: Model server client for LLM calls.
        """
        if not self.sweep_config:
            raise ValueError("sweep_config is required for preprocessing")

        logger.info("Running agentic preprocessing (saturation detection)")
        logger.info(
            f"Sweep config: num_probes={self.sweep_config.num_probes}, "
            f"probe_range=[{self.sweep_config.min_probe_rate:.1f}, {self.sweep_config.max_probe_rate:.1f}], "
            f"sessions_per_probe={self.sweep_config.num_sessions}"
        )

        detector = SaturationDetector(self.sweep_config)
        probe_rates = detector.get_probe_rates()

        # Run probe stages
        for probe_rate in probe_rates:
            if self.interrupt_sig:
                logger.info("Preprocessing interrupted by signal")
                break

            logger.info(f"Running probe at rate={probe_rate:.1f} sessions/sec")

            # Run probe stage
            result = await self._run_probe_stage(
                probe_rate,
                client,
                timeout=self.sweep_config.timeout,
            )

            # Collect session inference times
            inference_times = [
                sr.session_inference_time_ms
                for sr in result.session_results
                if sr.turns_completed > 0
            ]

            detector.add_probe_result(
                rate=probe_rate,
                inference_times_ms=inference_times,
                completed_sessions=result.completed_sessions,
                total_sessions=result.total_sessions,
            )

        # Detect saturation
        self.saturation_result = detector.detect_saturation()

        # Generate stages based on saturation
        stages = detector.generate_stages(self.saturation_result.saturation_rate)

        # Update arrival config with generated stages
        if self.arrival_config is None:
            self.arrival_config = SessionArrivalConfig()

        self.arrival_config.stages = [
            SessionStageConfig(rate=rate, duration=duration)
            for rate, duration in stages
        ]

        logger.info(
            f"Preprocessing complete: "
            f"saturation at {self.saturation_result.saturation_rate:.1f} sessions/sec, "
            f"baseline_p95={self.saturation_result.baseline_p95:.0f}ms, "
            f"generated {len(stages)} stages"
        )

    async def _run_probe_stage(
        self,
        rate: float,
        client: ModelServerClient,
        timeout: float,
    ) -> StageResult:
        """Run a single probe stage with timeout.

        Args:
            rate: Session arrival rate to test.
            client: Model server client.
            timeout: Maximum time for probe stage in seconds.

        Returns:
            StageResult with probe metrics.
        """
        result = StageResult(
            stage_id=-1,  # Probe stages use -1
            start_time=time.perf_counter(),
            end_time=0.0,
        )

        total_sessions = self.sweep_config.num_sessions if self.sweep_config else 100
        start_times = self._generate_arrival_times(total_sessions, rate, poisson=False)

        running_sessions: List[asyncio.Task] = []
        stage_start = time.perf_counter()

        with tqdm(total=total_sessions, desc=f"Probe rate={rate:.1f}") as pbar:
            for i, arrival_time in enumerate(start_times):
                if self.interrupt_sig:
                    break

                # Check timeout
                elapsed = time.perf_counter() - stage_start
                if elapsed > timeout:
                    logger.info(f"Probe timed out after {timeout:.0f}s with {i} sessions started")
                    break

                # Wait until arrival time
                current_time = time.perf_counter() - stage_start
                if arrival_time > current_time:
                    await asyncio.sleep(arrival_time - current_time)

                # Start session
                session = self._get_next_session()
                task = asyncio.create_task(
                    self._run_session(session, -1, client)
                )
                running_sessions.append(task)

                # Update progress
                completed = sum(1 for t in running_sessions if t.done())
                pbar.n = completed
                pbar.refresh()

        # Wait for running sessions with remaining timeout
        remaining_timeout = max(1.0, timeout - (time.perf_counter() - stage_start))

        if running_sessions:
            try:
                done, pending = await asyncio.wait(
                    running_sessions,
                    timeout=remaining_timeout,
                    return_when=asyncio.ALL_COMPLETED,
                )

                # Cancel pending tasks
                for task in pending:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

                # Collect results from completed tasks
                for task in done:
                    try:
                        res = task.result()
                        if isinstance(res, SessionResult):
                            result.session_results.append(res)
                    except Exception as e:
                        logger.debug(f"Probe session failed: {e}")

            except asyncio.TimeoutError:
                logger.warning("Probe wait timed out, cancelling remaining sessions")
                for task in running_sessions:
                    if not task.done():
                        task.cancel()
                        try:
                            await task
                        except asyncio.CancelledError:
                            pass

        result.end_time = time.perf_counter()

        logger.debug(
            f"Probe rate={rate:.1f} completed: "
            f"{result.completed_sessions}/{result.total_sessions} sessions"
        )

        return result
