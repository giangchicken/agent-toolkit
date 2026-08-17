"""Concurrency gating, token-bucket throttling, and the four defects fixed.

Time is never faked by patching ``time.monotonic``: the event loop reads it too,
so freezing it breaks the machinery under test. Instead the controller's
``_last_refill`` is backdated, which is arithmetically identical to the clock
advancing and leaves the loop alone.

Several tests are paired with a control that reproduces the original defect, so a
passing test means the fix works rather than that the measurement is blind.
"""

import ast
import asyncio
import gc
import inspect
import pathlib
import weakref

import pytest

from agent_toolkit.llm import traffic_control
from agent_toolkit.llm.traffic_control import TrafficController, get_traffic_controller

# Enough credits that the token bucket never interferes with a concurrency test.
UNTHROTTLED = 1_000_000


def _depth() -> int:
    """``len(inspect.stack())``, without the source-line lookup that makes it slow."""
    return len(inspect.stack(0))


class TestConcurrencyGate:
    async def test_a_burst_of_fifty_never_exceeds_max_concurrency(self) -> None:
        """T7's first criterion."""
        controller = TrafficController(
            max_concurrency=5, requests_per_minute=UNTHROTTLED
        )
        peak = 0

        async def worker() -> None:
            nonlocal peak
            async with controller:
                peak = max(peak, controller.active_requests)
                await asyncio.sleep(0)

        await asyncio.gather(*(worker() for _ in range(50)))

        assert peak <= 5
        # Non-vacuity: the gate was actually saturated, so the bound was tested.
        assert peak == 5

    async def test_every_request_gets_through(self) -> None:
        controller = TrafficController(
            max_concurrency=3, requests_per_minute=UNTHROTTLED
        )
        done = 0

        async def worker() -> None:
            nonlocal done
            async with controller:
                done += 1

        await asyncio.gather(*(worker() for _ in range(50)))
        assert done == 50
        assert controller.active_requests == 0

    async def test_acquisition_times_out_rather_than_waiting_forever(self) -> None:
        controller = TrafficController(
            max_concurrency=1, requests_per_minute=UNTHROTTLED, acquisition_timeout=0.01
        )
        async with controller:
            with pytest.raises(TimeoutError):
                async with controller:
                    pass


class TestActiveRequests:
    """T7's fifth criterion."""

    async def test_zero_before_and_after_a_normal_exit(self) -> None:
        controller = TrafficController(requests_per_minute=UNTHROTTLED)
        assert controller.active_requests == 0
        async with controller:
            assert controller.active_requests == 1
        assert controller.active_requests == 0

    async def test_zero_after_an_exception_inside_the_body(self) -> None:
        controller = TrafficController(requests_per_minute=UNTHROTTLED)
        with pytest.raises(RuntimeError):
            async with controller:
                assert controller.active_requests == 1
                raise RuntimeError("the call failed")
        assert controller.active_requests == 0

    async def test_it_counts_concurrent_holders(self) -> None:
        controller = TrafficController(
            max_concurrency=3, requests_per_minute=UNTHROTTLED
        )
        seen: list[int] = []
        release = asyncio.Event()

        async def worker() -> None:
            async with controller:
                seen.append(controller.active_requests)
                await release.wait()

        tasks = [asyncio.create_task(worker()) for _ in range(3)]
        while len(seen) < 3:
            await asyncio.sleep(0)
        assert controller.active_requests == 3
        release.set()
        await asyncio.gather(*tasks)
        assert controller.active_requests == 0

    def test_no_semaphore_private_is_read(self) -> None:
        """The original computed this from ``self._semaphore._value``.

        Checked over the parsed module rather than its text, so the docstring
        that *names* the defect does not read as committing it.
        """
        source = pathlib.Path(traffic_control.__file__).read_text(encoding="utf-8")
        offenders = [
            ast.unparse(node)
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.Attribute)
            and node.attr.startswith("_")
            and isinstance(node.value, ast.Attribute)
            and node.value.attr == "_semaphore"
        ]
        assert not offenders, offenders


class TestTokenBucket:
    """T7's second criterion: refill arithmetic, on a clock we control."""

    async def test_the_bucket_starts_full(self) -> None:
        controller = TrafficController(requests_per_minute=30)
        assert controller.available_credits == 30.0

    async def test_each_acquisition_spends_one_credit(self) -> None:
        controller = TrafficController(requests_per_minute=60)
        controller._last_refill = controller._last_refill + 10_000  # freeze refill
        for expected in (59.0, 58.0, 57.0):
            await controller._acquire_credit()
            assert controller.available_credits == pytest.approx(expected)

    async def test_refill_is_proportional_to_elapsed_time(self) -> None:
        controller = TrafficController(requests_per_minute=60)  # 1 credit/second
        controller._credits = 0.0
        controller._last_refill -= 10.0  # as if ten seconds had passed

        await controller._acquire_credit()

        # Ten seconds bought ten credits, one of which was just spent.
        assert controller.available_credits == pytest.approx(9.0, abs=0.01)

    async def test_refill_never_exceeds_the_bucket_size(self) -> None:
        controller = TrafficController(requests_per_minute=30)
        controller._credits = 0.0
        controller._last_refill -= 3600.0  # an hour of credits

        await controller._acquire_credit()

        assert controller.available_credits == pytest.approx(29.0, abs=0.01)

    async def test_an_empty_bucket_waits_for_exactly_one_credit(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        controller = TrafficController(requests_per_minute=60)  # 1 credit/second
        controller._credits = 0.0
        waits: list[float] = []
        real_sleep = asyncio.sleep

        async def fake_sleep(delay: float) -> None:
            waits.append(delay)
            controller._last_refill -= delay  # the clock "advances"
            await real_sleep(0)

        monkeypatch.setattr(traffic_control.asyncio, "sleep", fake_sleep)
        await controller._acquire_credit()

        assert waits == [pytest.approx(1.0, abs=0.01)]

    def test_zero_requests_per_minute_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="requests_per_minute must be > 0"):
            TrafficController(requests_per_minute=0)


class TestSustainedThrottlingDoesNotGrowTheStack:
    """T7's third criterion."""

    async def test_two_hundred_throttled_retries_stay_at_one_depth(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        controller = TrafficController(
            max_concurrency=100, requests_per_minute=60, acquisition_timeout=30.0
        )
        controller._credits = 0.0
        depths: list[int] = []
        real_sleep = asyncio.sleep

        async def fake_sleep(delay: float) -> None:
            # A tenth of a credit per wake, whatever was asked for. Fifty waiters
            # share one clock in reality, so most of them wake to find the credit
            # already taken and go round again; backdating by the full requested
            # delay per sleeper instead mints fifty credits at once and every
            # waiter succeeds on its first retry, which measures nothing.
            depths.append(_depth())
            controller._last_refill -= 0.1
            await real_sleep(0)

        monkeypatch.setattr(traffic_control.asyncio, "sleep", fake_sleep)

        # 50 waiters on an empty bucket: each loses the race for the refilled
        # credit repeatedly, which is what made the original recurse.
        await asyncio.gather(*(controller._acquire_credit() for _ in range(50)))

        assert len(depths) >= 200, f"only {len(depths)} retries -- not sustained"
        assert max(depths) - min(depths) <= 2, (
            f"stack grew from {min(depths)} to {max(depths)} frames"
        )

    async def test_the_stack_measurement_detects_recursion(self) -> None:
        """Non-vacuity: the metric above catches added frames when there are any."""
        depths: list[int] = []

        async def recursive(remaining: int) -> None:
            depths.append(_depth())
            if remaining:
                await recursive(remaining - 1)

        await recursive(20)
        assert max(depths) - min(depths) >= 20


class TestEventLoopLifetime:
    """T7's fourth criterion, and the reason the registry is not keyed by id()."""

    def test_a_controller_from_the_registry_survives_a_second_asyncio_run(self) -> None:
        async def go() -> int:
            controller = get_traffic_controller(
                "m", max_concurrency=2, requests_per_minute=UNTHROTTLED
            )
            async with controller:
                return controller.active_requests

        assert [asyncio.run(go()) for _ in range(3)] == [1, 1, 1]

    def test_the_defect_it_avoids_is_real(self) -> None:
        """Control: one controller reused across loops fails once contended.

        This is what the harvested code did -- build the controller during
        process-wide cached config resolution -- and it is why the registry keys
        per loop. Note the contention: an uncontended semaphore never binds a
        loop, so a test with a single task would pass either way.
        """
        controller = TrafficController(
            max_concurrency=1, requests_per_minute=UNTHROTTLED
        )

        async def contend() -> None:
            async def hold() -> None:
                async with controller:
                    await asyncio.sleep(0)

            await asyncio.gather(hold(), hold())

        asyncio.run(contend())
        with pytest.raises(RuntimeError, match="different event loop"):
            asyncio.run(contend())

    def test_id_of_a_loop_is_reused_so_it_cannot_be_the_key(self) -> None:
        """Why the registry deviates from the plan's ``(model, id(loop))``.

        Successive ``asyncio.run()`` calls allocate the new loop where the
        collected one stood, so equal ids do not mean the same loop.
        """
        ids: list[int] = []

        async def note() -> None:
            ids.append(id(asyncio.get_running_loop()))

        for _ in range(3):
            asyncio.run(note())

        assert len(set(ids)) < len(ids), "ids happened not to repeat on this run"

    async def test_the_same_provider_on_one_loop_is_the_same_controller(self) -> None:
        first = get_traffic_controller("shared", requests_per_minute=UNTHROTTLED)
        second = get_traffic_controller("shared", requests_per_minute=UNTHROTTLED)
        assert first is second

    async def test_different_providers_get_different_controllers(self) -> None:
        assert get_traffic_controller("a") is not get_traffic_controller("b")

    def test_a_finished_loops_controller_is_not_reused(self) -> None:
        """Reuse across loops is impossible because the entry does not survive.

        Stronger than comparing identities, which ``id()`` reuse makes
        unreliable: the first loop's controller is proved unreachable once its
        loop is gone. The registry holds the only strong reference to it, so the
        dead weakref is also proof the entry was dropped. Non-accumulation is
        measured as non-growth rather than as a count of one, because other
        tests in the session have live loops of their own in the same registry.
        """
        refs: list[weakref.ref[TrafficController]] = []

        async def note() -> None:
            refs.append(weakref.ref(get_traffic_controller("m")))

        before = len(traffic_control._controllers)
        asyncio.run(note())
        gc.collect()
        assert refs[0]() is None, "the controller outlived its event loop"
        assert len(traffic_control._controllers) <= before, "the registry accumulated"

    async def test_limits_are_ignored_after_the_first_call(self) -> None:
        first = get_traffic_controller("pinned", max_concurrency=2)
        second = get_traffic_controller("pinned", max_concurrency=99)
        assert second is first
        assert second.max_concurrency == 2

    def test_outside_a_running_loop_it_refuses(self) -> None:
        with pytest.raises(RuntimeError, match="no running event loop"):
            get_traffic_controller("m")


class TestCancellationReleasesTheSlot:
    """The fourth defect: ``except Exception`` never caught cancellation."""

    async def test_a_cancelled_wait_for_credit_does_not_leak_the_slot(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        controller = TrafficController(max_concurrency=1, requests_per_minute=60)
        controller._credits = 0.0
        waiting = asyncio.Event()
        real_sleep = asyncio.sleep

        async def fake_sleep(delay: float) -> None:
            waiting.set()
            await real_sleep(3600)  # block until cancelled

        monkeypatch.setattr(traffic_control.asyncio, "sleep", fake_sleep)

        async def blocked() -> None:
            async with controller:
                pass

        task = asyncio.create_task(blocked())
        await waiting.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        # The slot is free: a fresh acquisition would hang forever if it leaked.
        monkeypatch.undo()
        controller._credits = 5.0
        async with asyncio.timeout(1):
            async with controller:
                assert controller.active_requests == 1

    def test_exception_alone_would_not_have_caught_it(self) -> None:
        """Non-vacuity for the fix: cancellation is not an ``Exception``."""
        assert not issubclass(asyncio.CancelledError, Exception)
        assert issubclass(asyncio.CancelledError, BaseException)
