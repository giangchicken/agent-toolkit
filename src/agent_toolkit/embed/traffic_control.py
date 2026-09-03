"""logic. This route's own concurrency and rate budget."""

import asyncio
import time
import weakref
from types import TracebackType

from agent_toolkit.logging import get_logger

logger = get_logger(__name__)

__all__ = ["TrafficController", "get_traffic_controller"]


class TrafficController:
    def __init__(
        self,
        provider_name: str = "embed",
        max_concurrency: int = 5,
        requests_per_minute: int = 30,
        acquisition_timeout: float = 120.0,
    ) -> None:
        self.provider_name = provider_name
        self.max_concurrency = max_concurrency
        if requests_per_minute <= 0:
            raise ValueError("requests_per_minute must be > 0")
        self.rpm = requests_per_minute
        self.acquisition_timeout = acquisition_timeout

        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._in_flight = 0

        self._credits = float(requests_per_minute)
        self._last_refill = time.monotonic()
        self._refill_rate = requests_per_minute / 60.0
        self._lock = asyncio.Lock()

    async def _acquire_credit(self) -> None:
        while True:
            async with self._lock:
                now = time.monotonic()
                elapsed = now - self._last_refill

                new_credits = elapsed * self._refill_rate
                if new_credits > 0:
                    self._credits = min(float(self.rpm), self._credits + new_credits)
                    self._last_refill = now

                if self._credits >= 1:
                    self._credits -= 1.0
                    return

                wait_time = (1.0 - self._credits) / self._refill_rate

            logger.debug(
                "[%s] Rate limit active, waiting %.2fs for request credit",
                self.provider_name,
                wait_time,
            )
            await asyncio.sleep(wait_time)

    @property
    def available_credits(self) -> float:
        return self._credits

    @property
    def active_requests(self) -> int:
        return self._in_flight

    async def __aenter__(self) -> "TrafficController":
        start = time.monotonic()

        try:
            await asyncio.wait_for(
                self._semaphore.acquire(), timeout=self.acquisition_timeout
            )
        except TimeoutError:
            logger.error(
                "[%s] Concurrency limit (%d) exceeded for >%.1fs — %d requests in flight.",
                self.provider_name,
                self.max_concurrency,
                self.acquisition_timeout,
                self.active_requests,
            )
            raise

        try:
            await self._acquire_credit()
        except BaseException:
            self._semaphore.release()
            raise

        self._in_flight += 1

        wait_duration = time.monotonic() - start
        if wait_duration > 1.0:
            logger.warning(
                "[%s] Traffic control wait: %.2fs (credits: %.1f/%d, in-flight: %d)",
                self.provider_name,
                wait_duration,
                self._credits,
                self.rpm,
                self.active_requests,
            )

        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self._in_flight -= 1
        self._semaphore.release()
        return None


_controllers: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, dict[str, TrafficController]]" = weakref.WeakKeyDictionary()


def get_traffic_controller(
    provider_name: str,
    *,
    max_concurrency: int = 5,
    requests_per_minute: int = 30,
) -> TrafficController:
    loop = asyncio.get_running_loop()
    per_loop = _controllers.get(loop)
    if per_loop is None:
        per_loop = {}
        _controllers[loop] = per_loop

    controller = per_loop.get(provider_name)
    if controller is None:
        controller = TrafficController(
            provider_name,
            max_concurrency=max_concurrency,
            requests_per_minute=requests_per_minute,
        )
        per_loop[provider_name] = controller
    return controller
