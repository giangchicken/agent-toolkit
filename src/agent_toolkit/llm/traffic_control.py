"""Traffic control primitives for LLM providers.

Two mechanisms guard every LLM call:

1. **Concurrency gate** -- ``asyncio.Semaphore(max_concurrency)``
   Limits how many HTTP requests are in-flight at once.
   Prevents connection pool exhaustion and event-loop starvation.

2. **Request-rate throttle** -- token-bucket algorithm
   Limits how many API calls can be made per minute.
   Prevents hitting the provider's rate limit (429 / 524).

One **bucket credit** = one **API request** (not one LLM token).

Four defects in the harvested version are fixed here; each has a test named for
it in ``tests/test_traffic_control.py``:

- ``_acquire_credit`` retried by calling itself after sleeping. Under sustained
  throttling with many waiters, every retry added a frame. It is a loop now.
- ``active_requests`` returned ``max_concurrency - self._semaphore._value``,
  reading a CPython private. An explicit counter replaces it.
- The controller was built during config resolution, which was
  ``lru_cache``-d process-wide, so a second ``asyncio.run()`` in one process
  reused a semaphore whose waiters belonged to a closed loop.
  :func:`get_traffic_controller` keys them per event loop instead.
- ``__aenter__`` released the concurrency slot under ``except Exception``, with
  the comment "if rate limiter fails/cancels, release semaphore". Cancellation
  does not raise ``Exception``, so the case the comment names was the one case
  that leaked the slot -- permanently, since nothing ever releases it again.
"""

import asyncio
import time
import weakref
from types import TracebackType

from agent_toolkit.logging import get_logger

logger = get_logger(__name__)

__all__ = ["TrafficController", "get_traffic_controller"]


class TrafficController:
    """Controls concurrency and rate limits for LLM providers.

    Protects both the local system (resource exhaustion) and
    remote provider (rate limits).

    Usage::

        controller = TrafficController("GLM-5.1", max_concurrency=5, requests_per_minute=30)
        async with controller:
            # At most 5 calls are in-flight; at most 30/min total
            response = await sdk_complete(...)

    One instance belongs to one event loop, because its semaphore and lock do.
    Construct it inside the loop that will use it, or get it from
    :func:`get_traffic_controller`, which does that for you.
    """

    def __init__(
        self,
        provider_name: str = "llm",
        max_concurrency: int = 5,
        requests_per_minute: int = 30,
        acquisition_timeout: float = 120.0,
    ) -> None:
        """
        Args:
            provider_name: Label for logging.
            max_concurrency: Max simultaneous in-flight API requests.
            requests_per_minute: Max API requests per minute before local throttling.
            acquisition_timeout: Max seconds to wait for a slot before failing.
        """
        self.provider_name = provider_name
        self.max_concurrency = max_concurrency
        if requests_per_minute <= 0:
            raise ValueError("requests_per_minute must be > 0")
        self.rpm = requests_per_minute
        self.acquisition_timeout = acquisition_timeout

        # Concurrency Gate
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._in_flight = 0

        # Request-Rate Throttle (Token Bucket — one credit = one API request)
        self._credits = float(requests_per_minute)
        self._last_refill = time.monotonic()
        self._refill_rate = requests_per_minute / 60.0  # credits per second
        self._lock = asyncio.Lock()  # Protects credit state

    async def _acquire_credit(self) -> None:
        """Consume one request-rate credit, waiting if the bucket is empty.

        A loop rather than the original's tail call. The logic is unchanged, but
        a retry no longer costs a stack frame: with many waiters on an empty
        bucket, each one loses the race for the refilled credit and goes round
        again, so the recursion depth grew with the contention.
        """
        while True:
            async with self._lock:
                now = time.monotonic()
                elapsed = now - self._last_refill

                # Refill credits based on elapsed time
                new_credits = elapsed * self._refill_rate
                if new_credits > 0:
                    self._credits = min(float(self.rpm), self._credits + new_credits)
                    self._last_refill = now

                # Consume one credit (= one API request)
                if self._credits >= 1:
                    self._credits -= 1.0
                    return

                # Calculate wait time needed for 1 credit
                wait_time = (1.0 - self._credits) / self._refill_rate

            # Wait outside lock to avoid blocking other tasks. wait_time is
            # always positive here -- credits are below 1 and the refill rate is
            # positive -- so the original's `if wait_time > 0` guard is gone: as
            # a loop condition it would have meant spinning, not proceeding.
            logger.debug(
                "[%s] Rate limit active, waiting %.2fs for request credit",
                self.provider_name,
                wait_time,
            )
            await asyncio.sleep(wait_time)

    @property
    def available_credits(self) -> float:
        """Current number of available request credits (approximate)."""
        return self._credits

    @property
    def active_requests(self) -> int:
        """Number of requests currently between ``__aenter__`` and ``__aexit__``."""
        return self._in_flight

    async def __aenter__(self) -> "TrafficController":
        """
        Acquire concurrency slot AND request-rate credit.
        Raises asyncio.TimeoutError if system is overloaded.
        """
        start = time.monotonic()

        # 1. Acquire Concurrency Slot
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

        # 2. Acquire Request-Rate Credit
        try:
            await self._acquire_credit()
        except BaseException:
            # If rate limiter fails/cancels, release semaphore
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
        """Release concurrency slot."""
        self._in_flight -= 1
        self._semaphore.release()
        return None


# One controller per (event loop, provider name). Keyed by the loop *object* in a
# weak-keyed map, not by `id(loop)`: successive `asyncio.run()` calls in one
# process are observed to allocate the new loop at the same address as the
# collected one, so an id key would hand a fresh loop the controller belonging to
# the closed one -- exactly the defect this replaces. Weak keys also mean a
# finished loop's controllers are reclaimed rather than accumulating.
_controllers: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, dict[str, TrafficController]]" = weakref.WeakKeyDictionary()


def get_traffic_controller(
    provider_name: str,
    *,
    max_concurrency: int = 5,
    requests_per_minute: int = 30,
) -> TrafficController:
    """Return the controller for ``provider_name`` on the running event loop.

    Must be called from inside a running loop, which is the point: the semaphore
    and lock it holds bind to whichever loop first contends them, so the
    controller cannot be built before the loop exists.

    ``max_concurrency`` and ``requests_per_minute`` are used only when the
    controller is created. Later calls naming the same provider on the same loop
    get the existing one, and their limits are ignored -- one provider's budget
    cannot be raised halfway through a run by asking differently.
    """
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
