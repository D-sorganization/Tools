import threading
import time


class RateLimiter:
    """
    Thread-safe fixed-window rate limiter.

    Tracks requests per key (e.g., IP address) within a time window.
    Uses a fixed window algorithm: request counts are reset at the beginning of each time window.
    """

    def __init__(self, limit: int, window: int) -> None:
        """
        Initialize the rate limiter.

        Args:
            limit: Maximum number of requests allowed per window.
            window: Duration of the window in seconds.
        """
        self.limit = limit
        self.window = window
        # Storage: key -> (window_start_timestamp, count)
        self.hits: dict[str, tuple[int, int]] = {}
        self.lock = threading.Lock()
        self.last_global_window = 0

    def is_allowed(self, key: str) -> bool:
        """
        Check if a request is allowed for the given key.

        Args:
            key: Unique identifier for the client (e.g., IP address).

        Returns:
            True if allowed, False if limit exceeded.
        """
        now = time.time()
        current_window = int(now / self.window)

        with self.lock:
            # Cleanup mechanism: If the time window has moved forward,
            # previous window data is obsolete. Clear the storage to prevent memory leaks.
            if current_window > self.last_global_window:
                self.hits.clear()
                self.last_global_window = current_window

            last_window, count = self.hits.get(key, (0, 0))

            if last_window != current_window:
                # New window, reset count
                self.hits[key] = (current_window, 1)
                return True

            if count < self.limit:
                # Within limit
                self.hits[key] = (last_window, count + 1)
                return True

            # Limit exceeded
            return False
