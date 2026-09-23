import hashlib
from datetime import UTC, datetime, timedelta


class ResponseCache:
    """
    In Memory response cache with TTL(time-to-live)
    In production, replace this with Redis for:
    - Persistence across restarts
    - Shared cache across multiple instances
    - Build in TTL
    """

    def __init__(self, ttl_seconds: int = 300):
        self.ttl = ttl_seconds
        self._cache: dict[str, dict] = {}
        self._hits = 0
        self._misses = 0

    @staticmethod
    def _make_key(query: str) -> str:
        """Create a cache key from the normalized query"""
        normalized = query.lower().strip()
        return hashlib.sha256(normalized.encode()).hexdigest()

    def get(self, query: str) -> str | None:
        """Get a cached response if it hits and hasn't expired.
        return None on cache miss.
        """
        key = self._make_key(query)

        if key in self._cache:
            entry = self._cache[key]
            cached_at = datetime.fromisoformat(entry["timestamp"])
            if datetime.now(UTC) - cached_at < timedelta(seconds=self.ttl):
                self._hits += 1
                return entry["response"]
            else:
                del self._cache[key]

        self._misses += 1
        return None

    def set(self, query: str, response: str) -> None:
        """Cache a response"""
        key = self._make_key(query)
        self._cache[key] = {
            "response": response,
            "timestamp": datetime.now(UTC).isoformat(),
            "query": query,
        }

    @property
    def status(self) -> dict[str, int | float | str]:
        """Return cache performance statistics."""
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0.0
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": f"{hit_rate:.1%}",
            "cached_entries": len(self._cache),
        }
