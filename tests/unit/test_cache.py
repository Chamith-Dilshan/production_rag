from app.cache import ResponseCache


def test_cache_set_and_get_returns_value() -> None:
    cache = ResponseCache(ttl_seconds=60)

    cache.set("Hello world", "cached-response")

    assert cache.get("hello world") == "cached-response"
    assert cache.status["cached_entries"] == 1


def test_cache_expires_after_ttl() -> None:
    cache = ResponseCache(ttl_seconds=0)

    cache.set("expired prompt", "stale")

    assert cache.get("expired prompt") is None
    assert cache.status["misses"] >= 1
