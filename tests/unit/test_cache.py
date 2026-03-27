"""Unit tests for the cache module."""

import time
import pytest

from src.cache import AnalysisCache


class TestAnalysisCache:
    """Tests for AnalysisCache class."""

    def test_get_key_consistency(self):
        """Test that get_key produces consistent hashes."""
        key1 = AnalysisCache.get_key("image1", "prompt1", "model1")
        key2 = AnalysisCache.get_key("image1", "prompt1", "model1")

        assert key1 == key2
        assert len(key1) == 64  # SHA256 produces 64 char hex string

    def test_get_key_different_inputs(self):
        """Test that different inputs produce different keys."""
        key1 = AnalysisCache.get_key("image1", "prompt1", "model1")
        key2 = AnalysisCache.get_key("image2", "prompt1", "model1")
        key3 = AnalysisCache.get_key("image1", "prompt2", "model1")
        key4 = AnalysisCache.get_key("image1", "prompt1", "model2")

        assert key1 != key2
        assert key1 != key3
        assert key1 != key4

    def test_cache_set_and_get(self):
        """Test basic cache set and get operations."""
        cache = AnalysisCache(ttl=3600)
        key = "test_key"
        value = "test_result"

        cache.set(key, value)
        result = cache.get(key)

        assert result == value

    def test_cache_miss(self):
        """Test cache miss returns None."""
        cache = AnalysisCache(ttl=3600)

        result = cache.get("nonexistent_key")

        assert result is None

    def test_cache_ttl_expiration(self):
        """Test that cache entries expire after TTL."""
        cache = AnalysisCache(ttl=0)  # Immediate expiration

        cache.set("key", "value")
        time.sleep(0.1)  # Small delay to ensure expiration

        result = cache.get("key")

        assert result is None

    def test_cache_delete(self):
        """Test cache entry deletion."""
        cache = AnalysisCache(ttl=3600)

        cache.set("key", "value")
        assert cache.get("key") == "value"

        deleted = cache.delete("key")
        assert deleted is True
        assert cache.get("key") is None

    def test_cache_delete_nonexistent(self):
        """Test deleting non-existent key returns False."""
        cache = AnalysisCache(ttl=3600)

        deleted = cache.delete("nonexistent")

        assert deleted is False

    def test_cache_clear(self):
        """Test clearing all cache entries."""
        cache = AnalysisCache(ttl=3600)

        cache.set("key1", "value1")
        cache.set("key2", "value2")

        cache.clear()

        assert cache.get("key1") is None
        assert cache.get("key2") is None
        assert len(cache) == 0

    def test_cache_maxsize_eviction(self):
        """Test LRU eviction when maxsize is reached."""
        cache = AnalysisCache(ttl=3600, maxsize=3)

        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")
        cache.set("key4", "value4")  # Should evict key1

        assert cache.get("key1") is None  # Evicted
        assert cache.get("key2") == "value2"
        assert cache.get("key3") == "value3"
        assert cache.get("key4") == "value4"

    def test_cache_stats(self):
        """Test cache statistics."""
        cache = AnalysisCache(ttl=3600, maxsize=100)

        cache.set("key1", "value1")
        cache.set("key2", "value2")

        cache.get("key1")  # Hit
        cache.get("key1")  # Hit
        cache.get("nonexistent")  # Miss

        stats = cache.stats()

        assert stats["size"] == 2
        assert stats["hits"] == 2
        assert stats["misses"] == 1
        assert stats["maxsize"] == 100
        assert stats["ttl"] == 3600

    def test_cache_contains(self):
        """Test `in` operator for cache."""
        cache = AnalysisCache(ttl=3600)

        cache.set("key", "value")

        assert "key" in cache
        assert "nonexistent" not in cache

    def test_cache_repr(self):
        """Test string representation."""
        cache = AnalysisCache(ttl=3600, maxsize=100)

        repr_str = repr(cache)

        assert "AnalysisCache" in repr_str
        assert "3600" in repr_str
        assert "100" in repr_str

    def test_cache_invalid_ttl(self):
        """Test that negative TTL raises ValueError."""
        with pytest.raises(ValueError, match="TTL must be non-negative"):
            AnalysisCache(ttl=-1)

    def test_cache_invalid_maxsize(self):
        """Test that negative maxsize raises ValueError."""
        with pytest.raises(ValueError, match="Maxsize must be non-negative"):
            AnalysisCache(maxsize=-1)

    def test_prune_expired(self):
        """Test pruning expired entries."""
        cache = AnalysisCache(ttl=0)  # Immediate expiration

        cache.set("key1", "value1")
        cache.set("key2", "value2")
        time.sleep(0.1)

        removed = cache.prune_expired()

        assert removed == 2
        assert len(cache) == 0

    def test_lru_ordering(self):
        """Test that LRU ordering is maintained."""
        cache = AnalysisCache(ttl=3600, maxsize=3)

        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")

        # Access key1 (moves to end)
        cache.get("key1")

        # Add key4 (should evict key2, not key1)
        cache.set("key4", "value4")

        assert cache.get("key1") == "value1"  # Still present
        assert cache.get("key2") is None  # Evicted
