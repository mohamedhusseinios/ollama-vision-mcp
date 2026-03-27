"""
Caching module for Ollama Vision MCP Server

This module provides a TTL-based LRU cache for storing image analysis results
to avoid redundant API calls for repeated analyses of the same image.
"""

from __future__ import annotations

import hashlib
import time
from collections import OrderedDict
from threading import Lock
from typing import Any, Dict, Optional

from .exceptions import CacheError


class AnalysisCache:
    """
    Thread-safe LRU cache with TTL (Time-To-Live) for image analysis results.

    This cache stores the results of image analyses keyed by a hash of the
    image data, prompt, and model. Entries automatically expire after the
    configured TTL.

    Attributes:
        ttl: Time-to-live in seconds for cache entries
        maxsize: Maximum number of entries in the cache
        _cache: Internal cache storage (OrderedDict for LRU ordering)
        _lock: Thread lock for concurrent access

    Example:
        >>> cache = AnalysisCache(ttl=3600)  # 1 hour TTL
        >>> key = cache.get_key("base64imagedata", "Describe this", "llava-phi3")
        >>> cache.set(key, "A beautiful sunset...")
        >>> result = cache.get(key)
        >>> print(result)
        "A beautiful sunset..."
    """

    def __init__(self, ttl: int = 3600, maxsize: int = 1000) -> None:
        """
        Initialize the cache.

        Args:
            ttl: Time-to-live in seconds (default: 3600 = 1 hour)
            maxsize: Maximum number of cache entries (default: 1000)

        Raises:
            ValueError: If ttl or maxsize is negative
        """
        if ttl < 0:
            raise ValueError("TTL must be non-negative")
        if maxsize < 0:
            raise ValueError("Maxsize must be non-negative")

        self.ttl = ttl
        self.maxsize = maxsize
        self._cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._lock = Lock()

        # Statistics
        self._hits = 0
        self._misses = 0

    @staticmethod
    def get_key(image_data: str, prompt: str, model: str) -> str:
        """
        Generate a cache key from image data, prompt, and model.

        The key is a SHA256 hash of the concatenated components, ensuring
        that identical analyses produce the same key.

        Args:
            image_data: Base64-encoded image data
            prompt: The analysis prompt
            model: The model name

        Returns:
            SHA256 hash string as the cache key
        """
        components = f"{image_data}:{prompt}:{model}"
        return hashlib.sha256(components.encode()).hexdigest()

    def get(self, key: str) -> Optional[str]:
        """
        Retrieve a cached result if it exists and hasn't expired.

        Args:
            key: The cache key (from get_key())

        Returns:
            The cached result if found and valid, None otherwise

        Note:
            This method is thread-safe and automatically removes expired entries.
        """
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None

            entry = self._cache[key]
            current_time = time.time()

            # Check if entry has expired
            if current_time - entry["timestamp"] > self.ttl:
                # Remove expired entry
                del self._cache[key]
                self._misses += 1
                return None

            # Move to end for LRU (most recently used)
            self._cache.move_to_end(key)
            self._hits += 1
            return entry["value"]

    def set(self, key: str, value: str) -> None:
        """
        Store a result in the cache.

        If the cache is full, the least recently used entry is evicted.

        Args:
            key: The cache key (from get_key())
            value: The analysis result to cache

        Raises:
            CacheError: If there's an error storing the result
        """
        if not key:
            raise CacheError("Cache key cannot be empty", operation="set")

        try:
            with self._lock:
                current_time = time.time()

                # If key exists, update and move to end
                if key in self._cache:
                    self._cache[key] = {"value": value, "timestamp": current_time}
                    self._cache.move_to_end(key)
                    return

                # Evict oldest entry if at capacity
                if self.maxsize > 0 and len(self._cache) >= self.maxsize:
                    self._cache.popitem(last=False)  # Remove oldest (FIFO)

                # Add new entry
                self._cache[key] = {"value": value, "timestamp": current_time}
        except Exception as e:
            raise CacheError(
                f"Failed to cache result: {str(e)}", operation="set", reason=str(e)
            )

    def delete(self, key: str) -> bool:
        """
        Remove an entry from the cache.

        Args:
            key: The cache key to remove

        Returns:
            True if entry was removed, False if it didn't exist
        """
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    def clear(self) -> None:
        """Remove all entries from the cache."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0

    def stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with:
                - size: Current number of entries
                - hits: Number of cache hits
                - misses: Number of cache misses
                - hit_rate: Percentage of requests that were hits
                - ttl: Cache TTL in seconds
                - maxsize: Maximum cache size
        """
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = (self._hits / total_requests * 100) if total_requests > 0 else 0

            return {
                "size": len(self._cache),
                "maxsize": self.maxsize,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": f"{hit_rate:.2f}%",
                "ttl": self.ttl,
            }

    def prune_expired(self) -> int:
        """
        Remove all expired entries from the cache.

        Returns:
            Number of entries removed
        """
        current_time = time.time()
        removed = 0

        with self._lock:
            expired_keys = [
                key
                for key, entry in self._cache.items()
                if current_time - entry["timestamp"] > self.ttl
            ]
            for key in expired_keys:
                del self._cache[key]
                removed += 1

        return removed

    def __len__(self) -> int:
        """Return current number of entries in cache."""
        return len(self._cache)

    def __contains__(self, key: str) -> bool:
        """Check if key exists in cache (and hasn't expired)."""
        return self.get(key) is not None

    def __repr__(self) -> str:
        """Return detailed representation for debugging."""
        return (
            f"{self.__class__.__name__}("
            f"ttl={self.ttl}, "
            f"maxsize={self.maxsize}, "
            f"entries={len(self._cache)})"
        )
