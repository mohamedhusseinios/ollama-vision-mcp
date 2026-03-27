"""
Retry logic module for Ollama Vision MCP Server

This module provides retry utilities with exponential backoff and jitter
for handling transient failures in API calls and other operations.
"""

from __future__ import annotations

import asyncio
import logging
import random
from functools import wraps
from typing import Any, Callable, Set, TypeVar

from .exceptions import OllamaAPIError, TimeoutError

logger = logging.getLogger(__name__)

T = TypeVar("T")

# HTTP status codes that should be retried
RETRYABLE_STATUS_CODES: Set[int] = {
    408,  # Request Timeout
    429,  # Too Many Requests
    500,  # Internal Server Error
    502,  # Bad Gateway
    503,  # Service Unavailable
    504,  # Gateway Timeout
}


def get_status_code(error: Exception) -> int | None:
    """
    Extract HTTP status code from an exception if available.

    Args:
        error: The exception to extract status code from

    Returns:
        HTTP status code if available, None otherwise
    """
    if isinstance(error, OllamaAPIError):
        return error.status_code
    return getattr(error, "status_code", None)


def is_retryable(error: Exception) -> bool:
    """
    Determine if an error should be retried.

    Args:
        error: The exception to check

    Returns:
        True if the operation should be retried, False otherwise
    """
    status_code = get_status_code(error)
    if status_code and status_code in RETRYABLE_STATUS_CODES:
        return True

    # Check for timeout errors
    if isinstance(error, TimeoutError):
        return True

    # Check for connection errors
    error_name = type(error).__name__
    if error_name in (
        "ClientError",
        "ClientConnectionError",
        "ServerDisconnectedError",
    ):
        return True

    return False


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    retryable_exceptions: set[type] | None = None,
):
    """
    Decorator that retries a function with exponential backoff.

    This decorator will retry the decorated function on specified exceptions
    with exponentially increasing delays between retries.

    Args:
        max_retries: Maximum number of retry attempts (default: 3)
        base_delay: Base delay in seconds (default: 1.0)
        max_delay: Maximum delay in seconds (default: 60.0)
        exponential_base: Base for exponential calculation (default: 2.0)
        jitter: Add random jitter to delay (default: True)
        retryable_exceptions: Set of exception types to retry (default: auto-detect)

    Returns:
        Decorated function that retries on failure

    Example:
        >>> @retry_with_backoff(max_retries=3, base_delay=1.0)
        ... async def fetch_data():
        ...     # Make API call
        ...     return response

        >>> # Or with custom exceptions:
        >>> @retry_with_backoff(retryable_exceptions={ConnectionError, TimeoutError})
        ... async def connect():
        ...     # Connection logic
        ...     pass
    """

    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any):
            last_exception: Exception | None = None

            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)

                except Exception as e:
                    last_exception = e

                    # Check if we should retry
                    if retryable_exceptions:
                        should_retry = isinstance(e, tuple(retryable_exceptions))
                    else:
                        should_retry = is_retryable(e)

                    if not should_retry or attempt >= max_retries:
                        logger.error(
                            f"Operation failed (attempt {attempt + 1}/{max_retries + 1}): {e}"
                        )
                        raise

                    # Calculate delay with exponential backoff
                    delay = min(base_delay * (exponential_base**attempt), max_delay)

                    # Add jitter (randomness) to avoid thundering herd
                    if jitter:
                        delay = delay * (1 + random.uniform(-0.1, 0.1))

                    logger.warning(
                        f"Operation failed (attempt {attempt + 1}/{max_retries + 1}): {e}. "
                        f"Retrying in {delay:.2f} seconds..."
                    )

                    await asyncio.sleep(delay)

            # This should never be reached, but mypy needs it
            raise last_exception if last_exception else RuntimeError("Unexpected state")

        @wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any):
            """Sync wrapper - not supported, raises NotImplementedError."""
            raise NotImplementedError(
                "retry_with_backoff only supports async functions. "
                "Use the async version of your function."
            )

        # Return the appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            # For non-async functions, we'll use the async version anyway
            # since this module is designed for async operations
            return async_wrapper

    return decorator


class RetryContext:
    """
    Context manager for retry operations with state tracking.

    This class provides context for tracking retry attempts and managing
    retry state across multiple operations.

    Example:
        >>> async def operation():
        ...     ctx = RetryContext(max_retries=3)
        ...     async with ctx as retry_ctx:
        ...         result = await risky_operation()
        ...         return result
    """

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
    ) -> None:
        """
        Initialize retry context.

        Args:
            max_retries: Maximum retry attempts
            base_delay: Base delay in seconds
            max_delay: Maximum delay in seconds
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.attempt = 0
        self.last_error: Exception | None = None

    async def __aenter__(self) -> "RetryContext":
        """Enter async context."""
        return self

    async def __aexit__(
        self,
        exc_type: type | None,
        exc_val: Exception | None,
        exc_tb: Any,
    ) -> bool:
        """Handle exception in async context."""
        if exc_val is not None:
            self.last_error = exc_val

            if is_retryable(exc_val) and self.attempt < self.max_retries:
                # Calculate delay
                delay = min(self.base_delay * (2**self.attempt), self.max_delay)

                logger.warning(
                    f"Retry {self.attempt + 1}/{self.max_retries} after {delay:.2f}s: {exc_val}"
                )

                await asyncio.sleep(delay)
                self.attempt += 1
                return True  # Suppress exception and retry

        return False  # Let exception propagate

    def should_retry(self) -> bool:
        """Check if more retries are available."""
        return self.attempt < self.max_retries


async def retry_operation(
    operation: Callable[[], Any],
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    retryable_exceptions: set[type] | None = None,
):
    """
    Execute an operation with retry logic.

    This is a convenience function for retrying operations without using decorators.

    Args:
        operation: Async function to execute
        max_retries: Maximum retry attempts
        base_delay: Base delay in seconds
        max_delay: Maximum delay in seconds
        retryable_exceptions: Set of exception types to retry

    Returns:
        Result of the operation

    Raises:
        Exception: The last exception after all retries exhausted

    Example:
        >>> result = await retry_operation(
        ...     operation=lambda: api_call(),
        ...     max_retries=3,
        ...     base_delay=1.0,
        ... )
    """
    last_exception: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            result = await operation()
            return result

        except Exception as e:
            last_exception = e

            # Check if we should retry
            if retryable_exceptions:
                should_retry = isinstance(e, tuple(retryable_exceptions))
            else:
                should_retry = is_retryable(e)

            if not should_retry or attempt >= max_retries:
                raise

            # Calculate delay
            delay = min(base_delay * (2**attempt), max_delay)
            jitter = delay * random.uniform(0.1, 0.3)
            await asyncio.sleep(delay + jitter)

    raise last_exception if last_exception else RuntimeError("Unexpected state")
