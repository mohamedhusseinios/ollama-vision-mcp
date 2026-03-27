"""Unit tests for the retry module."""

import asyncio
import pytest

from src.retry import (
    retry_with_backoff,
    retry_operation,
    RetryContext,
    is_retryable,
    RETRYABLE_STATUS_CODES,
)
from src.exceptions import OllamaAPIError, TimeoutError


class TestIsRetryable:
    """Tests for is_retryable function."""

    def test_retryable_status_codes(self):
        """Test that known retryable status codes return True."""
        for status_code in RETRYABLE_STATUS_CODES:
            error = OllamaAPIError("Error", status_code=status_code)
            assert is_retryable(error) is True

    def test_non_retryable_status_codes(self):
        """Test that non-retryable status codes return False."""
        error = OllamaAPIError("Error", status_code=400)
        assert is_retryable(error) is False

    def test_timeout_error_is_retryable(self):
        """Test that TimeoutError is retryable."""
        error = TimeoutError("Timeout", timeout_seconds=60)
        assert is_retryable(error) is True

    def test_generic_error_not_retryable(self):
        """Test that generic errors are not retryable."""
        error = Exception("Generic error")
        assert is_retryable(error) is False


class TestRetryWithBackoff:
    """Tests for retry_with_backoff decorator."""

    @pytest.mark.asyncio
    async def test_success_on_first_try(self):
        """Test that successful calls don't retry."""
        call_count = 0

        @retry_with_backoff(max_retries=3, base_delay=0.01)
        async def successful_func():
            nonlocal call_count
            call_count += 1
            return "success"

        result = await successful_func()

        assert result == "success"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_retry_on_retryable_error(self):
        """Test retrying on retryable errors."""
        call_count = 0

        @retry_with_backoff(max_retries=3, base_delay=0.01, max_delay=0.05)
        async def flaky_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise OllamaAPIError("Failed", status_code=503)
            return "success"

        result = await flaky_func()

        assert result == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_fail_after_max_retries(self):
        """Test failure after max retries."""
        call_count = 0

        @retry_with_backoff(max_retries=2, base_delay=0.01, max_delay=0.05)
        async def always_failing_func():
            nonlocal call_count
            call_count += 1
            raise OllamaAPIError("Failed", status_code=503)

        with pytest.raises(OllamaAPIError):
            await always_failing_func()

        assert call_count == 3  # Initial + 2 retries

    @pytest.mark.asyncio
    async def test_no_retry_on_non_retryable_error(self):
        """Test no retry on non-retryable errors."""
        call_count = 0

        @retry_with_backoff(max_retries=3, base_delay=0.01)
        async def failing_with_400():
            nonlocal call_count
            call_count += 1
            raise OllamaAPIError("Bad Request", status_code=400)

        with pytest.raises(OllamaAPIError):
            await failing_with_400()

        assert call_count == 1  # No retry

    @pytest.mark.asyncio
    async def test_custom_retryable_exceptions(self):
        """Test custom retryable exceptions."""
        call_count = 0

        @retry_with_backoff(
            max_retries=2,
            base_delay=0.01,
            max_delay=0.05,
            retryable_exceptions={ValueError},
        )
        async def failing_with_value_error():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Custom error")
            return "success"

        result = await failing_with_value_error()

        assert result == "success"
        assert call_count == 3


class TestRetryOperation:
    """Tests for retry_operation function."""

    @pytest.mark.asyncio
    async def test_success_immediately(self):
        """Test success on first attempt."""
        call_count = 0

        async def operation():
            nonlocal call_count
            call_count += 1
            return "result"

        result = await retry_operation(operation, max_retries=3, base_delay=0.01)

        assert result == "result"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_retry_on_failure(self):
        """Test retry on failure."""
        call_count = 0

        async def operation():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise OllamaAPIError("Failed", status_code=500)
            return "success"

        result = await retry_operation(
            operation, max_retries=3, base_delay=0.01, max_delay=0.05
        )

        assert result == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_fail_after_all_retries(self):
        """Test failure after all retries exhausted."""
        call_count = 0

        async def always_fail():
            nonlocal call_count
            call_count += 1
            raise OllamaAPIError("Failed", status_code=500)

        with pytest.raises(OllamaAPIError):
            await retry_operation(
                always_fail, max_retries=2, base_delay=0.01, max_delay=0.05
            )

        assert call_count == 3  # Initial + 2 retries


class TestRetryContext:
    """Tests for RetryContext class."""

    @pytest.mark.asyncio
    async def test_context_success(self):
        """Test context manager on success."""
        call_count = 0

        async def operation():
            nonlocal call_count
            call_count += 1
            return "result"

        ctx = RetryContext(max_retries=3)
        async with ctx:
            result = await operation()

        assert result == "result"
        assert ctx.attempt == 0

    @pytest.mark.asyncio
    async def test_context_retries(self):
        """Test context manager retries on retryable errors."""
        call_count = 0

        ctx = RetryContext(max_retries=3, base_delay=0.01)

        while ctx.attempt <= ctx.max_retries:
            try:
                async with ctx:
                    call_count += 1
                    if call_count < 3:
                        raise OllamaAPIError("Failed", status_code=503)
                    break
            except OllamaAPIError:
                if ctx.attempt >= ctx.max_retries:
                    raise

        assert call_count == 3

    def test_should_retry(self):
        """Test should_retry method."""
        ctx = RetryContext(max_retries=3)

        assert ctx.should_retry() is True

        ctx.attempt = 3
        assert ctx.should_retry() is False
