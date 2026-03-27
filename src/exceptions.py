"""
Custom exceptions for Ollama Vision MCP Server

This module provides a custom exception hierarchy for better error handling
throughout the application. All custom exceptions inherit from OllamaVisionError
as the base class.
"""

from __future__ import annotations


class OllamaVisionError(Exception):
    """
    Base exception for all Ollama Vision MCP errors.

    All custom exceptions in this application should inherit from this class
    to provide a consistent error handling interface.

    Attributes:
        message: Human-readable error description

    Example:
        >>> raise OllamaVisionError("Something went wrong")
        OllamaVisionError: Something went wrong
    """

    def __init__(self, message: str) -> None:
        """
        Initialize the base exception.

        Args:
            message: Human-readable error description
        """
        self.message = message
        super().__init__(self.message)

    def __str__(self) -> str:
        """Return formatted error message for string conversion."""
        return self.message

    def __repr__(self) -> str:
        """Return detailed representation for debugging."""
        return f"{self.__class__.__name__}(message={self.message!r})"


class OllamaAPIError(OllamaVisionError):
    """
    Exception raised when the Ollama API request fails.

    This exception is raised when there are HTTP errors, connection failures,
    or invalid responses from the Ollama API server.

    Attributes:
        message: Human-readable error description
        status_code: HTTP status code returned by the API (if available)
        response_text: Raw response text from the API (if available)

    Example:
        >>> raise OllamaAPIError("API request failed", status_code=500, response_text="Internal Server Error")
        OllamaAPIError: API request failed (status_code=500)
    """

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        response_text: str | None = None,
    ) -> None:
        """
        Initialize the API error exception.

        Args:
            message: Human-readable error description
            status_code: HTTP status code returned by the API (optional)
            response_text: Raw response text from the API (optional)
        """
        self.status_code = status_code
        self.response_text = response_text
        super().__init__(message)

    def __str__(self) -> str:
        """Return formatted error message including status code if available."""
        if self.status_code is not None:
            return f"{self.message} (status_code={self.status_code})"
        return self.message

    def __repr__(self) -> str:
        """Return detailed representation for debugging."""
        return (
            f"{self.__class__.__name__}("
            f"message={self.message!r}, "
            f"status_code={self.status_code}, "
            f"response_text={self.response_text!r})"
        )


class ImageProcessingError(OllamaVisionError):
    """
    Exception raised when image processing fails.

    This exception is raised when there are issues loading, validating,
    or processing images from files, URLs, or base64 data.

    Attributes:
        message: Human-readable error description
        image_path: Path to the problematic image file (if applicable)
        reason: Specific reason for the failure

    Example:
        >>> raise ImageProcessingError("Failed to load image", image_path="/tmp/photo.jpg", reason="File not found")
        ImageProcessingError: Failed to load image: File not found (image_path=/tmp/photo.jpg)
    """

    def __init__(
        self, message: str, image_path: str | None = None, reason: str | None = None
    ) -> None:
        """
        Initialize the image processing error exception.

        Args:
            message: Human-readable error description
            image_path: Path to the problematic image file (optional)
            reason: Specific reason for the failure (optional)
        """
        self.image_path = image_path
        self.reason = reason
        super().__init__(message)

    def __str__(self) -> str:
        """Return formatted error message including image path and reason."""
        parts = [self.message]
        if self.reason:
            parts.append(f"reason={self.reason}")
        if self.image_path:
            parts.append(f"image_path={self.image_path}")

        if len(parts) > 1:
            return f"{parts[0]}: {' | '.join(parts[1:])}"
        return self.message

    def __repr__(self) -> str:
        """Return detailed representation for debugging."""
        return (
            f"{self.__class__.__name__}("
            f"message={self.message!r}, "
            f"image_path={self.image_path!r}, "
            f"reason={self.reason!r})"
        )


class ModelNotFoundError(OllamaVisionError):
    """
    Exception raised when a requested model is not available.

    This exception is raised when trying to use a vision model that
    is not installed or available in the Ollama server.

    Attributes:
        message: Human-readable error description
        model_name: Name of the requested model
        available_models: List of available models that can be used instead

    Example:
        >>> raise ModelNotFoundError("Model not found", model_name="llava:13b", available_models=["llava-phi3", "llava:7b"])
        ModelNotFoundError: Model 'llava:13b' not found. Available models: llava-phi3, llava:7b
    """

    def __init__(
        self,
        message: str,
        model_name: str | None = None,
        available_models: list[str] | None = None,
    ) -> None:
        """
        Initialize the model not found error exception.

        Args:
            message: Human-readable error description
            model_name: Name of the requested model (optional)
            available_models: List of available models (optional)
        """
        self.model_name = model_name
        self.available_models = available_models if available_models is not None else []
        super().__init__(message)

    def __str__(self) -> str:
        """Return formatted error message with model suggestions."""
        if self.model_name and self.available_models:
            models_str = ", ".join(self.available_models)
            return (
                f"Model '{self.model_name}' not found. Available models: {models_str}"
            )
        elif self.model_name:
            return f"Model '{self.model_name}' not found. No models available."
        return self.message

    def __repr__(self) -> str:
        """Return detailed representation for debugging."""
        return (
            f"{self.__class__.__name__}("
            f"message={self.message!r}, "
            f"model_name={self.model_name!r}, "
            f"available_models={self.available_models!r})"
        )


class TimeoutError(OllamaVisionError):
    """
    Exception raised when a request times out.

    This exception is raised when an operation (API request, image download,
    etc.) exceeds the configured timeout duration.

    Attributes:
        message: Human-readable error description
        timeout_seconds: The timeout duration that was exceeded
        operation: Description of the operation that timed out

    Example:
        >>> raise TimeoutError("Request timed out", timeout_seconds=120, operation="image analysis")
        TimeoutError: image analysis timed out after 120 seconds
    """

    def __init__(
        self,
        message: str,
        timeout_seconds: float | None = None,
        operation: str | None = None,
    ) -> None:
        """
        Initialize the timeout error exception.

        Args:
            message: Human-readable error description
            timeout_seconds: The timeout duration that was exceeded (optional)
            operation: Description of the operation that timed out (optional)
        """
        self.timeout_seconds = timeout_seconds
        self.operation = operation
        super().__init__(message)

    def __str__(self) -> str:
        """Return formatted error message with timeout details."""
        if self.operation and self.timeout_seconds is not None:
            return f"{self.operation} timed out after {self.timeout_seconds} seconds"
        elif self.operation:
            return f"{self.operation} timed out"
        elif self.timeout_seconds is not None:
            return f"Operation timed out after {self.timeout_seconds} seconds"
        return self.message

    def __repr__(self) -> str:
        """Return detailed representation for debugging."""
        return (
            f"{self.__class__.__name__}("
            f"message={self.message!r}, "
            f"timeout_seconds={self.timeout_seconds}, "
            f"operation={self.operation!r})"
        )


class CacheError(OllamaVisionError):
    """
    Exception raised when caching operations fail.

    This exception is raised when there are issues with the caching system,
    such as cache read/write failures, cache corruption, or invalid cache states.

    Attributes:
        message: Human-readable error description
        operation: The cache operation that failed (e.g., 'read', 'write', 'invalidate')
        reason: Specific reason for the failure

    Example:
        >>> raise CacheError("Cache write failed", operation="write", reason="Disk full")
        CacheError: Cache write failed: Disk full (operation=write)
    """

    def __init__(
        self, message: str, operation: str | None = None, reason: str | None = None
    ) -> None:
        """
        Initialize the cache error exception.

        Args:
            message: Human-readable error description
            operation: The cache operation that failed (optional)
            reason: Specific reason for the failure (optional)
        """
        self.operation = operation
        self.reason = reason
        super().__init__(message)

    def __str__(self) -> str:
        """Return formatted error message with operation and reason."""
        parts = [self.message]
        if self.reason:
            parts.append(f"reason={self.reason}")
        if self.operation:
            parts.append(f"operation={self.operation}")

        if len(parts) > 1:
            return f"{parts[0]}: {' | '.join(parts[1:])}"
        return self.message

    def __repr__(self) -> str:
        """Return detailed representation for debugging."""
        return (
            f"{self.__class__.__name__}("
            f"message={self.message!r}, "
            f"operation={self.operation!r}, "
            f"reason={self.reason!r})"
        )
