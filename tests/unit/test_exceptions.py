"""Unit tests for the exceptions module."""

import pytest

from src.exceptions import (
    OllamaVisionError,
    OllamaAPIError,
    ImageProcessingError,
    ModelNotFoundError,
    TimeoutError,
    CacheError,
)


class TestOllamaVisionError:
    """Tests for base OllamaVisionError."""

    def test_init_with_message(self):
        """Test initialization with message."""
        error = OllamaVisionError("Test error")

        assert error.message == "Test error"
        assert str(error) == "Test error"

    def test_repr(self):
        """Test string representation."""
        error = OllamaVisionError("Test error")

        assert "OllamaVisionError" in repr(error)
        assert "Test error" in repr(error)


class TestOllamaAPIError:
    """Tests for OllamaAPIError."""

    def test_init_with_status_code(self):
        """Test initialization with status code."""
        error = OllamaAPIError("API failed", status_code=500)

        assert error.message == "API failed"
        assert error.status_code == 500
        assert error.response_text is None

    def test_init_with_response_text(self):
        """Test initialization with response text."""
        error = OllamaAPIError(
            "API failed", status_code=500, response_text="Internal Server Error"
        )

        assert error.status_code == 500
        assert error.response_text == "Internal Server Error"

    def test_str_with_status_code(self):
        """Test string conversion includes status code."""
        error = OllamaAPIError("API failed", status_code=500)

        assert "500" in str(error)

    def test_str_without_status_code(self):
        """Test string conversion without status code."""
        error = OllamaAPIError("API failed")

        assert str(error) == "API failed"

    def test_repr_includes_all_fields(self):
        """Test repr includes all fields."""
        error = OllamaAPIError("API failed", status_code=500, response_text="Error")

        assert "status_code" in repr(error)
        assert "response_text" in repr(error)


class TestImageProcessingError:
    """Tests for ImageProcessingError."""

    def test_init_with_path_and_reason(self):
        """Test initialization with image path and reason."""
        error = ImageProcessingError(
            "Failed to process", image_path="/tmp/image.png", reason="File not found"
        )

        assert error.image_path == "/tmp/image.png"
        assert error.reason == "File not found"

    def test_str_includes_reason(self):
        """Test string includes reason."""
        error = ImageProcessingError("Failed", reason="Invalid format")

        assert "Invalid format" in str(error)

    def test_str_includes_path(self):
        """Test string includes path."""
        error = ImageProcessingError("Failed", image_path="/tmp/image.png")

        assert "/tmp/image.png" in str(error)

    def test_str_with_all_fields(self):
        """Test string with all fields."""
        error = ImageProcessingError(
            "Failed", image_path="/tmp/image.png", reason="Invalid format"
        )

        str_repr = str(error)
        assert "Failed" in str_repr
        assert "Invalid format" in str_repr
        assert "/tmp/image.png" in str_repr


class TestModelNotFoundError:
    """Tests for ModelNotFoundError."""

    def test_init_with_model_name(self):
        """Test initialization with model name."""
        error = ModelNotFoundError("Not found", model_name="llava:13b")

        assert error.model_name == "llava:13b"
        assert error.available_models == []

    def test_init_with_available_models(self):
        """Test initialization with available models."""
        error = ModelNotFoundError(
            "Not found",
            model_name="llava:13b",
            available_models=["llava-phi3", "llava:7b"],
        )

        assert error.available_models == ["llava-phi3", "llava:7b"]

    def test_str_with_suggestions(self):
        """Test string includes suggestions."""
        error = ModelNotFoundError(
            "Not found",
            model_name="llava:13b",
            available_models=["llava-phi3", "llava:7b"],
        )

        assert "llava:13b" in str(error)
        assert "llava-phi3" in str(error)

    def test_str_without_models(self):
        """Test string without models."""
        error = ModelNotFoundError("Not found", model_name="llava:13b")

        assert "llava:13b" in str(error)
        assert "No models available" in str(error)


class TestTimeoutError:
    """Tests for TimeoutError."""

    def test_init_with_timeout_and_operation(self):
        """Test initialization with timeout and operation."""
        error = TimeoutError(
            "Timed out", timeout_seconds=120, operation="image analysis"
        )

        assert error.timeout_seconds == 120
        assert error.operation == "image analysis"

    def test_str_with_all_fields(self):
        """Test string includes all fields."""
        error = TimeoutError(
            "Timed out", timeout_seconds=120, operation="image analysis"
        )

        assert "image analysis" in str(error)
        assert "120" in str(error)

    def test_str_with_operation_only(self):
        """Test string with operation only."""
        error = TimeoutError("Timed out", operation="image analysis")

        assert "image analysis" in str(error)
        assert "timed out" in str(error)


class TestCacheError:
    """Tests for CacheError."""

    def test_init_with_operation_and_reason(self):
        """Test initialization with operation and reason."""
        error = CacheError("Cache failed", operation="write", reason="Disk full")

        assert error.operation == "write"
        assert error.reason == "Disk full"

    def test_str_includes_fields(self):
        """Test string includes fields."""
        error = CacheError("Failed", operation="write", reason="Disk full")

        assert "Disk full" in str(error)
        assert "write" in str(error)


class TestExceptionHierarchy:
    """Tests for exception hierarchy."""

    def test_all_exceptions_inherit_from_base(self):
        """Test all exceptions inherit from OllamaVisionError."""
        assert issubclass(OllamaAPIError, OllamaVisionError)
        assert issubclass(ImageProcessingError, OllamaVisionError)
        assert issubclass(ModelNotFoundError, OllamaVisionError)
        assert issubclass(TimeoutError, OllamaVisionError)
        assert issubclass(CacheError, OllamaVisionError)

    def test_catch_all_vision_errors(self):
        """Test catching all errors with base exception."""
        errors = [
            OllamaAPIError("API error"),
            ImageProcessingError("Image error"),
            ModelNotFoundError("Model error"),
            TimeoutError("Timeout error"),
            CacheError("Cache error"),
        ]

        for error in errors:
            with pytest.raises(OllamaVisionError):
                raise error
