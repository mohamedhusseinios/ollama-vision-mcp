"""
Pytest configuration and fixtures for Ollama Vision MCP tests
"""

import asyncio
import base64
import tempfile
from pathlib import Path
from typing import Dict, Any, AsyncGenerator, Generator, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from PIL import Image, ImageDraw

from src.config import Config
from src.ollama_client import OllamaClient
from src.image_handler import ImageHandler


# Configure pytest-asyncio
pytest_plugins = ("pytest_asyncio",)


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create an instance of the event loop for session-scoped tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def sample_image_data() -> str:
    """
    Provide a valid base64 encoded test image.
    Creates a simple 200x100 pixel image with text.

    Returns:
        Base64 encoded string of a PNG image
    """
    # Create a simple test image in memory
    img = Image.new("RGB", (200, 100), color="white")
    draw = ImageDraw.Draw(img)
    draw.text((10, 40), "Test Image", fill="black")

    # Convert to base64
    import io

    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format="PNG")
    img_bytes = img_byte_arr.getvalue()

    return base64.b64encode(img_bytes).decode("utf-8")


@pytest.fixture
def temp_image_file(tmp_path: Path) -> Path:
    """
    Create a temporary image file for testing.

    Args:
        tmp_path: Pytest's temporary path fixture

    Returns:
        Path to the temporary image file
    """
    # Create a simple test image
    img = Image.new("RGB", (200, 100), color="blue")
    draw = ImageDraw.Draw(img)
    draw.text((10, 40), "Temp Test Image", fill="white")

    # Save to temporary file
    temp_file = tmp_path / "test_image.png"
    img.save(temp_file, format="PNG")

    return temp_file


@pytest.fixture
def temp_image_file_small(tmp_path: Path) -> Path:
    """
    Create a small temporary image file for testing.

    Args:
        tmp_path: Pytest's temporary path fixture

    Returns:
        Path to the small temporary image file
    """
    # Create a small test image (50x50 pixels)
    img = Image.new("RGB", (50, 50), color="green")

    # Save to temporary file
    temp_file = tmp_path / "small_test_image.png"
    img.save(temp_file, format="PNG")

    return temp_file


@pytest.fixture
def temp_image_file_large(tmp_path: Path) -> Path:
    """
    Create a large temporary image file for testing resize behavior.

    Args:
        tmp_path: Pytest's temporary path fixture

    Returns:
        Path to the large temporary image file
    """
    # Create a large test image (2000x2000 pixels)
    img = Image.new("RGB", (2000, 2000), color="red")
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, 2000, 2000], outline="black", width=5)

    # Save to temporary file
    temp_file = tmp_path / "large_test_image.png"
    img.save(temp_file, format="PNG")

    return temp_file


@pytest.fixture
def mock_config() -> Config:
    """
    Provide a mock configuration for testing.

    Returns:
        Config instance with test defaults
    """
    config = MagicMock(spec=Config)
    config.ollama_url = "http://localhost:11434"
    config.default_model = "llava-phi3"
    config.timeout = 120
    config.log_level = "INFO"
    config.cache_enabled = False
    config.cache_ttl = 3600
    config.model_preferences = ["llava-phi3", "llava:7b", "llava:13b", "bakllava"]
    return config


@pytest.fixture
def mock_ollama_client(mock_config: Config) -> AsyncMock:
    """
    Provide a mock Ollama client that doesn't require a real Ollama instance.

    Args:
        mock_config: Mock configuration fixture

    Returns:
        AsyncMock configured to simulate Ollama API responses
    """
    client = AsyncMock(spec=OllamaClient)

    # Import mock responses
    from tests.mocks.ollama_responses import (
        MOCK_MODELS_RESPONSE,
        MOCK_ANALYZE_RESPONSE,
        MOCK_ERROR_RESPONSES,
    )

    # Configure mock methods
    async def mock_check_connection() -> bool:
        """Simulate successful connection check."""
        return True

    async def mock_list_models() -> list:
        """Return mock list of available models."""
        return [model["name"] for model in MOCK_MODELS_RESPONSE["models"]]

    async def mock_analyze_image(
        image_data: str, prompt: str, model: Optional[str] = None
    ) -> str:
        """
        Simulate image analysis response.

        Args:
            image_data: Base64 encoded image data
            prompt: Analysis prompt
            model: Optional model name

        Returns:
            Mock analysis result
        """
        # Return different responses based on prompt type
        if "describe" in prompt.lower():
            return MOCK_ANALYZE_RESPONSE["describe"]
        elif "objects" in prompt.lower():
            return MOCK_ANALYZE_RESPONSE["objects"]
        elif "text" in prompt.lower() or "read" in prompt.lower():
            return MOCK_ANALYZE_RESPONSE["text"]
        else:
            return MOCK_ANALYZE_RESPONSE["default"]

    async def mock_ensure_model(model: str) -> bool:
        """Simulate ensuring model is available."""
        return True

    # Assign mock methods
    client.check_connection.side_effect = mock_check_connection
    client.list_models.side_effect = mock_list_models
    client.analyze_image.side_effect = mock_analyze_image
    client.ensure_model.side_effect = mock_ensure_model
    client.config = mock_config

    return client


@pytest.fixture
def mock_ollama_client_error(mock_config: Config) -> AsyncMock:
    """
    Provide a mock Ollama client that simulates errors.

    Args:
        mock_config: Mock configuration fixture

    Returns:
        AsyncMock configured to simulate error conditions
    """
    from tests.mocks.ollama_responses import MOCK_ERROR_RESPONSES

    client = AsyncMock(spec=OllamaClient)

    # Configure error methods
    async def mock_check_connection_error() -> bool:
        """Simulate failed connection check."""
        return False

    async def mock_list_models_empty() -> list:
        """Return empty list of models."""
        return []

    async def mock_analyze_image_timeout(*args, **kwargs):
        """Simulate timeout error."""
        raise asyncio.TimeoutError("Request timed out")

    async def mock_analyze_image_connection_error(*args, **kwargs):
        """Simulate connection error."""
        raise Exception(MOCK_ERROR_RESPONSES["connection_error"])

    # Assign error methods
    client.check_connection.side_effect = mock_check_connection_error
    client.list_models.side_effect = mock_list_models_empty
    client.analyze_image.side_effect = mock_analyze_image_connection_error
    client.ensure_model.side_effect = lambda model: False
    client.config = mock_config

    return client


@pytest.fixture
def mock_ollama_client_timeout(mock_config: Config) -> AsyncMock:
    """
    Provide a mock Ollama client that simulates timeout errors.

    Args:
        mock_config: Mock configuration fixture

    Returns:
        AsyncMock configured to simulate timeout errors
    """
    client = AsyncMock(spec=OllamaClient)

    async def mock_check_connection() -> bool:
        return True

    async def mock_list_models() -> list:
        return ["llava-phi3"]

    async def mock_analyze_image_timeout(*args, **kwargs):
        raise asyncio.TimeoutError("Request timed out after 120 seconds")

    client.check_connection.side_effect = mock_check_connection
    client.list_models.side_effect = mock_list_models
    client.analyze_image.side_effect = mock_analyze_image_timeout
    client.config = mock_config

    return client


@pytest.fixture
def mock_image_handler() -> MagicMock:
    """
    Provide a mock image handler for testing.

    Returns:
        MagicMock configured for image processing simulation
    """
    handler = MagicMock(spec=ImageHandler)

    async def mock_process_image(image_path: str) -> str:
        """
        Simulate image processing.

        Args:
            image_path: Path to image file or URL

        Returns:
            Mock base64 encoded image data
        """
        # Return a valid base64 string
        sample_bytes = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde"
        return base64.b64encode(sample_bytes).decode("utf-8")

    handler.process_image.side_effect = mock_process_image
    handler.validate_image_path.return_value = True

    return handler


@pytest.fixture
def temp_config_file(tmp_path: Path) -> Path:
    """
    Create a temporary configuration file for testing.

    Args:
        tmp_path: Pytest's temporary path fixture

    Returns:
        Path to the temporary config file
    """
    import json

    config_data = {
        "ollama_url": "http://localhost:11434",
        "default_model": "llava-phi3",
        "timeout": 120,
        "log_level": "INFO",
        "cache_enabled": False,
        "cache_ttl": 3600,
        "model_preferences": ["llava-phi3", "llava:7b", "llava:13b", "bakllava"],
    }

    config_file = tmp_path / "test_config.json"
    with open(config_file, "w") as f:
        json.dump(config_data, f, indent=2)

    return config_file


@pytest.fixture
def sample_url_image() -> str:
    """
    Provide a sample image URL for testing URL handling.

    Returns:
        A sample image URL string
    """
    return "https://example.com/test_image.png"


@pytest.fixture
def sample_base64_image() -> str:
    """
    Provide a sample base64 image string for testing.

    Returns:
        A sample base64 encoded image string
    """
    # Create a 1x1 pixel image
    img = Image.new("RGB", (1, 1), color="red")

    import io

    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format="PNG")
    img_bytes = img_byte_arr.getvalue()

    return base64.b64encode(img_bytes).decode("utf-8")
