"""
Unit tests for OllamaClient
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.mark.asyncio
class TestOllamaClient:
    """Test suite for OllamaClient class"""

    async def test_check_connection_success(self, mock_ollama_client):
        """Test successful connection check"""
        result = await mock_ollama_client.check_connection()
        assert result is True

    async def test_list_models_returns_vision_models(self, mock_ollama_client):
        """Test listing vision models"""
        models = await mock_ollama_client.list_models()

        assert isinstance(models, list)
        assert len(models) > 0
        assert "llava-phi3" in models

    async def test_analyze_image_default(self, mock_ollama_client, sample_image_data):
        """Test image analysis with default prompt"""
        result = await mock_ollama_client.analyze_image(
            sample_image_data, "Describe this image"
        )

        assert isinstance(result, str)
        assert len(result) > 0

    async def test_analyze_image_with_model(
        self, mock_ollama_client, sample_image_data
    ):
        """Test image analysis with specific model"""
        result = await mock_ollama_client.analyze_image(
            sample_image_data, "What objects are in this image?", model="llava:7b"
        )

        assert isinstance(result, str)

    async def test_ensure_model_success(self, mock_ollama_client):
        """Test ensuring model availability"""
        result = await mock_ollama_client.ensure_model("llava-phi3")
        assert result is True


class TestMockFixtures:
    """Test that mock fixtures work correctly"""

    def test_sample_image_data_is_valid_base64(self, sample_image_data):
        """Verify sample_image_data is valid base64"""
        import base64

        # Should not raise an exception
        decoded = base64.b64decode(sample_image_data)
        assert len(decoded) > 0

    def test_temp_image_file_exists(self, temp_image_file):
        """Verify temp_image_file creates a valid file"""
        assert temp_image_file.exists()
        assert temp_image_file.suffix == ".png"

    def test_mock_config_has_required_attributes(self, mock_config):
        """Verify mock_config has all required attributes"""
        assert hasattr(mock_config, "ollama_url")
        assert hasattr(mock_config, "default_model")
        assert hasattr(mock_config, "timeout")
        assert mock_config.default_model == "llava-phi3"

    async def test_mock_ollama_client_is_async(self, mock_ollama_client):
        """Verify mock_ollama_client methods are async"""
        import inspect

        assert inspect.iscoroutinefunction(mock_ollama_client.check_connection)
        assert inspect.iscoroutinefunction(mock_ollama_client.list_models)
        assert inspect.iscoroutinefunction(mock_ollama_client.analyze_image)


@pytest.mark.asyncio
class TestErrorScenarios:
    """Test error handling scenarios"""

    async def test_connection_failure(self, mock_ollama_client_error):
        """Test handling of connection failures"""
        result = await mock_ollama_client_error.check_connection()
        assert result is False

    async def test_empty_models_list(self, mock_ollama_client_error):
        """Test handling of no available models"""
        models = await mock_ollama_client_error.list_models()
        assert models == []

    async def test_timeout_error(self, mock_ollama_client_timeout, sample_image_data):
        """Test handling of timeout errors"""
        import asyncio

        with pytest.raises(asyncio.TimeoutError):
            await mock_ollama_client_timeout.analyze_image(
                sample_image_data, "Describe this image"
            )
