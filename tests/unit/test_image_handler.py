"""
Unit tests for ImageHandler
"""

import pytest
from pathlib import Path


@pytest.mark.asyncio
class TestImageHandler:
    """Test suite for ImageHandler class"""

    async def test_process_local_image_file(self, temp_image_file):
        """Test processing a local image file"""
        from src.image_handler import ImageHandler

        handler = ImageHandler()
        result = await handler.process_image(str(temp_image_file))

        assert isinstance(result, str)
        assert len(result) > 0

    async def test_process_small_image(self, temp_image_file_small):
        """Test processing a small image"""
        from src.image_handler import ImageHandler

        handler = ImageHandler()
        result = await handler.process_image(str(temp_image_file_small))

        assert isinstance(result, str)

    async def test_process_large_image(self, temp_image_file_large):
        """Test processing a large image (should resize)"""
        from src.image_handler import ImageHandler

        handler = ImageHandler()
        result = await handler.process_image(str(temp_image_file_large))

        assert isinstance(result, str)

    async def test_invalid_file_path_raises_error(self):
        """Test that invalid file path raises appropriate error"""
        from src.image_handler import ImageHandler

        handler = ImageHandler()

        with pytest.raises(FileNotFoundError):
            await handler.process_image("/nonexistent/image.png")

    async def test_mock_image_handler(self, mock_image_handler, sample_image_data):
        """Test using mock image handler"""
        result = await mock_image_handler.process_image("any_path.png")

        assert isinstance(result, str)
        assert len(result) > 0


class TestImageFileFixtures:
    """Test that image file fixtures work correctly"""

    def test_temp_image_file_is_png(self, temp_image_file):
        """Verify temp image file is PNG format"""
        assert temp_image_file.suffix == ".png"

    def test_temp_image_file_small_dimensions(self, temp_image_file_small):
        """Verify small image has correct dimensions"""
        from PIL import Image

        img = Image.open(temp_image_file_small)
        assert img.size == (50, 50)

    def test_temp_image_file_large_dimensions(self, temp_image_file_large):
        """Verify large image has correct dimensions"""
        from PIL import Image

        img = Image.open(temp_image_file_large)
        assert img.size == (2000, 2000)
