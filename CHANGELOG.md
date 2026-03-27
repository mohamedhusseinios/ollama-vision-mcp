# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - 2024-01-XX

### Added

#### New Tools
- **`compare_images`**: Compare two images and identify differences/similarities/changes
  - Supports 4 comparison types: `differences`, `similarities`, `sequence`, `quality`
  - Includes progress reporting for batch processing
  
- **`batch_analyze`**: Analyze multiple images with the same prompt
  - Process 5+ images in a single call
  - Includes progress reporting and error handling per image
  
- **`detect_objects_with_boxes`**: Experimental object detection with location descriptions
  - Returns approximate bounding box positions as relative locations
  - Supports category filtering: `people`, `vehicles`, `animals`, `text`, `all`
  - Marked as EXPERIMENTAL due to varying accuracy

#### MCP Resources
- **`models://list`**: List available vision models from Ollama
- **`status://health`**: Server health status with connection check
- **`config://current`**: Current server configuration

#### MCP Prompts
- **`analyze_workflow`**: Guided workflow for comprehensive image analysis
- **`compare_workflow`**: Workflow for comparing two images
- **`batch_workflow`**: Workflow for batch image processing

#### Core Infrastructure
- **`src/cache.py`**: Thread-safe LRU cache with TTL for repeated analyses
  - Configurable cache size (default: 1000 entries)
  - Configurable TTL (default: 3600 seconds)
  - Statistics tracking (hits, misses, hit rate)
  
- **`src/retry.py`**: Exponential backoff retry logic
  - Automatic retry on transient failures (500, 502, 503, 504, 429, 408)
  - Configurable max retries and delays
  - Jitter to prevent thundering herd
  
- **`src/exceptions.py`**: Custom exception hierarchy
  - `OllamaVisionError` (base exception)
  - `OllamaAPIError` (API failures)
  - `ImageProcessingError` (image handling errors)
  - `ModelNotFoundError` (model availability)
  - `TimeoutError` (request timeouts)
  - `CacheError` (caching issues)

#### FastMCP Implementation
- **`src/fastmcp_server.py`**: Modern FastMCP-based server implementation
  - Cleaner tool definitions using decorators
  - Built-in lifespan management
  - Automatic JSON schema generation
  - Progress reporting support via `ctx.report_progress()`
  - Type-safe context access

#### Testing Infrastructure
- **`tests/conftest.py`**: Comprehensive pytest fixtures
  - `mock_ollama_client`: Mock Ollama API responses
  - `sample_image_data`: Base64 encoded test image
  - `temp_image_file`: Temporary image file fixtures
  - Multiple error scenario fixtures
  
- **`tests/mocks/ollama_responses.py`**: Mock API responses
  - Mock model list responses
  - Mock analysis responses
  - Mock error scenarios
  - Mock streaming responses

- **`tests/unit/`**: Unit test suite
  - `test_cache.py`: Cache module tests
  - `test_exceptions.py`: Exception hierarchy tests
  - `test_retry.py`: Retry logic tests

### Changed

- **Python version**: Minimum Python version bumped from 3.8 to 3.10+
- **FastMCP dependency**: Added `fastmcp>=0.2.0` to dependencies
- **Version**: Bumped from 1.0.0 to 2.0.0
- **Export structure**: Added cache, retry, and exception exports to `src/__init__.py`

### Deprecated

- **`src/server.py`**: Legacy MCP SDK implementation
  - Use `src/fastmcp_server.py` for new implementations
  - Maintained for backward compatibility

### Migration Guide

#### From v1.x to v2.0

1. **Update Python version**: Ensure Python 3.10+
   ```bash
   python --version  # Should be 3.10 or higher
   ```

2. **Install new dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Using FastMCP (recommended)**:
   ```python
   # Old way (still works)
   from src.server import OllamaVisionServer
   
   # New way (recommended)
   from src.fastmcp_server import mcp, main
   ```

4. **Access new exceptions**:
   ```python
   from src.exceptions import (
       OllamaVisionError,
       OllamaAPIError,
       ImageProcessingError,
       ModelNotFoundError,
       TimeoutError,
       CacheError,
   )
   ```

5. **Use caching**:
   ```python
   from src.cache import AnalysisCache
   
   cache = AnalysisCache(ttl=3600)  # 1 hour TTL
   key = AnalysisCache.get_key(image_data, prompt, model)
   
   cached_result = cache.get(key)
   if cached_result:
       return cached_result
   
   result = await analyze_image(...)
   cache.set(key, result)
   ```

6. **Use retry logic**:
   ```python
   from src.retry import retry_with_backoff
   
   @retry_with_backoff(max_retries=3, base_delay=1.0)
   async def resilient_api_call():
       # Will retry on 500, 502, 503, 504, 429
       return await ollama_client.analyze_image(...)
   ```

## [1.0.0] - 2024-01-01

### Added
- Initial release
- Basic MCP server with 4 tools
- Image analysis using Ollama vision models
- Support for local files, URLs, and base64 images
- Configuration via environment variables and JSON file