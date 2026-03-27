"""
Ollama Vision MCP Server
A Model Context Protocol server for computer vision using Ollama

This package provides two server implementations:
1. OllamaVisionServer (legacy) - Raw MCP SDK implementation
2. FastMCP Server (recommended) - Modern FastMCP implementation with enhanced features

For new implementations, use FastMCP via:
    from src.fastmcp_server import mcp, main
"""

__version__ = "2.0.0"
__author__ = "Ollama Vision MCP Contributors"

from .server import OllamaVisionServer, main as legacy_main
from .ollama_client import OllamaClient
from .image_handler import ImageHandler
from .config import Config
from .cache import AnalysisCache
from .retry import (
    retry_with_backoff,
    retry_operation,
    RetryContext,
    is_retryable,
    RETRYABLE_STATUS_CODES,
)

from .exceptions import (
    OllamaVisionError,
    OllamaAPIError,
    ImageProcessingError,
    ModelNotFoundError,
    TimeoutError,
    CacheError,
)

__all__ = [
    "OllamaVisionServer",
    "OllamaClient",
    "ImageHandler",
    "Config",
    "AnalysisCache",
    "retry_with_backoff",
    "retry_operation",
    "RetryContext",
    "is_retryable",
    "RETRYABLE_STATUS_CODES",
    "OllamaVisionError",
    "OllamaAPIError",
    "ImageProcessingError",
    "ModelNotFoundError",
    "TimeoutError",
    "CacheError",
    "main",
    "legacy_main",
]

main = legacy_main
