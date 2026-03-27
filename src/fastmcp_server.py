#!/usr/bin/env python3
"""
Ollama Vision MCP Server - FastMCP Implementation
A Model Context Protocol server providing computer vision capabilities using Ollama

This module provides a modern FastMCP-based implementation with:
- Cleaner tool definitions using decorators
- Lifespan management for resource initialization
- Progress reporting for long-running operations
- Built-in caching and retry logic
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Annotated, Any

from fastmcp import Context, FastMCP
from pydantic import Field

from .config import Config
from .ollama_client import OllamaClient
from .image_handler import ImageHandler
from .cache import AnalysisCache
from .exceptions import (
    OllamaVisionError,
    OllamaAPIError,
    ImageProcessingError,
    ModelNotFoundError,
)

logger = logging.getLogger(__name__)


@dataclass
class AppContext:
    """Application context for lifespan management."""

    config: Config
    ollama_client: OllamaClient
    image_handler: ImageHandler
    cache: AnalysisCache


@asynccontextmanager
async def app_lifespan(server: FastMCP):
    """Manage application lifecycle with type-safe context."""
    logger.info("Initializing Ollama Vision MCP Server...")

    config = Config()
    ollama_client = OllamaClient(config)
    image_handler = ImageHandler()
    cache = AnalysisCache(ttl=config.cache_ttl, maxsize=1000)

    try:
        connected = await ollama_client.check_connection()
        if connected:
            logger.info("Successfully connected to Ollama")
            models = await ollama_client.list_models()
            if models:
                logger.info(f"Available vision models: {', '.join(models)}")
            else:
                logger.warning("No vision models found. Run 'ollama pull llava-phi3'")
        else:
            logger.warning(
                "Could not connect to Ollama. Server will retry on tool calls."
            )
    except Exception as e:
        logger.warning(f"Initial connection check failed: {e}")

    yield AppContext(
        config=config,
        ollama_client=ollama_client,
        image_handler=image_handler,
        cache=cache,
    )

    logger.info("Shutting down Ollama Vision MCP Server...")


mcp = FastMCP("ollama-vision-mcp", lifespan=app_lifespan)


async def analyze_with_cache(
    ctx: Context,
    image_data: str,
    prompt: str,
    model: str | None = None,
) -> str:
    """Analyze image with caching and progress reporting."""
    app_context: AppContext = ctx.request_context.lifespan_context

    effective_model = model or app_context.config.default_model

    cache_key = AnalysisCache.get_key(image_data, prompt, effective_model)
    cached_result = app_context.cache.get(cache_key)

    if cached_result:
        await ctx.info("Cache hit - returning cached result")
        return cached_result

    await ctx.info(f"Analyzing image with model: {effective_model}")

    result = await app_context.ollama_client.analyze_image(
        image_data, prompt, effective_model
    )

    app_context.cache.set(cache_key, result)

    return result


@mcp.tool
async def analyze_image(
    image_path: Annotated[
        str, Field(description="Path to image file, URL, or base64 encoded image data")
    ],
    prompt: Annotated[
        str | None, Field(description="Custom prompt for analysis (optional)")
    ] = None,
    model: Annotated[
        str | None,
        Field(description="Ollama model to use (optional, defaults to config default)"),
    ] = None,
    ctx: Context = None,
) -> str:
    """
    Analyze an image with optional custom prompt and model.

    This is the most flexible tool for image analysis. Provide a custom
    prompt to get specific information about the image.

    Examples:
        - "Describe this image" → prompt="Describe this image in detail"
        - "What objects are in this image?" → prompt="List all objects in this image"
        - "Is this image safe for work?" → prompt="Is this image appropriate for a workplace?"
    """
    app_context: AppContext = ctx.request_context.lifespan_context

    await ctx.info(f"Processing image from: {image_path[:50]}...")
    image_data = await app_context.image_handler.process_image(image_path)

    effective_prompt = prompt or "Describe this image in detail"
    effective_model = model or app_context.config.default_model

    return await analyze_with_cache(ctx, image_data, effective_prompt, effective_model)


@mcp.tool
async def describe_image(
    image_path: Annotated[str, Field(description="Path to image file or URL")],
    ctx: Context = None,
) -> str:
    """
    Get a comprehensive description of what's in the image.

    This tool provides a detailed description including:
        - Overall scene/content
        - Colors and composition
        - Notable details
        - Spatial relationships
    """
    app_context: AppContext = ctx.request_context.lifespan_context

    await ctx.info(f"Describing image from: {image_path[:50]}...")
    image_data = await app_context.image_handler.process_image(image_path)

    prompt = """Provide a comprehensive description of this image including:
        1. Overall scene and content
        2. Main subjects and objects
        3. Colors, lighting, and composition
        4. Any notable details or interesting elements
        5. Spatial relationships between elements
        
        Be thorough and descriptive."""

    return await analyze_with_cache(ctx, image_data, prompt)


@mcp.tool
async def identify_objects(
    image_path: Annotated[str, Field(description="Path to image file or URL")],
    ctx: Context = None,
) -> str:
    """
    List all identifiable objects in the image.

    This tool identifies and lists objects found in the image.
    Objects are presented as a structured list with brief descriptions.
    """
    app_context: AppContext = ctx.request_context.lifespan_context

    await ctx.info(f"Identifying objects in image from: {image_path[:50]}...")
    image_data = await app_context.image_handler.process_image(image_path)

    prompt = """Identify all objects, people, animals, and items in this image.
        
        Format your response as a structured list:
        - Object name: brief description if relevant
        
        Include everything you can identify, even small or background items.
        If you're uncertain about an object, indicate your uncertainty."""

    return await analyze_with_cache(ctx, image_data, prompt)


@mcp.tool
async def read_text(
    image_path: Annotated[str, Field(description="Path to image file or URL")],
    ctx: Context = None,
) -> str:
    """
    Extract visible text from the image (OCR-like capabilities).

    This tool attempts to read and transcribe all visible text in the image.
    Works best with clear, readable text. May struggle with handwritten,
    stylized, or low-resolution text.
    """
    app_context: AppContext = ctx.request_context.lifespan_context

    await ctx.info(f"Reading text from image: {image_path[:50]}...")
    image_data = await app_context.image_handler.process_image(image_path)

    prompt = """Extract and transcribe ALL visible text from this image.
        
        Rules:
        1. Include all text exactly as it appears
        2. Preserve line breaks and formatting where possible
        3. Indicate text location if relevant (e.g., "top left", "center")
        4. If text is unclear or illegible, indicate with [illegible]
        5. If no text is found, respond with "No text found in this image."
        
        Do not interpret or explain the text - just transcribe it."""

    return await analyze_with_cache(ctx, image_data, prompt)


@mcp.tool
async def compare_images(
    image_path_1: Annotated[str, Field(description="Path to first image file or URL")],
    image_path_2: Annotated[str, Field(description="Path to second image file or URL")],
    comparison_type: Annotated[
        str,
        Field(
            description="Type of comparison: 'differences', 'similarities', 'sequence', or 'quality'"
        ),
    ] = "differences",
    ctx: Context = None,
) -> str:
    """
    Compare two images and identify differences, similarities, or changes.

    Comparison types:
        - differences: Focus on what's different between the images
        - similarities: Focus on what's the same between the images
        - sequence: Treat images as before/after and describe changes
        - quality: Compare the technical quality of both images

    This tool is experimental - accuracy may vary depending on image complexity.
    """
    app_context: AppContext = ctx.request_context.lifespan_context

    await ctx.info(f"Comparing images...")
    await ctx.report_progress(0.0, 2.0, "Processing first image")

    image_data_1 = await app_context.image_handler.process_image(image_path_1)

    await ctx.report_progress(1.0, 2.0, "Processing second image")
    image_data_2 = await app_context.image_handler.process_image(image_path_2)

    await ctx.report_progress(1.5, 2.0, "Analyzing images")

    prompts = {
        "differences": """Compare these two images and identify ALL differences:
            - Objects present in one but not the other
            - Position or orientation changes
            - Color or lighting differences
            - Text or label differences
            - Any other visible differences
            
            Format as a structured comparison.""",
        "similarities": """Compare these two images and identify ALL similarities:
            - Common objects and elements
            - Similar colors and tones
            - Same people or subjects
            - Identical or near-identical regions
            
            Explain what makes these images alike.""",
        "sequence": """Treat these as before/after images and describe what changed:
            - What was added?
            - What was removed?
            - What moved?
            - What transformed?
            
            Provide a clear narrative of the changes.""",
        "quality": """Compare the technical quality of these two images:
            - Resolution and sharpness
            - Lighting and exposure
            - Color accuracy and saturation
            - Noise or artifacts
            - Overall image quality assessment
            
            Which image is technically superior and why?""",
    }

    effective_prompt = prompts.get(comparison_type, prompts["differences"])

    combined_prompt = f"""{effective_prompt}

        Image 1 data: {image_data_1[:200]}...
        
        Image 2 data: {image_data_2[:200]}...
        
        Please analyze both and provide the comparison."""

    combined_data = f"{image_data_1}|{image_data_2}"

    result = await app_context.ollama_client.analyze_image(
        combined_data, effective_prompt, app_context.config.default_model
    )

    await ctx.report_progress(2.0, 2.0, "Comparison complete")

    return result


@mcp.tool
async def batch_analyze(
    image_paths: Annotated[
        list[str], Field(description="List of image paths or URLs to analyze")
    ],
    prompt: Annotated[str, Field(description="Analysis prompt to apply to all images")],
    model: Annotated[
        str | None, Field(description="Ollama model to use (optional)")
    ] = None,
    ctx: Context = None,
) -> str:
    """
    Analyze multiple images with the same prompt.

    This tool processes a batch of images and applies the same analysis
    to each. Results are returned as a structured response.

    Maximum recommended batch size: 5 images to avoid timeout.
    """
    app_context: AppContext = ctx.request_context.lifespan_context

    effective_model = model or app_context.config.default_model
    total = len(image_paths)
    results = []

    await ctx.info(f"Starting batch analysis of {total} images")

    for i, image_path in enumerate(image_paths):
        progress = i / total
        await ctx.report_progress(progress, 1.0, f"Processing image {i + 1}/{total}")

        try:
            image_data = await app_context.image_handler.process_image(image_path)
            result = await analyze_with_cache(ctx, image_data, prompt, effective_model)
            results.append(f"### Image {i + 1}: {image_path}\n\n{result}")
        except Exception as e:
            results.append(f"### Image {i + 1}: {image_path}\n\nError: {str(e)}")
            logger.error(f"Failed to analyze image {image_path}: {e}")

    await ctx.report_progress(1.0, 1.0, "Batch complete")

    return "\n\n---\n\n".join(results)


@mcp.tool
async def detect_objects_with_boxes(
    image_path: Annotated[str, Field(description="Path to image file or URL")],
    object_category: Annotated[
        str | None,
        Field(
            description="Category to focus on (optional): 'people', 'vehicles', 'animals', 'text', 'all'"
        ),
    ] = "all",
    ctx: Context = None,
) -> str:
    """
    Detect objects with approximate bounding box descriptions.

    WARNING: This tool is EXPERIMENTAL. Bounding box accuracy varies
    significantly by model and image type. Results should not be used
    for critical applications.

    This tool attempts to identify objects and describe their locations
    using relative positioning (e.g., "top-left quadrant", "center").
    """
    app_context: AppContext = ctx.request_context.lifespan_context

    await ctx.info(f"Detecting objects in: {image_path[:50]}...")
    image_data = await app_context.image_handler.process_image(image_path)

    category_prompts = {
        "people": "Detect all people in this image. For each person, describe their approximate location (top-left, center-right, etc.), relative size in the frame, and any visible characteristics.",
        "vehicles": "Detect all vehicles (cars, trucks, motorcycles, bicycles, etc.) in this image. For each, describe location, relative size, color if visible, and type.",
        "animals": "Detect all animals in this image. For each, describe location, type/species, relative size, and any distinctive features.",
        "text": "Detect all text regions in this image. For each text block, describe its location, approximate size, and content if readable.",
        "all": """Detect ALL objects in this image and provide approximate locations.
            
            For each object found, provide:
            1. Object name/type
            2. Approximate location (use descriptors: "top-left", "center", "bottom-right", etc.)
            3. Relative size (small/medium/large)
            4. Brief description
            
            Format as a structured list.""",
    }

    effective_prompt = category_prompts.get(
        object_category or "all", category_prompts["all"]
    )
    effective_prompt += (
        "\n\nNote: This is an experimental feature. Location accuracy is approximate."
    )

    result = await analyze_with_cache(ctx, image_data, effective_prompt)

    return f"⚠️ EXPERIMENTAL - Location accuracy may vary\n\n{result}"


@mcp.resource("models://list")
async def list_available_models(ctx: Context) -> str:
    """List available vision models from Ollama."""
    import json

    app_context: AppContext = ctx.request_context.lifespan_context

    try:
        models = await app_context.ollama_client.list_models()

        if not models:
            return json.dumps(
                {
                    "status": "warning",
                    "message": "No vision models found. Run 'ollama pull llava-phi3' to install one.",
                    "models": [],
                },
                indent=2,
            )

        return json.dumps(
            {
                "status": "success",
                "models": models,
                "default_model": app_context.config.default_model,
            },
            indent=2,
        )

    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        return json.dumps(
            {"status": "error", "message": str(e), "models": []}, indent=2
        )


@mcp.resource("status://health")
async def health_check(ctx: Context) -> str:
    """Server health status."""
    import json
    from datetime import datetime

    app_context: AppContext = ctx.request_context.lifespan_context

    try:
        connected = await app_context.ollama_client.check_connection()
        models = await app_context.ollama_client.list_models() if connected else []

        return json.dumps(
            {
                "status": "healthy" if connected else "degraded",
                "ollama_connected": connected,
                "available_models": len(models),
                "models": models,
                "default_model": app_context.config.default_model,
                "cache_enabled": app_context.config.cache_enabled,
                "cache_ttl": app_context.config.cache_ttl,
                "timestamp": datetime.now().isoformat(),
            },
            indent=2,
        )

    except Exception as e:
        return json.dumps(
            {
                "status": "error",
                "message": str(e),
                "timestamp": datetime.now().isoformat(),
            },
            indent=2,
        )


@mcp.resource("config://current")
async def get_current_config(ctx: Context) -> str:
    """Current server configuration."""
    import json

    app_context: AppContext = ctx.request_context.lifespan_context

    return json.dumps(
        {
            "ollama_url": app_context.config.ollama_url,
            "default_model": app_context.config.default_model,
            "timeout": app_context.config.timeout,
            "log_level": app_context.config.log_level,
            "cache_enabled": app_context.config.cache_enabled,
            "cache_ttl": app_context.config.cache_ttl,
            "model_preferences": app_context.config.model_preferences,
        },
        indent=2,
    )


@mcp.prompt
def analyze_workflow(image_description: str) -> str:
    """Guided workflow for comprehensive image analysis."""
    return f"""Please analyze this image comprehensively:

Image context: {image_description}

Provide analysis covering:
1. Overall scene/content description
2. Key objects and their relationships
3. Colors, lighting, and composition
4. Any text or symbols present
5. Notable details or anomalies

Structure your response with clear headings."""


@mcp.prompt
def compare_workflow(image_a: str, image_b: str) -> str:
    """Workflow for comparing two images."""
    return f"""Compare these two images:

Image A: {image_a}
Image B: {image_b}

Please analyze both images and provide a detailed comparison:
1. What are the main differences?
2. What similarities exist?
3. Which image do you prefer and why?
4. Any other observations?"""


@mcp.prompt
def batch_workflow(images: list[str]) -> str:
    """Workflow for batch image processing."""
    image_list = "\n".join(f"- {img}" for img in images)
    return f"""Process the following images:

{image_list}

For each image:
1. Describe the main content
2. Identify key objects
3. Note any text present
4. Rate image quality (1-10)

Summarize common themes across all images."""


def main():
    """Main entry point for FastMCP server."""
    import sys

    try:
        mcp.run()
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
