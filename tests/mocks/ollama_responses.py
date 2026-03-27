"""
Mock Ollama API responses for testing
Provides realistic mock data that simulates Ollama API behavior
"""

from typing import Dict, Any, List

# Mock response for GET /api/tags endpoint
MOCK_MODELS_RESPONSE: Dict[str, Any] = {
    "models": [
        {
            "name": "llava-phi3",
            "modified_at": "2024-01-15T10:30:00.000000Z",
            "size": 2700000000,
            "digest": "abc123def456",
            "details": {
                "format": "gguf",
                "family": "llava",
                "parameter_size": "3.8B",
                "quantization_level": "Q4_K_M",
            },
        },
        {
            "name": "llava:7b",
            "modified_at": "2024-01-14T15:45:00.000000Z",
            "size": 4500000000,
            "digest": "def456ghi789",
            "details": {
                "format": "gguf",
                "family": "llava",
                "parameter_size": "7B",
                "quantization_level": "Q4_K_M",
            },
        },
        {
            "name": "llava:13b",
            "modified_at": "2024-01-13T08:20:00.000000Z",
            "size": 7800000000,
            "digest": "ghi789jkl012",
            "details": {
                "format": "gguf",
                "family": "llava",
                "parameter_size": "13B",
                "quantization_level": "Q4_K_M",
            },
        },
        {
            "name": "bakllava",
            "modified_at": "2024-01-12T12:15:00.000000Z",
            "size": 4200000000,
            "digest": "jkl012mno345",
            "details": {
                "format": "gguf",
                "family": "bakllava",
                "parameter_size": "7B",
                "quantization_level": "Q4_K_M",
            },
        },
        {
            "name": "llama2:7b",
            "modified_at": "2024-01-11T09:00:00.000000Z",
            "size": 3800000000,
            "digest": "mno345pqr678",
            "details": {
                "format": "gguf",
                "family": "llama",
                "parameter_size": "7B",
                "quantization_level": "Q4_K_M",
            },
        },
    ]
}

# Mock responses for POST /api/generate endpoint
MOCK_ANALYZE_RESPONSE: Dict[str, str] = {
    "default": """I can see a test image with the text "Test Image" written in black 
    on a white background. The image appears to be 200 pixels wide and 100 pixels tall. 
    The text is centered approximately in the middle of the image, positioned around 
    y=40 pixels from the top. The overall composition is simple and minimal.""",
    "describe": """This is a comprehensive description of the image:

    **Main Content:**
    - The image contains the text "Test Image" written in black color
    - The background is white (RGB: 255, 255, 255)
    - The text appears to be positioned at coordinates (10, 40)
    
    **Dimensions:**
    - Width: 200 pixels
    - Height: 100 pixels
    - Aspect ratio: 2:1
    
    **Colors:**
    - Background: White (#FFFFFF)
    - Text color: Black (#000000)
    
    **Composition:**
    - The text is located in the upper portion of the image
    - There's sufficient padding around the text
    - The overall design is clean and minimalist
    
    **Technical Details:**
    - Image format: PNG
    - Color space: RGB
    - No transparency detected""",
    "objects": """I can identify the following objects in the image:

    • Text/Object: "Test Image" - appears to be the main focal point
    • Background: White rectangular area
    • Text styling: Black color, standard font
    
    The image contains only one primary visual element - the text itself. 
    There are no other distinct objects, shapes, or patterns visible.""",
    "text": """Extracted text from the image:
    
    "Test Image"
    
    No additional text was found in the image. The text appears to be rendered 
    in a standard sans-serif font at a readable size.""",
    "detailed": """Detailed analysis of the test image:

    1. **Primary Subject:** Text string "Test Image"
       - Font: Sans-serif, standard weight
       - Color: Black (#000000)
       - Position: Started at x=10, y=40
       - Size: Approximately 20-24 points
    
    2. **Background:**
       - Color: Pure white (#FFFFFF)
       - Texture: Smooth, solid color
       - Extension: Fills entire 200x100 pixel canvas
    
    3. **Layout & Composition:**
       - Horizontal orientation
       - Text positioned in upper-center region
       - No visible borders or frames
       - Adequate white space surrounding the text
    
    4. **Quality Assessment:**
       - Clear, legible text
       - No compression artifacts
       - Sharp edges
       - High contrast between text and background
    
    5. **Potential Use Cases:**
       - Placeholder image
       - Test pattern
       - Simple demonstration graphic""",
}

# Full API response structure for generate endpoint
MOCK_GENERATE_RESPONSE: Dict[str, Any] = {
    "model": "llava-phi3",
    "created_at": "2024-01-15T10:35:00.000000Z",
    "response": MOCK_ANALYZE_RESPONSE["default"],
    "done": True,
    "context": [1, 2, 3, 4, 5],  # Token context
    "total_duration": 1500000000,  # 1.5 seconds in nanoseconds
    "load_duration": 50000000,  # 50 milliseconds
    "prompt_eval_count": 10,
    "prompt_eval_duration": 200000000,  # 200 milliseconds
    "eval_count": 50,
    "eval_duration": 1200000000,  # 1.2 seconds
}

# Streaming response chunks (for testing streaming functionality if added later)
MOCK_STREAM_RESPONSES: List[Dict[str, Any]] = [
    {
        "model": "llava-phi3",
        "created_at": "2024-01-15T10:35:00.000000Z",
        "response": "I ",
        "done": False,
    },
    {
        "model": "llava-phi3",
        "created_at": "2024-01-15T10:35:00.100000Z",
        "response": "can ",
        "done": False,
    },
    {
        "model": "llava-phi3",
        "created_at": "2024-01-15T10:35:00.200000Z",
        "response": "see ",
        "done": False,
    },
    {
        "model": "llava-phi3",
        "created_at": "2024-01-15T10:35:00.300000Z",
        "response": "a ",
        "done": False,
    },
    {
        "model": "llava-phi3",
        "created_at": "2024-01-15T10:35:00.400000Z",
        "response": "test ",
        "done": False,
    },
    {
        "model": "llava-phi3",
        "created_at": "2024-01-15T10:35:00.500000Z",
        "response": "image.",
        "done": True,
    },
]

# Mock responses for different error scenarios
MOCK_ERROR_RESPONSES: Dict[str, Any] = {
    "connection_error": {
        "error": "Failed to connect to Ollama server at http://localhost:11434",
        "type": "ConnectionError",
        "message": "Ollama may not be running. Please start Ollama and try again.",
    },
    "timeout_error": {
        "error": "Request timed out after 120 seconds",
        "type": "TimeoutError",
        "message": "The vision model is taking too long to respond. Consider using a smaller model or a simpler prompt.",
    },
    "model_not_found": {
        "error": "Model 'invalid-model' not found",
        "type": "ModelNotFoundError",
        "message": "The specified model is not available. Run 'ollama pull <model>' to download it.",
    },
    "invalid_image": {
        "error": "Invalid image data",
        "type": "ImageProcessingError",
        "message": "The provided image could not be processed. Ensure it's a valid image file (JPEG, PNG, etc.)",
    },
    "invalid_base64": {
        "error": "Invalid base64 encoding",
        "type": "EncodingError",
        "message": "The image data is not properly base64 encoded.",
    },
    "file_not_found": {
        "error": "Image file not found",
        "type": "FileNotFoundError",
        "message": "The specified image file does not exist.",
    },
    "url_error": {
        "error": "Failed to download image from URL",
        "type": "URLError",
        "message": "Could not retrieve the image from the provided URL.",
    },
    "api_error_500": {
        "error": "Internal server error",
        "type": "APIError",
        "message": "Ollama returned an internal server error (500). Check Ollama logs for details.",
    },
    "api_error_404": {
        "error": "Endpoint not found",
        "type": "APIError",
        "message": "The requested API endpoint does not exist.",
    },
    "rate_limit": {
        "error": "Rate limit exceeded",
        "type": "RateLimitError",
        "message": "Too many requests. Please wait before sending more requests.",
    },
}

# Mock responses for model pull operation
MOCK_PULL_RESPONSES: List[Dict[str, Any]] = [
    {"status": "pulling manifest"},
    {
        "status": "downloading",
        "digest": "sha256:abc123",
        "completed": 0,
        "total": 2700000000,
    },
    {
        "status": "downloading",
        "digest": "sha256:abc123",
        "completed": 270000000,
        "total": 2700000000,
    },
    {
        "status": "downloading",
        "digest": "sha256:abc123",
        "completed": 540000000,
        "total": 2700000000,
    },
    {
        "status": "downloading",
        "digest": "sha256:abc123",
        "completed": 1350000000,
        "total": 2700000000,
    },
    {
        "status": "downloading",
        "digest": "sha256:abc123",
        "completed": 2700000000,
        "total": 2700000000,
    },
    {"status": "verifying sha256 digest"},
    {"status": "writing manifest"},
    {"status": "removing any unused layers"},
    {"status": "success"},
]

# Empty response (for testing edge cases)
MOCK_EMPTY_RESPONSE: Dict[str, Any] = {"models": []}

# Response when no vision models are available
MOCK_NO_VISION_MODELS: Dict[str, Any] = {
    "models": [
        {
            "name": "llama2:7b",
            "modified_at": "2024-01-11T09:00:00.000000Z",
            "size": 3800000000,
            "digest": "mno345pqr678",
            "details": {
                "format": "gguf",
                "family": "llama",
                "parameter_size": "7B",
                "quantization_level": "Q4_K_M",
            },
        },
        {
            "name": "mistral:7b",
            "modified_at": "2024-01-10T14:30:00.000000Z",
            "size": 4100000000,
            "digest": "pqr678stu901",
            "details": {
                "format": "gguf",
                "family": "mistral",
                "parameter_size": "7B",
                "quantization_level": "Q4_K_M",
            },
        },
    ]
}
