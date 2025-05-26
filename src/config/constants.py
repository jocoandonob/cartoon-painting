"""
Constants and configuration values for the Stable Diffusion Image Generator.
"""

# Model configurations for different styles
MODEL_CONFIGS = {
    "Disney": {
        "base_model": "stabilityai/stable-diffusion-xl-base-1.0",
        "lora_path": "models/disney_style_xl.safetensors",
        "default_prompt": "disney style, animal focus, animal, cat",
        "use_safetensors": True,
        "is_sdxl": True,
        "pipeline": "sdxl"
    },
    "Flux": {
        "base_model": "black-forest-labs/FLUX.1-dev",
        "lora_path": "models/joco.safetensors",
        "default_prompt": "A cartoon style couple takes a selfie in front of an Egyptian pyramid, which is composed of a man and a woman, both wearing sunglasses. Men wear blue shirts, jeans, and white shoes, while women wear yellow hats, blue jackets, white tops, orange dresses, and pink sneakers. Sand and a group of tourists in the distance. Integrating reality and cartoon elements.",
        "use_safetensors": True,
        "is_sdxl": False,
        "pipeline": "flux"
    },
    "TextToImage": {
        "base_model": "stabilityai/stable-diffusion-xl-base-1.0",
        "lora_path": "models/pytorch_lora_weights.safetensors",
        "default_prompt": "Draw a picture of two female boxers fighting each other.",
        "use_safetensors": True,
        "is_sdxl": True,
        "pipeline": "sdxl"
    },
    "ClayAnimation": {
        "base_model": "runwayml/stable-diffusion-v1-5",
        "lora_path": "models/ClayAnimationRedmond15-ClayAnimation-Clay.safetensors",
        "default_prompt": "A cute blonde girl, ,Clay Animation, Clay,",
        "use_safetensors": True,
        "is_sdxl": False,
        "pipeline": "stable-diffusion"
    },
    "StoryboardSketch": {
        "base_model": "stabilityai/stable-diffusion-xl-base-1.0",
        "lora_path": "models/Storyboard_sketch.safetensors",
        "default_prompt": "storyboard sketch of a zombie basketball player dunking with both hands, action shot, motion blur, hero",
        "use_safetensors": True,
        "is_sdxl": True,
        "pipeline": "sdxl"
    },
    "GraphicNovel": {
        "base_model": "stabilityai/stable-diffusion-xl-base-1.0",
        "lora_path": "models/Graphic_Novel_Illustration-000007.safetensors",
        "default_prompt": "breathtaking highly detailed graphic novel illustration of morgan freeman riding a harley davidson motorcycle, dark and gritty",
        "use_safetensors": True,
        "is_sdxl": True,
        "pipeline": "sdxl"
    }
}

# File paths
STATIC_DIR = "static"
TEMPLATES_DIR = "templates"
CSS_DIR = "css"
MODELS_DIR = "models"

# UI Constants
DEFAULT_SEED = 123
DEFAULT_STEPS = 30
DEFAULT_GUIDANCE_SCALE = 7.5
DEFAULT_STRENGTH = 0.75

# Supported image formats
SUPPORTED_IMAGE_FORMATS = ["png", "jpg", "jpeg"] 