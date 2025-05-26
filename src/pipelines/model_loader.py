import torch
from diffusers import (
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLInpaintPipeline,
    EulerAncestralDiscreteScheduler,
    DiffusionPipeline,
    StableDiffusionPipeline,
    AutoPipelineForImage2Image,
    AutoPipelineForText2Image,
    AutoPipelineForInpainting
)
from transformers import CLIPTextModel, CLIPTokenizer
from huggingface_hub import login
import os
import streamlit as st
from src.config.constants import MODEL_CONFIGS

@st.cache_resource
def load_model(model_name, img2img=False, inpainting=False):
    try:
        config = MODEL_CONFIGS[model_name]
        
        # Check if HF token is set
        hf_token = os.getenv("HUGGINGFACE_TOKEN")
        if not hf_token:
            st.error("Hugging Face token not found. Please set the HUGGINGFACE_TOKEN environment variable.")
            st.stop()
        
        # Login to Hugging Face
        login(token=hf_token)
        
        # Load base model based on pipeline type
        if config["pipeline"] == "sdxl":
            if inpainting:
                # First load text2image pipeline
                base_pipe = StableDiffusionXLPipeline.from_pretrained(
                    config["base_model"],
                    torch_dtype=torch.float32,
                    variant="fp16",
                    use_safetensors=config["use_safetensors"],
                    token=hf_token
                )
                # Convert to inpainting pipeline
                pipe = AutoPipelineForInpainting.from_pipe(base_pipe)
            elif img2img:
                # Load base model
                base_pipe = DiffusionPipeline.from_pretrained(
                    "stabilityai/stable-diffusion-xl-base-1.0",
                    torch_dtype=torch.float16,
                    variant="fp16",
                    use_safetensors=True,
                    token=hf_token
                )
                
                # Load refiner model with shared components
                refiner_pipe = DiffusionPipeline.from_pretrained(
                    "stabilityai/stable-diffusion-xl-refiner-1.0",
                    text_encoder_2=base_pipe.text_encoder_2,
                    vae=base_pipe.vae,
                    torch_dtype=torch.float32,
                    use_safetensors=True,
                    variant="fp16",
                    token=hf_token
                )
                
                # Move models to GPU if available
                device = "cuda" if torch.cuda.is_available() else "cpu"
                base_pipe = base_pipe.to(device)
                refiner_pipe = refiner_pipe.to(device)
                
                # Load LoRA weights for base model
                base_pipe.load_lora_weights(config["lora_path"])
                
                return {"base": base_pipe, "refiner": refiner_pipe}
            else:
                pipe = StableDiffusionXLPipeline.from_pretrained(
                    config["base_model"],
                    torch_dtype=torch.float32,
                    variant="fp16",
                    use_safetensors=config["use_safetensors"],
                    token=hf_token
                )
        elif config["pipeline"] == "flux":
            pipe = DiffusionPipeline.from_pretrained(
                config["base_model"],
                torch_dtype=torch.float32,
                use_safetensors=config["use_safetensors"],
                token=hf_token
            )
        else:  # stable-diffusion
            tokenizer = CLIPTokenizer.from_pretrained(
                config["base_model"],
                subfolder="tokenizer",
                token=hf_token
            )
            text_encoder = CLIPTextModel.from_pretrained(
                config["base_model"],
                subfolder="text_encoder",
                token=hf_token
            )
            
            pipe = StableDiffusionPipeline.from_pretrained(
                config["base_model"],
                torch_dtype=torch.float32,
                use_safetensors=config["use_safetensors"],
                token=hf_token,
                tokenizer=tokenizer,
                text_encoder=text_encoder
            )
        
        # Set scheduler
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
        
        # Load LoRA weights
        pipe.load_lora_weights(config["lora_path"])
        
        # Move to GPU if available, otherwise keep on CPU
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe = pipe.to(device)
        
        return pipe
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop() 