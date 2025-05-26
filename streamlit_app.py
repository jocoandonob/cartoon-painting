import streamlit as st
import torch
from diffusers import StableDiffusionXLPipeline, EulerAncestralDiscreteScheduler, DiffusionPipeline, StableDiffusionPipeline, AutoPipelineForImage2Image
from transformers import CLIPTextModel, CLIPTokenizer
from PIL import Image
import io
import base64
import time
import os
import json
from datetime import datetime, timedelta
from huggingface_hub import login
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Model configurations
MODEL_CONFIGS = {
    # "HiDream": {
    #     "base_model": "HiDream-ai/HiDream-I1-Full",
    #     "lora_path": "models/Floating_Head_HiDream_v1_3000.safetensors",
    #     "default_prompt": "h3adfl0at 3D floating head of a an old latino man wearing a LA Dodgers baseball cap hat and thick rimmed brown sunglasses with light tinted lenses, he has a thick mustache and looks brooding",
    #     "use_safetensors": True,
    #     "is_sdxl": False,
    #     "pipeline": "stable-diffusion"
    # },
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

# Set page config
st.set_page_config(
    page_title="Text to Image Generator",
    page_icon="🎨",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Load CSS
def load_css():
    css_file = os.path.join("static", "css", "style.css")
    with open(css_file, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Load HTML templates
def load_template(template_name):
    template_file = os.path.join("templates", f"{template_name}.html")
    with open(template_file, "r", encoding="utf-8") as f:
        return f.read()

# Load CSS
load_css()

# Title and description
st.title("🎨 Style Image Generator")
st.markdown(load_template("cards").split("<!-- Description Card -->")[1], unsafe_allow_html=True)

# Tab selection
tab1, tab2 = st.tabs(["Text to Image", "Image to Image"])

# Initialize the model
@st.cache_resource
def load_model(model_name, img2img=False):
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
            pipe = StableDiffusionXLPipeline.from_pretrained(
                config["base_model"],
                torch_dtype=torch.float32,
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
                add_prefix_space=False,
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
        
        # If img2img is requested, create the image-to-image pipeline from the text-to-image pipeline
        if img2img:
            pipe = AutoPipelineForImage2Image.from_pipe(pipe).to(device)
        
        return pipe
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()

# Text to Image Tab
with tab1:
    # Create two columns for input and output
    col1, col2 = st.columns([1, 1])

    with col1:
        # Input section
        st.markdown(load_template("cards").split("<!-- Input Card -->")[1].split("<!-- Parameters Card -->")[0], unsafe_allow_html=True)
        
        # Model selection
        selected_model = st.selectbox(
            "Select Style:",
            options=list(MODEL_CONFIGS.keys()),
            format_func=lambda x: x
        )
        
        # Load model configuration
        model_config = MODEL_CONFIGS[selected_model]
        
        # Text input with a larger text area
        prompt = st.text_area(
            "Enter your prompt:",
            height=80,
            placeholder=f"Describe the image you want to generate... (e.g., '{model_config['default_prompt']}')",
            value=model_config["default_prompt"]
        )
        
        # Parameters section
        st.markdown(load_template("cards").split("<!-- Parameters Card -->")[1].split("<!-- Output Card -->")[0], unsafe_allow_html=True)
        
        num_inference_steps = st.slider("Number of inference steps", 20, 50, 30)
        guidance_scale = st.slider("Guidance scale", 1.0, 20.0, 7.5)
        
        # Seed control
        seed = st.number_input("Seed (for reproducibility)", value=123, step=1)
        
        # Generate button
        generate_button = st.button("🎨 Generate Image", type="primary")

    with col2:
        # Output section
        st.markdown(load_template("cards").split("<!-- Output Card -->")[1].split("<!-- Description Card -->")[0], unsafe_allow_html=True)
        
        # Placeholder for the generated image
        image_placeholder = st.empty()
        
        if generate_button and prompt:
            # Create loading animation
            loading_container = st.empty()
            loading_container.markdown(load_template("loading"), unsafe_allow_html=True)
            
            try:
                # Load the selected model
                with st.spinner("Loading model..."):
                    pipe = load_model(selected_model)
                
                # Create progress bar and time display
                progress_bar = st.progress(0)
                progress_text = st.empty()
                time_text = st.empty()
                
                # Track start time
                start_time = time.time()
                
                # Create a callback to update progress
                def progress_callback(step, timestep, latents):
                    current_time = time.time()
                    elapsed_time = current_time - start_time
                    
                    # Calculate progress
                    progress = min(int((step + 1) / num_inference_steps * 100), 100)
                    
                    # Calculate estimated time remaining
                    if step > 0:
                        time_per_step = elapsed_time / step
                        remaining_steps = num_inference_steps - step
                        estimated_time_remaining = time_per_step * remaining_steps
                        
                        # Format time remaining in minutes and seconds
                        minutes = int(estimated_time_remaining // 60)
                        seconds = int(estimated_time_remaining % 60)
                        time_str = f"{minutes}m {seconds}s" if minutes > 0 else f"{seconds}s"
                        
                        # Update progress and time displays
                        progress_bar.progress(progress)
                        progress_text.markdown(
                            f'<span style="color: #FFD700">Generating image... {progress}%</span>', 
                            unsafe_allow_html=True
                        )
                        time_text.markdown(
                            f'<span style="color: #FFD700">Time remaining: {time_str}</span>', 
                            unsafe_allow_html=True
                        )
                
                # Set deterministic seed
                generator = torch.Generator(device="cpu").manual_seed(seed)
                
                # Generate image with progress callback
                image = pipe(
                    prompt=prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                    callback=progress_callback,
                    callback_steps=1
                ).images[0]
                
                # Clear loading animation and progress
                loading_container.empty()
                progress_bar.empty()
                progress_text.empty()
                time_text.empty()
                
                # Display image
                image_placeholder.image(image, caption=f"Generated Image using {selected_model} style", use_container_width=True)
                
                # Add download button
                buf = io.BytesIO()
                image.save(buf, format="PNG")
                st.download_button(
                    label="⬇️ Download Image",
                    data=buf.getvalue(),
                    file_name=f"{selected_model.lower()}_style_output.png",
                    mime="image/png"
                )
                
            except Exception as e:
                # Clear all loading states
                loading_container.empty()
                progress_bar.empty()
                progress_text.empty()
                time_text.empty()
                st.error(f"Error generating image: {str(e)}")
        elif generate_button:
            st.warning("⚠️ Please enter a prompt first.")

# Image to Image Tab
with tab2:
    # Create two columns for input and output
    col1, col2 = st.columns([1, 1])

    with col1:
        # Input section
        st.markdown(load_template("cards").split("<!-- Input Card -->")[1].split("<!-- Parameters Card -->")[0], unsafe_allow_html=True)
        
        # Model selection
        selected_model = st.selectbox(
            "Select Style (Image to Image):",
            options=list(MODEL_CONFIGS.keys()),
            format_func=lambda x: x,
            key="img2img_model"
        )
        
        # Load model configuration
        model_config = MODEL_CONFIGS[selected_model]
        
        # Image upload
        uploaded_file = st.file_uploader("Upload an image to transform:", type=["png", "jpg", "jpeg"])
        
        if uploaded_file is not None:
            # Display the uploaded image
            init_image = Image.open(uploaded_file)
            st.image(init_image, caption="Uploaded Image", use_container_width=True)
        
        # Text input with a larger text area
        prompt = st.text_area(
            "Enter your prompt:",
            height=80,
            placeholder=f"Describe how you want to transform the image... (e.g., '{model_config['default_prompt']}')",
            value=model_config["default_prompt"],
            key="img2img_prompt"
        )
        
        # Parameters section
        st.markdown(load_template("cards").split("<!-- Parameters Card -->")[1].split("<!-- Output Card -->")[0], unsafe_allow_html=True)
        
        num_inference_steps = st.slider("Number of inference steps", 20, 50, 30, key="img2img_steps")
        guidance_scale = st.slider("Guidance scale", 1.0, 20.0, 7.5, key="img2img_guidance")
        strength = st.slider("Transformation strength", 0.0, 1.0, 0.75, key="img2img_strength")
        
        # Seed control
        seed = st.number_input("Seed (for reproducibility)", value=123, step=1, key="img2img_seed")
        
        # Generate button
        generate_button = st.button("🎨 Transform Image", type="primary", key="img2img_generate")

    with col2:
        # Output section
        st.markdown(load_template("cards").split("<!-- Output Card -->")[1].split("<!-- Description Card -->")[0], unsafe_allow_html=True)
        
        # Placeholder for the generated image
        image_placeholder = st.empty()
        
        if generate_button and prompt and uploaded_file is not None:
            # Create loading animation
            loading_container = st.empty()
            loading_container.markdown(load_template("loading"), unsafe_allow_html=True)
            
            try:
                # Load the selected model
                with st.spinner("Loading model..."):
                    pipe = load_model(selected_model, img2img=True)
                
                # Create progress bar and time display
                progress_bar = st.progress(0)
                progress_text = st.empty()
                time_text = st.empty()
                
                # Track start time
                start_time = time.time()
                
                # Create a callback to update progress
                def progress_callback(step, timestep, latents):
                    current_time = time.time()
                    elapsed_time = current_time - start_time
                    
                    # Calculate progress
                    progress = min(int((step + 1) / num_inference_steps * 100), 100)
                    
                    # Calculate estimated time remaining
                    if step > 0:
                        time_per_step = elapsed_time / step
                        remaining_steps = num_inference_steps - step
                        estimated_time_remaining = time_per_step * remaining_steps
                        
                        # Format time remaining in minutes and seconds
                        minutes = int(estimated_time_remaining // 60)
                        seconds = int(estimated_time_remaining % 60)
                        time_str = f"{minutes}m {seconds}s" if minutes > 0 else f"{seconds}s"
                        
                        # Update progress and time displays
                        progress_bar.progress(progress)
                        progress_text.markdown(
                            f'<span style="color: #FFD700">Transforming image... {progress}%</span>', 
                            unsafe_allow_html=True
                        )
                        time_text.markdown(
                            f'<span style="color: #FFD700">Time remaining: {time_str}</span>', 
                            unsafe_allow_html=True
                        )
                
                # Set deterministic seed
                generator = torch.Generator(device="cpu").manual_seed(seed)
                
                # Generate image with progress callback
                image = pipe(
                    prompt=prompt,
                    image=init_image,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    strength=strength,
                    generator=generator,
                    callback=progress_callback,
                    callback_steps=1
                ).images[0]
                
                # Clear loading animation and progress
                loading_container.empty()
                progress_bar.empty()
                progress_text.empty()
                time_text.empty()
                
                # Display image
                image_placeholder.image(image, caption=f"Transformed Image using {selected_model} style", use_container_width=True)
                
                # Add download button
                buf = io.BytesIO()
                image.save(buf, format="PNG")
                st.download_button(
                    label="⬇️ Download Image",
                    data=buf.getvalue(),
                    file_name=f"{selected_model.lower()}_style_transformed.png",
                    mime="image/png"
                )
                
            except Exception as e:
                # Clear all loading states
                loading_container.empty()
                progress_bar.empty()
                progress_text.empty()
                time_text.empty()
                st.error(f"Error transforming image: {str(e)}")
        elif generate_button:
            if not uploaded_file:
                st.warning("⚠️ Please upload an image first.")
            elif not prompt:
                st.warning("⚠️ Please enter a prompt first.") 