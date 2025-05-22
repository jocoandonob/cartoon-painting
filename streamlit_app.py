import streamlit as st
import torch
from diffusers import StableDiffusionXLPipeline, EulerAncestralDiscreteScheduler
from PIL import Image
import io
import base64
import time
import os

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
st.title("🎨 Disney Style Image Generator")
st.markdown(load_template("cards").split("<!-- Description Card -->")[1], unsafe_allow_html=True)

# Initialize the model
@st.cache_resource
def load_model():
    try:
        # Load SDXL base model
        pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float32,
            use_safetensors=True
        ).to("cpu")
        
        # Optional: Set the same scheduler Hugging Face might use
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
        
        # Load LoRA weights
        pipe.load_lora_weights("goofyai/disney_style_xl", weight_name="disney_style_xl.safetensors")
        
        # Move to GPU if available, otherwise keep on CPU
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe = pipe.to(device)
        
        return pipe
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()

# Load the model
with st.spinner("Loading model..."):
    try:
        pipe = load_model()
        st.success("✨ Model loaded successfully!")
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()

# Create two columns for input and output
col1, col2 = st.columns([1, 1])

with col1:
    # Input section
    st.markdown(load_template("cards").split("<!-- Input Card -->")[1].split("<!-- Parameters Card -->")[0], unsafe_allow_html=True)
    
    # Text input with a larger text area
    prompt = st.text_area(
        "Enter your prompt:",
        height=80,
        placeholder="Describe the image you want to generate... (e.g., 'a cute disney-style princess, full body, highly detailed')"
    )
    
    # Negative prompt
    negative_prompt = st.text_area(
        "Negative prompt (what to avoid):",
        height=68,
        placeholder="blurry, low quality, distorted",
        value="blurry, low quality, distorted, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, normal quality, jpeg artifacts, signature, watermark, username"
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
            # Create progress bar
            progress_bar = st.progress(0)
            progress_text = st.empty()
            
            # Create a callback to update progress
            def progress_callback(step, timestep, latents):
                progress = min(int((step + 1) / num_inference_steps * 100), 100)
                progress_bar.progress(progress)
                progress_text.markdown(f'<span style="color: #FFD700">Generating image... {progress}%</span>', unsafe_allow_html=True)
            
            # Set deterministic seed
            generator = torch.Generator(device="cpu").manual_seed(seed)
            
            # Generate image with progress callback
            image = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
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
            
            # Display image
            image_placeholder.image(image, caption="Generated Image", use_container_width=True)
            
            # Add download button
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            st.download_button(
                label="⬇️ Download Image",
                data=buf.getvalue(),
                file_name="disney_style_output.png",
                mime="image/png"
            )
            
        except Exception as e:
            loading_container.empty()
            st.error(f"Error generating image: {str(e)}")
    elif generate_button:
        st.warning("⚠️ Please enter a prompt first.") 