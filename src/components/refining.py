import streamlit as st
import torch
import io
import time
from PIL import Image
from src.pipelines.model_loader import load_model
from src.utils.template_loader import load_template
from src.config.constants import (
    MODEL_CONFIGS,
    DEFAULT_SEED,
    DEFAULT_STEPS,
    DEFAULT_GUIDANCE_SCALE,
    SUPPORTED_IMAGE_FORMATS
)

def make_image_grid(images, rows=1, cols=2):
    """Create a grid of images."""
    w, h = images[0].size
    grid = Image.new('RGB', size=(cols * w, rows * h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i % cols * w, i // cols * h))
    return grid

def render_refining_tab():
    # Create two columns for input and output
    col1, col2 = st.columns([1, 1])

    with col1:
        # Input section
        st.markdown(load_template("cards").split("<!-- Input Card -->")[1].split("<!-- Parameters Card -->")[0], unsafe_allow_html=True)
        
        # Model selection
        selected_model = st.selectbox(
            "Select Style (Refining):",
            options=list(MODEL_CONFIGS.keys()),
            format_func=lambda x: x,
            key="refine_model"
        )
        
        # Load model configuration
        model_config = MODEL_CONFIGS[selected_model]
        
        # Text input with a larger text area
        prompt = st.text_area(
            "Enter your prompt:",
            height=80,
            placeholder=f"Describe what you want to generate... (e.g., '{model_config['default_prompt']}')",
            value=model_config["default_prompt"],
            key="refine_prompt"
        )
        
        # Parameters section
        st.markdown(load_template("cards").split("<!-- Parameters Card -->")[1].split("<!-- Output Card -->")[0], unsafe_allow_html=True)
        
        num_inference_steps = st.slider("Number of inference steps", 20, 50, DEFAULT_STEPS, key="refine_steps")
        guidance_scale = st.slider("Guidance scale", 1.0, 20.0, DEFAULT_GUIDANCE_SCALE, key="refine_guidance")
        denoising_end = st.slider("Denoising end", 0.0, 1.0, 0.8, key="refine_denoising_end")
        
        # Seed control
        seed = st.number_input("Seed (for reproducibility)", value=DEFAULT_SEED, step=1, key="refine_seed")
        
        # Generate button
        generate_button = st.button("🎨 Generate & Refine", type="primary", key="refine_generate")

    with col2:
        # Output section
        st.markdown(load_template("cards").split("<!-- Output Card -->")[1].split("<!-- Description Card -->")[0], unsafe_allow_html=True)
        
        # Placeholder for the generated images
        base_image_placeholder = st.empty()
        refined_image_placeholder = st.empty()
        
        if generate_button and prompt:
            # Create loading animation
            loading_container = st.empty()
            loading_container.markdown(load_template("loading"), unsafe_allow_html=True)
            
            try:
                # Load the selected model
                with st.spinner("Loading models..."):
                    pipes = load_model(selected_model, img2img=True)
                    base_pipe = pipes["base"]
                    refiner_pipe = pipes["refiner"]
                
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
                generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(seed)
                
                # Generate image with base model
                with torch.no_grad():
                    # BASE: output_type="latent" for refiner, decode for display
                    progress_text.markdown(
                        '<span style="color: #FFD700">Stage 1: Generating with base model...</span>', 
                        unsafe_allow_html=True
                    )
                    base_result = base_pipe(
                        prompt=prompt,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        denoising_end=denoising_end,
                        output_type="latent",
                        generator=generator,
                        callback=progress_callback,
                        callback_steps=1,
                        return_dict=True
                    )
                    latents = base_result.images
                    # Decode latent to PIL for display
                    base_pil = base_pipe.decode_latents(latents)
                    base_pil = base_pipe.numpy_to_pil(base_pil)[0]
                    base_image_placeholder.image(base_pil, caption="Base Image", use_container_width=True)
                    
                    # REFINER: input latent, output PIL
                    progress_text.markdown(
                        '<span style="color: #FFD700">Stage 2: Refining with refiner model...</span>', 
                        unsafe_allow_html=True
                    )
                    refined_result = refiner_pipe(
                        prompt=prompt,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        denoising_start=denoising_end,
                        image=latents,
                        generator=generator,
                        callback=progress_callback,
                        callback_steps=1,
                        return_dict=True
                    )
                    refined_image = refined_result.images[0]
                    refined_image_placeholder.image(refined_image, caption="Refined Image", use_container_width=True)
                
                # Clear loading animation and progress
                loading_container.empty()
                progress_bar.empty()
                progress_text.empty()
                time_text.empty()
                
                # Add download button
                buf = io.BytesIO()
                refined_image.save(buf, format="PNG")
                st.download_button(
                    label="⬇️ Download Refined Image",
                    data=buf.getvalue(),
                    file_name=f"{selected_model.lower()}_style_refined.png",
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
            if not prompt:
                st.warning("⚠️ Please enter a prompt first.") 