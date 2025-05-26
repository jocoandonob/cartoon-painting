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
    DEFAULT_GUIDANCE_SCALE
)

def render_text_to_image_tab():
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
        
        num_inference_steps = st.slider("Number of inference steps", 20, 50, DEFAULT_STEPS)
        guidance_scale = st.slider("Guidance scale", 1.0, 20.0, DEFAULT_GUIDANCE_SCALE)
        
        # Seed control
        seed = st.number_input("Seed (for reproducibility)", value=DEFAULT_SEED, step=1)
        
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