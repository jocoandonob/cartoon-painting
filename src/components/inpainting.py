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
    DEFAULT_STRENGTH,
    SUPPORTED_IMAGE_FORMATS
)

def render_inpainting_tab():
    # Create two columns for input and output
    col1, col2 = st.columns([1, 1])

    with col1:
        # Input section
        st.markdown(load_template("cards").split("<!-- Input Card -->")[1].split("<!-- Parameters Card -->")[0], unsafe_allow_html=True)
        
        # Model selection
        selected_model = st.selectbox(
            "Select Style (Inpainting):",
            options=list(MODEL_CONFIGS.keys()),
            format_func=lambda x: x,
            key="inpaint_model"
        )
        
        # Load model configuration
        model_config = MODEL_CONFIGS[selected_model]
        
        # Image upload
        uploaded_file = st.file_uploader("Upload an image to inpaint:", type=SUPPORTED_IMAGE_FORMATS)
        
        if uploaded_file is not None:
            # Display the uploaded image
            init_image = Image.open(uploaded_file)
            st.image(init_image, caption="Original Image", use_container_width=True)
            
            # Mask upload
            mask_file = st.file_uploader("Upload a mask image (white areas will be inpainted):", type=SUPPORTED_IMAGE_FORMATS)
            if mask_file is not None:
                mask_image = Image.open(mask_file)
                st.image(mask_image, caption="Mask Image", use_container_width=True)
        
        # Text input with a larger text area
        prompt = st.text_area(
            "Enter your prompt:",
            height=80,
            placeholder=f"Describe what you want to generate in the masked area... (e.g., '{model_config['default_prompt']}')",
            value=model_config["default_prompt"],
            key="inpaint_prompt"
        )
        
        # Parameters section
        st.markdown(load_template("cards").split("<!-- Parameters Card -->")[1].split("<!-- Output Card -->")[0], unsafe_allow_html=True)
        
        num_inference_steps = st.slider("Number of inference steps", 20, 50, DEFAULT_STEPS, key="inpaint_steps")
        guidance_scale = st.slider("Guidance scale", 1.0, 20.0, DEFAULT_GUIDANCE_SCALE, key="inpaint_guidance")
        strength = st.slider("Inpainting strength", 0.0, 1.0, DEFAULT_STRENGTH, key="inpaint_strength")
        
        # Seed control
        seed = st.number_input("Seed (for reproducibility)", value=DEFAULT_SEED, step=1, key="inpaint_seed")
        
        # Generate button
        generate_button = st.button("🎨 Inpaint Image", type="primary", key="inpaint_generate")

    with col2:
        # Output section
        st.markdown(load_template("cards").split("<!-- Output Card -->")[1].split("<!-- Description Card -->")[0], unsafe_allow_html=True)
        
        # Placeholder for the generated image
        image_placeholder = st.empty()
        
        if generate_button and prompt and uploaded_file is not None and mask_file is not None:
            # Create loading animation
            loading_container = st.empty()
            loading_container.markdown(load_template("loading"), unsafe_allow_html=True)
            
            try:
                # Load the selected model
                with st.spinner("Loading model..."):
                    pipe = load_model(selected_model, inpainting=True)
                
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
                            f'<span style="color: #FFD700">Inpainting image... {progress}%</span>', 
                            unsafe_allow_html=True
                        )
                        time_text.markdown(
                            f'<span style="color: #FFD700">Time remaining: {time_str}</span>', 
                            unsafe_allow_html=True
                        )
                
                # Set deterministic seed
                generator = torch.Generator(device="cpu").manual_seed(seed)
                
                # Generate image with progress callback
                with torch.no_grad():
                    inpainted_image = pipe(
                        prompt=prompt,
                        image=init_image,
                        mask_image=mask_image,
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
                
                # Display inpainted image
                image_placeholder.image(inpainted_image, caption="Inpainted Image", use_container_width=True)
                
                # Add download button
                buf = io.BytesIO()
                inpainted_image.save(buf, format="PNG")
                st.download_button(
                    label="⬇️ Download Inpainted Image",
                    data=buf.getvalue(),
                    file_name=f"{selected_model.lower()}_style_inpainted.png",
                    mime="image/png"
                )
                
            except Exception as e:
                # Clear all loading states
                loading_container.empty()
                progress_bar.empty()
                progress_text.empty()
                time_text.empty()
                st.error(f"Error inpainting image: {str(e)}")
        elif generate_button:
            if not uploaded_file:
                st.warning("⚠️ Please upload an image first.")
            elif not mask_file:
                st.warning("⚠️ Please upload a mask image first.")
            elif not prompt:
                st.warning("⚠️ Please enter a prompt first.") 