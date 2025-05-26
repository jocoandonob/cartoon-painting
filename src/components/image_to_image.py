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

def render_image_to_image_tab():
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
        uploaded_file = st.file_uploader("Upload an image to transform:", type=SUPPORTED_IMAGE_FORMATS)
        
        if uploaded_file is not None:
            # Display the uploaded image
            init_image = Image.open(uploaded_file)
            st.image(init_image, caption="Original Image", use_container_width=True)
        
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
        
        num_inference_steps = st.slider("Number of inference steps", 20, 50, DEFAULT_STEPS, key="img2img_steps")
        guidance_scale = st.slider("Guidance scale", 1.0, 20.0, DEFAULT_GUIDANCE_SCALE, key="img2img_guidance")
        strength = st.slider("Transformation strength", 0.0, 1.0, DEFAULT_STRENGTH, key="img2img_strength")
        
        # Seed control
        seed = st.number_input("Seed (for reproducibility)", value=DEFAULT_SEED, step=1, key="img2img_seed")
        
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
                            f'<span style="color: #FFD700">Transforming image... {progress}%</span>', 
                            unsafe_allow_html=True
                        )
                        time_text.markdown(
                            f'<span style="color: #FFD700">Time remaining: {time_str}</span>', 
                            unsafe_allow_html=True
                        )
                
                # Set deterministic seed
                generator = torch.Generator(device="cpu").manual_seed(seed)
                
                # Generate image with base model
                with torch.no_grad():
                    # First stage: Generate with base model
                    progress_text.markdown(
                        '<span style="color: #FFD700">Stage 1: Generating with base model...</span>', 
                        unsafe_allow_html=True
                    )
                    base_image = base_pipe(
                        prompt=prompt,
                        image=init_image,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        strength=strength,
                        generator=generator,
                        callback=progress_callback,
                        callback_steps=1
                    ).images[0]
                    
                    # Second stage: Refine with refiner model
                    progress_text.markdown(
                        '<span style="color: #FFD700">Stage 2: Refining with refiner model...</span>', 
                        unsafe_allow_html=True
                    )
                    refined_image = refiner_pipe(
                        prompt=prompt,
                        image=base_image,
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
                
                # Display refined image
                image_placeholder.image(refined_image, caption="Refined Image", use_container_width=True)
                
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
                st.error(f"Error transforming image: {str(e)}")
        elif generate_button:
            if not uploaded_file:
                st.warning("⚠️ Please upload an image first.")
            elif not prompt:
                st.warning("⚠️ Please enter a prompt first.") 