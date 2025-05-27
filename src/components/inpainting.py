import streamlit as st
import torch
import io
from PIL import Image
from src.pipelines.model_loader import load_model
from src.utils.template_loader import load_template
from src.config.constants import (
    MODEL_CONFIGS, DEFAULT_SEED, DEFAULT_STEPS, DEFAULT_GUIDANCE_SCALE, DEFAULT_STRENGTH, SUPPORTED_IMAGE_FORMATS
)

def make_image_grid(images, rows=1, cols=3):
    w, h = images[0].size
    grid = Image.new('RGB', size=(cols * w, rows * h))
    for i, image in enumerate(images):
        grid.paste(image.resize((w, h)), box=(i % cols * w, i // cols * h))
    return grid

def render_inpainting_tab():
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown(load_template("cards").split("<!-- Input Card -->")[1].split("<!-- Parameters Card -->")[0], unsafe_allow_html=True)
        selected_model = st.selectbox(
            "Select Style (Inpainting):",
            options=list(MODEL_CONFIGS.keys()),
            format_func=lambda x: x,
            key="inpaint_model"
        )
        model_config = MODEL_CONFIGS[selected_model]
        uploaded_file = st.file_uploader("Upload an image to inpaint:", type=SUPPORTED_IMAGE_FORMATS)
        mask_file = st.file_uploader("Upload a mask image (white areas will be inpainted):", type=SUPPORTED_IMAGE_FORMATS)
        if uploaded_file is not None:
            init_image = Image.open(uploaded_file)
            st.image(init_image, caption="Original Image", use_container_width=True)
        else:
            init_image = None
        if mask_file is not None:
            mask_image = Image.open(mask_file)
            st.image(mask_image, caption="Mask Image", use_container_width=True)
        else:
            mask_image = None
        prompt = st.text_area(
            "Enter your prompt:",
            height=80,
            placeholder=f"Describe what you want to generate in the masked area... (e.g., '{model_config['default_prompt']}')",
            value=model_config["default_prompt"],
            key="inpaint_prompt"
        )
        st.markdown(load_template("cards").split("<!-- Parameters Card -->")[1].split("<!-- Output Card -->")[0], unsafe_allow_html=True)
        num_inference_steps = st.slider("Number of inference steps", 20, 100, 75, key="inpaint_steps")
        guidance_scale = st.slider("Guidance scale", 1.0, 20.0, DEFAULT_GUIDANCE_SCALE, key="inpaint_guidance")
        high_noise_frac = st.slider("Refiner high noise fraction", 0.0, 1.0, 0.7, key="inpaint_high_noise_frac")
        seed = st.number_input("Seed (for reproducibility)", value=DEFAULT_SEED, step=1, key="inpaint_seed")
        generate_button = st.button("🎨 Inpaint Image", type="primary", key="inpaint_generate")

    with col2:
        st.markdown(load_template("cards").split("<!-- Output Card -->")[1].split("<!-- Description Card -->")[0], unsafe_allow_html=True)
        image_placeholder = st.empty()
        if generate_button and prompt and init_image is not None and mask_image is not None:
            loading_container = st.empty()
            loading_container.markdown(load_template("loading"), unsafe_allow_html=True)
            try:
                with st.spinner("Loading models..."):
                    pipes = load_model(selected_model, inpainting=True)
                    base_pipe = pipes["base"]
                    refiner_pipe = pipes["refiner"]
                generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(seed)
                # BASE: output_type="latent"
                base_result = base_pipe(
                    prompt=prompt,
                    image=init_image,
                    mask_image=mask_image,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    denoising_end=high_noise_frac,
                    output_type="latent",
                    generator=generator,
                    return_dict=True
                )
                latents = base_result.images
                # REFINER: input latent, output PIL
                refined_result = refiner_pipe(
                    prompt=prompt,
                    image=latents,
                    mask_image=mask_image,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    denoising_start=high_noise_frac,
                    generator=generator,
                    return_dict=True
                )
                refined_image = refined_result.images[0]
                loading_container.empty()
                # --- Compose grid ---
                w, h = refined_image.size
                init_resized = init_image.resize((w, h))
                mask_resized = mask_image.resize((w, h))
                grid = make_image_grid([init_resized, mask_resized, refined_image], rows=1, cols=3)
                image_placeholder.image(grid, caption="Original | Mask | Inpainted", use_container_width=True)
                buf = io.BytesIO()
                refined_image.save(buf, format="PNG")
                st.download_button(
                    label="⬇️ Download Inpainted Image",
                    data=buf.getvalue(),
                    file_name=f"{selected_model.lower()}_inpainted_refined.png",
                    mime="image/png"
                )
            except Exception as e:
                loading_container.empty()
                st.error(f"Error inpainting image: {str(e)}")
        elif generate_button:
            if init_image is None:
                st.warning("⚠️ Please upload an image first.")
            elif mask_image is None:
                st.warning("⚠️ Please upload a mask image first.")
            elif not prompt:
                st.warning("⚠️ Please enter a prompt first.") 