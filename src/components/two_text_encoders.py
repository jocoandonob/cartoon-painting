import streamlit as st
import torch
import io
from PIL import Image
from src.utils.template_loader import load_template
from diffusers import StableDiffusionXLPipeline

def render_two_text_encoders_tab():
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown(load_template("cards").split("<!-- Input Card -->")[1].split("<!-- Parameters Card -->")[0], unsafe_allow_html=True)
        prompt = st.text_area(
            "Prompt (OAI CLIP-ViT/L-14):",
            height=68,
            value="Astronaut in a jungle, cold color palette, muted colors, detailed, 8k",
            key="twoenc_prompt"
        )
        prompt_2 = st.text_area(
            "Prompt 2 (OpenCLIP-ViT/bigG-14):",
            height=68,
            value="Van Gogh painting",
            key="twoenc_prompt2"
        )
        num_inference_steps = st.slider("Number of inference steps", 20, 50, 30, key="twoenc_steps")
        guidance_scale = st.slider("Guidance scale", 1.0, 20.0, 7.5, key="twoenc_guidance")
        seed = st.number_input("Seed (for reproducibility)", value=42, step=1, key="twoenc_seed")
        generate_button = st.button("🎨 Generate Image", type="primary", key="twoenc_generate")

    with col2:
        st.markdown(load_template("cards").split("<!-- Output Card -->")[1].split("<!-- Description Card -->")[0], unsafe_allow_html=True)
        image_placeholder = st.empty()
        if generate_button and prompt and prompt_2:
            loading_container = st.empty()
            loading_container.markdown(load_template("loading"), unsafe_allow_html=True)
            try:
                with st.spinner("Loading SDXL pipeline..."):
                    pipe = StableDiffusionXLPipeline.from_pretrained(
                        "stabilityai/stable-diffusion-xl-base-1.0",
                        torch_dtype=torch.float32,
                        variant="fp16",
                        use_safetensors=True
                    ).to("cuda" if torch.cuda.is_available() else "cpu")
                generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(seed)
                result = pipe(
                    prompt=prompt,
                    prompt_2=prompt_2,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                    return_dict=True
                )
                image = result.images[0]
                loading_container.empty()
                image_placeholder.image(image, caption="SDXL Two Text-Encoders Result", use_container_width=True)
                buf = io.BytesIO()
                image.save(buf, format="PNG")
                st.download_button(
                    label="⬇️ Download Image",
                    data=buf.getvalue(),
                    file_name="sdxl_two_text_encoders.png",
                    mime="image/png"
                )
            except Exception as e:
                loading_container.empty()
                st.error(f"Error generating image: {str(e)}")
        elif generate_button:
            if not prompt:
                st.warning("⚠️ Please enter the main prompt.")
            elif not prompt_2:
                st.warning("⚠️ Please enter the secondary prompt.") 