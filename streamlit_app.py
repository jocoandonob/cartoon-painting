import streamlit as st
from dotenv import load_dotenv
from src.utils.template_loader import load_css, load_template
from src.components.text_to_image import render_text_to_image_tab
from src.components.image_to_image import render_image_to_image_tab

# Load environment variables from .env file
load_dotenv()

# Set page config
st.set_page_config(
    page_title="Text to Image Generator",
    page_icon="🎨",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Load CSS
load_css()

# Title and description
st.title("🎨 Style Image Generator")
st.markdown(load_template("cards").split("<!-- Description Card -->")[1], unsafe_allow_html=True)

# Tab selection
tab1, tab2 = st.tabs(["Text to Image", "Image to Image"])

# Render tabs
with tab1:
    render_text_to_image_tab()

with tab2:
    render_image_to_image_tab() 