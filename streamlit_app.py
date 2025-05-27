import streamlit as st
from dotenv import load_dotenv
from src.utils.template_loader import load_css, load_template
from src.components.text_to_image import render_text_to_image_tab
from src.components.image_to_image import render_image_to_image_tab
from src.components.inpainting import render_inpainting_tab
from src.components.refining import render_refining_tab
from src.components.two_text_encoders import render_two_text_encoders_tab

# Load environment variables from .env file
load_dotenv()

# Set page config
st.set_page_config(
    page_title="AI Image Generator",
    page_icon="🎨",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Load CSS
load_css()

# Set theme colors for better visibility
st.markdown("""
    <style>
        .stTabs [data-baseweb="tab-list"] {
            gap: 2rem;
        }
        .stTabs [data-baseweb="tab"] {
            height: 4rem;
            white-space: pre-wrap;
            background-color: #262730;
            border-radius: 4px 4px 0 0;
            gap: 1rem;
            padding-top: 10px;
            padding-bottom: 10px;
        }
        .stTabs [aria-selected="true"] {
            background-color: #4B4B4B;
        }
        .stTabs [data-baseweb="tab-panel"] {
            padding-top: 2rem;
        }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.title("🎨 AI Image Generator")
st.markdown(load_template("cards").split("<!-- Description Card -->")[1], unsafe_allow_html=True)

# Tab selection
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Text to Image", "Image to Image", "Inpainting", "Refining", "Two Text-Encoders"])

# Render tabs
with tab1:
    render_text_to_image_tab()

with tab2:
    render_image_to_image_tab()

with tab3:
    render_inpainting_tab()

with tab4:
    render_refining_tab()

with tab5:
    render_two_text_encoders_tab() 