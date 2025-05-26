import os
import streamlit as st
from src.config.constants import STATIC_DIR, TEMPLATES_DIR, CSS_DIR

def load_css():
    css_file = os.path.join(STATIC_DIR, CSS_DIR, "style.css")
    with open(css_file, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def load_template(template_name):
    template_file = os.path.join(TEMPLATES_DIR, f"{template_name}.html")
    with open(template_file, "r", encoding="utf-8") as f:
        return f.read() 