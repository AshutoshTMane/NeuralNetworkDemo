import streamlit as st

from components.neural_network_creator import render_creator_section
from components.trainer_section import render_training_section
from components.evaluation_section import render_evaluation_section

# Initialize session state for navigation
if "selected_section" not in st.session_state:
    st.session_state.selected_section = "Creator"  # Default section

def set_section(section_name):
    """Updates the selected section in session state"""
    st.session_state.selected_section = section_name

def sidebar():
    """Dynamically updates sidebar content based on selected section"""
    st.sidebar.title("Neural Network Visualizer")

    # Display section based on what user has selected
    section = st.session_state.selected_section

    if section == "Creator":
        render_creator_section()
    elif section == "Training":
        render_training_section()
    elif section == "Evaluation":
        render_evaluation_section()

# Main UI: Buttons to change sidebar content
st.title("Neural Network Visualizer")
st.button("Go to Creator", on_click=set_section, args=("Creator",))
st.button("Go to Training", on_click=set_section, args=("Training",))
st.button("Go to Evaluation", on_click=set_section, args=("Evaluation",))

# Render sidebar dynamically
sidebar()
