import streamlit as st

from components.neural_network_creator import render_creator_section
from components.trainer_section import render_training_section
from components.evaluation_section import render_evaluation_section

# Initialize session state if not already set
if "selected_info" not in st.session_state:
    st.session_state.selected_info = "Welcome! Click an item to see details."

def update_sidebar_info(new_info):
    """Updates the sidebar based on interactions in the main content."""
    st.session_state.selected_info = new_info

def sidebar():
    """Sidebar that updates dynamically based on user interactions in the main content."""
    st.sidebar.title("Dynamic Info Panel")
    st.sidebar.write(st.session_state.selected_info) 