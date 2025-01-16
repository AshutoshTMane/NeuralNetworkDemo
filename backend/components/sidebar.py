import streamlit as st

from components.neural_network_creator import render_creator_section
from components.trainer_section import render_training_section
from components.evaluation_section import render_evaluation_section

def sidebar():
    # Navigation menu
    st.sidebar.title("Neural Network Visualizer")
    menu = st.sidebar.radio("Navigation", ["Creator", "Training", "Evaluation"])

    # Section-specific controls
    if menu == "Creator":
        render_creator_section()
    elif menu == "Training":
        render_training_section()
    elif menu == "Evaluation":
        render_evaluation_section()
