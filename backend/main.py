import streamlit as st

from components.neural_network_creator import render_creator_section
from components.premade_datasets import render_dataset_selection_section
from components.pretrained_model import render_pretrained_model_section
from components.trainer_section import render_training_section
from components.evaluation_section import render_evaluation_section

st.set_page_config(layout="wide")

# Add custom CSS for styling
theme_page_1 = """
    <style>
    body {
        background-color: #ffffff;
        color: #333333;
    }
    .top-bar {
        text-align: center;
        padding: 10px;
        color: white;
        font-family: 'Arial Black', sans-serif;
        background-color: #264653;
        border-radius: 10px;
    }
    
    .stButton>button {
        background-color: #2a9d8f;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 12px;
        width: 100%;
        text-align: center;
        cursor: pointer;
        transition: background-color 0.3s ease, box-shadow 0.3s ease, color 0.3s ease;
    }

    .stButton>button:hover {
        background-color: #21867a;
    }

    .stButton>button:focus {
        outline: none;
        box-shadow: 0 0 15px 5px rgba(42, 157, 143, 0.8);
    }

    /* Additional styling for section containers */
    .section-container {
        margin: 10px;
        padding: 15px;
        border: 1px solid #ccc;
        border-radius: 10px;
        background-color: #f9f9f9;
    }

    
    </style>
"""

theme_page_2 = """
    <style>
    body {
        background-color: #1e1e1e;
        color: #ffffff;
    }
    .top-bar {
        text-align: center;
        padding: 10px;
        color: white;
        font-family: 'Arial Black', sans-serif;
        background-color: #6a5acd;
        border-radius: 10px;
    }
    
    </style>
"""

def render_page_1():

    st.markdown(theme_page_1, unsafe_allow_html=True)
    # Initialize session state for model creation and training status
    if "model_created" not in st.session_state:
        st.session_state["model_created"] = False
    if "model_trained" not in st.session_state:
        st.session_state["model_trained"] = False

    # Initialize session state for section visibility (default to False)
    if "show_creator" not in st.session_state:
        st.session_state["show_creator"] = False
    if "show_training" not in st.session_state:
        st.session_state["show_training"] = False
    if "show_evaluation" not in st.session_state:
        st.session_state["show_evaluation"] = False

    # Top bar navigation
    st.markdown('<div class="top-bar"><h2>Neural Network Visualizer</h2></div>', unsafe_allow_html=True)

    # Layout: Three horizontal columns for buttons
    col1, col2, col3 = st.columns(3)

    # Button to toggle the visibility of each section in columns
    with col1:
        if st.button("Creator"):
            st.session_state["show_creator"] = not st.session_state["show_creator"]

    with col2:
        if st.button("Training"):
            st.session_state["show_training"] = not st.session_state["show_training"]

    with col3:
        if st.button("Evaluation"):
            st.session_state["show_evaluation"] = not st.session_state["show_evaluation"]

    # Dynamic content rendering based on visibility status
    # Use a container to render content dynamically
    content = st.container()

    with content:
        active_sections = [
            st.session_state["show_creator"],
            st.session_state["show_training"],
            st.session_state["show_evaluation"]
        ]
        num_active_sections = sum(active_sections)

        # Adjust the layout dynamically based on the number of active sections
        if num_active_sections == 1:
            # Only one section active, so take up the full width
            if st.session_state["show_creator"]:
                render_dataset_selection_section()
                render_creator_section()
                render_pretrained_model_section()
            elif st.session_state["show_training"]:
                render_training_section()
            elif st.session_state["show_evaluation"]:
                render_evaluation_section()

        elif num_active_sections == 2:
            # Two sections active, each takes half the width
            col1, col2 = st.columns(2)
            with col1:
                if st.session_state["show_creator"]:
                    render_dataset_selection_section()
                    render_creator_section()
                    render_pretrained_model_section()
                elif st.session_state["show_training"]:
                    render_training_section()
                elif st.session_state["show_evaluation"]:
                    render_evaluation_section()
            with col2:
                if st.session_state["show_evaluation"]:
                    render_evaluation_section()
                elif st.session_state["show_training"]:
                    render_training_section()
                elif st.session_state["show_creator"]:
                    render_dataset_selection_section()
                    render_creator_section()
                    render_pretrained_model_section()

        elif num_active_sections == 3:
            # All three sections active, each takes one-third of the width
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.session_state["show_creator"]:
                    render_dataset_selection_section()
                    render_creator_section()
                    render_pretrained_model_section()
            with col2:
                if st.session_state["show_training"]:
                    render_training_section()
            with col3:
                if st.session_state["show_evaluation"]:
                    render_evaluation_section()


# Function to display Page 2
def render_page_2():
    st.markdown(theme_page_2, unsafe_allow_html=True)
    st.markdown('<div class="top-bar"><h2>Neural Network Visualizer - Page 2</h2></div>', unsafe_allow_html=True)
    render_training_section()
    render_evaluation_section()


# Main function
def main():
    # Select slider widget
    option = st.select_slider('Choose an option', options=['Off', 'On'])
    
    # Display different content based on the slider option
    if option == 'Off':
        render_page_1()
    else:
        render_page_2()



if __name__ == "__main__":
    main()
