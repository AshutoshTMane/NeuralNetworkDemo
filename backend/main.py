import streamlit as st

# Importing components
from components.neural_network_creator import render_creator_section
from components.trainer_section import render_training_section
from components.evaluation_section import render_evaluation_section

st.set_page_config(layout="wide")

# Add custom CSS for styling
st.markdown("""
    <style>
    .top-bar {
        text-align: center;
        padding: 10px;
        color: white;
        font-family: 'Arial Black', sans-serif;
        background-color: #264653;
        border-radius: 10px;
    }
    
    /* Style for section containers to make them look like cards */
    .st-expanderHeader {
        background-color: #2a9d8f;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 12px;
        text-align: center;
        cursor: pointer;
        transition: background-color 0.3s ease;
    }
    
    .st-expanderHeader:hover {
        background-color: #21867a;
    }

    /* Style for content inside the expander */
    .st-expanderContent {
        padding: 15px;
        border-radius: 8px;
        background-color: #f0f0f0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
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
""", unsafe_allow_html=True)

def main():
    # Top bar navigation
    st.markdown('<div class="top-bar"><h2>Neural Network Visualizer</h2></div>', unsafe_allow_html=True)

    # Define sections and their corresponding functions
    sections = {
        "Creator": render_creator_section,
        "Training": render_training_section,
        "Evaluation": render_evaluation_section,
    }

    # Layout: Three horizontal sections
    col1, col2, col3 = st.columns(3)

    # Create expandable sections with styling
    with col1:
        with st.expander("Creator", expanded=True):  # You can set expanded=True to default open
            render_creator_section()

    with col2:
        with st.expander("Training", expanded=True):  # You can set expanded=True to default open
            render_training_section()

    with col3:
        with st.expander("Evaluation", expanded=True):  # You can set expanded=True to default open
            render_evaluation_section()

if __name__ == "__main__":
    main()
