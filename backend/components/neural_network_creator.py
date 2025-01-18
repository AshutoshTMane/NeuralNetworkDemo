import streamlit as st
import pandas as pd

from model.creation import create_model
from components.visualization import visualize_interactive_model

def get_dataset_info():
    """Returns the input size and output size based on the dataset."""
    if "dataset" in st.session_state:
        df = st.session_state["dataset"]

        if isinstance(df, pd.DataFrame):  # For tabular data (e.g., Iris)
            input_size = df.shape[1] - 1  # Exclude target column
            output_size = len(df["target"].unique())  # Number of unique classes for classification
        else:
            st.error("Unsupported dataset type. Please use a tabular dataset.")
            # Handle other cases like image or audio datasets (e.g., MNIST, SpeechCommands)
            input_size = 784  # Default for MNIST (28x28 images)
            output_size = 10  # Default output size for MNIST (10 classes)
        
        return input_size, output_size
    else:
        return 784, 10  # Default values if no dataset is selected

def render_creator_section():
    st.subheader("Model Maker")

    # Automatically set input and output sizes based on the dataset
    input_size, output_size = get_dataset_info()

    # Sidebar inputs for model configuration
    if "hidden_layers" not in st.session_state:
        st.session_state.hidden_layers = []
    if "activations" not in st.session_state:
        st.session_state.activations = []

    # Display automatically detected input/output sizes
    st.write(f"Input Size: {input_size}")
    st.write(f"Output Size: {output_size}")

    # Add or edit hidden layers
    if st.button("Add Hidden Layer"):
        st.session_state.hidden_layers.append(256)
        st.session_state.activations.append("ReLU")

    # Inject CSS to open all expanders by default
    st.markdown(
        """
        <style>
        .streamlit-expanderHeader {
            background-color: #f0f0f0; /* Optional: styling */
        }
        .streamlit-expanderContent {
            display: block !important; /* Make expander open by default */
        }
        </style>
        """, unsafe_allow_html=True
    )

    # Create hidden layer inputs with delete functionality
    for i, (layer_size, activation) in enumerate(zip(st.session_state.hidden_layers, st.session_state.activations)):
        # Dynamic expander title showing neurons and activation
        expander_title = f"Hidden Layer {i+1} - Neurons: {layer_size}, Activation: {activation}"
        
        with st.expander(expander_title):
            col1, col2 = st.columns([10, 1])  # 10 for input size, 1 for delete button
            
            with col1:
                st.session_state.hidden_layers[i] = st.number_input(
                    f"Size for Layer {i+1}",
                    value=layer_size,
                    key=f"layer_size_{i}"  # Unique key for each layer size
                )
                st.session_state.activations[i] = st.selectbox(
                    f"Activation for Layer {i+1}",
                    ["ReLU", "Sigmoid", "Tanh"],
                    index=["ReLU", "Sigmoid", "Tanh"].index(activation),
                    key=f"activation_{i}"  # Unique key for each activation function
                )
            
            # Delete button in the second column (styled as an "X")
            with col2:
                delete_key = f"delete_{i}"  # Unique delete button key
                if st.button("❌", key=delete_key):
                    # Remove the layer and its activation from session state
                    st.session_state.hidden_layers.pop(i)
                    st.session_state.activations.pop(i)

    if st.button("Build Model"):
        if not st.session_state.get("dataset_selected", False):
            st.error("No dataset selected. Please select a dataset before creating a model.")
        else:
            model = create_model(input_size, st.session_state.hidden_layers, output_size, st.session_state.activations)
            st.text(model)
            st.session_state["current_model"] = model
            html_file = visualize_interactive_model(input_size, st.session_state.hidden_layers, output_size)
            with open(html_file, "r") as f:
                html_content = f.read()
            st.components.v1.html(html_content, height=600, width=800)
