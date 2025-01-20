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
        return 0, 0  # Default values if no dataset is selected

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

    # Function to calculate the suggested number of neurons for the next layer
    def calculate_next_layer_size():
        if len(st.session_state.hidden_layers) == 0:
            return max((input_size + output_size) // 2, 1)  # First hidden layer size
        else:
            # Progressively reduce the size based on the previous layer and output size
            prev_size = st.session_state.hidden_layers[-1]
            return max((prev_size + output_size) // 2, output_size)

    # Add a new hidden layer with default values
    if st.button("Add Hidden Layer"):
        suggested_neurons = calculate_next_layer_size()
        st.session_state.hidden_layers.append(suggested_neurons)
        st.session_state.activations.append("ReLU")

    # Create hidden layer inputs with delete and save functionality
    for i, (layer_size, activation) in enumerate(zip(st.session_state.hidden_layers, st.session_state.activations)):
        # The title dynamically updates based on saved values
        expander_title = f"Hidden Layer {i+1} - Neurons: {layer_size}, Activation: {activation}"
        
        with st.expander(expander_title, expanded=True):
            col1, col2, col3 = st.columns([6, 3, 1])  # Adjust column ratios for input, save, and delete
            
            with col1:
                # Inputs for layer size and activation
                updated_layer_size = st.number_input(
                    f"Size for Layer {i+1}",
                    value=layer_size,
                    key=f"layer_size_input_{i}"  # Unique key for this input
                )
                updated_activation = st.selectbox(
                    f"Activation for Layer {i+1}",
                    ["ReLU", "Sigmoid", "Tanh"],
                    index=["ReLU", "Sigmoid", "Tanh"].index(activation),
                    key=f"activation_input_{i}"  # Unique key for this selectbox
                )
            
            with col2:
                # Save button to commit changes to the layer
                if st.button("Save Changes", key=f"save_button_{i}"):
                    # Update the session state and refresh UI
                    st.session_state.hidden_layers[i] = updated_layer_size
                    st.session_state.activations[i] = updated_activation
            
            with col3:
                # Delete button to remove the layer
                if st.button("❌", key=f"delete_button_{i}"):
                    # Remove the layer and refresh UI
                    st.session_state.hidden_layers.pop(i)
                    st.session_state.activations.pop(i)

    if st.button("Build Model"):
        if not st.session_state.get("dataset_selected", False):
            st.error("No dataset selected. Please select a dataset before creating a model.")
        else:
            model = create_model(input_size, st.session_state.hidden_layers, output_size, st.session_state.activations)
            #st.text(model)
            st.session_state["current_model"] = model
            html_file = visualize_interactive_model(input_size, st.session_state.hidden_layers, output_size)
            with open(html_file, "r") as f:
                html_content = f.read()
            st.components.v1.html(html_content, height=600, width=800)
