import streamlit as st
import streamlit.components.v1 as components

from model.training import train_model
from components.visualization import visualize_interactive_model


def sidebar():
    # Sidebar for user input
    st.sidebar.header("Neural Network Configuration")

    # Persist state with Streamlit's session state
    if "hidden_layers" not in st.session_state:
        st.session_state.hidden_layers = []
    if "activations" not in st.session_state:
        st.session_state.activations = []

    input_size = st.sidebar.number_input("Input Layer Size", min_value=1, value=None, step=1)
    output_size = st.sidebar.number_input("Output Layer Size", min_value=1, value=None, step=1)

    # Button to add a new hidden layer
    if st.sidebar.button("Add Hidden Layer"):
        st.session_state.hidden_layers.append(256)  # Default size
        st.session_state.activations.append("ReLU")  # Default activation function

    # Display, edit, or delete existing hidden layers
    for i, (layer_size, activation) in enumerate(zip(st.session_state.hidden_layers, st.session_state.activations)):
        with st.sidebar.expander(f"Hidden Layer {i+1} ({layer_size} nodes, {activation})"):
            new_size = st.number_input(f"Edit Size for Hidden Layer {i+1}", min_value=1, value=layer_size, step=1, key=f"size_{i}")
            new_activation = st.selectbox(
                f"Activation for Hidden Layer {i+1}",
                ["ReLU", "Sigmoid", "Tanh"],
                index=["ReLU", "Sigmoid", "Tanh"].index(activation),
                key=f"activation_{i}"
            )
            if st.button(f"Save Changes to Layer {i+1}", key=f"save_{i}"):
                st.session_state.hidden_layers[i] = new_size
                st.session_state.activations[i] = new_activation

            if st.button(f"Delete Layer {i+1}", key=f"delete_{i}"):
                st.session_state.hidden_layers.pop(i)
                st.session_state.activations.pop(i)
                break  # Prevent index issues after deletion

    # Visualize model button
    if st.sidebar.button("Build Model"):
        if st.session_state.hidden_layers:
            html_file = visualize_interactive_model(input_size, st.session_state.hidden_layers, output_size)
            with open(html_file, "r") as f:
                html_content = f.read()
            components.html(html_content, height=600, width=800)
        else:
            st.error("Please add at least one hidden layer.")
        
    
    if st.sidebar.button("Train Model"):
        train_model(input_size, output_size)