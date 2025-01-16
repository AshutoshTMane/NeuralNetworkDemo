import streamlit as st

from components.visualization import visualize_interactive_model

def render_creator_section():
    st.header("Neural Network Creator")

    # Sidebar inputs for model configuration
    if "hidden_layers" not in st.session_state:
        st.session_state.hidden_layers = []
    if "activations" not in st.session_state:
        st.session_state.activations = []

    input_size = st.number_input("Input Layer Size", min_value=1, value=784, step=1)
    output_size = st.number_input("Output Layer Size", min_value=1, value=10, step=1)

    # Add or edit hidden layers
    if st.button("Add Hidden Layer"):
        st.session_state.hidden_layers.append(256)
        st.session_state.activations.append("ReLU")

    for i, (layer_size, activation) in enumerate(zip(st.session_state.hidden_layers, st.session_state.activations)):
        with st.expander(f"Hidden Layer {i+1}"):
            st.session_state.hidden_layers[i] = st.number_input(
                f"Size for Layer {i+1}",
                value=layer_size,
                key=f"layer_size_{i}"
            )
            st.session_state.activations[i] = st.selectbox(
                f"Activation for Layer {i+1}",
                ["ReLU", "Sigmoid", "Tanh"],
                index=["ReLU", "Sigmoid", "Tanh"].index(activation),
                key=f"activation_{i}"
            )

    if st.button("Build Model"):
        if not st.session_state.get("dataset_selected", False):
            st.error("No dataset selected. Please select a dataset before creating a model.")
        else:
            html_file = visualize_interactive_model(input_size, st.session_state.hidden_layers, output_size)
            with open(html_file, "r") as f:
                html_content = f.read()
            st.components.v1.html(html_content, height=600, width=800)
