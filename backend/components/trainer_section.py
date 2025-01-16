import streamlit as st

from model.training import train_model


def render_training_section():
    st.header("Training")

    # Training parameters
    epochs = st.slider("Epochs", min_value=1, max_value=50, value=10)
    learning_rate = st.number_input("Learning Rate", min_value=0.001, max_value=1.0, value=0.01, step=0.001)

    if st.button("Train Model"):
        train_model(st.session_state.hidden_layers, epochs, learning_rate)
        st.success("Model training complete!")
