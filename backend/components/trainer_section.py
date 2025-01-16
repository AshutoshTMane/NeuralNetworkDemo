import streamlit as st

from model.training import train_model

def render_training_section():
    st.header("Training")

    # Store the slider's value in session state to avoid re-rendering
    if "epochs" not in st.session_state:
        st.session_state.epochs = 10  # Default value

    epochs = st.slider("Epochs", min_value=1, max_value=50, value=st.session_state.epochs, key="training_epochs_slider")
    st.session_state.epochs = epochs  # Update the value in session state

    learning_rate = st.number_input("Learning Rate", min_value=0.001, max_value=1.0, value=0.01, step=0.001, key="training_lr_input")

    if st.button("Train Model", key="train_button"):
        train_model(st.session_state.hidden_layers, epochs, learning_rate)
        st.success("Model training complete!")