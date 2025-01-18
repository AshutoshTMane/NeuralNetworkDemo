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
    
    if "input_size" not in st.session_state or "output_size" not in st.session_state:
        # Center-align the success message
        st.markdown(
            f'<div style="text-align: center; color: #f39c12;">{ "Please select a dataset and configure the model before training." }</div>',
            unsafe_allow_html=True
        )

    # Train the model when the button is clicked
    if st.button("Train Model", key="train_button"):
        # Ensure the dataset and model are selected/created
        if not st.session_state.get("dataset_selected", False) or not st.session_state.get("model_selected", False):
            st.error("No dataset or model selected. Please select both before training.")
        else:
            # Retrieve the pre-created model from session state
            model = st.session_state.get("current_model", None)
            if model is None:
                st.error("Model not found. Please create a model before training.")
                return
            
            # Debug output to verify training parameters
            st.write(f"Training with: epochs={epochs}, learning_rate={learning_rate}")

            # Call the training function with the retrieved model
            train_model(model, epochs, learning_rate)
            st.success("Model training complete!")
