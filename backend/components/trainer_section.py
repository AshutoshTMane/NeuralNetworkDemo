import streamlit as st
from model.training import train_model

def render_training_section():
    st.header("Training")

    # Store the slider's value in session state to avoid re-rendering
    if "epochs" not in st.session_state:
        st.session_state.epochs = 10  # Default value

    epochs = st.slider("Epochs", min_value=2, max_value=500, value=st.session_state.epochs, key="training_epochs_slider")
    st.session_state.epochs = epochs  # Update the value in session state

    learning_rate = st.number_input(
        "Learning Rate",
        min_value=0.0001,  # Minimum value allowed
        max_value=1.0,     # Maximum value allowed
        value=0.01,        # Default value
        step=0.0001,       # Smaller step for finer control
        format="%.4f",     # Display up to 4 decimal places
        key="training_lr_input"
    )
    
    if not st.session_state.get("dataset_selected", False) or not st.session_state.get("model_selected", False):
        # Center-align the error message
        st.markdown(
            f'<div style="text-align: center; color: #f39c12;">{"Please select a dataset and configure the model before training."}</div>',
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
            
                # Get dataset loader (use the one selected by the user)
            dataset = st.session_state.get("dataset", None)
            if dataset is None:
                st.error("No dataset loaded. Please load a dataset before training.")
                return
            
            # Debug output to verify training parameters
            st.write(f"Training with: epochs={epochs}, learning_rate={learning_rate}")

            # Call the training function with the retrieved model
            train_model(model, dataset, epochs, learning_rate)
