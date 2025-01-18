import streamlit as st
import torch
import torchvision.models as models

def render_pretrained_model_section():
    st.subheader("Load Pretrained Model")

    if "pretrained_model_open" not in st.session_state:
        st.session_state["pretrained_model_open"] = False

    if st.button("Select Pretrained Model Options"):
        st.session_state["pretrained_model_open"] = not st.session_state["pretrained_model_open"]

    if st.session_state["pretrained_model_open"]:
        col1, col2 = st.columns(2)

        with col1:
            uploaded_file = st.file_uploader("Upload Pretrained Model", type=["h5", "pt", "pth"])
            if uploaded_file:
                try:
                    # Save the uploaded file temporarily
                    with open("temp_model_file.pt", "wb") as f:
                        f.write(uploaded_file.read())
                    
                    # Load the model
                    model = torch.load("temp_model_file.pt")
                    
                    st.success("Pretrained model uploaded and loaded successfully!")
                    
                    # Display model structure
                    st.write("Model structure:")
                    st.text(model)
                    
                    # Optional: Save the model in session state for further use
                    st.session_state["current_model"] = model
                    
                except Exception as e:
                    st.error(f"Error loading the model: {e}")

        with col2:
            st.subheader("Available Pretrained Models")
            pretrained_models = ["ResNet18", "MobileNetV2", "VGG16"]
            model_mapping = {
                "ResNet18": models.resnet18(pretrained=True),
                "MobileNetV2": models.mobilenet_v2(pretrained=True),
                "VGG16": models.vgg16(pretrained=True),
            }
            selected_model = st.selectbox("Choose a Pretrained Model", pretrained_models)

            if st.button("Load Selected Model"):
                if selected_model in model_mapping:
                    try:
                        # Load the selected pretrained model
                        model = model_mapping[selected_model]

                        # Store the model in session state
                        st.session_state["current_model"] = model
                        st.session_state["model_selected"] = True

                        # Provide user feedback
                        st.success(f"Loaded {selected_model} successfully!")
                        
                        # Display the model structure
                        st.write("Model structure:")
                        st.text(model)
                    except Exception as e:
                        st.error(f"Error loading {selected_model}: {e}")
                else:
                    st.error("Selected model is not available.")