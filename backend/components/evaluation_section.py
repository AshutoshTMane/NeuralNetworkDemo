import streamlit as st
from PIL import Image
from model.evaluation import evaluate_model, predict_single_sample
import torchvision.transforms as transforms
from streamlit_drawable_canvas import st_canvas 

def render_evaluation_section():
    st.header("Evaluation and Prediction")

    model, test_loader = st.session_state.get('trained_model', None), st.session_state.get('test_loader', None)

    # Tabs for evaluation and prediction
    tab1, tab2, tab3 = st.tabs(["User Test Data", "Premade Test Data", "Predict Single Sample"])

    # Tab 1: Evaluate with User Uploaded Test Data
    with tab1:
        st.subheader("Batch Evaluation: User Uploaded Test Data")
        test_data = st.file_uploader("Upload Test Data", type=["csv", "txt"])
        if st.button("Evaluate with Uploaded Data", key="evaluate_uploaded"):
            if model is None:
                st.error("Model not loaded. Please complete the model setup first.")
            elif test_data is not None:
                evaluate_model(model, test_data)  # Assuming `evaluate_model` can handle user-uploaded data
                st.success("Evaluation complete with uploaded test data!")
            else:
                st.error("Please upload test data.")

    # Tab 2: Evaluate with Preloaded Test Data
    with tab2:
        st.subheader("Batch Evaluation: Preloaded Test Data")
        test_loader = st.session_state.test_loader
        if st.button("Evaluate Preloaded Data", key="evaluate_preloaded"):
            if model is None:
                st.error("Model not loaded. Please complete the model setup first.")
            elif test_loader is None:
                st.error("No preloaded test data found. Please ensure test data is loaded.")
            else:
                evaluate_model(model, test_loader)
                st.success("Evaluation complete with preloaded test data!")

    # Prediction Tab
    with tab3:
        st.subheader("Predict Single Sample")

        # Check if the model is loaded before allowing prediction
        if model is None:
            st.error("Model not loaded. Please complete the model setup first.")
            return

        # Option to draw (for MNIST-like datasets)
        if st.checkbox("Draw Input (for MNIST)"):
            canvas_result = st_canvas(
                stroke_width=10,
                stroke_color="#000000",
                background_color="#FFFFFF",
                width=200,
                height=200,
                drawing_mode="freedraw",
                key="canvas"
            )

            if canvas_result.image_data is not None:
                # Process canvas image for prediction
                image = Image.fromarray(canvas_result.image_data.astype("uint8")).convert("L")
                image = image.resize((28, 28))  # Resize for MNIST
                transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

                if st.button("Predict Drawing"):
                    pred = predict_single_sample(model, image, transform)
                    st.write(f"**Predicted Digit:** {pred}")

        # Option to upload an image or data file
        else:
            uploaded_file = st.file_uploader("Upload Image or Data File")
            if uploaded_file is not None:
                if uploaded_file.type.startswith("image/"):
                    image = Image.open(uploaded_file).convert("L").resize((28, 28))
                    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

                    if st.button("Predict Uploaded Image"):
                        pred = predict_single_sample(model, image, transform)
                        st.write(f"**Predicted Digit:** {pred}")
                else:
                    st.error("Only image files are supported for single sample prediction.")

