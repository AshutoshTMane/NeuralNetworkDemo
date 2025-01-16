import streamlit as st
from PIL import Image
from model.evaluation import evaluate_model, predict_single_sample
import torchvision.transforms as transforms
from streamlit_drawable_canvas import st_canvas 

def render_evaluation_section():
    st.header("Evaluation and Prediction")

    model, test_loader = st.session_state.get('model', None), st.session_state.get('test_loader', None)

    # Tabs for evaluation and prediction
    tab1, tab2 = st.tabs(["Evaluate Model", "Predict Single Sample"])

    # Evaluation Tab
    with tab1:
        st.subheader("Batch Evaluation")
        test_data = st.file_uploader("Upload Test Data", type=["csv", "txt"])
        if st.button("Evaluate Model"):
            if test_data is not None:
                evaluate_model(model, test_loader)
            else:
                st.error("Please upload test data.")

    # Prediction Tab
    with tab2:
        st.subheader("Predict Single Sample")

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

