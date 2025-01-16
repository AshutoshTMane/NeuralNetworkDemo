import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris, load_digits, load_breast_cancer

def render_dataset_selection_section():
    st.header("Dataset Selection")

    if "dataset_selected" not in st.session_state:
        st.session_state["dataset_selected"] = False

    st.subheader("Choose Your Dataset")
    col1, col2 = st.columns(2)

    with col1:
        uploaded_file = st.file_uploader("Upload Your Dataset", type=["csv", "xlsx", "json"])
        if uploaded_file:
            st.session_state["dataset_selected"] = True
            st.success("Dataset uploaded successfully!")
            st.write("Dataset preview:")
            # Display a preview of the uploaded dataset
            try:
                if uploaded_file.name.endswith(".csv"):
                    df = pd.read_csv(uploaded_file)
                elif uploaded_file.name.endswith(".xlsx"):
                    df = pd.read_excel(uploaded_file)
                elif uploaded_file.name.endswith(".json"):
                    df = pd.read_json(uploaded_file)
                st.write(df.head())
                st.session_state["dataset"] = df
            except Exception as e:
                st.error(f"Error reading dataset: {e}")

    with col2:
        st.subheader("Use a Predefined Dataset")
        predefined_datasets = ["Iris", "MNIST", "Breast Cancer"]
        selected_dataset = st.selectbox("Select a Predefined Dataset", predefined_datasets)
        if st.button("Load Predefined Dataset"):
            st.session_state["dataset_selected"] = True
            st.success(f"Loaded {selected_dataset} dataset successfully!")

            if selected_dataset == "Iris":
                data = load_iris()
                df = pd.DataFrame(data.data, columns=data.feature_names)
                df['target'] = data.target
            elif selected_dataset == "MNIST":
                data = load_digits()
                df = pd.DataFrame(data.data)
                df['target'] = data.target
            elif selected_dataset == "Breast Cancer":
                data = load_breast_cancer()
                df = pd.DataFrame(data.data, columns=data.feature_names)
                df['target'] = data.target

            #st.write("Dataset preview:")
            #st.write(df.sample(1))
            st.session_state["dataset"] = df

    if not st.session_state["dataset_selected"]:
        st.warning("Please select a dataset to proceed.")


def selected_dataset():
    if not st.session_state.get("dataset_selected", False):
        st.error("No dataset selected. Please select a dataset before creating a model.")
        return None