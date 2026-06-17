import streamlit as st
import pandas as pd

from components.data_exploration import display_dataset_information
from components.visualization import display_visualizations

st.title("AI Business Intelligence Copilot")

st.header("Upload Your Dataset")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.success("File uploaded successfully!")

    st.header("Dataset Preview")
    st.dataframe(df.head())

    # Data exploration
    display_dataset_information(df)

    # Data visualization
    display_visualizations(df)