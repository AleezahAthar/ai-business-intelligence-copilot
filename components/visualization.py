import streamlit as st
import plotly.express as px


def display_visualizations(df):
    st.header("Data Visualization")

    chart_type = st.selectbox(
        "Select Chart Type",
        ["Bar Chart", "Line Chart", "Scatter Plot"]
    )

    x_axis = st.selectbox("Select X-axis", df.columns)
    y_axis = st.selectbox("Select Y-axis", df.columns)

    if chart_type == "Bar Chart":
        fig = px.bar(df, x=x_axis, y=y_axis)

    elif chart_type == "Line Chart":
        fig = px.line(df, x=x_axis, y=y_axis)

    else:
        fig = px.scatter(df, x=x_axis, y=y_axis)

    st.plotly_chart(fig, use_container_width=True)