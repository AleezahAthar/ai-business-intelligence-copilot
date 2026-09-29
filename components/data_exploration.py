import streamlit as st

def display_dataset_information(df):

    st.header("Dataset Information")

    rows, columns = df.shape

    st.write(f"Rows: {rows}")
    st.write(f"Columns: {columns}")

    st.subheader("Column Names")
    st.write(df.columns.tolist())

    
    st.subheader("Data Types")
    data_types = {
    "Column": df.columns,
    "Data Type": [str(dtype) for dtype in df.dtypes]
    }
    st.table(data_types)

    st.subheader("Summary Statistics")
    st.dataframe(df.describe())

    st.subheader("Missing Values")
    st.dataframe(df.isnull().sum())

    st.subheader("Duplicate Rows")
    st.write(f"Duplicate rows: {df.duplicated().sum()}")

    st.subheader("Unique Values Per Column")
    st.dataframe(df.nunique())

    st.subheader("Sample Rows")
    st.dataframe(df.sample(min(5, len(df))))
    
    st.subheader("Numeric Columns")
    st.write(df.select_dtypes(include="number").columns.tolist())

    st.subheader("Categorical Columns")
    st.write(df.select_dtypes(include="object").columns.tolist())


