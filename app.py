import hashlib
from io import BytesIO
from pathlib import Path

import pandas as pd
import streamlit as st

from components.data_cleaning import (
    display_reset_section,
    initialize_dataframes,
)
from components.data_exploration import display_dataset_information
from components.data_qa import display_data_qa
from components.visualization import display_visualizations


SAMPLE_PATH = (
    Path(__file__).resolve().parent / "data" / "sample_sales.csv"
)


st.set_page_config(
    page_title="AI Business Intelligence Copilot",
    page_icon="📊",
    layout="wide",
)

st.title("AI Business Intelligence Copilot")

st.write(
    "Upload, prepare, explore, visualize, and ask questions "
    "about business data through one interactive application."
)

dataset_bytes = None
file_name = None

with st.sidebar:
    st.header("Dataset")

    dataset_source = st.radio(
        "Choose how to begin",
        options=["Try sample data", "Upload a CSV"],
        key="dataset_source",
    )

    if dataset_source == "Try sample data":
        st.caption(
            "Explore fictional sales data across six months, "
            "with duplicates and missing values."
        )

        try:
            dataset_bytes = SAMPLE_PATH.read_bytes()
            file_name = SAMPLE_PATH.name

        except OSError:
            st.error(
                "The sample dataset could not be loaded. "
                "Please upload a CSV instead."
            )

    else:
        uploaded_file = st.file_uploader(
            "Upload a CSV file",
            type=["csv"],
        )

        if uploaded_file is not None:
            dataset_bytes = uploaded_file.getvalue()
            file_name = uploaded_file.name


if dataset_bytes is not None:
    try:
        uploaded_df = pd.read_csv(BytesIO(dataset_bytes))

        # Detect changes to the source or file contents.
        dataset_id = (
            f"{dataset_source}:"
            f"{hashlib.sha256(dataset_bytes).hexdigest()}"
        )

        initialize_dataframes(
            df=uploaded_df,
            file_name=file_name,
            file_id=dataset_id,
        )

        working_df = st.session_state["working_df"]

        with st.sidebar:
            if dataset_source == "Try sample data":
                st.success("Sample dataset loaded")
            else:
                st.success("File uploaded successfully")

            st.write(f"**File:** {file_name}")
            st.write(f"**Rows:** {len(working_df):,}")
            st.write(f"**Columns:** {len(working_df.columns)}")

            st.caption(
                "Loading a different dataset starts "
                "a fresh working copy."
            )

        if dataset_source == "Try sample data":
            st.info(
                "You're exploring fictional sample sales data. "
                "Try removing duplicates, creating a chart, "
                "or asking 'What is the total revenue by Region?'"
            )

        overview_tab, cleaning_tab, visualization_tab, qa_tab = st.tabs(
            [
                "Overview",
                "Data Cleaning",
                "Visualization",
                "Q&A",
            ]
        )

        # Overview
        with overview_tab:
            st.header("Dataset Overview")
            st.subheader("Dataset Preview")

            if len(working_df) > 1:
                preview_rows = st.slider(
                    "Number of rows to preview",
                    min_value=1,
                    max_value=min(50, len(working_df)),
                    value=min(10, len(working_df)),
                    key=f"preview_rows_{dataset_id}",
                )
            else:
                preview_rows = len(working_df)

            st.dataframe(
                working_df.head(preview_rows),
                use_container_width=True,
            )

            display_dataset_information(working_df)

        # Data Cleaning
        with cleaning_tab:
            display_reset_section()

            st.divider()
            st.subheader("Download Working Dataset")

            st.download_button(
                label="Download cleaned CSV",
                data=st.session_state["working_df"]
                .to_csv(index=False)
                .encode("utf-8"),
                file_name=(
                    f"{file_name.rsplit('.', 1)[0]}_cleaned.csv"
                ),
                mime="text/csv",
            )

        working_df = st.session_state["working_df"]

        # Visualization
        with visualization_tab:
            display_visualizations(working_df)

        # Q&A
        with qa_tab:
            display_data_qa(working_df)

    except pd.errors.EmptyDataError:
        st.error("The CSV file is empty.")

    except pd.errors.ParserError:
        st.error(
            "The CSV file could not be parsed. "
            "Check that it uses a valid CSV format."
        )

    except UnicodeDecodeError:
        st.error(
            "The CSV file could not be read as UTF-8. "
            "Save it with UTF-8 encoding and upload it again."
        )

    except Exception:
        st.error(
            "The dataset could not be displayed. "
            "Try resetting the working dataset or loading "
            "another CSV."
        )

else:
    st.info(
        "Upload a CSV using the sidebar, or select "
        "'Try sample data' to explore the app."
    )

    st.subheader("Application Workflow")

    st.write(
        """
        1. Upload a CSV dataset or try the sample.
        2. Review its structure and data quality.
        3. Clean and prepare a separate working copy.
        4. Create visualizations from the working dataset.
        5. Answer questions about the working dataset.
        6. Download the prepared dataset.
        """
    )