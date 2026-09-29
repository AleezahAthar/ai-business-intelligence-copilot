import hashlib

import pandas as pd
import streamlit as st

from components.data_cleaning import (
    display_reset_section,
    initialize_dataframes,
)
from components.data_exploration import display_dataset_information
from components.visualization import display_visualizations


# Browser tab and page settings
st.set_page_config(
    page_title="AI Business Intelligence Copilot",
    page_icon="📊",
    layout="wide",
)

# Main application heading
st.title("AI Business Intelligence Copilot")

st.write(
    "Upload, prepare, explore, and visualize business data "
    "through one interactive application."
)

# The uploaded file controls the dataset used throughout the app.
with st.sidebar:
    st.header("Dataset")

    uploaded_file = st.file_uploader(
        "Upload a CSV file",
        type=["csv"],
    )


if uploaded_file is not None:

    try:
        # Read the uploaded CSV into a DataFrame.
        uploaded_df = pd.read_csv(uploaded_file)

        # The file fingerprint detects changed contents even when
        # the new upload has the same filename.
        initialize_dataframes(
            df=uploaded_df,
            file_name=uploaded_file.name,
            file_id=hashlib.sha256(uploaded_file.getvalue()).hexdigest(),
        )

        # Use the working copy so cleaning does not alter the original.
        working_df = st.session_state["working_df"]

        with st.sidebar:
            st.success("File uploaded successfully")
            st.write(f"**File:** {uploaded_file.name}")
            st.write(f"**Rows:** {len(working_df):,}")
            st.write(f"**Columns:** {len(working_df.columns)}")

        overview_tab, cleaning_tab, visualization_tab = st.tabs(
            [
                "Overview",
                "Data Cleaning",
                "Visualization",
            ]
        )

        # -----------------------------
        # Overview tab
        # -----------------------------
        with overview_tab:
            st.header("Dataset Overview")

            st.subheader("Dataset Preview")

            # Starting at 1 allows CSVs with fewer than five rows.
            preview_rows = st.slider(
                "Number of rows to preview",
                min_value=1,
                max_value=min(50, len(working_df)),
                value=min(10, len(working_df)),
            )

            st.dataframe(
                working_df.head(preview_rows),
                use_container_width=True,
            )

            display_dataset_information(working_df)

        # -----------------------------
        # Data Cleaning tab
        # -----------------------------
        with cleaning_tab:
            display_reset_section()

            st.divider()
            st.subheader("Download Working Dataset")

            # Export the current working copy, including cleaning changes.
            st.download_button(
                label="Download cleaned CSV",
                data=st.session_state["working_df"]
                .to_csv(index=False)
                .encode("utf-8"),
                file_name=(
                    f"{uploaded_file.name.rsplit('.', 1)[0]}_cleaned.csv"
                ),
                mime="text/csv",
            )

        # Retrieve the DataFrame again after the cleaning controls.
        working_df = st.session_state["working_df"]

        # -----------------------------
        # Visualization tab
        # -----------------------------
        with visualization_tab:
            display_visualizations(working_df)

    except pd.errors.EmptyDataError:
        st.error("The uploaded CSV file is empty.")

    except pd.errors.ParserError:
        st.error(
            "The CSV file could not be parsed. "
            "Check that it uses a valid CSV format."
        )

    except Exception as error:
        st.error(f"An unexpected error occurred: {error}")

else:
    st.info("Upload a CSV file using the sidebar to begin.")

    st.subheader("Application Workflow")

    st.write(
        """
        1. Upload a CSV dataset.
        2. Review its structure and data quality.
        3. Clean and prepare a separate working copy.
        4. Create visualizations from the cleaned dataset.
        5. Use the cleaned dataset for future AI analysis.
        """
    )