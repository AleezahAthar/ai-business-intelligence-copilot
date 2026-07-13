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

# File upload belongs in the sidebar because it controls the entire application
with st.sidebar:
    st.header("Dataset")

    uploaded_file = st.file_uploader(
        "Upload a CSV file",
        type=["csv"],
    )


if uploaded_file is not None:

    try:
        # Read the newly uploaded file
        uploaded_df = pd.read_csv(uploaded_file)

        # Preserve an original copy and create a separate working copy
        initialize_dataframes(
            df=uploaded_df,
            file_name=uploaded_file.name,
        )

        # Always retrieve the latest working DataFrame from session state
        working_df = st.session_state["working_df"]

        # Show compact dataset information in the sidebar
        with st.sidebar:
            st.success("File uploaded successfully")

            st.write(f"**File:** {uploaded_file.name}")
            st.write(f"**Rows:** {len(working_df):,}")
            st.write(f"**Columns:** {len(working_df.columns)}")

        # Separate the major parts of the application
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

            preview_rows = st.slider(
                "Number of rows to preview",
                min_value=5,
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

        # Retrieve the DataFrame again because cleaning controls may update it
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