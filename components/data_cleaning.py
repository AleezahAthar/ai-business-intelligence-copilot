import pandas as pd
import streamlit as st


def initialize_dataframes(
    df: pd.DataFrame,
    file_name: str,
) -> None:
    """
    Store an original and working copy of an uploaded DataFrame.

    The DataFrames are recreated only when the user uploads
    a file with a different filename.
    """

    current_file_name = st.session_state.get("uploaded_file_name")

    if current_file_name != file_name:
        st.session_state["original_df"] = df.copy(deep=True)
        st.session_state["working_df"] = df.copy(deep=True)
        st.session_state["uploaded_file_name"] = file_name

        # Clear messages left over from a previously uploaded dataset.
        st.session_state.pop("duplicate_message", None)
        st.session_state.pop("reset_message", None)


def reset_working_dataframe() -> None:
    """
    Replace the working DataFrame with a fresh copy
    of the original uploaded DataFrame.
    """

    if "original_df" in st.session_state:
        st.session_state["working_df"] = (
            st.session_state["original_df"].copy(deep=True)
        )


def remove_duplicate_rows() -> int:
    """
    Remove exact duplicate rows from the working DataFrame.

    Returns:
        int: The number of duplicate rows removed.
    """

    working_df = st.session_state["working_df"]

    rows_before = len(working_df)

    cleaned_df = (
        working_df
        .drop_duplicates()
        .reset_index(drop=True)
    )

    rows_after = len(cleaned_df)
    rows_removed = rows_before - rows_after

    st.session_state["working_df"] = cleaned_df

    return rows_removed


def display_reset_message() -> None:
    """
    Display the reset message once after a rerun.
    """

    message = st.session_state.pop("reset_message", None)

    if message is not None:
        st.success(message)


def display_duplicate_message() -> None:
    """
    Display the duplicate-removal message once after a rerun.
    """

    message = st.session_state.pop("duplicate_message", None)

    if message is not None:
        st.success(message)


def display_reset_section() -> None:
    """
    Display the current data-preparation controls.
    """

    st.header("Data Preparation")

    st.write(
        "The original uploaded dataset is preserved. "
        "Cleaning operations only affect the working dataset."
    )

    if st.button(
        "Reset working dataset",
        type="secondary",
    ):
        reset_working_dataframe()

        st.session_state["reset_message"] = (
            "The working dataset has been reset "
            "to the original upload."
        )

        st.rerun()

    # The reset message appears directly below the reset button.
    display_reset_message()

    st.divider()

    st.subheader("Remove Duplicate Rows")

    working_df = st.session_state["working_df"]

    duplicate_count = int(
        working_df.duplicated().sum()
    )

    st.write(
        f"Duplicate rows found: **{duplicate_count}**"
    )

    if st.button(
        "Remove duplicate rows",
        type="primary",
        disabled=duplicate_count == 0,
    ):
        rows_removed = remove_duplicate_rows()

        st.session_state["duplicate_message"] = (
            f"Removed {rows_removed} duplicate row(s)."
        )

        st.rerun()

    # The duplicate message appears directly below its button.
    display_duplicate_message()