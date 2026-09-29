import pandas as pd
import streamlit as st


def initialize_dataframes(
    df: pd.DataFrame,
    file_name: str,
    file_id: str,
) -> None:
    """
    Store original and working copies of an uploaded DataFrame.

    Recreate both copies when the uploaded file's contents change.
    """

    current_file_id = st.session_state.get("uploaded_file_id")

    if current_file_id != file_id:
        st.session_state["original_df"] = df.copy(deep=True)
        st.session_state["working_df"] = df.copy(deep=True)
        st.session_state["uploaded_file_name"] = file_name
        st.session_state["uploaded_file_id"] = file_id

        # Clear messages from a previously uploaded dataset.
        st.session_state.pop("duplicate_message", None)
        st.session_state.pop("missing_message", None)
        st.session_state.pop("reset_message", None)


def reset_working_dataframe() -> None:
    """Restore the working DataFrame from the original upload."""

    if "original_df" in st.session_state:
        st.session_state["working_df"] = (
            st.session_state["original_df"].copy(deep=True)
        )


def remove_duplicate_rows() -> int:
    """Remove exact duplicate rows and return the number removed."""

    working_df = st.session_state["working_df"]
    rows_before = len(working_df)

    cleaned_df = (
        working_df
        .drop_duplicates()
        .reset_index(drop=True)
    )

    rows_removed = rows_before - len(cleaned_df)
    st.session_state["working_df"] = cleaned_df

    return rows_removed


def fill_missing_values(column: str) -> tuple[int, object]:
    """
    Fill missing values in one column.

    Numeric columns use their median. Other columns use their
    most common nonmissing value.

    Returns the number of values filled and the value used.
    """

    working_df = st.session_state["working_df"].copy()
    series = working_df[column]
    missing_count = int(series.isna().sum())

    if pd.api.types.is_numeric_dtype(series):
        fill_value = series.median()
    else:
        fill_value = series.mode(dropna=True).iloc[0]

    working_df[column] = series.fillna(fill_value)
    st.session_state["working_df"] = working_df

    return missing_count, fill_value


def display_reset_message() -> None:
    """Display the reset message once after a rerun."""

    message = st.session_state.pop("reset_message", None)

    if message is not None:
        st.success(message)


def display_duplicate_message() -> None:
    """Display the duplicate-removal message once after a rerun."""

    message = st.session_state.pop("duplicate_message", None)

    if message is not None:
        st.success(message)


def display_missing_message() -> None:
    """Display the missing-value message once after a rerun."""

    message = st.session_state.pop("missing_message", None)

    if message is not None:
        st.success(message)


def display_reset_section() -> None:
    """Display the data-cleaning controls."""

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

    display_reset_message()

    st.divider()

    # -----------------------------
    # Duplicate rows
    # -----------------------------
    st.subheader("Remove Duplicate Rows")

    working_df = st.session_state["working_df"]
    duplicate_count = int(working_df.duplicated().sum())

    st.write(f"Duplicate rows found: **{duplicate_count}**")

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

    display_duplicate_message()

    st.divider()

    # -----------------------------
    # Missing values
    # -----------------------------
    st.subheader("Fill Missing Values")

    working_df = st.session_state["working_df"]

    missing_columns = [
        column
        for column in working_df.columns
        if working_df[column].isna().any()
    ]

    if not missing_columns:
        st.info("No missing values remain in the working dataset.")
    else:
        selected_column = st.selectbox(
            "Column with missing values",
            options=missing_columns,
        )

        series = working_df[selected_column]
        missing_count = int(series.isna().sum())
        has_existing_value = series.notna().any()

        st.write(
            f"Missing values in **{selected_column}**: "
            f"**{missing_count}**"
        )

        if pd.api.types.is_numeric_dtype(series):
            st.caption(
                "Numeric column: blanks will be filled "
                "with the column median."
            )
        else:
            st.caption(
                "Text column: blanks will be filled "
                "with the most common existing value."
            )

        if not has_existing_value:
            st.warning(
                "This column is entirely blank, so there is "
                "no existing value to use."
            )

        if st.button(
            "Fill missing values in this column",
            disabled=not has_existing_value,
        ):
            values_filled, fill_value = fill_missing_values(
                selected_column
            )

            st.session_state["missing_message"] = (
                f"Filled {values_filled} missing value(s) in "
                f"{selected_column} with {fill_value}."
            )

            st.rerun()

    display_missing_message()