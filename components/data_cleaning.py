import pandas as pd
import streamlit as st


def initialize_dataframes(
    df: pd.DataFrame,
    file_name: str,
    file_id: str,
) -> None:
    """Store original and working copies when the upload changes."""

    current_file_id = st.session_state.get("uploaded_file_id")

    if current_file_id != file_id:
        st.session_state["original_df"] = df.copy(deep=True)
        st.session_state["working_df"] = df.copy(deep=True)
        st.session_state["uploaded_file_name"] = file_name
        st.session_state["uploaded_file_id"] = file_id

        for message_key in (
            "duplicate_message",
            "conversion_message",
            "missing_message",
            "reset_message",
        ):
            st.session_state.pop(message_key, None)


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


def parse_numeric_values(series: pd.Series) -> pd.Series:
    """
    Parse numeric-looking text.

    Remove currency symbols and thousands separators.
    Convert entries with a percent sign to decimal fractions.
    Unrecognized text becomes a missing value.
    """

    text_values = series.astype("string").str.strip()
    percent_entries = text_values.str.endswith("%", na=False)

    normalized = (
        text_values
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.replace("%", "", regex=False)
    )

    numeric_values = pd.to_numeric(
        normalized,
        errors="coerce",
    )

    numeric_values.loc[percent_entries] = (
        numeric_values.loc[percent_entries] / 100
    )

    return numeric_values


def convert_column_to_numeric(
    column: str,
    converted_values: pd.Series,
) -> None:
    """Apply a previewed numeric conversion to the working copy."""

    working_df = st.session_state["working_df"].copy()
    working_df[column] = converted_values
    st.session_state["working_df"] = working_df


def fill_missing_values(column: str) -> tuple[int, object]:
    """
    Fill numeric blanks with the median and other blanks
    with the most common nonmissing value.
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


def display_message(message_key: str) -> None:
    """Display a success message once after a rerun."""

    message = st.session_state.pop(message_key, None)

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

    display_message("reset_message")

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

    display_message("duplicate_message")

    st.divider()

    # -----------------------------
    # Numeric conversion
    # -----------------------------
    st.subheader("Convert Text to Numeric")

    st.write(
        "Choose a column containing numeric-looking text. "
        "Preview any values that cannot be converted before applying."
    )

    working_df = st.session_state["working_df"]

    candidate_columns = [
        column
        for column in working_df.columns
        if (
            not pd.api.types.is_numeric_dtype(working_df[column])
            and parse_numeric_values(working_df[column]).notna().any()
        )
    ]

    if not candidate_columns:
        st.info("No numeric-looking text columns were found.")
    else:
        selected_column = st.selectbox(
            "Text column to convert",
            options=candidate_columns,
        )

        original_values = working_df[selected_column]
        converted_values = parse_numeric_values(
            original_values
        )

        # These are existing values that parsing would turn into blanks.
        invalid_mask = (
            original_values.notna()
            & converted_values.isna()
        )
        invalid_count = int(invalid_mask.sum())

        st.write(
            f"Numeric values recognized: "
            f"**{int(converted_values.notna().sum())}**"
        )

        if invalid_count:
            examples = (
                original_values[invalid_mask]
                .astype(str)
                .unique()[:5]
                .tolist()
            )

            st.warning(
                f"{invalid_count} nonblank value(s) cannot be "
                f"converted and will become missing. "
                f"Examples: {examples}"
            )
        else:
            st.success(
                "All nonblank values in this column can be converted."
            )

        if st.button("Convert selected column to numeric"):
            convert_column_to_numeric(
                selected_column,
                converted_values,
            )

            st.session_state["conversion_message"] = (
                f"Converted {selected_column} to numeric. "
                f"{invalid_count} unrecognized value(s) "
                f"became missing."
            )
            st.rerun()

    display_message("conversion_message")

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
        has_existing_value = bool(series.notna().any())

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

    display_message("missing_message")