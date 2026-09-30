import pandas as pd
import streamlit as st


def display_data_qa(df: pd.DataFrame) -> None:
    """Answer a small set of questions using the working dataset."""

    st.header("Ask About Your Data")

    st.write(
        "Choose a question to calculate an answer "
        "from your current working dataset."
    )

    if df.empty:
        st.info("The working dataset has no rows to analyze.")
        return

    numeric_columns = df.select_dtypes(
        include="number"
    ).columns.tolist()

    questions = ["How many rows are in the dataset?"]

    if numeric_columns:
        questions.extend([
            "What is the total of a numeric column?",
            "What is the average of a numeric column?",
            "What is the highest value in a numeric column?",
            "What is the lowest value in a numeric column?",
            "What is the total of a numeric column by group?",
        ])

    question = st.selectbox(
        "Question",
        options=questions,
    )

    if question == "How many rows are in the dataset?":
        st.metric("Number of rows", f"{len(df):,}")
        return

    value_column = st.selectbox(
        "Numeric column",
        options=numeric_columns,
    )

    if question == "What is the total of a numeric column by group?":
        group_column = st.selectbox(
            "Group by",
            options=df.columns.tolist(),
        )

        # Keep group labels separate from numeric values.
        analysis_df = pd.DataFrame({
            "Group": (
                df[group_column]
                .astype("string")
                .fillna("(Missing)")
            ),
            "Value": df[value_column],
        })

        result = (
            analysis_df
            .groupby("Group", dropna=False)["Value"]
            .sum(min_count=1)
            .reset_index(name="Total")
            .sort_values(
                "Total",
                ascending=False,
                na_position="last",
            )
        )

        st.write(f"Total **{value_column}** by **{group_column}**")
        st.dataframe(result, use_container_width=True)
        st.caption(
            "Missing numeric values are excluded. "
            "A group with no numeric values has a blank total."
        )
        return

    values = df[value_column]

    if question == "What is the total of a numeric column?":
        answer = values.sum(min_count=1)
        label = f"Total {value_column}"

    elif question == "What is the average of a numeric column?":
        answer = values.mean()
        label = f"Average {value_column}"

    elif question == "What is the highest value in a numeric column?":
        answer = values.max()
        label = f"Highest {value_column}"

    else:
        answer = values.min()
        label = f"Lowest {value_column}"

    if pd.isna(answer):
        st.info("This column has no nonmissing numeric values.")
        return

    st.metric(label, f"{answer:,.2f}")

    missing_count = int(values.isna().sum())

    if missing_count:
        st.caption(
            f"Excluded {missing_count} missing value(s) "
            "from this calculation."
        )