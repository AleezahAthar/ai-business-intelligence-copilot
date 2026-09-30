import pandas as pd
import plotly.express as px
import streamlit as st


def display_visualizations(df: pd.DataFrame) -> None:
    """Show chart options appropriate for the uploaded dataset."""

    st.header("Data Visualization")

    if df.empty:
        st.info("The working dataset has no rows to visualize.")
        return

    # These columns can be used as numeric chart values.
    numeric_columns = df.select_dtypes(
        include="number"
    ).columns.tolist()

    chart_type = st.selectbox(
        "Select Chart Type",
        ["Bar Chart", "Line Chart", "Scatter Plot"],
    )

    if chart_type == "Bar Chart":
        group_column = st.selectbox(
            "Group records by",
            df.columns.tolist(),
        )

        metric_options = ["Count of rows"]

        if numeric_columns:
            metric_options.append("Sum of a numeric column")

        metric = st.selectbox(
            "What should the bars show?",
            metric_options,
        )

        # Give missing group labels a visible name.
        groups = (
            df[group_column]
            .astype("string")
            .fillna("(Missing)")
        )

        if metric == "Count of rows":
            chart_df = pd.DataFrame({
                "Group": groups,
            })

            chart_df = (
                chart_df
                .groupby("Group", dropna=False)
                .size()
                .reset_index(name="Count")
            )

            fig = px.bar(
                chart_df,
                x="Group",
                y="Count",
                title=f"Count of rows by {group_column}",
                labels={"Group": group_column},
            )

        else:
            value_column = st.selectbox(
                "Numeric column to sum",
                numeric_columns,
            )

            # Group labels and values have separate names, even if
            # the user selects the same source column for both.
            chart_df = pd.DataFrame({
                "Group": groups,
                "Value": df[value_column],
            })

            chart_df = (
                chart_df
                .groupby("Group", dropna=False)["Value"]
                .sum()
                .reset_index(name="Total")
            )

            fig = px.bar(
                chart_df,
                x="Group",
                y="Total",
                title=f"Total {value_column} by {group_column}",
                labels={
                    "Group": group_column,
                    "Total": f"Total {value_column}",
                },
            )

    elif chart_type == "Line Chart":
        if not numeric_columns:
            st.info(
                "A line chart needs at least one numeric column."
            )
            return

        x_axis = st.selectbox(
            "Select X-axis",
            df.columns.tolist(),
        )

        y_axis = st.selectbox(
            "Select numeric Y-axis",
            numeric_columns,
        )

        # Plotly connects points in the dataset's current row order.
        fig = px.line(
            df,
            x=x_axis,
            y=y_axis,
            title=f"{y_axis} by {x_axis}",
        )

    else:
        if len(numeric_columns) < 2:
            st.info(
                "A scatter plot needs at least two numeric columns."
            )
            return

        x_axis = st.selectbox(
            "Select numeric X-axis",
            numeric_columns,
        )

        y_axis = st.selectbox(
            "Select numeric Y-axis",
            numeric_columns,
            index=1,
        )

        fig = px.scatter(
            df,
            x=x_axis,
            y=y_axis,
            title=f"{y_axis} vs. {x_axis}",
        )

    st.plotly_chart(fig, use_container_width=True)