import pandas as pd
import plotly.express as px
import streamlit as st

from components.analysis_engine import AnalysisResult
from components.analysis_planner import AnalysisPlan


def create_auto_chart(
    result: AnalysisResult,
    plan: AnalysisPlan,
):
    """
    Create a Plotly chart from a completed analysis result.

    Returns None when:
    - the result is not a table
    - there is no data
    - the analysis does not need a chart
    """

    if result.result_type != "table":
        return None

    if result.data is None or result.data.empty:
        return None

    chart_type = choose_auto_chart_type(plan)

    if chart_type == "bar":
        return create_auto_bar_chart(
            result.data,
            plan,
        )

    if chart_type == "line":
        return create_auto_line_chart(
            result.data,
            plan,
        )

    if chart_type == "scatter":
        return create_auto_scatter_chart(
            result.data,
            plan,
        )

    return None


def choose_auto_chart_type(
    plan: AnalysisPlan,
) -> str:
    """
    Decide which chart type should be used.

    The application controls obvious analytical cases rather
    than relying completely on the AI recommendation.
    """

    if plan.operation == "trend":
        return "line"

    if plan.operation in {
        "ranking",
        "comparison",
    }:
        return "bar"

    if plan.operation == "percentage_change":
        return "line"

    return plan.visualization


def create_auto_bar_chart(
    data: pd.DataFrame,
    plan: AnalysisPlan,
):
    """
    Create an automatic bar chart for rankings and comparisons.
    """

    if not plan.group_by:
        return None

    x_column = plan.group_by[0]

    if (
        x_column not in data.columns
        or plan.metric not in data.columns
    ):
        return None

    title = build_auto_chart_title(plan)

    fig = px.bar(
        data,
        x=x_column,
        y=plan.metric,
        title=title,
        labels={
            x_column: x_column,
            plan.metric: format_metric_name(
                plan.metric,
                plan.aggregation,
            ),
        },
    )

    return fig


def create_auto_line_chart(
    data: pd.DataFrame,
    plan: AnalysisPlan,
):
    """
    Create an automatic line chart for time-based analyses.
    """

    if plan.date_column is None:
        return None

    if (
        plan.date_column not in data.columns
        or plan.metric not in data.columns
    ):
        return None

    title = build_auto_chart_title(plan)

    fig = px.line(
        data,
        x=plan.date_column,
        y=plan.metric,
        markers=True,
        title=title,
        labels={
            plan.date_column: plan.date_column,
            plan.metric: format_metric_name(
                plan.metric,
                plan.aggregation,
            ),
        },
    )

    return fig


def create_auto_scatter_chart(
    data: pd.DataFrame,
    plan: AnalysisPlan,
):
    """
    Create a scatter chart when a future analysis plan
    explicitly supports two numeric dimensions.

    Current planner operations rarely require scatter charts,
    so this safely returns None unless suitable columns exist.
    """

    numeric_columns = (
        data.select_dtypes(
            include="number"
        )
        .columns
        .tolist()
    )

    if len(numeric_columns) < 2:
        return None

    x_column = numeric_columns[0]
    y_column = numeric_columns[1]

    fig = px.scatter(
        data,
        x=x_column,
        y=y_column,
        title=f"{y_column} vs. {x_column}",
    )

    return fig


def build_auto_chart_title(
    plan: AnalysisPlan,
) -> str:
    """
    Build a readable title for an automatically generated chart.
    """

    metric_name = format_metric_name(
        plan.metric,
        plan.aggregation,
    )

    if plan.operation == "ranking":
        group_name = ", ".join(
            plan.group_by
        )

        if plan.limit is not None:
            return (
                f"Top {plan.limit} {group_name} "
                f"by {metric_name}"
            )

        return (
            f"{group_name} ranked by "
            f"{metric_name}"
        )

    if plan.operation == "comparison":
        group_name = ", ".join(
            plan.group_by
        )

        return (
            f"{metric_name} by {group_name}"
        )

    if plan.operation == "trend":
        return (
            f"{plan.time_granularity.title()} "
            f"{metric_name} Trend"
        )

    if plan.operation == "percentage_change":
        return (
            f"{plan.time_granularity.title()} "
            f"{plan.metric} Percentage Change"
        )

    return f"{metric_name} Analysis"


def format_metric_name(
    metric: str | None,
    aggregation: str | None,
) -> str:
    """
    Convert metric + aggregation into a readable label.
    """

    if metric is None:
        return "Value"

    aggregation_labels = {
        "sum": "Total",
        "mean": "Average",
        "count": "Count of",
        "min": "Minimum",
        "max": "Maximum",
    }

    prefix = aggregation_labels.get(
        aggregation,
        "",
    )

    if prefix:
        return f"{prefix} {metric}"

    return metric


# ---------------------------------------------------------
# EXISTING MANUAL VISUALIZATION PAGE
# ---------------------------------------------------------


def display_visualizations(
    df: pd.DataFrame,
) -> None:
    """
    Show chart options appropriate for the uploaded dataset.
    """

    st.header("Data Visualization")

    if df.empty:
        st.info(
            "The working dataset has no rows to visualize."
        )
        return

    numeric_columns = (
        df.select_dtypes(
            include="number"
        )
        .columns
        .tolist()
    )

    chart_type = st.selectbox(
        "Select Chart Type",
        [
            "Bar Chart",
            "Line Chart",
            "Scatter Plot",
        ],
    )

    # -----------------------------------------------------
    # BAR CHART
    # -----------------------------------------------------

    if chart_type == "Bar Chart":
        group_column = st.selectbox(
            "Group records by",
            df.columns.tolist(),
        )

        metric_options = [
            "Count of rows"
        ]

        if numeric_columns:
            metric_options.append(
                "Sum of a numeric column"
            )

        metric = st.selectbox(
            "What should the bars show?",
            metric_options,
        )

        groups = (
            df[group_column]
            .astype("string")
            .fillna("(Missing)")
        )

        if metric == "Count of rows":
            chart_df = pd.DataFrame(
                {
                    "Group": groups,
                }
            )

            chart_df = (
                chart_df
                .groupby(
                    "Group",
                    dropna=False,
                )
                .size()
                .reset_index(
                    name="Count"
                )
            )

            fig = px.bar(
                chart_df,
                x="Group",
                y="Count",
                title=(
                    f"Count of rows by "
                    f"{group_column}"
                ),
                labels={
                    "Group": group_column,
                },
            )

        else:
            value_column = st.selectbox(
                "Numeric column to sum",
                numeric_columns,
            )

            chart_df = pd.DataFrame(
                {
                    "Group": groups,
                    "Value": df[
                        value_column
                    ],
                }
            )

            chart_df = (
                chart_df
                .groupby(
                    "Group",
                    dropna=False,
                )["Value"]
                .sum()
                .reset_index(
                    name="Total"
                )
            )

            fig = px.bar(
                chart_df,
                x="Group",
                y="Total",
                title=(
                    f"Total {value_column} "
                    f"by {group_column}"
                ),
                labels={
                    "Group":
                        group_column,
                    "Total":
                        f"Total {value_column}",
                },
            )

    # -----------------------------------------------------
    # LINE CHART
    # -----------------------------------------------------

    elif chart_type == "Line Chart":
        if not numeric_columns:
            st.info(
                "A line chart needs at least "
                "one numeric column."
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

        fig = px.line(
            df,
            x=x_axis,
            y=y_axis,
            title=f"{y_axis} by {x_axis}",
        )

    # -----------------------------------------------------
    # SCATTER PLOT
    # -----------------------------------------------------

    else:
        if len(numeric_columns) < 2:
            st.info(
                "A scatter plot needs at least "
                "two numeric columns."
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
            title=(
                f"{y_axis} vs. {x_axis}"
            ),
        )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )