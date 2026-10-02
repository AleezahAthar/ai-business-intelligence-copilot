from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd

from components.analysis_planner import (
    AnalysisPlan,
    FilterCondition,
)


@dataclass
class AnalysisResult:
    """
    Standard result returned by the analysis engine.

    scalar:
        Used for answers such as total revenue or row count.

    table:
        Used for rankings, comparisons, trends,
        and percentage-change analyses.
    """

    operation: str
    result_type: Literal["scalar", "table"]
    value: Any = None
    data: pd.DataFrame | None = None


def execute_analysis(
    df: pd.DataFrame,
    plan: AnalysisPlan,
) -> AnalysisResult:
    """
    Execute a validated analysis plan using pandas.

    This function assumes the plan has already passed
    plan_validator.py.
    """

    if plan.operation == "unsupported":
        raise ValueError(
            "An unsupported analysis plan cannot be executed."
        )

    # Never modify the original dataframe.
    working_df = df.copy()

    # Apply any filters requested by the user.
    working_df = apply_filters(
        df=working_df,
        filters=plan.filters,
    )

    if plan.operation == "row_count":
        return execute_row_count(working_df)

    if plan.operation == "aggregate":
        return execute_aggregate(
            df=working_df,
            plan=plan,
        )

    if plan.operation == "ranking":
        return execute_ranking(
            df=working_df,
            plan=plan,
        )

    if plan.operation == "comparison":
        return execute_comparison(
            df=working_df,
            plan=plan,
        )

    if plan.operation == "trend":
        return execute_trend(
            df=working_df,
            plan=plan,
        )

    if plan.operation == "percentage_change":
        return execute_percentage_change(
            df=working_df,
            plan=plan,
        )

    raise ValueError(
        f"Unknown operation: {plan.operation}"
    )


def execute_row_count(
    df: pd.DataFrame,
) -> AnalysisResult:
    """Return the number of rows in the dataframe."""

    return AnalysisResult(
        operation="row_count",
        result_type="scalar",
        value=len(df),
    )


def execute_aggregate(
    df: pd.DataFrame,
    plan: AnalysisPlan,
) -> AnalysisResult:
    """
    Perform one overall aggregation.

    Example:
        total Revenue
        average Price
        maximum Sales
    """

    value = aggregate_series(
        series=df[plan.metric],
        aggregation=plan.aggregation,
    )

    return AnalysisResult(
        operation="aggregate",
        result_type="scalar",
        value=value,
    )


def execute_ranking(
    df: pd.DataFrame,
    plan: AnalysisPlan,
) -> AnalysisResult:
    """
    Group, aggregate, sort, and optionally limit results.

    Example:
        Top 5 products by total revenue.
    """

    result = aggregate_by_group(
        df=df,
        group_by=plan.group_by,
        metric=plan.metric,
        aggregation=plan.aggregation,
    )

    ascending = plan.sort == "ascending"

    result = result.sort_values(
        by=plan.metric,
        ascending=ascending,
    )

    if plan.limit is not None:
        result = result.head(plan.limit)

    result = result.reset_index(drop=True)

    return AnalysisResult(
        operation="ranking",
        result_type="table",
        data=result,
    )


def execute_comparison(
    df: pd.DataFrame,
    plan: AnalysisPlan,
) -> AnalysisResult:
    """
    Compare an aggregated metric across groups.

    Example:
        Revenue across regions.
    """

    result = aggregate_by_group(
        df=df,
        group_by=plan.group_by,
        metric=plan.metric,
        aggregation=plan.aggregation,
    )

    if plan.sort is not None:
        ascending = plan.sort == "ascending"

        result = result.sort_values(
            by=plan.metric,
            ascending=ascending,
        )

    result = result.reset_index(drop=True)

    return AnalysisResult(
        operation="comparison",
        result_type="table",
        data=result,
    )


def execute_trend(
    df: pd.DataFrame,
    plan: AnalysisPlan,
) -> AnalysisResult:
    """
    Aggregate a metric over time.

    Example:
        Monthly revenue trend.
    """

    working_df = df.copy()

    working_df[plan.date_column] = pd.to_datetime(
        working_df[plan.date_column],
        errors="coerce",
    )

    # Remove rows whose dates could not be parsed.
    working_df = working_df.dropna(
        subset=[plan.date_column]
    )

    working_df[plan.date_column] = convert_to_time_period(
        series=working_df[plan.date_column],
        granularity=plan.time_granularity,
    )

    result = (
        working_df
        .groupby(
            plan.date_column,
            dropna=False,
        )[plan.metric]
        .agg(plan.aggregation)
        .reset_index()
        .sort_values(plan.date_column)
        .reset_index(drop=True)
    )

    return AnalysisResult(
        operation="trend",
        result_type="table",
        data=result,
    )


def execute_percentage_change(
    df: pd.DataFrame,
    plan: AnalysisPlan,
) -> AnalysisResult:
    """
    Calculate percentage change between time periods.

    Example:
        Month-over-month revenue growth.
    """

    working_df = df.copy()

    working_df[plan.date_column] = pd.to_datetime(
        working_df[plan.date_column],
        errors="coerce",
    )

    working_df = working_df.dropna(
        subset=[plan.date_column]
    )

    if plan.time_granularity is not None:
        working_df[plan.date_column] = convert_to_time_period(
            series=working_df[plan.date_column],
            granularity=plan.time_granularity,
        )

    result = (
        working_df
        .groupby(
            plan.date_column,
            dropna=False,
        )[plan.metric]
        .agg(plan.aggregation)
        .reset_index()
        .sort_values(plan.date_column)
        .reset_index(drop=True)
    )

    change_column = (
        f"{plan.metric} Percentage Change (%)"
    )

    result[change_column] = (
        result[plan.metric]
        .pct_change(fill_method=None)
        .mul(100)
    )

    # Avoid displaying infinity when the previous value was zero.
    result[change_column] = result[
        change_column
    ].replace(
        [float("inf"), float("-inf")],
        pd.NA,
    )

    return AnalysisResult(
        operation="percentage_change",
        result_type="table",
        data=result,
    )


def apply_filters(
    df: pd.DataFrame,
    filters: list[FilterCondition],
) -> pd.DataFrame:
    """
    Apply validated filters to a dataframe.

    Filters are applied sequentially,
    meaning multiple filters behave like AND conditions.
    """

    filtered_df = df.copy()

    for condition in filters:
        column = condition.column
        operator = condition.operator
        value = condition.value

        if operator == "equals":
            filtered_df = filtered_df[
                filtered_df[column] == value
            ]

        elif operator == "not_equals":
            filtered_df = filtered_df[
                filtered_df[column] != value
            ]

        elif operator == "greater_than":
            numeric_value = convert_filter_value_to_number(
                value
            )

            filtered_df = filtered_df[
                filtered_df[column] > numeric_value
            ]

        elif operator == "greater_than_or_equal":
            numeric_value = convert_filter_value_to_number(
                value
            )

            filtered_df = filtered_df[
                filtered_df[column] >= numeric_value
            ]

        elif operator == "less_than":
            numeric_value = convert_filter_value_to_number(
                value
            )

            filtered_df = filtered_df[
                filtered_df[column] < numeric_value
            ]

        elif operator == "less_than_or_equal":
            numeric_value = convert_filter_value_to_number(
                value
            )

            filtered_df = filtered_df[
                filtered_df[column] <= numeric_value
            ]

        elif operator == "contains":
            filtered_df = filtered_df[
                filtered_df[column]
                .astype(str)
                .str.contains(
                    str(value),
                    case=False,
                    na=False,
                )
            ]

        else:
            raise ValueError(
                f"Unsupported filter operator: {operator}"
            )

    return filtered_df


def aggregate_series(
    series: pd.Series,
    aggregation: str,
):
    """Apply one supported aggregation to a pandas Series."""

    if aggregation == "sum":
        return series.sum()

    if aggregation == "mean":
        return series.mean()

    if aggregation == "count":
        return series.count()

    if aggregation == "min":
        return series.min()

    if aggregation == "max":
        return series.max()

    raise ValueError(
        f"Unsupported aggregation: {aggregation}"
    )


def aggregate_by_group(
    df: pd.DataFrame,
    group_by: list[str],
    metric: str,
    aggregation: str,
) -> pd.DataFrame:
    """
    Group a dataframe and aggregate the requested metric.
    """

    return (
        df
        .groupby(
            group_by,
            dropna=False,
        )[metric]
        .agg(aggregation)
        .reset_index()
    )


def convert_to_time_period(
    series: pd.Series,
    granularity: str,
) -> pd.Series:
    """
    Convert datetime values into consistent time periods.

    The returned values remain timestamps so they are easy
    to sort and chart later.
    """

    if granularity == "day":
        return series.dt.floor("D")

    if granularity == "week":
        return (
            series
            .dt.to_period("W")
            .dt.start_time
        )

    if granularity == "month":
        return (
            series
            .dt.to_period("M")
            .dt.start_time
        )

    if granularity == "quarter":
        return (
            series
            .dt.to_period("Q")
            .dt.start_time
        )

    if granularity == "year":
        return (
            series
            .dt.to_period("Y")
            .dt.start_time
        )

    raise ValueError(
        f"Unsupported time granularity: {granularity}"
    )


def convert_filter_value_to_number(
    value,
) -> int | float:
    """
    Convert a filter value into a numeric value.

    Raises a clear error instead of letting pandas fail
    with a harder-to-understand comparison error.
    """

    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Filter value '{value}' could not be "
            "converted to a number."
        ) from error

    if number.is_integer():
        return int(number)

    return number