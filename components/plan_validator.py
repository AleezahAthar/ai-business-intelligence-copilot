from pydantic import BaseModel

import pandas as pd

from components.analysis_planner import AnalysisPlan


class ValidationResult(BaseModel):
    """Result returned after validating an analysis plan."""

    is_valid: bool
    reason: str
    plan: AnalysisPlan | None = None


def validate_analysis_plan(
    plan: AnalysisPlan,
    df: pd.DataFrame,
) -> ValidationResult:
    """
    Validate an AI-generated analysis plan against the actual dataframe.

    The validator checks whether:
    - referenced columns exist
    - metrics are appropriate for aggregations
    - grouping columns exist
    - filters reference real columns
    - trend analysis contains a valid date column
    - required fields exist for each operation

    Returns a ValidationResult containing:
    - is_valid
    - reason
    - validated plan
    """

    if plan.operation == "unsupported":
        return ValidationResult(
            is_valid=False,
            reason=(
                plan.unsupported_reason
                or "The requested analysis is not supported."
            ),
            plan=None,
        )

    # ---------------------------------------------------------
    # ROW COUNT
    # ---------------------------------------------------------

    if plan.operation == "row_count":
        return ValidationResult(
            is_valid=True,
            reason=(
                "Valid row-count plan. "
                "No metric, grouping, or aggregation is required."
            ),
            plan=plan,
        )

    # ---------------------------------------------------------
    # METRIC CHECKS
    # ---------------------------------------------------------

    operations_requiring_metric = {
        "aggregate",
        "ranking",
        "trend",
        "comparison",
        "percentage_change",
    }

    if plan.operation in operations_requiring_metric:
        if plan.metric is None:
            return ValidationResult(
                is_valid=False,
                reason=(
                    f"The '{plan.operation}' operation requires "
                    "a metric, but no metric was provided."
                ),
                plan=None,
            )

        if plan.metric not in df.columns:
            return ValidationResult(
                is_valid=False,
                reason=(
                    f"The metric column '{plan.metric}' does not "
                    "exist in the dataset."
                ),
                plan=None,
            )

    # ---------------------------------------------------------
    # AGGREGATION CHECKS
    # ---------------------------------------------------------

    operations_requiring_aggregation = {
        "aggregate",
        "ranking",
        "trend",
        "comparison",
        "percentage_change",
    }

    if plan.operation in operations_requiring_aggregation:
        if plan.aggregation is None:
            return ValidationResult(
                is_valid=False,
                reason=(
                    f"The '{plan.operation}' operation requires "
                    "an aggregation, but none was provided."
                ),
                plan=None,
            )

    # Sum, mean, min, and max require numeric data.
    numeric_aggregations = {
        "sum",
        "mean",
        "min",
        "max",
    }

    if (
        plan.metric is not None
        and plan.aggregation in numeric_aggregations
    ):
        if not pd.api.types.is_numeric_dtype(
            df[plan.metric]
        ):
            return ValidationResult(
                is_valid=False,
                reason=(
                    f"The aggregation '{plan.aggregation}' "
                    f"requires a numeric metric, but "
                    f"'{plan.metric}' has dtype "
                    f"'{df[plan.metric].dtype}'."
                ),
                plan=None,
            )

    # ---------------------------------------------------------
    # GROUP BY CHECKS
    # ---------------------------------------------------------

    for column in plan.group_by:
        if column not in df.columns:
            return ValidationResult(
                is_valid=False,
                reason=(
                    f"The grouping column '{column}' does not "
                    "exist in the dataset."
                ),
                plan=None,
            )

    operations_requiring_group_by = {
        "ranking",
        "comparison",
    }

    if (
        plan.operation in operations_requiring_group_by
        and not plan.group_by
    ):
        return ValidationResult(
            is_valid=False,
            reason=(
                f"The '{plan.operation}' operation requires "
                "at least one grouping column."
            ),
            plan=None,
        )

    # ---------------------------------------------------------
    # FILTER CHECKS
    # ---------------------------------------------------------

    for filter_condition in plan.filters:
        if filter_condition.column not in df.columns:
            return ValidationResult(
                is_valid=False,
                reason=(
                    f"The filter references column "
                    f"'{filter_condition.column}', but that "
                    "column does not exist in the dataset."
                ),
                plan=None,
            )

        column_series = df[filter_condition.column]

        numeric_operators = {
            "greater_than",
            "greater_than_or_equal",
            "less_than",
            "less_than_or_equal",
        }

        if (
            filter_condition.operator in numeric_operators
            and not pd.api.types.is_numeric_dtype(
                column_series
            )
        ):
            return ValidationResult(
                is_valid=False,
                reason=(
                    f"The filter operator "
                    f"'{filter_condition.operator}' requires "
                    f"a numeric column, but "
                    f"'{filter_condition.column}' has dtype "
                    f"'{column_series.dtype}'."
                ),
                plan=None,
            )

    # ---------------------------------------------------------
    # RANKING CHECKS
    # ---------------------------------------------------------

    if plan.operation == "ranking":
        if plan.sort is None:
            return ValidationResult(
                is_valid=False,
                reason=(
                    "A ranking operation requires a sort "
                    "direction."
                ),
                plan=None,
            )

        if plan.limit is not None and plan.limit <= 0:
            return ValidationResult(
                is_valid=False,
                reason=(
                    "The ranking limit must be greater than zero."
                ),
                plan=None,
            )

    # ---------------------------------------------------------
    # TREND CHECKS
    # ---------------------------------------------------------

    if plan.operation == "trend":
        if plan.date_column is None:
            return ValidationResult(
                is_valid=False,
                reason=(
                    "A trend operation requires a date column, "
                    "but none was provided."
                ),
                plan=None,
            )

        if plan.date_column not in df.columns:
            return ValidationResult(
                is_valid=False,
                reason=(
                    f"The date column '{plan.date_column}' does "
                    "not exist in the dataset."
                ),
                plan=None,
            )

        if plan.time_granularity is None:
            return ValidationResult(
                is_valid=False,
                reason=(
                    "A trend operation requires a time "
                    "granularity such as day, month, or year."
                ),
                plan=None,
            )

        if not is_date_column(
            df[plan.date_column]
        ):
            return ValidationResult(
                is_valid=False,
                reason=(
                    f"The column '{plan.date_column}' could not "
                    "be interpreted as a valid date column."
                ),
                plan=None,
            )

    # ---------------------------------------------------------
    # PERCENTAGE CHANGE CHECKS
    # ---------------------------------------------------------

    if plan.operation == "percentage_change":
        if plan.date_column is None:
            return ValidationResult(
                is_valid=False,
                reason=(
                    "A percentage-change operation requires "
                    "a date column."
                ),
                plan=None,
            )

        if plan.date_column not in df.columns:
            return ValidationResult(
                is_valid=False,
                reason=(
                    f"The date column '{plan.date_column}' does "
                    "not exist in the dataset."
                ),
                plan=None,
            )

        if not is_date_column(
            df[plan.date_column]
        ):
            return ValidationResult(
                is_valid=False,
                reason=(
                    f"The column '{plan.date_column}' could not "
                    "be interpreted as a valid date column."
                ),
                plan=None,
            )

    # ---------------------------------------------------------
    # VALID PLAN
    # ---------------------------------------------------------

    return ValidationResult(
        is_valid=True,
        reason=build_success_reason(plan),
        plan=plan,
    )


def is_date_column(series: pd.Series) -> bool:
    """
    Determine whether a Series is already datetime or can
    reasonably be converted to datetime.
    """

    if pd.api.types.is_datetime64_any_dtype(series):
        return True

    try:
        converted = pd.to_datetime(
            series,
            errors="coerce",
        )
    except Exception:
        return False

    valid_ratio = converted.notna().mean()

    return valid_ratio >= 0.8


def build_success_reason(
    plan: AnalysisPlan,
) -> str:
    """
    Create a readable explanation of why a plan passed validation.
    """

    if plan.operation == "aggregate":
        return (
            f"Valid aggregate plan: apply "
            f"'{plan.aggregation}' to '{plan.metric}'."
        )

    if plan.operation == "ranking":
        groups = ", ".join(plan.group_by)

        limit_text = (
            f" and return the top {plan.limit}"
            if plan.limit is not None
            else ""
        )

        return (
            f"Valid ranking plan: group by {groups}, apply "
            f"'{plan.aggregation}' to '{plan.metric}', sort "
            f"{plan.sort}{limit_text}."
        )

    if plan.operation == "comparison":
        groups = ", ".join(plan.group_by)

        return (
            f"Valid comparison plan: compare "
            f"'{plan.metric}' grouped by {groups} using "
            f"'{plan.aggregation}'."
        )

    if plan.operation == "trend":
        return (
            f"Valid trend plan: aggregate '{plan.metric}' "
            f"using '{plan.aggregation}' by "
            f"{plan.time_granularity} from "
            f"'{plan.date_column}'."
        )

    if plan.operation == "percentage_change":
        return (
            f"Valid percentage-change plan using "
            f"'{plan.metric}' and date column "
            f"'{plan.date_column}'."
        )

    return "The analysis plan is valid."