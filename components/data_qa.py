from dataclasses import dataclass
from numbers import Number
from typing import Callable

import pandas as pd
import streamlit as st
from google.genai import errors
from pydantic import ValidationError

from components.analysis_engine import (
    AnalysisResult,
    execute_analysis,
)
from components.analysis_planner import (
    AnalysisPlan,
    build_dataset_schema,
    create_analysis_plan,
)
from components.insight_generator import (
    generate_business_insight,
)
from components.plan_validator import (
    ValidationResult,
    validate_analysis_plan,
)
from components.visualization import (
    create_auto_chart,
)


@dataclass
class DataQAResponse:
    """
    Complete response from the Ask Your Data pipeline.

    plan:
        The structured plan created by Gemini.

    validation:
        The result of checking the plan against the dataframe.

    result:
        The verified pandas calculation.

    insight:
        Gemini's explanation of the verified result.

    insight_error:
        Stores an error message if the calculation succeeds
        but the optional business insight cannot be generated.
    """

    plan: AnalysisPlan
    validation: ValidationResult
    result: AnalysisResult | None = None
    insight: str | None = None
    insight_error: str | None = None


def run_data_qa(
    df: pd.DataFrame,
    question: str,
    api_key: str,
    progress_callback: Callable[[str], None] | None = None,
) -> DataQAResponse:
    """
    Run the complete natural-language analysis pipeline.

    Flow:
        1. Read the dataset structure.
        2. Ask Gemini to create an analysis plan.
        3. Validate the plan against the actual dataframe.
        4. Execute the calculation locally with pandas.
        5. Ask Gemini to explain the verified result.
    """

    def report(message: str) -> None:
        if progress_callback is not None:
            progress_callback(message)

    # ---------------------------------------------------------
    # STEP 1: UNDERSTAND DATASET STRUCTURE
    # ---------------------------------------------------------

    report(
        "Step 1 of 5: Reading the dataset structure "
        "and identifying available columns."
    )

    schema = build_dataset_schema(df)

    # ---------------------------------------------------------
    # STEP 2: CREATE ANALYSIS PLAN
    # ---------------------------------------------------------

    report(
        "Step 2 of 5: Asking Gemini to translate your "
        "question into a structured analysis plan."
    )

    plan = create_analysis_plan(
        question=question,
        schema=schema,
        api_key=api_key,
    )

    report(
        f"Gemini selected the '{plan.operation}' "
        "analysis operation."
    )

    # ---------------------------------------------------------
    # STEP 3: VALIDATE PLAN
    # ---------------------------------------------------------

    report(
        "Step 3 of 5: Checking the AI-generated plan "
        "against the actual dataset."
    )

    validation = validate_analysis_plan(
        plan=plan,
        df=df,
    )

    if not validation.is_valid:
        report(
            "Validation stopped the analysis because the "
            "plan cannot safely be executed."
        )

        return DataQAResponse(
            plan=plan,
            validation=validation,
            result=None,
            insight=None,
        )

    report(
        "Validation passed. The requested columns and "
        "operations are supported."
    )

    # ---------------------------------------------------------
    # STEP 4: EXECUTE WITH PANDAS
    # ---------------------------------------------------------

    report(
        "Step 4 of 5: Running the verified calculation "
        "locally with pandas."
    )

    result = execute_analysis(
        df=df,
        plan=validation.plan,
    )

    report(
        "Pandas finished the calculation successfully."
    )

    # ---------------------------------------------------------
    # STEP 5: GENERATE BUSINESS INSIGHT
    # ---------------------------------------------------------

    report(
        "Step 5 of 5: Asking Gemini to explain the "
        "verified pandas result."
    )

    try:
        insight = generate_business_insight(
            question=question,
            plan=validation.plan,
            result=result,
            api_key=api_key,
        )

        insight_error = None

        report(
            "Business insight generated. "
            "The final response is ready."
        )

    except Exception as error:
        # The calculation itself is still valid even if the
        # optional explanation service fails.
        insight = None
        insight_error = str(error)

        report(
            "The calculation completed successfully, but "
            "the business insight could not be generated."
        )

    return DataQAResponse(
        plan=plan,
        validation=validation,
        result=result,
        insight=insight,
        insight_error=insight_error,
    )


def build_plan_summary(
    plan: AnalysisPlan,
) -> str:
    """
    Convert the structured analysis plan into a readable sentence.
    """

    if plan.operation == "unsupported":
        return (
            plan.unsupported_reason
            or "This question cannot be handled by the "
            "current analytics engine."
        )

    if plan.operation == "row_count":
        return (
            "Count the rows in the current working dataset."
        )

    if plan.operation == "aggregate":
        return (
            f"Calculate the {plan.aggregation} of "
            f"'{plan.metric}'."
        )

    if plan.operation == "ranking":
        group_text = ", ".join(
            plan.group_by
        )

        summary = (
            f"Group the data by {group_text}, calculate "
            f"the {plan.aggregation} of '{plan.metric}', "
            f"and sort the results {plan.sort}."
        )

        if plan.limit is not None:
            summary += (
                f" Return the first {plan.limit} result(s)."
            )

        return summary

    if plan.operation == "comparison":
        group_text = ", ".join(
            plan.group_by
        )

        return (
            f"Compare the {plan.aggregation} of "
            f"'{plan.metric}' across {group_text}."
        )

    if plan.operation == "trend":
        return (
            f"Calculate the {plan.aggregation} of "
            f"'{plan.metric}' by {plan.time_granularity} "
            f"using '{plan.date_column}'."
        )

    if plan.operation == "percentage_change":
        return (
            f"Calculate the percentage change in "
            f"'{plan.metric}' over time using "
            f"'{plan.date_column}'."
        )

    return (
        f"Run the '{plan.operation}' analysis."
    )


def display_plan(
    plan: AnalysisPlan,
) -> None:
    """
    Explain how the user's question was interpreted.

    The internal JSON plan is intentionally not displayed
    in the user-facing application.
    """

    st.markdown(
        "#### What the app understood"
    )

    st.write(
        build_plan_summary(plan)
    )


def display_analysis_result(
    result: AnalysisResult,
    plan: AnalysisPlan,
) -> None:
    """
    Display a completed pandas analysis result.

    Automatically generates a chart when the analysis
    type benefits from visualization.
    """

    st.markdown(
        "#### Calculation"
    )

    st.caption(
        build_calculation_caption(plan)
    )

    # ---------------------------------------------------------
    # SCALAR RESULT
    # ---------------------------------------------------------

    if result.result_type == "scalar":
        display_scalar_result(
            result=result,
            plan=plan,
        )

        return

    # ---------------------------------------------------------
    # TABLE RESULT
    # ---------------------------------------------------------

    if result.result_type == "table":
        if result.data is None or result.data.empty:
            st.info(
                "The analysis ran successfully, but it "
                "produced no matching rows."
            )
            return

        # -----------------------------------------------------
        # AUTOMATIC VISUALIZATION
        # -----------------------------------------------------

        chart = create_auto_chart(
            result=result,
            plan=plan,
        )

        if chart is not None:
            st.markdown(
                "#### Visualization"
            )

            st.caption(
                "The chart type was selected based on "
                "the validated analysis."
            )

            st.plotly_chart(
                chart,
                use_container_width=True,
            )

        # -----------------------------------------------------
        # RESULT TABLE
        # -----------------------------------------------------

        st.markdown(
            "#### Result data"
        )

        st.caption(
            "These values were calculated locally "
            "from the working dataset."
        )

        st.dataframe(
            result.data,
            use_container_width=True,
        )

        return

    st.warning(
        "The analysis returned an unknown result type."
    )


def display_business_insight(
    insight: str | None,
    insight_error: str | None = None,
) -> None:
    """
    Display Gemini's explanation of the verified result.
    """

    st.markdown(
        "#### Business Insight"
    )

    if insight:
        safe_insight = insight.replace(
            "$",
            r"\$",
        )

        st.markdown(
            safe_insight
        )

        st.caption(
            "This explanation was generated from the "
            "verified calculation result, not from the "
            "original dataset rows."
        )

        return

    st.info(
        "The calculation completed successfully, but a "
        "business explanation could not be generated."
    )

    if insight_error:
        with st.expander(
            "Technical details"
        ):
            st.caption(
                insight_error
            )

def display_scalar_result(
    result: AnalysisResult,
    plan: AnalysisPlan,
) -> None:
    """
    Display a single calculated value.
    """

    value = result.value

    if value is None or pd.isna(value):
        st.info(
            "There are no non-missing values available "
            "for this calculation."
        )
        return

    label = build_metric_label(plan)

    st.metric(
        label,
        format_scalar_value(
            value=value,
            plan=plan,
        ),
    )


def build_metric_label(
    plan: AnalysisPlan,
) -> str:
    """
    Create a readable label for scalar results.
    """

    if plan.operation == "row_count":
        return "Number of rows"

    labels = {
        "sum": "Total",
        "mean": "Average",
        "count": "Count of",
        "min": "Lowest",
        "max": "Highest",
    }

    prefix = labels.get(
        plan.aggregation,
        "Result for",
    )

    return f"{prefix} {plan.metric}"


def format_scalar_value(
    value,
    plan: AnalysisPlan,
) -> str:
    """
    Format scalar calculation results for display.
    """

    if plan.operation == "row_count":
        return f"{int(value):,}"

    if plan.aggregation == "count":
        return f"{int(value):,}"

    if isinstance(value, Number):
        return f"{value:,.2f}"

    return str(value)


def build_calculation_caption(
    plan: AnalysisPlan,
) -> str:
    """
    Explain exactly what pandas calculated.
    """

    if plan.operation == "row_count":
        return (
            "The app counted all rows in the working dataset."
        )

    if plan.operation == "aggregate":
        return (
            f"The app calculated the {plan.aggregation} "
            f"of '{plan.metric}'."
        )

    if plan.operation == "ranking":
        group_text = ", ".join(
            plan.group_by
        )

        limit_text = ""

        if plan.limit is not None:
            limit_text = (
                f" and kept the first "
                f"{plan.limit} result(s)"
            )

        return (
            f"The app grouped by {group_text}, calculated "
            f"the {plan.aggregation} of '{plan.metric}', "
            f"sorted the result {plan.sort}"
            f"{limit_text}."
        )

    if plan.operation == "comparison":
        group_text = ", ".join(
            plan.group_by
        )

        return (
            f"The app grouped the data by {group_text} "
            f"and calculated the {plan.aggregation} of "
            f"'{plan.metric}' for each group."
        )

    if plan.operation == "trend":
        return (
            f"The app converted '{plan.date_column}' into "
            f"{plan.time_granularity} periods and calculated "
            f"the {plan.aggregation} of '{plan.metric}' "
            f"for each period."
        )

    if plan.operation == "percentage_change":
        return (
            f"The app calculated the {plan.aggregation} "
            f"of '{plan.metric}' over time and then "
            "calculated the percentage change between periods."
        )

    return (
        "The requested calculation was completed locally."
    )


def display_guided_questions(
    df: pd.DataFrame,
) -> None:
    """
    Display dropdown-based questions.

    Guided questions use the same validator and pandas
    analysis engine as AI questions, but do not use Gemini.
    """

    st.caption(
        "Guided mode does not use AI. Choose an analysis "
        "and the app will calculate the result locally."
    )

    numeric_columns = (
        df.select_dtypes(
            include="number"
        )
        .columns
        .tolist()
    )

    questions = {
        "How many rows are in the dataset?":
            "row_count",
    }

    if numeric_columns:
        questions.update(
            {
                "What is the total of a numeric column?":
                    "sum",
                "What is the average of a numeric column?":
                    "mean",
                "What is the highest value in a numeric column?":
                    "max",
                "What is the lowest value in a numeric column?":
                    "min",
                "What is the total by group?":
                    "comparison",
            }
        )

    selected_question = st.selectbox(
        "Choose a question",
        options=list(questions),
        key="qa_guided_question",
    )

    selection = questions[
        selected_question
    ]

    # ---------------------------------------------------------
    # BUILD GUIDED PLAN
    # ---------------------------------------------------------

    if selection == "row_count":
        plan = AnalysisPlan(
            operation="row_count",
        )

    elif selection == "comparison":
        value_column = st.selectbox(
            "Choose a numeric column",
            options=numeric_columns,
            key="qa_guided_value",
        )

        group_column = st.selectbox(
            "Choose a grouping column",
            options=df.columns.tolist(),
            key="qa_guided_group",
        )

        plan = AnalysisPlan(
            operation="comparison",
            metric=value_column,
            aggregation="sum",
            group_by=[group_column],
            sort="descending",
            visualization="bar",
        )

    else:
        value_column = st.selectbox(
            "Choose a numeric column",
            options=numeric_columns,
            key="qa_guided_value",
        )

        plan = AnalysisPlan(
            operation="aggregate",
            metric=value_column,
            aggregation=selection,
            visualization="none",
        )

    # ---------------------------------------------------------
    # EXPLAIN PLAN
    # ---------------------------------------------------------

    st.markdown(
        "#### Analysis plan"
    )

    st.write(
        build_plan_summary(plan)
    )

    # ---------------------------------------------------------
    # VALIDATE
    # ---------------------------------------------------------

    validation = validate_analysis_plan(
        plan=plan,
        df=df,
    )

    if not validation.is_valid:
        st.warning(
            validation.reason
        )
        return

    st.caption(
        f"Validation passed: {validation.reason}"
    )

    # ---------------------------------------------------------
    # EXECUTE
    # ---------------------------------------------------------

    result = execute_analysis(
        df=df,
        plan=validation.plan,
    )

    display_analysis_result(
        result=result,
        plan=validation.plan,
    )


def display_ai_questions(
    df: pd.DataFrame,
) -> None:
    """
    Display natural-language dataset questions.
    """

    st.write(
        "Ask a question about your working dataset."
    )

    st.caption(
        "Gemini interprets your question and creates an "
        "analysis plan. Your dataset rows stay local. "
        "The plan is validated, the calculation is performed "
        "with pandas, and Gemini then explains only the "
        "verified result."
    )

    # ---------------------------------------------------------
    # LOAD API KEY
    # ---------------------------------------------------------

    try:
        api_key = st.secrets.get(
            "GEMINI_API_KEY",
            "",
        ).strip()

    except FileNotFoundError:
        api_key = ""

    if not api_key:
        st.info(
            "AI questions are currently unavailable because "
            "no Gemini API key was found. Switch to Guided "
            "questions to explore your data."
        )
        return

    # ---------------------------------------------------------
    # QUESTION FORM
    # ---------------------------------------------------------

    with st.form("qa_ai_form"):
        question = st.text_input(
            "Your question",
            placeholder=(
                "Which 5 products generated "
                "the most revenue?"
            ),
            max_chars=1000,
        )

        submitted = st.form_submit_button(
            "Analyze",
            type="primary",
        )

    if not submitted:
        return

    question = question.strip()

    if not question:
        st.warning(
            "Enter a question first."
        )
        return

    # ---------------------------------------------------------
    # RUN PIPELINE
    # ---------------------------------------------------------

    status_box = st.status(
        "Starting analysis...",
        expanded=True,
    )

    try:
        response = run_data_qa(
            df=df,
            question=question,
            api_key=api_key,
            progress_callback=status_box.write,
        )

        # -----------------------------------------------------
        # SHOW WHAT GEMINI UNDERSTOOD
        # -----------------------------------------------------

        display_plan(
            response.plan
        )

        # -----------------------------------------------------
        # HANDLE INVALID / UNSUPPORTED PLAN
        # -----------------------------------------------------

        if not response.validation.is_valid:
            status_box.update(
                label="Analysis stopped",
                state="error",
                expanded=False,
            )

            if response.plan.operation == "unsupported":
                st.info(
                    response.validation.reason
                )

            else:
                st.warning(
                    response.validation.reason
                )

            return

        # -----------------------------------------------------
        # SUCCESS
        # -----------------------------------------------------

        status_box.update(
            label="Analysis complete",
            state="complete",
            expanded=False,
        )

        st.success(
            "The analysis plan passed validation and "
            "the calculation completed successfully."
        )

        st.caption(
            response.validation.reason
        )

        st.caption(
            "Gemini determined what analysis was needed. "
            "The numerical result was calculated from the "
            "working dataframe, not generated by the AI."
        )

        # -----------------------------------------------------
        # DISPLAY VERIFIED RESULT
        # -----------------------------------------------------

        display_analysis_result(
            result=response.result,
            plan=response.validation.plan,
        )

        # -----------------------------------------------------
        # DISPLAY BUSINESS INSIGHT
        # -----------------------------------------------------

        display_business_insight(
            insight=response.insight,
            insight_error=response.insight_error,
        )

    # ---------------------------------------------------------
    # GEMINI/API ERRORS
    # ---------------------------------------------------------

    except errors.APIError as error:
        status_box.update(
            label="AI service error",
            state="error",
            expanded=False,
        )

        if error.code == 429:
            st.warning(
                "The AI service has reached its request "
                "limit. Please try again later, or use "
                "Guided questions."
            )

        elif error.code in (
            408,
            500,
            502,
            503,
            504,
        ):
            st.warning(
                "The AI service is busy or taking longer "
                "than expected. Please try again, or use "
                "Guided questions."
            )

        elif error.code in (
            400,
            401,
            403,
            404,
        ):
            st.error(
                "AI questions are currently unavailable "
                "for this request. Please use Guided "
                "questions."
            )

        else:
            st.warning(
                "The AI service could not complete your "
                "request. Please try again shortly."
            )

    # ---------------------------------------------------------
    # PLAN / VALIDATION ERRORS
    # ---------------------------------------------------------

    except (
        ValidationError,
        ValueError,
    ) as error:
        status_box.update(
            label="Analysis could not be completed",
            state="error",
            expanded=False,
        )

        st.warning(
            "The question could not be converted into "
            "a valid analysis."
        )

        st.caption(
            str(error)
        )

    # ---------------------------------------------------------
    # UNEXPECTED ERRORS
    # ---------------------------------------------------------

    except Exception as error:
        status_box.update(
            label="Analysis failed",
            state="error",
            expanded=False,
        )

        st.warning(
            "The analysis could not be completed. "
            "The service may be temporarily unavailable "
            "or the connection may have been interrupted."
        )

        st.caption(
            str(error)
        )


def display_data_qa(
    df: pd.DataFrame,
) -> None:
    """
    Display guided and AI questions for the working dataset.
    """

    st.header(
        "Ask About Your Data"
    )

    st.write(
        "Use Guided questions for predefined calculations, "
        "or AI questions to describe what you want in "
        "natural language."
    )

    if df.empty:
        st.info(
            "The working dataset has no rows to analyze."
        )
        return

    mode = st.radio(
        "Question mode",
        options=[
            "Guided questions",
            "AI questions",
        ],
        horizontal=True,
        key="qa_mode",
    )

    if mode == "Guided questions":
        display_guided_questions(df)

    else:
        display_ai_questions(df)