import json

import pandas as pd
from google import genai
from google.genai import types

from components.analysis_engine import AnalysisResult
from components.analysis_planner import (
    AnalysisPlan,
    MODEL_NAME,
)


MAX_RESULT_ROWS = 50


def generate_business_insight(
    question: str,
    plan: AnalysisPlan,
    result: AnalysisResult,
    api_key: str,
) -> str:
    """
    Generate a short natural-language explanation of a verified
    pandas analysis result.

    Important:
    - Gemini does not receive the original dataset.
    - Gemini receives only the user's question,
      validated analysis plan, and verified pandas result.
    - Gemini explains the result but does not calculate it.
    """

    verified_result = build_verified_result_payload(
        result=result,
    )

    instructions = """
    You are the interpretation layer of a business analytics
    application.

    A separate pandas analytics engine has already performed
    and verified the calculation.

    Your job is ONLY to explain the verified result clearly.

    You must follow these rules:

    1. Treat the supplied pandas result as the source of truth.

    2. Do not recalculate or modify the result.

    3. Do not invent:
       - causes
       - explanations
       - trends not shown in the result
       - business context not supplied
       - future predictions
       - recommendations unsupported by the data

    4. Do not claim causation.

    5. Do not introduce numbers that cannot be derived directly
       from the verified result.

    6. You may make simple comparisons that are directly supported
       by the supplied values.

    7. If the result does not support a conclusion, say so.

    8. Keep the explanation concise:
       normally 2 to 4 sentences.

    9. Use plain business language.

    10. Do not mention implementation details such as:
        - pandas
        - JSON
        - the analysis engine
        - the prompt
        - the AI model

    11. Answer the user's original question directly.

    12. For rankings:
        identify the highest or lowest result when appropriate.

    13. For comparisons:
        summarize the most important differences visible in
        the supplied result.

    14. For trends:
        describe only observable changes across the provided
        time periods.

    15. For aggregate results:
        clearly state the calculated value.

    16. For percentage-change results:
        describe the observed percentage changes without
        attributing a cause.

    The verified result has already been calculated.
    Your role is interpretation, not computation.
    """

    request_data = {
        "original_question": question,
        "validated_analysis_plan": plan.model_dump(
            mode="json"
        ),
        "verified_result": verified_result,
    }

    with genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(
            timeout=120000,
        ),
    ) as client:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=json.dumps(
                request_data,
                default=str,
            ),
            config=types.GenerateContentConfig(
                system_instruction=instructions,
                temperature=0.2,
                automatic_function_calling=(
                    types.AutomaticFunctionCallingConfig(
                        disable=True
                    )
                ),
            ),
        )

    if not response.text:
        raise ValueError(
            "Gemini returned no business insight."
        )

    return response.text.strip()


def build_verified_result_payload(
    result: AnalysisResult,
) -> dict:
    """
    Convert an AnalysisResult into a JSON-safe structure
    that can be sent to Gemini.

    Only the calculated result is included.
    The original dataframe is never included.
    """

    if result.result_type == "scalar":
        return {
            "result_type": "scalar",
            "value": make_json_safe(
                result.value
            ),
        }

    if result.result_type == "table":
        if result.data is None:
            return {
                "result_type": "table",
                "columns": [],
                "rows": [],
                "row_count": 0,
                "truncated": False,
            }

        total_rows = len(result.data)

        result_for_ai = result.data.head(
            MAX_RESULT_ROWS
        ).copy()

        records = result_for_ai.to_dict(
            orient="records"
        )

        safe_records = [
            {
                str(key): make_json_safe(value)
                for key, value in row.items()
            }
            for row in records
        ]

        return {
            "result_type": "table",
            "columns": [
                str(column)
                for column in result.data.columns
            ],
            "rows": safe_records,
            "row_count": total_rows,
            "truncated": (
                total_rows > MAX_RESULT_ROWS
            ),
        }

    raise ValueError(
        f"Unsupported result type: "
        f"{result.result_type}"
    )


def make_json_safe(value):
    """
    Convert common pandas and NumPy values into values
    that can safely be serialized as JSON.
    """

    if pd.isna(value):
        return None

    if isinstance(
        value,
        pd.Timestamp,
    ):
        return value.isoformat()

    if hasattr(
        value,
        "item",
    ):
        try:
            return value.item()
        except (ValueError, AttributeError):
            pass

    return value