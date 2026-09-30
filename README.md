# AI Business Intelligence Copilot

An interactive application for uploading, cleaning, exploring, visualizing, and asking questions about CSV datasets.

**[Try the Live Demo](https://ai-bi-copilot-aleezah.streamlit.app/)**

No installation required. Start with the built-in fictional sales dataset or upload your own CSV.

The application preserves the original data while maintaining a separate working copy for analysis.


## Features

- **CSV upload:** Load a dataset and inspect its structure.
- **Sample data:** Explore 124 fictional sales records across six months, including four duplicate rows and missing values for demonstrating cleaning.
- **Dataset overview:** View previews, data types, summary statistics, missing values, duplicates, and unique-value counts.
- **Data cleaning:** Remove duplicate rows and fill selected missing values using numeric medians or categorical modes.
- **Numeric conversion:** Convert supported numeric text, including currency, comma-separated numbers, and percentages.
- **Reset and export:** Restore the original dataset or download the current working copy as a CSV.
- **Visualization:** Create bar charts, scatter plots, and basic line charts.
- **Guided questions:** Calculate row counts, totals, averages, minimums, maximums, and totals by group using dropdown selections.
- **AI questions:** Ask supported calculation questions in natural language.
- **Service error handling:** Receive helpful retry messages when the AI service is unavailable or reaches its request limit. Guided questions remain available.

## How AI Questions Work

Gemini translates a question into a structured query plan. The application validates the operation and column selections, then uses pandas to calculate the answer from the working dataset.

The model receives the question and column names/types. Dataset rows are not included in the AI request, although any data typed into the question is sent.

The application does not execute model-generated Python code.

### Supported Questions

Examples using the sample dataset:

- How many rows are in the dataset?
- What is the total revenue?
- What is the average customer rating?
- What is the highest revenue?
- What is the lowest unit price?
- What is the total revenue by Region?

AI questions currently support one calculation at a time: row count, sum, mean, minimum, maximum, or sum by group.

## Architecture

1. **Input:** Upload a CSV or load the sample dataset.
2. **Session state:** Preserve the original dataset and initialize a working copy.
3. **Preparation:** Clean or convert values in the working copy.
4. **Analysis:** Explore the data, create charts, or select a guided calculation.
5. **AI interpretation:** Translate a natural-language question into a supported query plan.
6. **Validation and calculation:** Validate the plan and calculate the result locally with pandas.
7. **Export:** Download the working dataset.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Application logic |
| Streamlit | Interactive interface and session state |
| pandas | CSV processing, cleaning, and calculations |
| Plotly | Interactive charts |
| Google Gen AI SDK | Gemini API integration |
| Pydantic | Structured query-plan validation |

The current AI model is `gemini-3.1-flash-lite`.

## Project Structure

```text
app.py
components/
    data_cleaning.py
    data_exploration.py
    data_qa.py
    visualization.py
data/
    sample_sales.csv
requirements.txt
.gitignore
README.md
```

For local AI configuration, create `.streamlit/secrets.toml`. This file must remain excluded from Git.

## Run Locally

### 1. Clone the Repository

```bash
git clone https://github.com/AleezahAthar/ai-business-intelligence-copilot.git
cd ai-business-intelligence-copilot
```

### 2. Create a Python 3.11 Virtual Environment

macOS or Linux:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Configure AI Questions — Optional

Create `.streamlit/secrets.toml` in the project folder:

```toml
GEMINI_API_KEY = "your_gemini_api_key"
```

Keep this file out of version control. For a hosted deployment, configure the key through the hosting platform's secrets settings.

Without an API key, sample data, uploads, cleaning, charts, export, and guided questions remain available.

### 5. Start the App

```bash
python -m streamlit run app.py
```

## Try the Demo Workflow

1. Select **Try sample data** in the sidebar.
2. Inspect the dataset in **Overview**.
3. Remove duplicates in **Data Cleaning**: the row count should change from 124 to 120.
4. Create a bar chart of Revenue by Region or a scatter plot of Quantity versus Revenue.
5. Ask **“What is the total revenue by Region?”** in AI questions.
6. Compare the answer with the corresponding guided calculation.
7. Download the prepared CSV or reset to the original sample.

## Data Handling

- The original and working datasets are maintained in Streamlit session state.
- Uploaded data is processed in the environment running the application. For a hosted app, this is the hosting server.
- The application does not currently save uploads or query history to a database.
- AI requests include the question and column metadata.
- Missing numeric values are excluded from supported aggregate calculations. A column or group with no non-missing numeric values produces a missing result.
- Cleaning can change subsequent chart and calculation results.
- Hosted AI requests use the app owner's configured Gemini project and quota.

## Current Limitations

This project is a working prototype.

- Input is limited to CSV files.
- AI questions do not yet support filtering, date ranges, distinct counts, ranking, forecasting, explanations of causes, or multiple calculations.
- Numeric text may need conversion before it can be used in calculations.
- Basic line charts use row order; they do not automatically aggregate data into time-series trends.
- Session state is temporary and is not persistent storage.
- Saved queries and query history are not yet implemented.
- AI interpretation can be imperfect; review the displayed calculation and compare with guided questions when needed.
- AI availability depends on the external service and project quota. Retry messages are provided, but the application does not implement its own automatic retry loop.

## Planned Improvements

- Saved queries and question history.
- Date-aware visualizations and time aggregation.
- Additional validated calculation operations.
- Persistent storage and deployment improvements.