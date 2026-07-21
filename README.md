This project develops an LLM-powered pipeline that automatically transforms natural language data science requests into executable Python workflows.

The goal is to create a safer and more reliable AI assistant for scientific computing by combining large language models (LLMs) with automated validation, execution checks, and iterative code improvement.

Instead of directly trusting generated code, the pipeline introduces multiple safety layers that review, test, and refine the generated workflow before producing final results.

Project Motivation

Large language models can generate powerful data analysis code, but they can also produce:

Incorrect implementations
Invalid assumptions about datasets
Runtime errors
Poor statistical choices
Code that does not match the user's intent

This project explores how an LLM can be integrated into a controlled data science workflow where generated code is continuously evaluated and improved.

The pipeline acts as an AI data scientist that:

Understands a user's scientific request
Generates an analysis workflow
Checks the generated code for correctness
Fixes problems automatically
Executes the final validated workflow
Produces plots, analysis, and results
Pipeline Architecture

The workflow consists of several major components:

User Request
      |
      v
+----------------+
| Intent Parsing |
+----------------+
      |
      v
+----------------+
| LLM Code       |
| Generation     |
+----------------+
      |
      v
+----------------+
| Safety / Code  |
| Validation     |
+----------------+
      |
      v
+----------------+
| Error Detection|
| & Fix Loop     |
+----------------+
      |
      v
+----------------+
| Code Execution |
+----------------+
      |
      v
Results, Graphs, Analysis
Core Components
1. User Request Processing

The user provides a natural language request describing the desired analysis.

Example:

Fit the best polynomial curve to this dataset using AIC model selection,
show uncertainty bounds, and plot the results.

The pipeline interprets:

The requested task
Required libraries
Expected outputs
Dataset structure
2. Dataset Understanding

The pipeline reads structured datasets such as:

JSON files
CSV files

The dataset schema is extracted and provided to the LLM so that generated code is based on the actual data structure.

Example information provided:

Column names
Data types
Available variables
Number of samples

This reduces hallucinated variables and incorrect assumptions.

3. LLM Code Generation

The LLM generates Python code based on:

User instructions
Dataset schema
Workflow requirements
Safety constraints

The generated code can perform tasks such as:

Data visualization
Curve fitting
Statistical analysis
Machine learning workflows
Dataset augmentation
4. Code Validation and Safety Checks

Generated code is not immediately executed.

The pipeline performs validation steps including:

Syntax checking
Runtime error detection
Output verification
Logical consistency checks

If problems are found, the code is returned to the LLM with the error information.

5. Iterative Code Repair

The pipeline uses an automated feedback loop:

Generate Code
      |
      v
Run Validation
      |
      |
  Errors Found?
      |
     Yes
      |
      v
LLM Fixes Code
      |
      v
Validate Again

This allows the system to improve generated workflows without requiring manual debugging.

Example Workflow
Input

Dataset:

fatigue_data.json

User request:

Fit the best curve using first, second, or third degree polynomial regression.
Choose the best model using AIC.
Display uncertainty bounds and plot the result.
Generated Workflow

The pipeline:

Reads the dataset
Extracts variables
Fits polynomial models:
Degree 1
Degree 2
Degree 3
Calculates AIC scores
Selects the best model
Generates uncertainty estimates
Creates visualization
Returns results
Dataset Augmentation and Evaluation

The pipeline also supports generating synthetic datasets to evaluate workflow robustness.

Example process:

Fit an initial model
Add Gaussian noise using a selected standard deviation
Generate synthetic points
Combine original and synthetic data
Train/test split
Refit the model
Compare performance

This allows evaluation of how generated workflows handle noisy scientific data.

Installation

Clone the repository:

git clone <repository-url>
cd <repository-name>

Create a virtual environment:

python -m venv venv

Activate the environment:

Windows:

venv\Scripts\activate

Linux/Mac:

source venv/bin/activate

Install dependencies:

pip install -r requirements.txt
Running the Pipeline

Basic usage:

python pipeline.py \
--data data/example.json \
--request "Analyze this dataset and create a visualization"

Additional options:

python pipeline.py \
--data data/fatigue_data.json \
--request "Fit the best polynomial model and show uncertainty bounds" \
--max-fixes 3
Configuration

The pipeline allows configuration of:

Generation model
Reviewer model
Maximum repair attempts
Execution timeout
Output directories

Example:

MODEL = "gemma4"
REVIEWER_MODEL = "gemma4"

MAX_FIXES = 2
TIMEOUT = 600
Technologies Used
Programming
Python
Machine Learning / Data Science
NumPy
Pandas
SciPy
Matplotlib
scikit-learn
LLM Framework
Ollama
Local LLM inference
Development Tools
Git
PyCharm
VS Code
Research Goals

This project investigates:

How LLMs can automate scientific workflows
How validation improves reliability of generated code
How iterative feedback improves LLM performance
How local LLMs can support scientific computing tasks

The broader goal is creating AI systems that assist researchers while maintaining transparency, reliability, and control.

Future Improvements

Potential future extensions:

More advanced code verification
Better scientific reasoning evaluation
Support for additional datasets and file formats
Multi-agent generation and review systems
Improved uncertainty estimation
Automated experiment comparison
Project Status

Current capabilities:

✅ Natural language workflow requests
✅ Dataset schema understanding
✅ LLM-generated Python analysis code
✅ Automated validation
✅ Iterative code repair
✅ Scientific plotting workflows
✅ Dataset augmentation experiments

Author

Ayyoub K. Aggour

Electrical Engineering Student
University of Maryland
