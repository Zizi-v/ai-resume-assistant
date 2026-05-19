# AI Resume Assistant – NLP & Analytics Project

A Streamlit application that compares a CV with a job description and gives practical CV improvement suggestions using NLP and analytics.

## Why this project matters

This project demonstrates practical AI adoption thinking: it turns a real job-search problem into an interactive tool. It combines NLP, analytics, a simple user interface, and business-oriented recommendations.

## Features

- Upload or paste a CV and job description
- Extract text from PDF, DOCX, or TXT files
- Calculate CV-job similarity using TF-IDF vectorisation and cosine similarity
- Detect matched and missing skills
- Identify job-description keywords that are missing or underused in the CV
- Generate tailored CV improvement suggestions
- Download the improvement suggestions as a text file

## Tech stack

- Python
- Streamlit
- scikit-learn
- pandas
- pypdf
- python-docx

## Project structure

```text
ai-resume-assistant/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── src/
│   └── resume_matcher.py
├── samples/
│   ├── sample_cv.txt
│   └── sample_job_description.txt
└── tests/
    └── test_resume_matcher.py
```

## How to run locally

1. Clone the repository or download the project folder.
2. Open a terminal inside the project folder.
3. Create a virtual environment:

```bash
python -m venv .venv
```

4. Activate the virtual environment:

On macOS/Linux:

```bash
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

5. Install dependencies:

```bash
pip install -r requirements.txt
```

6. Run the Streamlit app:

```bash
streamlit run app.py
```

## How to use

1. Upload or paste your CV.
2. Upload or paste a job description.
3. Click **Analyse match**.
4. Review the similarity score, matched skills, missing skills, keyword gaps, and improvement suggestions.
5. Download the suggestions and use them to tailor your CV.

## Example CV project bullet

```text
AI Resume Assistant – NLP & Analytics Project
Personal Project – 2026

Built an AI-powered Resume Assistant using Python and Streamlit to compare CVs with job descriptions.
Applied NLP techniques including TF-IDF vectorisation and cosine similarity to analyse resume-job matching.
Identified missing skills, keyword gaps, and generated tailored CV improvement suggestions.
Designed an interactive interface to support internship and job application optimisation.
```

## Future improvements

- Add semantic embeddings for deeper meaning-based matching
- Add multilingual CV/job-description support
- Add export to PDF report
- Add a database for tracking multiple applications
- Add role-specific recommendation templates for AI Adoption, BI, and Data Analyst roles

## Disclaimer

This tool is for learning, CV tailoring, and decision support. It is not an official ATS score and does not guarantee interview selection.
