
from __future__ import annotations

import io

import pandas as pd
import streamlit as st
from docx import Document
from pypdf import PdfReader

from src.resume_matcher import analyse_resume


st.set_page_config(
    page_title="AI Resume Assistant",
    page_icon="📄",
    layout="wide",
)


def extract_text_from_upload(uploaded_file) -> str:
    """Extract text from PDF, DOCX, or TXT uploads."""
    if uploaded_file is None:
        return ""

    file_name = uploaded_file.name.lower()
    file_bytes = uploaded_file.read()

    if file_name.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(file_bytes))
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    if file_name.endswith(".docx"):
        document = Document(io.BytesIO(file_bytes))
        return "\n".join(paragraph.text for paragraph in document.paragraphs)

    if file_name.endswith(".txt"):
        return file_bytes.decode("utf-8", errors="ignore")

    raise ValueError("Unsupported file type. Please upload PDF, DOCX, or TXT.")


def show_skill_table(title: str, skills: dict[str, list[str]]) -> None:
    rows = []
    for category, values in skills.items():
        rows.append(
            {
                "Category": category,
                "Skills": ", ".join(values) if values else "—",
            }
        )
    st.subheader(title)
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


st.title("AI Resume Assistant – NLP & Analytics Project")
st.write(
    "Compare a CV with a job description, identify keyword gaps, and generate practical CV improvement suggestions."
)

with st.sidebar:
    st.header("Input")
    cv_upload = st.file_uploader("Upload CV", type=["pdf", "docx", "txt"])
    job_upload = st.file_uploader("Upload job description", type=["pdf", "docx", "txt"])
    st.caption("You can also paste text below if you do not want to upload files.")

col1, col2 = st.columns(2)

with col1:
    cv_text_area = st.text_area("Paste CV text", height=280, placeholder="Paste your CV text here...")

with col2:
    job_text_area = st.text_area(
        "Paste job description", height=280, placeholder="Paste the vacancy text here..."
    )

try:
    uploaded_cv_text = extract_text_from_upload(cv_upload) if cv_upload else ""
    uploaded_job_text = extract_text_from_upload(job_upload) if job_upload else ""
except ValueError as error:
    st.error(str(error))
    st.stop()

cv_text = uploaded_cv_text or cv_text_area
job_text = uploaded_job_text or job_text_area

analyse_button = st.button("Analyse match", type="primary")

if analyse_button:
    if not cv_text.strip() or not job_text.strip():
        st.warning("Please upload or paste both a CV and a job description.")
        st.stop()

    result = analyse_resume(cv_text, job_text)
    match_percentage = round(result.similarity_score * 100, 1)

    st.header("Results")
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    metric_col1.metric("CV-job similarity", f"{match_percentage}%")
    metric_col2.metric(
        "Matched skills",
        sum(len(values) for values in result.matched_skills.values()),
    )
    metric_col3.metric(
        "Missing relevant skills",
        sum(len(values) for values in result.missing_skills.values()),
    )

    st.progress(min(result.similarity_score, 1.0))

    if match_percentage < 35:
        st.info("Positioning: weak match. The CV needs stronger role-specific evidence.")
    elif match_percentage < 60:
        st.info("Positioning: moderate match. The CV is relevant but should be tailored further.")
    else:
        st.success("Positioning: strong match. Improve wording and measurable evidence.")

    tab1, tab2, tab3 = st.tabs(["Skill match", "Keyword gaps", "Suggestions"])

    with tab1:
        skill_col1, skill_col2 = st.columns(2)
        with skill_col1:
            show_skill_table("Matched skills", result.matched_skills)
        with skill_col2:
            show_skill_table("Missing skills from CV", result.missing_skills)

    with tab2:
        st.subheader("Top job keywords missing or underused in the CV")
        st.dataframe(result.keyword_gaps, use_container_width=True, hide_index=True)
        st.bar_chart(result.keyword_gaps.set_index("keyword")["gap_score"])

    with tab3:
        st.subheader("Tailored CV improvement suggestions")
        for suggestion in result.improvement_suggestions:
            st.write(f"- {suggestion}")

        suggestions_text = "\n".join(f"- {item}" for item in result.improvement_suggestions)
        st.download_button(
            "Download suggestions",
            data=suggestions_text,
            file_name="cv_improvement_suggestions.txt",
            mime="text/plain",
        )

    st.caption(
        "Note: this tool supports CV tailoring and learning. It does not replace recruiter judgement or official ATS scoring."
    )
else:
    st.info("Upload or paste a CV and job description, then click Analyse match.")
