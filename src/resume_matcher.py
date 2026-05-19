"""Core analysis functions for the AI Resume Assistant.

The functions are intentionally simple and explainable so the project is easy to
present in an internship interview. They combine NLP similarity scoring with
skill extraction and keyword-gap analysis.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


SKILL_CATEGORIES: dict[str, list[str]] = {
    "Programming & Data": [
        "python",
        "r",
        "sql",
        "excel",
        "pandas",
        "numpy",
        "scikit-learn",
        "etl",
        "data cleaning",
        "data preprocessing",
        "data modelling",
        "data visualization",
        "github",
        "git",
    ],
    "AI, ML & NLP": [
        "machine learning",
        "artificial intelligence",
        "ai",
        "nlp",
        "natural language processing",
        "tf-idf",
        "cosine similarity",
        "classification",
        "clustering",
        "regression",
        "model evaluation",
        "predictive analytics",
    ],
    "BI & Analytics": [
        "power bi",
        "dashboard",
        "dashboards",
        "kpi",
        "reporting",
        "business intelligence",
        "analytics",
        "data analysis",
        "visual analytics",
        "stakeholder",
        "requirements gathering",
    ],
    "Business & Domain": [
        "process improvement",
        "operations",
        "finance",
        "accounting",
        "industrial management",
        "business process",
        "adoption",
        "change management",
        "user adoption",
        "communication",
        "collaboration",
    ],
}


@dataclass(frozen=True)
class MatchResult:
    similarity_score: float
    matched_skills: dict[str, list[str]]
    missing_skills: dict[str, list[str]]
    keyword_gaps: pd.DataFrame
    improvement_suggestions: list[str]


def clean_text(text: str) -> str:
    """Lowercase text and remove excessive spacing/punctuation noise."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9+#.\-\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _contains_skill(text: str, skill: str) -> bool:
    """Return True when a skill phrase appears as a clean phrase match."""
    text = clean_text(text)
    skill = clean_text(skill)
    pattern = r"(?<![a-z0-9+#.\-])" + re.escape(skill) + r"(?![a-z0-9+#.\-])"
    return re.search(pattern, text) is not None


def compute_similarity(cv_text: str, job_text: str) -> float:
    """Compute CV-job similarity using TF-IDF vectors and cosine similarity."""
    documents = [clean_text(cv_text), clean_text(job_text)]
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
    matrix = vectorizer.fit_transform(documents)
    return float(cosine_similarity(matrix[0:1], matrix[1:2])[0][0])


def extract_skill_gaps(cv_text: str, job_text: str) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Compare job-required skills with skills found in the CV."""
    matched: dict[str, list[str]] = {}
    missing: dict[str, list[str]] = {}

    for category, skills in SKILL_CATEGORIES.items():
        job_skills = [skill for skill in skills if _contains_skill(job_text, skill)]
        matched[category] = [skill for skill in job_skills if _contains_skill(cv_text, skill)]
        missing[category] = [skill for skill in job_skills if not _contains_skill(cv_text, skill)]

    return matched, missing


def extract_keyword_gaps(cv_text: str, job_text: str, top_n: int = 15) -> pd.DataFrame:
    """Find high-value job-description keywords that are weak or absent in the CV."""
    cv_clean = clean_text(cv_text)
    job_clean = clean_text(job_text)

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=120,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9+#.\-]{1,}\b",
    )
    matrix = vectorizer.fit_transform([cv_clean, job_clean])
    terms = np.array(vectorizer.get_feature_names_out())
    cv_scores = matrix[0].toarray().ravel()
    job_scores = matrix[1].toarray().ravel()

    df = pd.DataFrame(
        {
            "keyword": terms,
            "cv_tfidf": cv_scores,
            "job_tfidf": job_scores,
            "gap_score": job_scores - cv_scores,
        }
    )

    df = df[df["job_tfidf"] > 0].sort_values("gap_score", ascending=False)
    df = df[~df["keyword"].isin(ENGLISH_STOP_WORDS)]
    return df.head(top_n).reset_index(drop=True)


def generate_suggestions(
    similarity_score: float,
    matched_skills: dict[str, list[str]],
    missing_skills: dict[str, list[str]],
    keyword_gaps: pd.DataFrame,
) -> list[str]:
    """Create practical CV improvement suggestions from analysis results."""
    suggestions: list[str] = []

    percentage = round(similarity_score * 100)
    if percentage < 35:
        suggestions.append(
            "The CV-job match is still low. Add a targeted project summary and 3-5 bullets using the vacancy language."
        )
    elif percentage < 60:
        suggestions.append(
            "The match is moderate. Improve the CV by making relevant tools, business impact, and project outcomes more explicit."
        )
    else:
        suggestions.append(
            "The match is strong. Focus on sharpening evidence, metrics, and role-specific wording."
        )

    for category, skills in missing_skills.items():
        if skills:
            suggestions.append(
                f"Add evidence for {category}: " + ", ".join(skills[:5]) + "."
            )

    if not keyword_gaps.empty:
        top_keywords = ", ".join(keyword_gaps["keyword"].head(6).tolist())
        suggestions.append(
            f"Use important job-description keywords naturally in the profile, project, or skills section: {top_keywords}."
        )

    suggestions.append(
        "Add measurable outcomes where possible, for example: improved matching quality, reduced manual screening time, or supported faster CV tailoring."
    )
    return suggestions


def analyse_resume(cv_text: str, job_text: str) -> MatchResult:
    """Run the complete CV-job analysis pipeline."""
    similarity = compute_similarity(cv_text, job_text)
    matched, missing = extract_skill_gaps(cv_text, job_text)
    gaps = extract_keyword_gaps(cv_text, job_text)
    suggestions = generate_suggestions(similarity, matched, missing, gaps)

    return MatchResult(
        similarity_score=similarity,
        matched_skills=matched,
        missing_skills=missing,
        keyword_gaps=gaps,
        improvement_suggestions=suggestions,
    )


def keyword_frequency(text: str, keywords: Iterable[str]) -> pd.DataFrame:
    """Count keyword occurrences in text for simple reporting."""
    cleaned = clean_text(text)
    counts = Counter()
    for keyword in keywords:
        counts[keyword] = len(re.findall(re.escape(clean_text(keyword)), cleaned))
    return pd.DataFrame(counts.items(), columns=["keyword", "count"]).sort_values("count", ascending=False)
