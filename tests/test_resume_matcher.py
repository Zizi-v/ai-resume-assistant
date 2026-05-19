from src.resume_matcher import analyse_resume, compute_similarity, extract_skill_gaps


def test_similarity_returns_valid_score():
    score = compute_similarity("Python SQL dashboard", "Python dashboard analytics")
    assert 0 <= score <= 1


def test_skill_gap_detection():
    cv = "I use Python and Power BI for dashboards."
    job = "The role requires Python, Power BI, NLP, and user adoption."
    matched, missing = extract_skill_gaps(cv, job)
    assert "python" in matched["Programming & Data"]
    assert "power bi" in matched["BI & Analytics"]
    assert "nlp" in missing["AI, ML & NLP"]


def test_complete_analysis_pipeline():
    cv = "Python SQL Power BI dashboard data analysis"
    job = "Python SQL dashboard analytics NLP adoption"
    result = analyse_resume(cv, job)
    assert result.similarity_score >= 0
    assert not result.keyword_gaps.empty
    assert len(result.improvement_suggestions) > 0
