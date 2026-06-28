"""
agents/scoring_agent.py
------------------------
ATS Scoring Agent + Recommendation Agent

Responsibilities:
  - Calculate a composite ATS score (0-100) using weighted criteria:
      Skills Match   → 50%
      Experience     → 20%
      Projects       → 20%
      Keyword Match  → 10%
  - Determine recommendation tier based on score:
      90-100 → SHORTLIST
      70-89  → REVIEW
      <70    → REJECT
  - Generate a human-readable scoring rationale via Groq LLM
"""

import logging
import os
import re
from typing import List

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from models.state import ResumeState

logger = logging.getLogger(__name__)

# ── Score weights ─────────────────────────────────────────────────────────────
WEIGHT_SKILLS      = 0.50   # 50%
WEIGHT_EXPERIENCE  = 0.20   # 20%
WEIGHT_PROJECTS    = 0.20   # 20%
WEIGHT_KEYWORDS    = 0.10   # 10%

# ── Recommendation thresholds ────────────────────────────────────────────────
SHORTLIST_MIN = 90
REVIEW_MIN    = 70


# ── LLM system prompt ─────────────────────────────────────────────────────────
SCORING_SYSTEM = """You are an expert HR ATS scoring assistant.
Given a candidate's profile and job requirements, write a concise (2-3 sentences)
reasoning paragraph explaining the ATS score. Focus on strengths, gaps, and fit.
Be direct and professional. Do NOT repeat the score number — just explain the reasoning."""


def _score_experience(experience_summary: str, experience_requirement: str) -> float:
    """
    Heuristic experience score (0-100).
    Checks if years/keywords from the JD requirement appear in the resume experience.
    """
    if not experience_summary:
        return 30.0   # Minimum if no experience found

    # Extract year numbers from the requirement
    req_years = re.findall(r"\d+", experience_requirement)
    exp_years = re.findall(r"\d+", experience_summary)

    if req_years and exp_years:
        try:
            req = int(req_years[0])
            exp = max(int(y) for y in exp_years)
            if exp >= req:
                return 100.0
            elif exp >= req * 0.7:
                return 70.0
            else:
                return 40.0
        except ValueError:
            pass

    # Fallback: if experience text is reasonably long, give partial credit
    if len(experience_summary.split()) > 30:
        return 65.0
    elif len(experience_summary.split()) > 10:
        return 50.0
    return 30.0


def _score_projects(projects: List[str], required_skills: List[str]) -> float:
    """
    Project relevance score (0-100).
    Checks how many required skills appear in project descriptions.
    """
    if not projects:
        return 20.0  # Baseline for no projects listed

    if not required_skills:
        return 60.0  # Can't judge relevance without JD skills

    # Flatten project text and lowercase
    project_text = " ".join(projects).lower()
    required_lower = [s.lower() for s in required_skills]

    hits = sum(1 for skill in required_lower if skill in project_text)
    ratio = hits / len(required_lower) if required_lower else 0
    # Scale: 0 hits → 20, all hits → 100
    return round(20 + (ratio * 80), 1)


def _score_keywords(resume_text: str, jd_text: str) -> float:
    """
    Keyword relevance score (0-100).
    Counts overlapping significant words between resume and JD.
    """
    if not resume_text or not jd_text:
        return 0.0

    # Common stop words to ignore
    stopwords = {
        "and", "or", "the", "a", "an", "in", "on", "at", "to", "for",
        "of", "with", "is", "are", "was", "be", "have", "has", "will",
        "that", "this", "we", "you", "your", "our", "their", "as", "by",
    }

    def extract_words(text: str):
        words = re.findall(r"\b[a-zA-Z][a-zA-Z+#.]{2,}\b", text.lower())
        return {w for w in words if w not in stopwords}

    resume_words = extract_words(resume_text)
    jd_words = extract_words(jd_text)

    if not jd_words:
        return 50.0

    overlap = resume_words & jd_words
    ratio = len(overlap) / len(jd_words)
    return round(min(ratio * 100 * 1.5, 100.0), 1)  # Boost and cap at 100


def _determine_recommendation(score: float) -> str:
    """Map ATS score to recommendation tier."""
    if score >= SHORTLIST_MIN:
        return "SHORTLIST"
    elif score >= REVIEW_MIN:
        return "REVIEW"
    else:
        return "REJECT"


def _generate_reason(state: ResumeState, score: float) -> str:
    """
    Ask Groq LLM to write a concise scoring rationale.
    Returns a default string on failure.
    """
    from dotenv import load_dotenv
    load_dotenv()

    try:
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.3,
            api_key=os.environ.get("GROQ_API_KEY"),
        )

        user_content = f"""CANDIDATE: {state.get('candidate_name', 'Unknown')}
ATS SCORE: {score}/100
MATCHED SKILLS: {', '.join(state.get('matched_skills', [])) or 'None'}
MISSING SKILLS: {', '.join(state.get('missing_skills', [])) or 'None'}
EXPERIENCE: {state.get('experience_summary', 'N/A')}
PROJECTS: {'; '.join(state.get('projects', [])) or 'None listed'}
JOB REQUIREMENTS: {state.get('experience_requirements', 'N/A')}

Write the ATS scoring rationale (2-3 sentences)."""

        response = llm.invoke([
            SystemMessage(content=SCORING_SYSTEM),
            HumanMessage(content=user_content),
        ])
        return response.content.strip()

    except Exception as e:
        logger.warning(f"LLM reason generation failed: {e}. Using default reason.")
        matched = state.get("matched_skills", [])
        missing = state.get("missing_skills", [])
        return (
            f"Candidate matched {len(matched)} of the required skills. "
            f"{'Strong technical fit.' if score >= 80 else 'Notable gaps exist.'} "
            f"Missing: {', '.join(missing[:3]) if missing else 'none'}."
        )


def ats_scorer_node(state: ResumeState) -> ResumeState:
    """
    LangGraph node: ATS Scoring + Recommendation Agent.

    Calculates composite score, determines recommendation, and
    generates a human-readable rationale.

    Args:
        state: Current pipeline state.

    Returns:
        Updated state with ats_score, recommendation, score_reason.
    """
    try:
        skill_match_pct = state.get("skill_match_percentage", 0.0)
        experience_summary = state.get("experience_summary", "")
        experience_req = state.get("experience_requirements", "")
        projects = state.get("projects", [])
        required_skills = state.get("required_skills", [])
        resume_text = state.get("resume_text", "")
        jd_text = state.get("jd_text", "")

        # ── Component scores (0-100 each) ─────────────────────────────────────
        skills_score     = skill_match_pct                                          # Already 0-100
        experience_score = _score_experience(experience_summary, experience_req)
        projects_score   = _score_projects(projects, required_skills)
        keywords_score   = _score_keywords(resume_text, jd_text)

        # ── Weighted composite score ───────────────────────────────────────────
        composite = (
            skills_score     * WEIGHT_SKILLS +
            experience_score * WEIGHT_EXPERIENCE +
            projects_score   * WEIGHT_PROJECTS +
            keywords_score   * WEIGHT_KEYWORDS
        )
        final_score = round(min(max(composite, 0), 100), 1)

        recommendation = _determine_recommendation(final_score)
        reason = _generate_reason(state, final_score)

        logger.info(
            f"ATS Scorer → Score: {final_score} | "
            f"Skills: {skills_score:.1f} | Exp: {experience_score:.1f} | "
            f"Projects: {projects_score:.1f} | Keywords: {keywords_score:.1f} | "
            f"Recommendation: {recommendation}"
        )

        return {
            **state,
            "ats_score": final_score,
            "score_reason": reason,
            "recommendation": recommendation,
            # Store sub-scores for transparency (used in UI)
            "_sub_scores": {
                "skills": round(skills_score, 1),
                "experience": round(experience_score, 1),
                "projects": round(projects_score, 1),
                "keywords": round(keywords_score, 1),
            },
        }

    except Exception as e:
        logger.error(f"ats_scorer_node failed: {e}")
        return {
            **state,
            "ats_score": 0.0,
            "score_reason": f"Scoring failed: {e}",
            "recommendation": "REJECT",
            "error": str(e),
        }
