"""
models/state.py
----------------
Defines the shared state TypedDict that is passed between all LangGraph nodes.
Every agent reads from and writes to this state object.
"""

from typing import TypedDict, List, Optional


class ResumeState(TypedDict):
    """
    Central state object shared across all agents in the AgentHire AI pipeline.
    Each field is populated progressively as the graph nodes execute.
    """

    # ── Raw inputs ────────────────────────────────────────────────────────────
    resume_text: str          # Full extracted text from the PDF resume
    jd_text: str              # Full job description text entered by the recruiter

    # ── Resume Analyzer outputs ───────────────────────────────────────────────
    candidate_name: str       # Extracted candidate name
    extracted_skills: List[str]   # Skills found in the resume
    experience_summary: str   # Short narrative of total experience
    projects: List[str]       # List of project names / one-liners
    education: str            # Highest / notable education entry
    certifications: List[str] # Any certifications found

    # ── JD Analyzer outputs ───────────────────────────────────────────────────
    required_skills: List[str]    # Must-have skills from the JD
    preferred_skills: List[str]   # Nice-to-have skills
    experience_requirements: str  # e.g. "3+ years in Python"
    responsibilities: List[str]   # Key job responsibilities

    # ── Skill Matcher outputs ─────────────────────────────────────────────────
    matched_skills: List[str]     # Skills present in both resume and JD
    missing_skills: List[str]     # Required JD skills absent from resume
    skill_match_percentage: float # e.g. 75.0

    # ── ATS Scorer outputs ────────────────────────────────────────────────────
    ats_score: float              # 0-100 composite score
    score_reason: str             # Human-readable scoring rationale

    # ── Recommendation Agent output ───────────────────────────────────────────
    recommendation: str           # "SHORTLIST" | "REVIEW" | "REJECT"

    # ── Interview Question Agent outputs ──────────────────────────────────────
    interview_questions: List[str]  # 5-8 targeted questions

    # ── Internal / meta ───────────────────────────────────────────────────────
    error: Optional[str]          # Stores any pipeline error message

    # RAG context snippets injected by workflow.py before the graph runs
    # (private convention: prefix "_" indicates internal pipeline keys)
    _rag_resume_context: Optional[str]  # Retrieved resume chunks for LLM context
    _rag_jd_context: Optional[str]      # Retrieved JD chunks for LLM context

    # Sub-score breakdown stored by scoring_agent for UI transparency
    _sub_scores: Optional[dict]  # {skills, experience, projects, keywords}
