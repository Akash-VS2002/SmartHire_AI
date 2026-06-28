"""
agents/interview_agent.py
--------------------------
Interview Question Generator Agent

Responsibilities:
  - Generate 5-8 targeted interview questions using Groq LLM + RAG context
  - Question categories:
      • Technical depth questions (core skills match)
      • Project-based questions (candidate's own projects)
      • Skill gap questions (missing skills — probe learning ability)
  - Uses retrieved RAG context for grounded, specific questions
"""

import logging
import os
import re
from typing import List

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from models.state import ResumeState

logger = logging.getLogger(__name__)

# ── System prompt ─────────────────────────────────────────────────────────────
INTERVIEW_SYSTEM = """You are a senior technical recruiter and interviewer AI.
Your task is to generate targeted, insightful interview questions for a candidate.

Generate exactly 7 questions in three categories:
  1. Technical Questions (3 questions) — test depth in matched skills
  2. Project Questions (2 questions) — explore the candidate's own projects
  3. Skill Gap Questions (2 questions) — probe missing skills and learning ability

Format your response as a numbered list (1. to 7.) with NO extra headers, bullets, or markdown.
Each question should be on its own line. Questions must be specific to the candidate's profile.
"""


def _parse_questions(text: str) -> List[str]:
    """
    Extract numbered questions from LLM response.

    Handles formats like:
      "1. What is..."
      "1) What is..."
    """
    lines = text.strip().splitlines()
    questions: List[str] = []

    for line in lines:
        line = line.strip()
        # Match lines starting with a number followed by . or )
        match = re.match(r"^\d+[.)]\s+(.+)$", line)
        if match:
            question = match.group(1).strip()
            if question:
                questions.append(question)

    # Fallback: if regex fails, split by newline and filter empties
    if not questions:
        questions = [l.strip() for l in lines if l.strip() and len(l.strip()) > 10]

    return questions[:8]  # Cap at 8


def interview_generator_node(state: ResumeState) -> ResumeState:
    """
    LangGraph node: Interview Question Generator Agent.

    Uses candidate profile + RAG context to produce targeted questions.

    Args:
        state: Current pipeline state (fully populated by prior agents).

    Returns:
        Updated state with interview_questions list.
    """
    from dotenv import load_dotenv
    load_dotenv()

    candidate_name = state.get("candidate_name", "Unknown")
    matched_skills = state.get("matched_skills", [])
    missing_skills = state.get("missing_skills", [])
    projects = state.get("projects", [])
    experience = state.get("experience_summary", "")
    ats_score = state.get("ats_score", 0.0)
    jd_responsibilities = state.get("responsibilities", [])

    # RAG context injected by workflow.py
    rag_context = state.get("_rag_resume_context", "")
    jd_context = state.get("_rag_jd_context", "")

    user_content = f"""CANDIDATE: {candidate_name}
ATS SCORE: {ats_score}/100

MATCHED SKILLS: {', '.join(matched_skills[:10]) if matched_skills else 'None'}
MISSING SKILLS: {', '.join(missing_skills[:10]) if missing_skills else 'None'}
EXPERIENCE: {experience[:300] if experience else 'Not provided'}
PROJECTS: {'; '.join(projects[:4]) if projects else 'None listed'}

JOB RESPONSIBILITIES:
{chr(10).join(f'- {r}' for r in jd_responsibilities[:5]) if jd_responsibilities else 'N/A'}

RELEVANT RESUME CONTEXT (from RAG):
{rag_context[:800] if rag_context else 'N/A'}

RELEVANT JD CONTEXT (from RAG):
{jd_context[:400] if jd_context else 'N/A'}

Generate 7 interview questions (3 technical, 2 project, 2 skill-gap)."""

    try:
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.5,   # Slightly higher temp for creative/diverse questions
            api_key=os.environ.get("GROQ_API_KEY"),
        )

        messages = [
            SystemMessage(content=INTERVIEW_SYSTEM),
            HumanMessage(content=user_content),
        ]

        response = llm.invoke(messages)
        raw_text = response.content.strip()
        questions = _parse_questions(raw_text)

        # Ensure we have at least some questions
        if not questions:
            questions = [
                f"Can you walk us through your experience with {skill}?"
                for skill in (matched_skills[:3] or ["your core skills"])
            ]

        logger.info(f"Interview Agent → Generated {len(questions)} questions for '{candidate_name}'.")

        return {
            **state,
            "interview_questions": questions,
        }

    except Exception as e:
        logger.error(f"interview_generator_node failed: {e}")
        # Provide default questions on failure
        fallback = [
            f"Tell me about a project where you used {matched_skills[0] if matched_skills else 'your core skills'}.",
            "How do you stay current with emerging technologies in your field?",
            f"What steps would you take to learn {missing_skills[0] if missing_skills else 'a new technology'} quickly?",
        ]
        return {
            **state,
            "interview_questions": fallback,
            "error": str(e),
        }
