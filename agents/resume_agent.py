"""
agents/resume_agent.py
-----------------------
Resume Analyzer Agent

Responsibilities:
  - Read the raw resume text (+ RAG context)
  - Extract structured information via Groq LLM
  - Populate state fields: candidate_name, extracted_skills,
    experience_summary, projects, education, certifications
"""

import json
import logging
import re
from typing import Any, Dict

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from models.state import ResumeState

logger = logging.getLogger(__name__)

# ── System prompt ─────────────────────────────────────────────────────────────
RESUME_ANALYZER_SYSTEM = """You are an expert HR Resume Analyzer AI.
Your task is to extract structured information from a candidate's resume.

Always respond with valid JSON only — no extra commentary, no markdown fences.

JSON schema (all fields required, use empty list/string if not found):
{
  "name": "<Full candidate name>",
  "education": "<Highest or most recent degree and institution>",
  "experience": "<1-3 sentence summary of total experience and key roles>",
  "skills": ["skill1", "skill2", ...],
  "projects": ["Project title – one-line description", ...],
  "certifications": ["cert1", "cert2", ...]
}
"""


def _parse_llm_json(text: str) -> Dict[str, Any]:
    """
    Robustly parse JSON from LLM response.
    Strips markdown fences if present, then loads JSON.
    """
    # Strip ```json ... ``` or ``` ... ``` fences
    text = re.sub(r"```(?:json)?", "", text).replace("```", "").strip()
    # Find the first { ... } block
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        text = match.group(0)
    return json.loads(text)


def resume_analyzer_node(state: ResumeState) -> ResumeState:
    """
    LangGraph node: Resume Analyzer Agent.

    Reads `resume_text` and optional RAG context from state,
    calls Groq LLM, and returns enriched state.

    Args:
        state: Current pipeline state.

    Returns:
        Updated state with resume fields populated.
    """
    import os
    from dotenv import load_dotenv
    load_dotenv()

    resume_text = state.get("resume_text", "").strip()
    if not resume_text:
        logger.error("resume_analyzer_node: No resume text in state.")
        return {**state, "error": "Resume text is empty.", "candidate_name": "Unknown"}

    # Pull relevant RAG context if a vector store is attached via state
    rag_context = state.get("_rag_resume_context", "")  # injected by workflow.py

    # Build prompt
    user_content = f"""RESUME TEXT:
{resume_text[:4000]}  

ADDITIONAL CONTEXT (from vector search):
{rag_context[:1000] if rag_context else "N/A"}

Extract all information and return valid JSON."""

    try:
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0,
            api_key=os.environ.get("GROQ_API_KEY"),
        )

        messages = [
            SystemMessage(content=RESUME_ANALYZER_SYSTEM),
            HumanMessage(content=user_content),
        ]

        response = llm.invoke(messages)
        raw_json = response.content.strip()

        parsed = _parse_llm_json(raw_json)

        # Normalise and merge into state
        name = parsed.get("name", "Unknown Candidate")
        skills = parsed.get("skills", [])
        experience = parsed.get("experience", "")
        projects = parsed.get("projects", [])
        education = parsed.get("education", "")
        certifications = parsed.get("certifications", [])

        logger.info(f"Resume Analyzer → Candidate: '{name}', Skills: {len(skills)}")

        return {
            **state,
            "candidate_name": name,
            "extracted_skills": [s.strip() for s in skills if s.strip()],
            "experience_summary": experience,
            "projects": projects,
            "education": education,
            "certifications": certifications,
        }

    except json.JSONDecodeError as e:
        logger.error(f"JSON parse error in resume_analyzer_node: {e}")
        # Fallback: heuristic name from text cleaner
        from utils.text_cleaner import extract_candidate_name_heuristic
        return {
            **state,
            "candidate_name": extract_candidate_name_heuristic(resume_text),
            "extracted_skills": [],
            "experience_summary": resume_text[:300],
            "projects": [],
            "education": "",
            "certifications": [],
            "error": f"JSON parse error: {e}",
        }
    except Exception as e:
        logger.error(f"resume_analyzer_node failed: {e}")
        return {**state, "error": str(e), "candidate_name": "Unknown"}
