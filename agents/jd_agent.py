"""
agents/jd_agent.py
-------------------
Job Description Analyzer Agent

Responsibilities:
  - Parse the raw job description text
  - Extract required skills, preferred skills, experience requirements,
    and key responsibilities via Groq LLM
  - Populate state: required_skills, preferred_skills,
    experience_requirements, responsibilities
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
JD_ANALYZER_SYSTEM = """You are an expert HR Job Description Analyzer AI.
Your task is to extract structured information from a job description.

Always respond with valid JSON only — no extra commentary, no markdown fences.

JSON schema (all fields required, use empty lists/strings if not found):
{
  "required_skills": ["skill1", "skill2", ...],
  "preferred_skills": ["skill1", "skill2", ...],
  "experience_requirements": "<e.g. '3+ years in Python and ML'>",
  "responsibilities": ["responsibility1", "responsibility2", ...]
}

Rules:
- required_skills: Must-have technical and soft skills explicitly stated as required/must.
- preferred_skills: Nice-to-have / bonus skills labeled preferred/plus/bonus.
- If the JD does not distinguish, put all skills in required_skills.
"""


def _parse_llm_json(text: str) -> Dict[str, Any]:
    """Strip markdown fences and parse the first JSON object from LLM output."""
    text = re.sub(r"```(?:json)?", "", text).replace("```", "").strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        text = match.group(0)
    return json.loads(text)


def jd_analyzer_node(state: ResumeState) -> ResumeState:
    """
    LangGraph node: Job Description Analyzer Agent.

    Reads `jd_text` from state, calls Groq LLM, and returns
    state enriched with JD structured data.

    Args:
        state: Current pipeline state.

    Returns:
        Updated state with JD analysis fields.
    """
    import os
    from dotenv import load_dotenv
    load_dotenv()

    jd_text = state.get("jd_text", "").strip()
    if not jd_text:
        logger.error("jd_analyzer_node: No JD text in state.")
        return {
            **state,
            "required_skills": [],
            "preferred_skills": [],
            "experience_requirements": "",
            "responsibilities": [],
            "error": "Job description text is empty.",
        }

    # Optionally include RAG context from vector store
    rag_context = state.get("_rag_jd_context", "")

    user_content = f"""JOB DESCRIPTION TEXT:
{jd_text[:4000]}

ADDITIONAL CONTEXT (from vector search):
{rag_context[:800] if rag_context else "N/A"}

Extract all information and return valid JSON."""

    try:
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0,
            api_key=os.environ.get("GROQ_API_KEY"),
        )

        messages = [
            SystemMessage(content=JD_ANALYZER_SYSTEM),
            HumanMessage(content=user_content),
        ]

        response = llm.invoke(messages)
        raw_json = response.content.strip()
        parsed = _parse_llm_json(raw_json)

        required = parsed.get("required_skills", [])
        preferred = parsed.get("preferred_skills", [])
        experience_req = parsed.get("experience_requirements", "")
        responsibilities = parsed.get("responsibilities", [])

        logger.info(
            f"JD Analyzer → Required skills: {len(required)}, "
            f"Preferred: {len(preferred)}"
        )

        return {
            **state,
            "required_skills": [s.strip() for s in required if s.strip()],
            "preferred_skills": [s.strip() for s in preferred if s.strip()],
            "experience_requirements": experience_req,
            "responsibilities": responsibilities,
        }

    except json.JSONDecodeError as e:
        logger.error(f"JSON parse error in jd_analyzer_node: {e}")
        # Fallback: crude keyword extraction
        words = re.findall(r"\b[A-Za-z][a-zA-Z+#.]{2,}\b", jd_text)
        skills_guess = list(set(words))[:20]
        return {
            **state,
            "required_skills": skills_guess,
            "preferred_skills": [],
            "experience_requirements": "",
            "responsibilities": [],
            "error": f"JD JSON parse error: {e}",
        }
    except Exception as e:
        logger.error(f"jd_analyzer_node failed: {e}")
        return {
            **state,
            "required_skills": [],
            "preferred_skills": [],
            "experience_requirements": "",
            "responsibilities": [],
            "error": str(e),
        }
