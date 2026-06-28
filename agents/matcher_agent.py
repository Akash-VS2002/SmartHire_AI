"""
agents/matcher_agent.py
------------------------
Skill Matching Agent

Responsibilities:
  - Compare extracted resume skills against required JD skills
  - Use semantic similarity (SentenceTransformer cosine) — NOT just keyword matching
  - Compute: matched_skills, missing_skills, skill_match_percentage
"""

import logging
from typing import List, Tuple

from rag.embeddings import get_embeddings, cosine_similarity
from models.state import ResumeState

logger = logging.getLogger(__name__)

# Semantic match threshold — skills with cosine similarity >= this value
# are considered "matched" even if the exact wording differs.
# e.g. "ML" ↔ "Machine Learning" both score above 0.75
SIMILARITY_THRESHOLD = 0.72


def _semantic_match(
    resume_skills: List[str],
    jd_skills: List[str],
    threshold: float = SIMILARITY_THRESHOLD,
) -> Tuple[List[str], List[str]]:
    """
    Perform semantic skill matching using cosine similarity of embeddings.

    For each required JD skill, find the best-matching resume skill.
    If similarity >= threshold → matched; otherwise → missing.

    Args:
        resume_skills: Skills extracted from the resume.
        jd_skills:     Required skills from the job description.
        threshold:     Cosine similarity cutoff for a positive match.

    Returns:
        Tuple of (matched_jd_skills, missing_jd_skills).
    """
    if not resume_skills or not jd_skills:
        return [], list(jd_skills)

    # Embed all skills in a single batch for efficiency
    all_skills = resume_skills + jd_skills
    all_embeddings = get_embeddings(all_skills)

    resume_embeddings = all_embeddings[:len(resume_skills)]
    jd_embeddings = all_embeddings[len(resume_skills):]

    matched: List[str] = []
    missing: List[str] = []

    for jd_skill, jd_emb in zip(jd_skills, jd_embeddings):
        best_score = 0.0
        for res_emb in resume_embeddings:
            score = cosine_similarity(jd_emb, res_emb)
            if score > best_score:
                best_score = score

        if best_score >= threshold:
            matched.append(jd_skill)
            logger.debug(f"  MATCH: '{jd_skill}' (score={best_score:.2f})")
        else:
            missing.append(jd_skill)
            logger.debug(f"  MISS : '{jd_skill}' (best_score={best_score:.2f})")

    return matched, missing


def skill_matcher_node(state: ResumeState) -> ResumeState:
    """
    LangGraph node: Skill Matching Agent.

    Reads extracted_skills and required_skills from state,
    performs semantic matching, and returns enriched state.

    Args:
        state: Current pipeline state.

    Returns:
        Updated state with matched_skills, missing_skills, skill_match_percentage.
    """
    resume_skills: List[str] = state.get("extracted_skills", [])
    required_skills: List[str] = state.get("required_skills", [])
    preferred_skills: List[str] = state.get("preferred_skills", [])

    # Combine required + preferred for comprehensive matching
    all_jd_skills = list(dict.fromkeys(required_skills + preferred_skills))  # deduplicate preserving order

    if not all_jd_skills:
        logger.warning("skill_matcher_node: No JD skills to match against.")
        return {
            **state,
            "matched_skills": [],
            "missing_skills": [],
            "skill_match_percentage": 0.0,
        }

    if not resume_skills:
        logger.warning("skill_matcher_node: No resume skills found.")
        return {
            **state,
            "matched_skills": [],
            "missing_skills": all_jd_skills,
            "skill_match_percentage": 0.0,
        }

    try:
        matched, missing = _semantic_match(
            resume_skills=resume_skills,
            jd_skills=all_jd_skills,
            threshold=SIMILARITY_THRESHOLD,
        )

        total = len(all_jd_skills)
        percentage = round((len(matched) / total) * 100, 1) if total > 0 else 0.0

        logger.info(
            f"Skill Matcher → Matched: {len(matched)}/{total} "
            f"({percentage}%) | Missing: {len(missing)}"
        )

        return {
            **state,
            "matched_skills": matched,
            "missing_skills": missing,
            "skill_match_percentage": percentage,
        }

    except Exception as e:
        logger.error(f"skill_matcher_node failed: {e}")
        return {
            **state,
            "matched_skills": [],
            "missing_skills": all_jd_skills,
            "skill_match_percentage": 0.0,
            "error": str(e),
        }
