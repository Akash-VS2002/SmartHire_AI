"""
graph/workflow.py
-----------------
LangGraph multi-agent workflow for AgentHire AI.

Pipeline:
  START
    ↓
  resume_analyzer   ← Resume Analyzer Agent
    ↓
  jd_analyzer       ← Job Description Analyzer Agent
    ↓
  skill_matcher     ← Skill Matching Agent (semantic similarity)
    ↓
  ats_scorer        ← ATS Scoring + Recommendation Agent
    ↓
  interview_generator ← Interview Question Generator
    ↓
  END

Each node receives the full ResumeState dict and returns an updated copy.
The RAG vector store is initialised before the graph runs and context
is injected into the state as private '_rag_*' keys.
"""

import logging
from typing import Optional, Dict, Any

from langgraph.graph import StateGraph, END, START

from models.state import ResumeState
from agents.resume_agent import resume_analyzer_node
from agents.jd_agent import jd_analyzer_node
from agents.matcher_agent import skill_matcher_node
from agents.scoring_agent import ats_scorer_node
from agents.interview_agent import interview_generator_node
from rag.vectorstore import ResumeVectorStore
from utils.text_cleaner import clean_text, extract_candidate_name_heuristic

logger = logging.getLogger(__name__)


def _build_graph() -> StateGraph:
    """
    Construct and compile the LangGraph StateGraph.

    Returns:
        Compiled LangGraph runnable.
    """
    graph = StateGraph(ResumeState)  # type: ignore[arg-type]

    # ── Register nodes ────────────────────────────────────────────────────────
    graph.add_node("resume_analyzer",    resume_analyzer_node)
    graph.add_node("jd_analyzer",        jd_analyzer_node)
    graph.add_node("skill_matcher",      skill_matcher_node)
    graph.add_node("ats_scorer",         ats_scorer_node)
    graph.add_node("interview_generator", interview_generator_node)

    # ── Define edges (linear pipeline) ───────────────────────────────────────
    graph.add_edge(START,                "resume_analyzer")
    graph.add_edge("resume_analyzer",    "jd_analyzer")
    graph.add_edge("jd_analyzer",        "skill_matcher")
    graph.add_edge("skill_matcher",      "ats_scorer")
    graph.add_edge("ats_scorer",         "interview_generator")
    graph.add_edge("interview_generator", END)

    return graph.compile()


# Compile once at module level — reused across all runs
_COMPILED_GRAPH = _build_graph()


def run_pipeline(
    resume_text: str,
    jd_text: str,
    vector_store: Optional[ResumeVectorStore] = None,
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Execute the full AgentHire AI pipeline for a single candidate.

    Steps:
      1. Clean input texts.
      2. Index texts into ChromaDB (if vector_store provided or create new).
      3. Retrieve RAG context snippets for both resume and JD.
      4. Run the compiled LangGraph pipeline.
      5. Return the final state dict.

    Args:
        resume_text:  Extracted raw text from a candidate's PDF resume.
        jd_text:      Job description text entered by the recruiter.
        vector_store: Pre-built ResumeVectorStore (shared across candidates
                      so the JD is indexed only once). If None, a fresh
                      store is created.
        session_id:   Optional identifier for this analysis session.

    Returns:
        Final ResumeState dict with all agent outputs populated.
    """
    if not resume_text.strip():
        logger.error("run_pipeline: Empty resume text provided.")
        return {"error": "Resume text is empty.", "candidate_name": "Unknown"}

    if not jd_text.strip():
        logger.error("run_pipeline: Empty JD text provided.")
        return {"error": "Job description is empty.", "candidate_name": "Unknown"}

    # ── 1. Clean texts ────────────────────────────────────────────────────────
    clean_resume = clean_text(resume_text)
    clean_jd = clean_text(jd_text)

    # Quick heuristic name for indexing before LLM runs
    heuristic_name = extract_candidate_name_heuristic(clean_resume)
    safe_name = heuristic_name.replace(" ", "_")[:30]

    # ── 2. RAG — index texts ──────────────────────────────────────────────────
    vs = vector_store or ResumeVectorStore(session_id=session_id)

    # Index resume (always, per candidate)
    vs.add_resume(clean_resume, candidate_name=safe_name)

    # Index JD only if not already present (check collection count)
    if vs._jd_collection is None:
        vs.add_job_description(clean_jd)

    # ── 3. RAG — retrieve context ─────────────────────────────────────────────
    resume_query = "technical skills experience projects certifications"
    jd_query     = "required skills responsibilities experience qualifications"

    rag_resume_ctx = vs.get_resume_context(resume_query, candidate_name=safe_name, top_k=4)
    rag_jd_ctx     = vs.get_jd_context(jd_query, top_k=4)

    # ── 4. Build initial state ────────────────────────────────────────────────
    initial_state: ResumeState = {
        # Core inputs
        "resume_text": clean_resume,
        "jd_text": clean_jd,
        # RAG context (private — consumed by agents)
        "_rag_resume_context": rag_resume_ctx,
        "_rag_jd_context": rag_jd_ctx,
        # Pre-fill defaults so every key exists (agents may override)
        "candidate_name": heuristic_name,
        "extracted_skills": [],
        "experience_summary": "",
        "projects": [],
        "education": "",
        "certifications": [],
        "required_skills": [],
        "preferred_skills": [],
        "experience_requirements": "",
        "responsibilities": [],
        "matched_skills": [],
        "missing_skills": [],
        "skill_match_percentage": 0.0,
        "ats_score": 0.0,
        "score_reason": "",
        "recommendation": "REJECT",
        "interview_questions": [],
        "error": None,
    }

    # ── 5. Run LangGraph pipeline ─────────────────────────────────────────────
    logger.info(f"Starting pipeline for candidate: '{heuristic_name}'")
    final_state = _COMPILED_GRAPH.invoke(initial_state)
    logger.info(
        f"Pipeline complete → {final_state.get('candidate_name')} | "
        f"Score: {final_state.get('ats_score')} | "
        f"Rec: {final_state.get('recommendation')}"
    )

    return final_state


def run_batch_pipeline(
    resume_texts: list[str],
    jd_text: str,
    session_id: Optional[str] = None,
) -> list[Dict[str, Any]]:
    """
    Run the pipeline for multiple candidates sharing the same JD.

    The JD is indexed into ChromaDB only once; each resume is
    processed sequentially through the same pipeline.

    Args:
        resume_texts: List of raw resume text strings.
        jd_text:      Shared job description text.
        session_id:   Optional session identifier.

    Returns:
        List of final state dicts, one per candidate, sorted by ATS score desc.
    """
    # Shared vector store — JD indexed once
    vs = ResumeVectorStore(session_id=session_id)
    clean_jd = clean_text(jd_text)
    vs.add_job_description(clean_jd)

    results = []
    for i, resume_text in enumerate(resume_texts):
        logger.info(f"Processing resume {i + 1}/{len(resume_texts)} …")
        try:
            result = run_pipeline(
                resume_text=resume_text,
                jd_text=jd_text,
                vector_store=vs,
                session_id=session_id,
            )
            results.append(result)
        except Exception as e:
            logger.error(f"Pipeline failed for resume {i + 1}: {e}")
            results.append({
                "candidate_name": f"Candidate {i + 1}",
                "ats_score": 0.0,
                "error": str(e),
            })

    # Sort by ATS score descending (best candidates first)
    results.sort(key=lambda r: r.get("ats_score", 0), reverse=True)

    # Cleanup vector store
    vs.reset()

    return results
