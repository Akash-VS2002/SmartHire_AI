"""
app.py
------
AgentHire AI — Streamlit HR Dashboard
======================================

A premium, dark-themed ATS dashboard where recruiters can:
  1. Upload multiple candidate PDF resumes
  2. Enter a job description
  3. Trigger multi-agent AI analysis
  4. View ranked candidates with scores, skills, recommendations,
     and AI-generated interview questions

Run with: streamlit run app.py
"""

import os
import sys
import time
import logging
from pathlib import Path

# Ensure project root is in Python path (for module imports)
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AgentHire AI — Intelligent ATS",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS — dark theme + premium design ──────────────────────────────────
st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Root variables ── */
:root {
    --bg-primary:    #0D1117;
    --bg-secondary:  #161B22;
    --bg-card:       #1C2333;
    --bg-card-hover: #1E2A3A;
    --border:        rgba(48, 54, 68, 0.8);
    --accent-blue:   #3B82F6;
    --accent-cyan:   #22D3EE;
    --accent-purple: #A78BFA;
    --text-primary:  #E6EDF3;
    --text-muted:    #8B949E;
    --green:         #10B981;
    --amber:         #F59E0B;
    --red:           #EF4444;
    --green-bg:      rgba(16, 185, 129, 0.12);
    --amber-bg:      rgba(245, 158, 11, 0.12);
    --red-bg:        rgba(239, 68, 68, 0.12);
}

/* ── Base overrides ── */
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

.stApp {
    background: var(--bg-primary);
    color: var(--text-primary);
}

/* ── Hide default Streamlit elements ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1rem; padding-bottom: 2rem; max-width: 1400px; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--bg-secondary);
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] .block-container { padding-top: 0; }

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #3B82F6, #8B5CF6);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.6rem 1.4rem;
    font-family: 'Inter', sans-serif;
    font-weight: 600;
    font-size: 0.95rem;
    letter-spacing: 0.03em;
    transition: all 0.2s ease;
    box-shadow: 0 4px 15px rgba(59, 130, 246, 0.3);
    cursor: pointer;
    width: 100%;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(59, 130, 246, 0.45);
    background: linear-gradient(135deg, #2563EB, #7C3AED);
}
.stButton > button:active {
    transform: translateY(0);
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: var(--bg-card);
    border: 2px dashed var(--border);
    border-radius: 12px;
    padding: 1rem;
    transition: border-color 0.2s;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--accent-blue);
}

/* ── Text areas & inputs ── */
textarea, .stTextArea textarea {
    background: var(--bg-card) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.88rem !important;
}
textarea:focus {
    border-color: var(--accent-blue) !important;
    box-shadow: 0 0 0 2px rgba(59,130,246,0.2) !important;
}

/* ── Expanders ── */
[data-testid="stExpander"] {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    margin-bottom: 0.75rem;
    overflow: hidden;
}
[data-testid="stExpander"] summary {
    color: var(--text-primary) !important;
    font-weight: 600 !important;
    padding: 0.75rem 1rem !important;
}

/* ── Metrics ── */
[data-testid="metric-container"] {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1rem;
}
[data-testid="metric-container"] label {
    color: var(--text-muted) !important;
    font-size: 0.8rem !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: var(--text-primary) !important;
    font-size: 1.8rem !important;
    font-weight: 700 !important;
}

/* ── Progress bars ── */
.stProgress > div > div > div {
    background: linear-gradient(90deg, #3B82F6, #8B5CF6) !important;
    border-radius: 9999px !important;
}
.stProgress > div > div {
    background: var(--bg-secondary) !important;
    border-radius: 9999px !important;
    height: 8px !important;
}

/* ── Dividers ── */
hr { border-color: var(--border) !important; margin: 1.5rem 0; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg-secondary); }
::-webkit-scrollbar-thumb { background: #30363D; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent-blue); }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Helper: Render HTML components
# ═══════════════════════════════════════════════════════════════════════════════

def recommendation_badge(rec: str) -> str:
    """Return styled HTML badge for recommendation tier."""
    styles = {
        "SHORTLIST": ("var(--green)",   "var(--green-bg)",  "✅ SHORTLIST"),
        "REVIEW":    ("var(--amber)",   "var(--amber-bg)",  "🔍 REVIEW"),
        "REJECT":    ("var(--red)",     "var(--red-bg)",    "❌ REJECT"),
    }
    color, bg, label = styles.get(rec, ("var(--text-muted)", "transparent", rec))
    return (
        f'<span style="'
        f'background:{bg}; color:{color}; border:1px solid {color}; '
        f'padding:0.25rem 0.75rem; border-radius:9999px; '
        f'font-size:0.78rem; font-weight:700; letter-spacing:0.06em;'
        f'">{label}</span>'
    )


def skill_chip(skill: str, matched: bool) -> str:
    """Return a styled skill chip."""
    if matched:
        return (
            f'<span style="'
            f'background:rgba(16,185,129,0.15); color:#10B981; '
            f'border:1px solid rgba(16,185,129,0.4); '
            f'padding:0.2rem 0.6rem; border-radius:9999px; '
            f'font-size:0.78rem; font-weight:500; margin:2px; display:inline-block;">'
            f'✓ {skill}</span>'
        )
    else:
        return (
            f'<span style="'
            f'background:rgba(239,68,68,0.12); color:#EF4444; '
            f'border:1px solid rgba(239,68,68,0.35); '
            f'padding:0.2rem 0.6rem; border-radius:9999px; '
            f'font-size:0.78rem; font-weight:500; margin:2px; display:inline-block;">'
            f'✗ {skill}</span>'
        )


def score_color(score: float) -> str:
    """Return a gradient color string based on score."""
    if score >= 90:
        return "#10B981"   # green
    elif score >= 70:
        return "#F59E0B"   # amber
    else:
        return "#EF4444"   # red


def render_score_gauge(score: float) -> str:
    """SVG donut gauge for the ATS score."""
    color = score_color(score)
    pct = score / 100
    # SVG circle math: circumference = 2π×r = 2π×40 ≈ 251.3
    circumference = 251.3
    dash_offset = circumference * (1 - pct)

    return f"""
    <div style="position:relative; width:90px; height:90px; margin:auto;">
      <svg width="90" height="90" viewBox="0 0 90 90">
        <!-- Background circle -->
        <circle cx="45" cy="45" r="40" fill="none"
                stroke="rgba(255,255,255,0.07)" stroke-width="8"/>
        <!-- Score arc -->
        <circle cx="45" cy="45" r="40" fill="none"
                stroke="{color}" stroke-width="8"
                stroke-linecap="round"
                stroke-dasharray="{circumference}"
                stroke-dashoffset="{dash_offset:.1f}"
                transform="rotate(-90 45 45)"
                style="transition:stroke-dashoffset 1s ease;"/>
        <!-- Score text -->
        <text x="45" y="49" text-anchor="middle"
              fill="{color}" font-size="16" font-weight="700"
              font-family="Inter, sans-serif">{int(score)}</text>
      </svg>
    </div>
    """


# ═══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    # Logo / Brand
    st.markdown("""
    <div style="padding: 1.5rem 0.5rem 1rem; text-align:center;">
        <div style="font-size:2.5rem; margin-bottom:0.3rem;">🤖</div>
        <div style="font-size:1.3rem; font-weight:800; color:#E6EDF3;
                    background:linear-gradient(135deg,#3B82F6,#A78BFA);
                    -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
            AgentHire AI
        </div>
        <div style="font-size:0.75rem; color:#8B949E; letter-spacing:0.1em;
                    text-transform:uppercase; margin-top:0.2rem;">
            Intelligent ATS Platform
        </div>
    </div>
    <hr style="border-color:rgba(48,54,68,0.8); margin:0.5rem 0 1rem;">
    """, unsafe_allow_html=True)

    # API Key check
    api_key = os.environ.get("GROQ_API_KEY", "")
    if api_key:
        st.markdown("""
        <div style="background:rgba(16,185,129,0.1); border:1px solid rgba(16,185,129,0.3);
                    border-radius:8px; padding:0.5rem 0.75rem; font-size:0.8rem;
                    color:#10B981; margin-bottom:1rem;">
            🔑 &nbsp; Groq API Connected
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background:rgba(239,68,68,0.1); border:1px solid rgba(239,68,68,0.3);
                    border-radius:8px; padding:0.5rem 0.75rem; font-size:0.8rem;
                    color:#EF4444; margin-bottom:0.5rem;">
            ⚠️ &nbsp; GROQ_API_KEY not found in .env
        </div>""", unsafe_allow_html=True)
        api_key_input = st.text_input(
            "Enter Groq API Key",
            type="password",
            placeholder="gsk_...",
            key="api_key_input",
        )
        if api_key_input:
            os.environ["GROQ_API_KEY"] = api_key_input
            st.success("API key set for this session!")

    st.markdown("### 📂 Upload Resumes")
    uploaded_files = st.file_uploader(
        label="Drop PDF resumes here",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        key="resume_uploader",
    )

    if uploaded_files:
        st.markdown(
            f'<div style="font-size:0.82rem; color:#8B949E; margin-top:0.4rem;">'
            f'📄 {len(uploaded_files)} file(s) ready</div>',
            unsafe_allow_html=True,
        )

    st.markdown("### 📋 Job Description")
    jd_text = st.text_area(
        label="Paste job description",
        height=260,
        placeholder=(
            "Paste the full job description here...\n\n"
            "Include: Required skills, responsibilities, experience requirements, etc."
        ),
        label_visibility="collapsed",
        key="jd_input",
    )

    st.markdown("---")
    analyze_btn = st.button("🚀 Analyze Candidates", key="analyze_btn", use_container_width=True)

    st.markdown("""
    <div style="margin-top:2rem; padding:0.75rem; background:rgba(255,255,255,0.03);
                border-radius:8px; font-size:0.74rem; color:#8B949E; line-height:1.6;">
        <strong style="color:#E6EDF3;">AI Agents Active:</strong><br>
        🔵 Resume Analyzer<br>
        🟣 JD Analyzer<br>
        🟡 Skill Matcher<br>
        🟢 ATS Scorer<br>
        🔴 Recommendation<br>
        💬 Interview Generator
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Main Content Area
# ═══════════════════════════════════════════════════════════════════════════════

# Hero header
st.markdown("""
<div style="padding: 0.5rem 0 1.5rem;">
    <h1 style="font-size:2rem; font-weight:800; color:#E6EDF3; margin:0;
               background:linear-gradient(135deg,#3B82F6 30%,#A78BFA);
               -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
        Resume Screening & Job Matching
    </h1>
    <p style="color:#8B949E; font-size:0.92rem; margin:0.4rem 0 0;">
        Multi-agent AI pipeline powered by LangGraph · Groq LLM · ChromaDB RAG
    </p>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Session state
# ═══════════════════════════════════════════════════════════════════════════════
if "results" not in st.session_state:
    st.session_state.results = []
if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False


# ═══════════════════════════════════════════════════════════════════════════════
# Analysis trigger
# ═══════════════════════════════════════════════════════════════════════════════
if analyze_btn:
    # Validate inputs
    if not uploaded_files:
        st.error("⚠️ Please upload at least one PDF resume.")
        st.stop()
    if not jd_text.strip():
        st.error("⚠️ Please enter a job description.")
        st.stop()
    if not os.environ.get("GROQ_API_KEY"):
        st.error("⚠️ GROQ_API_KEY is required. Add it to your .env file or enter it in the sidebar.")
        st.stop()

    # Import pipeline modules
    from utils.pdf_reader import extract_text_from_uploaded_file
    from graph.workflow import run_batch_pipeline

    # ── Extract resume texts ──────────────────────────────────────────────────
    resume_texts = []
    file_names = []

    with st.status("📄 Reading PDF resumes...", expanded=True) as status:
        for uf in uploaded_files:
            st.write(f"Processing: **{uf.name}**")
            text = extract_text_from_uploaded_file(uf)
            if text:
                resume_texts.append(text)
                file_names.append(uf.name)
                st.write(f"  ✅ Extracted {len(text):,} characters")
            else:
                st.write(f"  ⚠️ Could not extract text from **{uf.name}** — skipped")
        status.update(label=f"✅ {len(resume_texts)} resume(s) ready", state="complete")

    if not resume_texts:
        st.error("❌ Could not extract text from any uploaded resume. Please check your PDFs.")
        st.stop()

    # ── Run multi-agent pipeline ──────────────────────────────────────────────
    progress_bar = st.progress(0, text="🤖 Initialising AI agents…")
    agent_steps = [
        "🔵 Resume Analyzer",
        "🟣 JD Analyzer",
        "🟡 Skill Matcher",
        "🟢 ATS Scorer",
        "💬 Interview Generator",
    ]

    results_container = st.empty()
    results = []
    errors = []

    import uuid
    session_id = str(uuid.uuid4())[:8]

    from rag.vectorstore import ResumeVectorStore
    from utils.text_cleaner import clean_text
    from graph.workflow import run_pipeline

    # Shared vector store — JD indexed once
    vs = ResumeVectorStore(session_id=session_id)
    clean_jd = clean_text(jd_text)
    vs.add_job_description(clean_jd)

    total = len(resume_texts)
    for i, (resume_text, fname) in enumerate(zip(resume_texts, file_names)):
        # Update progress for each agent step
        base = i / total
        for step_idx, step_name in enumerate(agent_steps):
            prog = base + (step_idx / len(agent_steps)) / total
            progress_bar.progress(
                min(prog, 0.99),
                text=f"Candidate {i + 1}/{total}: {step_name}…"
            )
            time.sleep(0.05)  # Brief visual delay for perceived responsiveness

        try:
            result = run_pipeline(
                resume_text=resume_text,
                jd_text=jd_text,
                vector_store=vs,
                session_id=session_id,
            )
            result["_source_file"] = fname
            results.append(result)
        except Exception as e:
            logger.error(f"Pipeline error for {fname}: {e}")
            errors.append(f"Error processing {fname}: {str(e)[:100]}")
            results.append({
                "candidate_name": fname.replace(".pdf", ""),
                "ats_score": 0.0,
                "recommendation": "REJECT",
                "matched_skills": [],
                "missing_skills": [],
                "skill_match_percentage": 0.0,
                "score_reason": str(e),
                "interview_questions": [],
                "experience_summary": "",
                "_source_file": fname,
                "error": str(e),
            })

    # Sort by ATS score descending
    results.sort(key=lambda r: r.get("ats_score", 0), reverse=True)

    # Cleanup
    vs.reset()

    progress_bar.progress(1.0, text="✅ Analysis complete!")
    time.sleep(0.5)
    progress_bar.empty()

    if errors:
        for err in errors:
            st.warning(err)

    st.session_state.results = results
    st.session_state.analysis_done = True
    st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# Results Display
# ═══════════════════════════════════════════════════════════════════════════════

if st.session_state.analysis_done and st.session_state.results:
    results = st.session_state.results

    # ── Summary metrics row ───────────────────────────────────────────────────
    total = len(results)
    shortlist_count = sum(1 for r in results if r.get("recommendation") == "SHORTLIST")
    review_count    = sum(1 for r in results if r.get("recommendation") == "REVIEW")
    reject_count    = sum(1 for r in results if r.get("recommendation") == "REJECT")
    avg_score       = sum(r.get("ats_score", 0) for r in results) / total if total else 0

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("📋 Total Candidates", total)
    col2.metric("✅ Shortlisted",  shortlist_count)
    col3.metric("🔍 For Review",   review_count)
    col4.metric("❌ Rejected",     reject_count)
    col5.metric("📊 Avg ATS Score", f"{avg_score:.1f}")

    st.markdown("---")

    # ── Candidate ranking table ───────────────────────────────────────────────
    st.markdown("""
    <h2 style="font-size:1.2rem; font-weight:700; color:#E6EDF3; margin-bottom:1rem;">
        🏆 Candidate Rankings
    </h2>
    """, unsafe_allow_html=True)

    # Build table HTML
    table_rows = ""
    for rank, r in enumerate(results, 1):
        name        = r.get("candidate_name", "Unknown")
        score       = r.get("ats_score", 0.0)
        match_pct   = r.get("skill_match_percentage", 0.0)
        rec         = r.get("recommendation", "REJECT")
        source_file = r.get("_source_file", "")

        sc = score_color(score)
        badge = recommendation_badge(rec)
        row_bg = (
            "rgba(16,185,129,0.05)"  if rec == "SHORTLIST" else
            "rgba(245,158,11,0.05)"  if rec == "REVIEW"    else
            "rgba(239,68,68,0.03)"
        )

        table_rows += f"""
        <tr style="border-bottom:1px solid rgba(48,54,68,0.6);
                   background:{row_bg}; transition:background 0.2s;">
          <td style="padding:0.9rem 1rem; font-weight:700; color:#8B949E;">#{rank}</td>
          <td style="padding:0.9rem 1rem;">
            <div style="font-weight:600; color:#E6EDF3;">{name}</div>
            <div style="font-size:0.75rem; color:#8B949E;">{source_file}</div>
          </td>
          <td style="padding:0.9rem 1rem; text-align:center;">
            <span style="font-size:1.4rem; font-weight:800; color:{sc};">{score:.1f}</span>
            <span style="font-size:0.75rem; color:#8B949E;">/100</span>
          </td>
          <td style="padding:0.9rem 1rem;">
            <div style="font-size:0.82rem; color:#8B949E; margin-bottom:4px;">{match_pct:.0f}%</div>
            <div style="background:rgba(255,255,255,0.07); border-radius:9999px; height:6px; overflow:hidden;">
              <div style="height:6px; border-radius:9999px; width:{match_pct:.0f}%;
                          background:linear-gradient(90deg,#3B82F6,#8B5CF6);"></div>
            </div>
          </td>
          <td style="padding:0.9rem 1rem;">{badge}</td>
        </tr>
        """

    st.markdown(f"""
    <div style="background:var(--bg-card,#1C2333); border:1px solid rgba(48,54,68,0.8);
                border-radius:14px; overflow:hidden; margin-bottom:2rem;">
      <table style="width:100%; border-collapse:collapse; font-family:'Inter',sans-serif;">
        <thead>
          <tr style="background:rgba(255,255,255,0.04);">
            <th style="padding:0.75rem 1rem; text-align:left; font-size:0.75rem;
                       color:#8B949E; font-weight:600; text-transform:uppercase;
                       letter-spacing:0.08em;">Rank</th>
            <th style="padding:0.75rem 1rem; text-align:left; font-size:0.75rem;
                       color:#8B949E; font-weight:600; text-transform:uppercase;
                       letter-spacing:0.08em;">Candidate</th>
            <th style="padding:0.75rem 1rem; text-align:center; font-size:0.75rem;
                       color:#8B949E; font-weight:600; text-transform:uppercase;
                       letter-spacing:0.08em;">ATS Score</th>
            <th style="padding:0.75rem 1rem; text-align:left; font-size:0.75rem;
                       color:#8B949E; font-weight:600; text-transform:uppercase;
                       letter-spacing:0.08em;">Skill Match</th>
            <th style="padding:0.75rem 1rem; text-align:left; font-size:0.75rem;
                       color:#8B949E; font-weight:600; text-transform:uppercase;
                       letter-spacing:0.08em;">Recommendation</th>
          </tr>
        </thead>
        <tbody>{table_rows}</tbody>
      </table>
    </div>
    """, unsafe_allow_html=True)

    # ── Detailed candidate cards ──────────────────────────────────────────────
    st.markdown("""
    <h2 style="font-size:1.2rem; font-weight:700; color:#E6EDF3; margin-bottom:1rem;">
        🔍 Detailed Candidate Analysis
    </h2>
    """, unsafe_allow_html=True)

    for rank, r in enumerate(results, 1):
        name        = r.get("candidate_name", "Unknown")
        score       = r.get("ats_score", 0.0)
        match_pct   = r.get("skill_match_percentage", 0.0)
        rec         = r.get("recommendation", "REJECT")
        matched     = r.get("matched_skills", [])
        missing     = r.get("missing_skills", [])
        experience  = r.get("experience_summary", "N/A")
        education   = r.get("education", "N/A")
        certifications = r.get("certifications", [])
        projects    = r.get("projects", [])
        reason      = r.get("score_reason", "")
        questions   = r.get("interview_questions", [])
        sub_scores  = r.get("_sub_scores", {})
        source_file = r.get("_source_file", "")
        has_error   = bool(r.get("error"))

        sc = score_color(score)
        icon = "✅" if rec == "SHORTLIST" else "🔍" if rec == "REVIEW" else "❌"

        with st.expander(
            f"{icon} #{rank} — {name}  ·  ATS Score: {score:.1f}/100  ·  {rec}",
            expanded=(rank == 1),
        ):
            # Top row: gauge + key stats
            g_col, s_col = st.columns([1, 3])

            with g_col:
                st.markdown(render_score_gauge(score), unsafe_allow_html=True)
                st.markdown(
                    f'<p style="text-align:center; font-size:0.75rem; color:#8B949E; margin-top:0.3rem;">'
                    f'{source_file}</p>',
                    unsafe_allow_html=True,
                )

            with s_col:
                m1, m2, m3 = st.columns(3)
                m1.metric("ATS Score",     f"{score:.1f}/100")
                m2.metric("Skill Match",   f"{match_pct:.1f}%")
                m3.metric("Recommendation", rec)

                st.markdown(
                    f'<p style="font-size:0.85rem; color:#8B949E; line-height:1.6; margin-top:0.5rem;">'
                    f'<strong style="color:#E6EDF3;">Scoring Rationale:</strong> {reason}</p>',
                    unsafe_allow_html=True,
                )

            # Sub-scores breakdown
            if sub_scores:
                st.markdown("**Score Breakdown:**")
                sc_cols = st.columns(4)
                labels = ["skills", "experience", "projects", "keywords"]
                names_ = ["Skills (50%)", "Experience (20%)", "Projects (20%)", "Keywords (10%)"]
                for col_, lbl, nm in zip(sc_cols, labels, names_):
                    v = sub_scores.get(lbl, 0)
                    col_.metric(nm, f"{v:.0f}")

            st.markdown("---")

            # Skills section
            sk_col, ms_col = st.columns(2)

            with sk_col:
                st.markdown(
                    '<p style="font-weight:600; color:#10B981; font-size:0.85rem;'
                    ' margin-bottom:0.5rem;">✅ Matched Skills</p>',
                    unsafe_allow_html=True,
                )
                if matched:
                    chips = " ".join(skill_chip(s, True) for s in matched)
                    st.markdown(f'<div style="line-height:2;">{chips}</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<span style="color:#8B949E; font-size:0.82rem;">No matched skills found</span>', unsafe_allow_html=True)

            with ms_col:
                st.markdown(
                    '<p style="font-weight:600; color:#EF4444; font-size:0.85rem;'
                    ' margin-bottom:0.5rem;">❌ Missing Skills</p>',
                    unsafe_allow_html=True,
                )
                if missing:
                    chips = " ".join(skill_chip(s, False) for s in missing)
                    st.markdown(f'<div style="line-height:2;">{chips}</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<span style="color:#8B949E; font-size:0.82rem;">No missing critical skills</span>', unsafe_allow_html=True)

            st.markdown("---")

            # Profile details
            p1, p2 = st.columns(2)
            with p1:
                st.markdown(
                    f'<p style="font-size:0.82rem;"><strong style="color:#A78BFA;">🎓 Education:</strong>'
                    f'<br><span style="color:#8B949E;">{education or "N/A"}</span></p>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<p style="font-size:0.82rem;"><strong style="color:#A78BFA;">💼 Experience Summary:</strong>'
                    f'<br><span style="color:#8B949E;">{experience or "N/A"}</span></p>',
                    unsafe_allow_html=True,
                )
                if certifications:
                    certs_str = " · ".join(certifications)
                    st.markdown(
                        f'<p style="font-size:0.82rem;"><strong style="color:#A78BFA;">🏆 Certifications:</strong>'
                        f'<br><span style="color:#8B949E;">{certs_str}</span></p>',
                        unsafe_allow_html=True,
                    )

            with p2:
                if projects:
                    st.markdown('<strong style="color:#A78BFA; font-size:0.82rem;">🛠️ Projects:</strong>', unsafe_allow_html=True)
                    for proj in projects[:5]:
                        st.markdown(
                            f'<p style="font-size:0.8rem; color:#8B949E; margin:0.2rem 0;">'
                            f'▸ {proj}</p>',
                            unsafe_allow_html=True,
                        )

            # Interview questions
            if questions:
                st.markdown("---")
                st.markdown(
                    '<p style="font-weight:700; color:#22D3EE; font-size:0.9rem;">'
                    '💬 AI-Generated Interview Questions</p>',
                    unsafe_allow_html=True,
                )
                for qi, q in enumerate(questions, 1):
                    cat_color = (
                        "#3B82F6" if qi <= 3 else
                        "#A78BFA" if qi <= 5 else
                        "#F59E0B"
                    )
                    cat_label = (
                        "Technical" if qi <= 3 else
                        "Project"   if qi <= 5 else
                        "Skill Gap"
                    )
                    st.markdown(
                        f'<div style="background:rgba(255,255,255,0.03); border-left:3px solid {cat_color};'
                        f' padding:0.6rem 0.9rem; border-radius:0 8px 8px 0; margin:0.4rem 0;">'
                        f'<span style="font-size:0.7rem; color:{cat_color}; font-weight:600;'
                        f' text-transform:uppercase; letter-spacing:0.06em;">{cat_label}</span><br>'
                        f'<span style="font-size:0.85rem; color:#E6EDF3;">{qi}. {q}</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

            # Error notice
            if has_error:
                st.markdown(
                    f'<div style="background:rgba(239,68,68,0.1); border:1px solid rgba(239,68,68,0.3);'
                    f' border-radius:8px; padding:0.5rem 0.75rem; font-size:0.78rem; color:#EF4444;'
                    f' margin-top:0.5rem;">⚠️ Pipeline note: {r.get("error", "")[:150]}</div>',
                    unsafe_allow_html=True,
                )

    # ── Reset button ──────────────────────────────────────────────────────────
    st.markdown("---")
    if st.button("🔄 New Analysis", key="reset_btn"):
        st.session_state.results = []
        st.session_state.analysis_done = False
        st.rerun()

else:
    # ── Welcome / empty state ─────────────────────────────────────────────────
    st.markdown("""
    <div style="text-align:center; padding:4rem 2rem; color:#8B949E;">
        <div style="font-size:4rem; margin-bottom:1rem;">🤖</div>
        <h3 style="color:#E6EDF3; font-weight:700; margin-bottom:0.5rem;">
            Ready to screen candidates
        </h3>
        <p style="max-width:500px; margin:0 auto 2rem; line-height:1.7;">
            Upload PDF resumes and paste your job description in the sidebar,
            then click <strong style="color:#3B82F6;">Analyze Candidates</strong> to
            launch the multi-agent AI pipeline.
        </p>
        <div style="display:flex; justify-content:center; gap:1.5rem; flex-wrap:wrap;
                    max-width:700px; margin:0 auto;">
    """, unsafe_allow_html=True)

    features = [
        ("🔵", "Resume Analyzer",    "Extracts name, skills, experience & projects"),
        ("🟣", "JD Analyzer",        "Parses required & preferred skills"),
        ("🟡", "Skill Matcher",      "Semantic similarity matching (not just keywords)"),
        ("🟢", "ATS Scorer",         "Weighted 0-100 composite score"),
        ("❤️", "Recommendation",     "Auto shortlist, review, or reject"),
        ("💬", "Interview AI",       "Generates 7 targeted questions per candidate"),
    ]

    cols = st.columns(3)
    for idx, (icon, title, desc) in enumerate(features):
        with cols[idx % 3]:
            st.markdown(f"""
            <div style="background:rgba(255,255,255,0.03); border:1px solid rgba(48,54,68,0.6);
                        border-radius:12px; padding:1.2rem; text-align:center; margin-bottom:1rem;">
                <div style="font-size:1.8rem; margin-bottom:0.5rem;">{icon}</div>
                <div style="font-weight:600; color:#E6EDF3; margin-bottom:0.3rem; font-size:0.9rem;">{title}</div>
                <div style="font-size:0.78rem; color:#8B949E; line-height:1.5;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)
