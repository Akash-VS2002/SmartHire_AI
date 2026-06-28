# 🤖 AgentHire AI — Intelligent Resume Screening & Job Matching ATS

<div align="center">

![AgentHire AI](https://img.shields.io/badge/AgentHire-AI%20ATS-3B82F6?style=for-the-badge&logo=robot&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-Multi--Agent-10B981?style=for-the-badge)
![Groq](https://img.shields.io/badge/Groq-LLM-F59E0B?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-EF4444?style=for-the-badge&logo=streamlit&logoColor=white)

**An AI-powered ATS system that uses multi-agent AI to screen resumes, calculate ATS scores, match skills semantically, rank candidates, and generate targeted interview questions.**

</div>

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 📄 **PDF Resume Upload** | Upload multiple candidate PDFs simultaneously |
| 🔍 **AI Resume Analysis** | LLM extracts name, skills, experience, projects, certifications |
| 📋 **JD Parsing** | Extracts required/preferred skills and responsibilities |
| 🧠 **Semantic Skill Matching** | Uses SentenceTransformer embeddings (not just keywords) |
| 📊 **ATS Scoring** | Weighted composite: Skills 50% + Exp 20% + Projects 20% + Keywords 10% |
| 🏆 **Auto Recommendation** | SHORTLIST ≥ 90 · REVIEW ≥ 70 · REJECT < 70 |
| 💬 **Interview Questions** | 7 AI-generated questions per candidate (Technical + Project + Skill Gap) |
| 🗄️ **ChromaDB RAG** | Semantic retrieval provides grounded context to every agent |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     AgentHire AI Pipeline                       │
│                                                                 │
│  PDF Resumes ──► PDF Reader ──► Text Cleaner ──► ChromaDB       │
│                                                      │          │
│              Job Description ──────────────────► ChromaDB       │
│                                                      │          │
│                           LangGraph StateGraph       │          │
│                    ┌─────────────────────────┐       │          │
│                    │  START                  │       │          │
│                    │    │                   RAG      │          │
│                    │  Resume Analyzer ◄──── context  │          │
│                    │    │                            │          │
│                    │  JD Analyzer ◄──────────────────┘          │
│                    │    │                                        │
│                    │  Skill Matcher (Semantic Cosine Sim)        │
│                    │    │                                        │
│                    │  ATS Scorer (Weighted Formula)              │
│                    │    │                                        │
│                    │  Interview Generator ◄── RAG context        │
│                    │    │                                        │
│                    │  END                                        │
│                    └─────────────────────────┘                  │
│                           │                                     │
│                    Streamlit Dashboard                          │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
SmartHire_AI/
│
├── app.py                    # Streamlit HR dashboard (entry point)
│
├── agents/
│   ├── resume_agent.py       # Resume Analyzer (Groq LLM + RAG)
│   ├── jd_agent.py           # JD Analyzer (Groq LLM)
│   ├── matcher_agent.py      # Semantic Skill Matcher (SentenceTransformer)
│   ├── scoring_agent.py      # ATS Scorer + Recommendation logic
│   └── interview_agent.py    # Interview Question Generator
│
├── graph/
│   └── workflow.py           # LangGraph StateGraph pipeline
│
├── rag/
│   ├── embeddings.py         # SentenceTransformer embedding wrapper
│   └── vectorstore.py        # ChromaDB vector store (ephemeral/per-session)
│
├── utils/
│   ├── pdf_reader.py         # PyPDF text extraction with error handling
│   └── text_cleaner.py       # Text cleaning, chunking, name heuristic
│
├── models/
│   └── state.py              # ResumeState TypedDict (shared pipeline state)
│
├── requirements.txt
├── .env.example
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone / Navigate to the project

```bash
cd agenthire_ai
```

### 2. Create a virtual environment

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up your API key

```bash
# Copy the template
copy .env.example .env   # Windows
cp .env.example .env     # macOS/Linux

# Edit .env and add your Groq API key:
# GROQ_API_KEY=gsk_your_actual_key_here
```

Get a free Groq API key at **https://console.groq.com**

### 5. Run the app

```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

---

## 🤖 Agent Details

### A. Resume Analyzer Agent (`agents/resume_agent.py`)
- Uses **Groq LLaMA-3.3-70B** to extract structured JSON from resume text
- Retrieves relevant chunks from ChromaDB via RAG before prompting
- Outputs: `name`, `education`, `experience`, `skills`, `projects`, `certifications`

### B. JD Analyzer Agent (`agents/jd_agent.py`)
- Parses job description into structured format
- Outputs: `required_skills`, `preferred_skills`, `experience_requirements`, `responsibilities`

### C. Skill Matching Agent (`agents/matcher_agent.py`)
- Uses **SentenceTransformer cosine similarity** (threshold: 0.72)
- Matches "ML" ↔ "Machine Learning", "NLP" ↔ "Natural Language Processing" etc.
- Outputs: `matched_skills`, `missing_skills`, `skill_match_percentage`

### D. ATS Scoring Agent (`agents/scoring_agent.py`)
- **Weighted formula:**
  ```
  Score = Skills(50%) + Experience(20%) + Projects(20%) + Keywords(10%)
  ```
- Generates LLM rationale for the score
- Outputs: `ats_score`, `score_reason`, `recommendation`

### E. Recommendation Logic (inside Scoring Agent)
```
≥ 90  →  ✅ SHORTLIST (Strong fit)
≥ 70  →  🔍 REVIEW   (Potential fit, needs assessment)
< 70  →  ❌ REJECT   (Significant gaps)
```

### F. Interview Question Generator (`agents/interview_agent.py`)
- Generates **7 targeted questions** per candidate:
  - 3 Technical questions (matched skills depth)
  - 2 Project questions (candidate's own work)
  - 2 Skill Gap questions (missing skills + learning ability)
- Uses RAG-retrieved context for highly specific questions

---

## 📊 Output Example

```
Candidate:       Akash Sharma
ATS Score:       88.5/100
Skill Match:     85.7%
Recommendation:  🔍 REVIEW

Matched Skills:  ✓ Python  ✓ Machine Learning  ✓ LangChain  ✓ RAG
Missing Skills:  ✗ Docker  ✗ AWS  ✗ Kubernetes

Score Reason:
  Strong GenAI background with practical LangChain and RAG experience.
  Projects demonstrate solid ML fundamentals. Missing cloud deployment
  skills (Docker, AWS) which are required for this role.

Interview Questions:
  [Technical]  1. Explain the RAG architecture and how you implemented it.
  [Technical]  2. What's the difference between embeddings and fine-tuning?
  [Technical]  3. How does LangGraph manage state across multiple agents?
  [Project]    4. Walk me through your most complex ML project end-to-end.
  [Project]    5. How did you handle model evaluation and deployment?
  [Skill Gap]  6. How would you approach learning Docker for MLOps?
  [Skill Gap]  7. Describe your experience with cloud platforms — what would
                   your first steps be to get AWS-certified?
```

---

## 🔧 Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `GROQ_API_KEY` | **Required** | Your Groq API key |
| `LOG_LEVEL` | `INFO` | Logging verbosity |

---

## 🔮 Future Roadmap

- [ ] **Email Automation** — Auto-send shortlist notifications to candidates
- [ ] **Ranking Dashboard** — Visual leaderboard with filtering/sorting
- [ ] **Database Storage** — Persist results to PostgreSQL / SQLite
- [ ] **Resume Improvement Agent** — Suggest resume improvements to candidates
- [ ] **Bulk Export** — Export results to CSV / Excel / PDF report
- [ ] **Authentication** — Multi-recruiter login with role-based access
- [ ] **Webhook Integration** — Connect with Workday, Greenhouse, Lever

---

## 🛡️ Tech Stack

| Component | Technology |
|-----------|-----------|
| **Orchestration** | LangGraph StateGraph |
| **LLM** | Groq (LLaMA-3.3-70B-Versatile) |
| **Embeddings** | SentenceTransformer all-MiniLM-L6-v2 |
| **Vector DB** | ChromaDB (ephemeral in-memory) |
| **PDF Processing** | PyPDF |
| **UI** | Streamlit |
| **Language** | Python 3.10+ |

---

<div align="center">
Built with ❤️ using LangGraph · Groq · ChromaDB · Streamlit
</div>
