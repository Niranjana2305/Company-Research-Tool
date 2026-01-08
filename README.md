# Company Research Tool

AI-powered company and employee research tool using Google Gemini with search grounding. Results are cached in a database for fast subsequent lookups.

## Features

- Search companies by name with optional context for disambiguation
- Auto-enriches company data (industry, size, domain, email) via Gemini AI
- Discovers employee information (name, title, department, seniority, email, profile URL)
- Caches results in SQLite (local) or PostgreSQL (production)
- Bulk email search using AI-generated query templates

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager
- Google Gemini API key

## Setup

### 1. Install uv (if not installed)

**Windows (PowerShell):**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**macOS/Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Clone and install dependencies

```bash
git clone <repository-url>
cd company-research-tool
uv sync
```

### 3. Configure environment

```bash
cp .env.example .env
```

Edit `.env` and add your credentials:

```env
# SQLite for local development
DATABASE_URL=sqlite:///dev.db

# Or PostgreSQL for production
# DATABASE_URL=postgresql://user:password@host:5432/database

# Required: Get from https://aistudio.google.com/apikey
GEMINI_API_KEY=your_api_key_here

# Optional: Override default model
# GEMINI_MODEL=gemini-2.5-flash
```

### 4. Run the app

```bash
uv run streamlit run app.py
```

Open http://localhost:8501 in your browser.

## Environment Variables

| Variable         | Required | Description                                |
| ---------------- | -------- | ------------------------------------------ |
| `DATABASE_URL`   | Yes      | Database connection string                 |
| `GEMINI_API_KEY` | Yes      | Google Gemini API key                      |
| `GEMINI_MODEL`   | No       | Model to use (default: `gemini-2.5-flash`) |

## Tech Stack

- **UI:** Streamlit
- **ORM:** SQLModel / SQLAlchemy
- **AI:** Google Gemini with search grounding
- **Database:** SQLite (dev) / PostgreSQL (prod)

## Project Structure

```
├── app.py           # Main Streamlit application
├── models.py        # SQLModel database models
├── pyproject.toml   # Project dependencies
├── .env.example     # Environment template
└── .env             # Local environment (git-ignored)
```
