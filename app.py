import os
import json
import re
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from sqlmodel import Session, select, create_engine, SQLModel
from sqlalchemy import func, text as sa_text

from models import Company, Employee
from google import genai
from google.genai import types
import streamlit as st

# -------------------------------------------------------------------
# 1. Setup & Database Initialization
# -------------------------------------------------------------------
load_dotenv()

# Prioritize the Postgres URL if it exists, otherwise use local SQLite
DATABASE_URL = os.getenv("DATABASE_URL") or "sqlite:///company_research.db"

if DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+psycopg2://", 1)
elif DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+psycopg2://", 1)

engine = create_engine(DATABASE_URL, echo=False)

# This creates tables and applies the schema from models.py
SQLModel.metadata.create_all(engine)

# -------------------------------------------------------------------
# 2. Gemini Client Configuration
# -------------------------------------------------------------------
client = None
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

if GEMINI_API_KEY:
    client = genai.Client(api_key=GEMINI_API_KEY)

GEMINI_PROMPT_COMPANY = """
You are a factual assistant. Using Google Search grounding, return a JSON object only (no commentary):
{
  "company": {
    "name": "string",
    "industry": "string",
    "employee_size": "integer",
    "domain": "string",
    "email": "string"
  },
  "employees": [
    {
      "full_name":"string",
      "title":"string",
      "department":"string",
      "seniority":"string",
      "profile_url":"string",
      "email":"string"
    }
  ]
}
Keep employees list to the top 10 relevant individuals.
"""

# -------------------------------------------------------------------
# 3. Helper Functions
# -------------------------------------------------------------------
def normalize_company_key(s: str) -> str:
    return " ".join(s.split()).strip().lower() if s else ""

def strip_code_fences(text: str) -> str:
    if not text: return text
    text = re.sub(r"```(?:json|js|txt)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()

def safe_json_parse(text: str) -> Optional[Dict[str, Any]]:
    text = strip_code_fences(text)
    try:
        return json.loads(text)
    except:
        return None

def get_company_by_name(session: Session, raw_user_input: str) -> Optional[Company]:
    raw_norm = normalize_company_key(raw_user_input)
    return session.exec(select(Company).where(Company.name_normalized == raw_norm)).first()

# -------------------------------------------------------------------
# 4. Core Logic: Gemini Fetch
# -------------------------------------------------------------------
def fetch_from_gemini(company_query: str, context: str = "") -> Dict[str, Any]:
    if not client: raise RuntimeError("GEMINI_API_KEY not set.")

    tool = types.Tool(google_search=types.GoogleSearch())
    cfg = types.GenerateContentConfig(
        tools=[tool], 
        response_mime_type="application/json"
    )

    query_text = f'Query: "{company_query}"'
    if context.strip(): query_text += f'\nContext: "{context.strip()}"'

    resp = client.models.generate_content(
        model=GEMINI_MODEL, 
        contents=GEMINI_PROMPT_COMPANY + "\n" + query_text,
        config=cfg,
    )
    
    parsed = safe_json_parse(resp.text)
    if not parsed:
        raise ValueError("Failed to parse Gemini response as JSON.")
    return parsed

# ... (upsert_company_and_employees function stays the same as before) ...

# -------------------------------------------------------------------
# 5. Streamlit User Interface
# -------------------------------------------------------------------
st.set_page_config(page_title="Company Research", layout="wide")
st.title("🏢 Company Research Tool")

with st.sidebar:
    st.header("System Status")
    # Visually confirm the DB type for the user
    db_type = "PostgreSQL" if "postgresql" in DATABASE_URL else "SQLite"
    st.success(f"Connected to: **{db_type}**")
    st.info(f"AI Model: `{GEMINI_MODEL}`")
    
    if not GEMINI_API_KEY:
        st.error("API Key missing")

# ... (Remaining Streamlit search form and display logic) ...
