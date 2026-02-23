import os
import json
import re
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from sqlmodel import Session, select, create_engine, SQLModel
from sqlalchemy import func
from google import genai
from google.genai import types
import streamlit as st

# Import your models explicitly
from models import Company, Employee

# -------------------------------------------------------------------
# 1. Setup & Environment
# -------------------------------------------------------------------
load_dotenv()
st.set_page_config(page_title="Company Research", layout="wide")
st.title("🏢 Company Research Tool")

@st.cache_resource
def get_engine():
    db_url = os.getenv("DATABASE_URL") or "sqlite:///company_research.db"
    # Ensure driver compatibility for Render/Neon
    if db_url.startswith("postgres://"):
        db_url = db_url.replace("postgres://", "postgresql+psycopg2://", 1)
    elif db_url.startswith("postgresql://") and "+psycopg2" not in db_url:
        db_url = db_url.replace("postgresql://", "postgresql+psycopg2://", 1)
    return create_engine(db_url, echo=False)

engine = get_engine()

def init_db():
    try:
        SQLModel.metadata.create_all(engine)
    except Exception as e:
        st.error(f"❌ Database connection failed: {e}")
        st.stop()

init_db()

# -------------------------------------------------------------------
# 2. Gemini Configuration
# -------------------------------------------------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

@st.cache_resource
def get_gemini_client():
    if not GEMINI_API_KEY:
        return None
    return genai.Client(api_key=GEMINI_API_KEY)

client = get_gemini_client()

if not client:
    st.warning("⚠️ **Gemini API Key missing.** Please add `GEMINI_API_KEY` to your environment variables.")
    st.stop()

GEMINI_PROMPT_COMPANY = """
You are a factual assistant. Using Google Search grounding, return a JSON object only (no commentary):
{
  "company": {
    "name": "string", "industry": "string", "employee_size": "integer", "domain": "string", "email": "string"
  },
  "employees": [
    {"full_name":"string", "title":"string", "department":"string", "seniority":"string", "profile_url":"string", "email":"string"}
  ]
}
"""

# -------------------------------------------------------------------
# 3. Helper & Logic Functions
# -------------------------------------------------------------------
def normalize_company_key(s: str) -> str:
    return " ".join(s.split()).strip().lower() if s else ""

def safe_json_parse(text: str) -> Optional[Dict[str, Any]]:
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if not match: return None
    try:
        return json.loads(match.group())
    except:
        return None

def fetch_from_gemini(company_query: str, context: str = "") -> Dict[str, Any]:
    tool = types.Tool(google_search=types.GoogleSearch())
    cfg = types.GenerateContentConfig(tools=[tool], response_mime_type="application/json")
    query_text = f'Research: "{company_query}"'
    if context: query_text += f'\nContext: {context}'
    
    resp = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=GEMINI_PROMPT_COMPANY + "\n" + query_text,
        config=cfg
    )
    parsed = safe_json_parse(resp.text)
    if not parsed: raise ValueError("AI returned invalid JSON.")
    return parsed

def upsert_company_and_employees(data: Dict[str, Any], raw_user_input: str = "") -> Company:
    company_data = data.get("company") or {}
    employees = data.get("employees") or []
    company_name = company_data.get("name") or raw_user_input or "Unknown"
    norm_name = normalize_company_key(company_name)

    with Session(engine) as session:
        existing_company = session.exec(select(Company).where(Company.name_normalized == norm_name)).first()
        if existing_company:
            for key in ["industry", "domain", "email"]:
                if company_data.get(key): setattr(existing_company, key, company_data[key])
            if company_data.get("employee_size"):
                try: existing_company.employee_size = int(company_data["employee_size"])
                except: pass
            company_obj = existing_company
        else:
            company_obj = Company(
                name=company_name,
                name_normalized=norm_name,
                industry=company_data.get("industry"),
                domain=company_data.get("domain"),
                email=company_data.get("email"),
                employee_size=int(company_data["employee_size"]) if str(company_data.get("employee_size")).isdigit() else None
            )
            session.add(company_obj)
        
        session.flush()

        for e in employees:
            full_name = (e.get("full_name") or "").strip()
            if not full_name: continue
            existing_emp = session.exec(select(Employee).where(Employee.full_name == full_name, Employee.company_id == company_obj.id)).first()
            if not existing_emp:
                session.add(Employee(
                    full_name=full_name,
                    title=e.get("title"),
                    department=e.get("department"),
                    profile_url=e.get("profile_url"),
                    company_id=company_obj.id
                ))
        session.commit()
        session.refresh(company_obj)
        return company_obj

# -------------------------------------------------------------------
# 4. Main UI Logic
# -------------------------------------------------------------------
with st.sidebar:
    st.info(f"📍 DB: {'Cloud Postgres' if 'postgresql' in os.getenv('DATABASE_URL','') else 'Local SQLite'}")
    st.info(f"🤖 Model: {GEMINI_MODEL}")

with st.form("search_form"):
    q = st.text_input("Enter Company Name")
    context_input = st.text_area("Additional Context (Optional)")
    submitted = st.form_submit_button("Start Research", type="primary")

if submitted and q:
    with st.spinner("🤖 Gemini is researching..."):
        try:
            with Session(engine) as session:
                raw_norm = normalize_company_key(q)
                company = session.exec(select(Company).where(Company.name_normalized == raw_norm)).first()
            
            if not company:
                data = fetch_from_gemini(q, context_input)
                company = upsert_company_and_employees(data, q)
                st.success("New data retrieved and saved!")
            else:
                st.info("Loaded from existing database.")

            # --- Display Results ---
            col1, col2 = st.columns([1, 2])
            with col1:
                st.subheader("Company Details")
                st.markdown(f"**Name:** {company.name}")
                st.markdown(f"**Industry:** {company.industry or 'N/A'}")
                st.markdown(f"**Domain:** {company.domain or 'N/A'}")
                st.markdown(f"**Size:** {company.employee_size or 'N/A'}")

            with col2:
                st.subheader("Key Personnel")
                with Session(engine) as session:
                    emps = session.exec(select(Employee).where(Employee.company_id == company.id)).all()
                if emps:
                    st.table([{"Name": e.full_name, "Title": e.title, "Dept": e.department, "URL": e.profile_url} for e in emps])
                else:
                    st.write("No employees found.")

        except Exception as e:
            st.error(f"An error occurred: {e}")
