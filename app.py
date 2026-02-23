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

DATABASE_URL = os.getenv("DATABASE_URL") or "sqlite:///company_research.db"
engine = create_engine(DATABASE_URL, echo=False)

# This creates tables and applies the schema from models.py
SQLModel.metadata.create_all(engine)

# -------------------------------------------------------------------
# 2. Gemini Client Configuration
# -------------------------------------------------------------------
client = None
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

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
# 3. Helper Functions (Normalization & Parsing)
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
    # Check normalized index first (fastest)
    company = session.exec(select(Company).where(Company.name_normalized == raw_norm)).first()
    if company: return company
    # Fallback to standard name check
    return session.exec(select(Company).where(func.lower(Company.name) == raw_norm)).first()

# -------------------------------------------------------------------
# 4. Core Logic: Gemini Fetch & DB Upsert
# -------------------------------------------------------------------
def fetch_from_gemini(company_query: str, context: str = "") -> Dict[str, Any]:
    if not client: raise RuntimeError("GEMINI_API_KEY not set.")

    tool = types.Tool(google_search=types.GoogleSearch())
    cfg = types.GenerateContentConfig(tools=[tool], response_mime_type="application/json")

    query_text = f'Query: "{company_query}"'
    if context.strip(): query_text += f'\nContext: "{context.strip()}"'

    resp = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=GEMINI_PROMPT_COMPANY + "\n" + query_text,
        config=cfg,
    )
    
    parsed = safe_json_parse(resp.text)
    if not parsed:
        raise ValueError("Failed to parse Gemini response as JSON.")
    return parsed

def upsert_company_and_employees(data: Dict[str, Any], raw_user_input: str = "") -> Company:
    company_data = data.get("company") or {}
    employees = data.get("employees") or []
    
    company_name = company_data.get("name") or raw_user_input or "Unknown"
    norm_name = normalize_company_key(company_name)

    with Session(engine) as session:
        existing_company = session.exec(
            select(Company).where(Company.name_normalized == norm_name)
        ).first()

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

            existing_emp = session.exec(
                select(Employee).where(Employee.full_name == full_name, Employee.company_id == company_obj.id)
            ).first()

            if not existing_emp:
                session.add(Employee(
                    full_name=full_name,
                    title=e.get("title"),
                    department=e.get("department"),
                    seniority=e.get("seniority"),
                    profile_url=e.get("profile_url"),
                    email=e.get("email"),
                    company_id=company_obj.id
                ))

        session.commit()
        session.refresh(company_obj)
        return company_obj

# -------------------------------------------------------------------
# 5. Streamlit User Interface
# -------------------------------------------------------------------
st.set_page_config(page_title="Company Research", layout="wide")
st.title("🏢 Company Research Tool")

with st.sidebar:
    st.header("Settings")
    if not GEMINI_API_KEY:
        st.error("API Key missing in .env")

with st.form("search_form"):
    q = st.text_input("Company Name", placeholder="e.g. NVIDIA")
    user_context = st.text_area("Context", placeholder="Helps Gemini identify the right company...")
    submitted = st.form_submit_button("Research", type="primary")

if submitted and q:
    with Session(engine) as session:
        company = get_company_by_name(session, q)

    if not company:
        with st.spinner(f"🔍 Searching for '{q}'..."):
            try:
                data = fetch_from_gemini(q, user_context)
                company = upsert_company_and_employees(data, raw_user_input=q)
                st.toast("New data found and saved!", icon="✅")
            except Exception as e:
                st.error(f"Error: {e}")
                st.stop()
    else:
        st.toast("Loaded from local database.", icon="🗄️")

    # Display Results
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Details")
        st.markdown(f"**Industry:** {company.industry or 'N/A'}")
        st.markdown(f"**Domain:** {company.domain or 'N/A'}")
        st.markdown(f"**Size:** {company.employee_size or 'N/A'} employees")
        st.markdown(f"**Contact:** {company.email or 'N/A'}")

    with col2:
        st.subheader("Key Personnel")
        with Session(engine) as session:
            employees = session.exec(select(Employee).where(Employee.company_id == company.id)).all()
        
        if employees:
            st.table([{
                "Name": e.full_name,
                "Title": e.title,
                "Dept": e.department,
                "LinkedIn": e.profile_url
            } for e in employees])
        else:
            st.info("No employee records found.")
