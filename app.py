import json
import re
import os
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
from sqlmodel import Session, select, create_engine
from sqlalchemy import func
from models import Company, Employee
from sqlmodel import SQLModel
import google.generativeai as genai
import streamlit as st

load_dotenv()
DB_FILE = os.getenv("DB_FILE", "company_research.db")
DATABASE_URL = f"sqlite:///{DB_FILE}"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", None)

engine = create_engine(DATABASE_URL, echo=False)
SQLModel.metadata.create_all(engine)

client = None
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    client = genai.GenerativeModel("gemini-2.0-flash-exp")

def normalize_name(s: str) -> str:
    return " ".join(s.split()).strip().casefold() if s else ""

def has_name_normalized_field() -> bool:
    return hasattr(Company, "name_normalized")

def get_company_by_name(session: Session, name: str) -> Optional[Company]:
    if not name:
        return None
    stmt = select(Company).where(Company.name == name)
    res = session.exec(stmt).first()
    if res:
        return res
    if has_name_normalized_field():
        norm = normalize_name(name)
        stmt = select(Company).where(Company.name_normalized == norm)
        res = session.exec(stmt).first()
        if res:
            return res
    stmt = select(Company).where(func.lower(Company.name) == name.lower())
    return session.exec(stmt).first()

def safe_json_parse(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start:end+1])
            except Exception:
                return None
    return None

def parse_employee_list_from_text(text: str) -> List[Dict[str, Any]]:
    employees: List[Dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = re.split(r"\s*\|\s*|\s*-\s*|\s*,\s*", line)
        emp = {"full_name": None, "title": None, "department": None, "seniority": None, "profile_url": None}
        if len(parts) >= 1:
            emp["full_name"] = parts[0].strip()
        if len(parts) >= 2:
            emp["title"] = parts[1].strip()
        if len(parts) >= 3:
            emp["department"] = parts[2].strip()
        if len(parts) >= 4:
            emp["seniority"] = parts[3].strip()
        if len(parts) >= 5:
            emp["profile_url"] = parts[4].strip()
        if emp["full_name"]:
            employees.append(emp)
    return employees

GEMINI_PROMPT_COMPANY = """
You are a factual assistant. Using Google Search grounding, return EXACTLY a JSON object:
{
  "company": {
    "name": "string",
    "industry": "string|null",
    "employee_size": "integer|null",
    "domain": "string|null"
  },
  "employees": [
    {"full_name":"string","title":"string|null","department":"string|null","seniority":"string|null","profile_url":"string|null"}
  ]
}
Keep employees <=10. Only publicly listed.
"""

def fetch_from_gemini(company_query: str) -> Dict[str, Any]:
    if not client:
        raise RuntimeError("GEMINI_API_KEY not set; can't call Gemini.")
    resp = client.generate_content(GEMINI_PROMPT_COMPANY + f'\nQuery: "{company_query}"')
    text = getattr(resp, "text", str(resp))
    parsed = safe_json_parse(text)
    if parsed:
        return parsed
    return {"company": {"name": company_query, "industry": None, "employee_size": None, "domain": None},
            "employees": parse_employee_list_from_text(text)}

def upsert_company_and_employees(data: Dict[str, Any]) -> Company:
    company_data = data.get("company") or {}
    employees = data.get("employees") or []
    company_name = company_data.get("name") or "Unknown"
    industry = company_data.get("industry")
    employee_size = company_data.get("employee_size")
    domain = company_data.get("domain")

    with Session(engine) as session:
        existing = get_company_by_name(session, company_name)
        if existing:
            if industry: 
                existing.industry = industry
            if employee_size is not None:
                try: 
                    existing.employee_size = int(employee_size)
                except Exception:
                    pass
            if domain: 
                existing.domain = domain
            if has_name_normalized_field(): 
                existing.name_normalized = normalize_name(company_name)
                session.add(existing)
                session.commit()
                session.refresh(existing)
            company_obj = existing
        else:
            company_obj = Company(
                name=company_name,
                **({"name_normalized": normalize_name(company_name)} if has_name_normalized_field() else {}),
                industry=industry or None,
                employee_size=int(employee_size) if isinstance(employee_size, (int, str)) and str(employee_size).isdigit() else None,
                domain=domain or None
            )
            session.add(company_obj)
            session.commit()
            session.refresh(company_obj)

        seen = set()
        for e in employees:
            full_name = (e.get("full_name") or "").strip()
            if not full_name:
                continue
            key = normalize_name(full_name)
            if key in seen:
                continue
            seen.add(key)

            stmt = select(Employee).where((Employee.company_id == company_obj.id) & (func.lower(Employee.full_name) == full_name.lower()))
            existing_emp = session.exec(stmt).first()
            if existing_emp:
                if e.get("title") and (existing_emp.title in (None, "Not found")): 
                    existing_emp.title = e.get("title")
                if e.get("department") and (existing_emp.department in (None, "Not found")): 
                    existing_emp.department = e.get("department")
                if e.get("seniority") and (existing_emp.seniority in (None, "Not found")): 
                    existing_emp.seniority = e.get("seniority")
                if e.get("profile_url") and (existing_emp.profile_url in (None, "Not found")): 
                    existing_emp.profile_url = e.get("profile_url")
                session.add(existing_emp)
            else:
                new_emp = Employee(
                    full_name=full_name,
                    title=e.get("title") or "Not found",
                    department=e.get("department") or "Not found",
                    seniority=e.get("seniority") or "Not found",
                    profile_url=e.get("profile_url") or "Not found",
                    company_id=company_obj.id
                )
                session.add(new_emp)
        session.commit()
        session.refresh(company_obj)
        return company_obj

# Streamlit App 
st.set_page_config(page_title="Company Research", layout="wide")
st.title("🔎 Company Research Tool")

with st.form("company_form"):
    q = st.text_input("Enter company name", "")
    industry_in = st.text_input("Industry (optional)", "")
    emp_in = st.text_input("Employee Size (optional)", "")
    domain_in = st.text_input("Domain/Website (optional)", "")
    submitted = st.form_submit_button("Search")

if submitted and q:
    user_inputs = {"industry": industry_in, "employee_size": emp_in, "domain": domain_in}
    with Session(engine) as session:
        company = get_company_by_name(session, q)

    if company:
        st.success("✅ Found in local database (Gemini NOT called)")
    else:
        if not client:
            st.error("❌ Not in DB and GEMINI_API_KEY not configured")
            st.stop()
        st.info("🌐 Fetching from Gemini...")
        try:
            data = fetch_from_gemini(q)
            company = upsert_company_and_employees(data)
        except Exception as e:
            st.error(f"Error fetching/saving: {e}")
            st.stop()

    st.subheader("🏢 Company Information")
    st.write(f"**Name:** {company.name}")
    st.write(f"**Industry:** {company.industry or 'Not found'}")
    st.write(f"**Employee size:** {company.employee_size or 'Not found'}")
    st.write(f"**Domain:** {company.domain or 'Not found'}")

    with Session(engine) as session:
        employees = session.exec(select(Employee).where(Employee.company_id == company.id)).all()
    st.subheader("👥 Employees")
    if employees:
        st.table([{
            "Name": e.full_name,
            "Title": e.title,
            "Department": e.department,
            "Seniority": e.seniority,
            "Profile": e.profile_url
        } for e in employees])
    else:
        st.info("No employees stored for this company.")




