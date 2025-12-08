# app.py
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

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL not set in .env")

engine = create_engine(DATABASE_URL, echo=False, future=True)
SQLModel.metadata.create_all(engine)

try:
    with engine.connect() as conn:
        conn.execute(sa_text("ALTER TABLE company ADD COLUMN IF NOT EXISTS email VARCHAR(255);"))
        conn.commit()
except Exception:
    pass

client = None
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    client = genai.Client(api_key=GEMINI_API_KEY)
def normalize_name(s: str) -> str:
    return " ".join(s.split()).strip().casefold() if s else ""

def has_name_normalized_field() -> bool:
    return hasattr(Company, "name_normalized")

def get_company_by_name(session: Session, raw_user_input: str) -> Optional[Company]:
    if not raw_user_input:
        return None

    raw_norm = raw_user_input.strip().lower()

    # If we have name_normalized, treat it as the lowercase raw user input key.
    if has_name_normalized_field():
        company = session.exec(select(Company).where(Company.name_normalized == raw_norm)).first()
        if company:
            return company

    # Exact name match 
    company = session.exec(select(Company).where(Company.name == raw_user_input)).first()
    if company:
        return company

    # Case-insensitive match on Company.name
    return session.exec(select(Company).where(func.lower(Company.name) == raw_norm)).first()

def strip_code_fences(text: str) -> str:
    if not text:
        return text
    text = re.sub(r"```(?:json|js|txt)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()

def safe_json_parse(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None

    text = strip_code_fences(text)

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : i + 1]
                try:
                    parsed = json.loads(candidate)
                    if isinstance(parsed, dict):
                        return parsed
                except Exception:
                    # try to fix common problems: trailing commas
                    candidate_fixed = re.sub(r",\s*}", "}", candidate)
                    candidate_fixed = re.sub(r",\s*\]", "]", candidate_fixed)
                    try:
                        parsed = json.loads(candidate_fixed)
                        if isinstance(parsed, dict):
                            return parsed
                    except Exception:
                        return None
    return None

def parse_employee_list_from_text(text: str) -> List[Dict[str, Any]]:
    employees: List[Dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = re.split(r"\s*\|\s*|\s*-\s*|\s*;\s*|\t|\s{2,}", line)
        emp = {
            "full_name": None,
            "title": None,
            "department": None,
            "seniority": None,
            "profile_url": None,
            "email": None,
        }
        if len(parts) >= 1:
            emp["full_name"] = parts[0].strip()
        if len(parts) >= 2:
            emp["title"] = parts[1].strip()
        if len(parts) >= 3:
            emp["department"] = parts[2].strip()
        if len(parts) >= 4:
            emp["seniority"] = parts[3].strip()
        if len(parts) >= 5:
            # sometimes URL or email
            maybe = parts[4].strip()
            if "@" in maybe:
                emp["email"] = maybe
            else:
                emp["profile_url"] = maybe
        if not emp["email"]:
            m = re.search(r"([a-zA-Z0-9.\-_+]+@[a-zA-Z0-9\-_]+\.[a-zA-Z0-9.\-_]+)", line)
            if m:
                emp["email"] = m.group(1)
        if emp["full_name"]:
            employees.append(emp)
    return employees

GEMINI_PROMPT_COMPANY = """
You are a factual assistant. Using Google Search grounding, return a JSON object only (no commentary):

{
  "company": {
    "name": "string or null",
    "industry": "string or null",
    "employee_size": "integer or null",
    "domain": "string or null",
    "email": "string or null"
  },
  "employees": [
    {
      "full_name":"string",
      "title":"string or null",
      "department":"string or null",
      "seniority":"string or null",
      "profile_url":"string or null",
      "email":"string or null"
    }
  ]
}

If you don't know a field, use null. Keep employees <= 10.
Consider any additional context provided to disambiguate companies with the same name.
"""

def fetch_from_gemini(company_query: str, context: str = "") -> Dict[str, Any]:
    if not client:
        raise RuntimeError("GEMINI_API_KEY not set so can't call Gemini.")

    tool = types.Tool(google_search=types.GoogleSearch())
    cfg = types.GenerateContentConfig(tools=[tool])

    query_text = f'Query: "{company_query}"'
    if context and context.strip():
        query_text += f'\nContext: "{context.strip()}"'

    contents = GEMINI_PROMPT_COMPANY + "\n" + query_text
    resp = client.models.generate_content(model="gemini-2.5-flash", contents=contents, config=cfg)

    text = getattr(resp, "text", None) or str(resp)
    parsed = safe_json_parse(text)
    if parsed:
        return parsed
    fallback_employees = parse_employee_list_from_text(text)
    return {
        "company": {"name": company_query, "industry": None, "employee_size": None, "domain": None, "email": None},
        "employees": fallback_employees,
    }

def upsert_company_and_employees(data: Dict[str, Any], raw_user_input: str = "") -> Company:
    company_data = data.get("company") or {}
    employees = data.get("employees") or []
    company_name = company_data.get("name") or raw_user_input or "Unknown"

    normalized_user_input = raw_user_input.strip().lower() if raw_user_input else None

    with Session(engine) as session:
        existing = None
        if normalized_user_input and has_name_normalized_field():
            existing = session.exec(select(Company).where(Company.name_normalized == normalized_user_input)).first()

        if not existing and company_data.get("name"):
            existing = session.exec(select(Company).where(Company.name == company_data["name"])).first()

        if not existing:
            existing = session.exec(select(Company).where(func.lower(Company.name) == (company_name or "").lower())).first()

        if existing:
            if company_data.get("industry"):
                existing.industry = company_data["industry"]
            if company_data.get("employee_size") not in (None, ""):
                try:
                    existing.employee_size = int(company_data["employee_size"])
                except Exception:
                    pass
            if company_data.get("domain"):
                existing.domain = company_data["domain"]
            if company_data.get("email"):
                existing.email = company_data["email"]
            if has_name_normalized_field() and normalized_user_input:
                existing.name_normalized = normalized_user_input
            session.add(existing)
            session.commit()
            session.refresh(existing)
            company_obj = existing
        else:
            company_obj = Company(
                name=company_name,
                **({"name_normalized": normalized_user_input} if has_name_normalized_field() and normalized_user_input else {}),
                industry=company_data.get("industry"),
                employee_size=int(company_data["employee_size"]) if str(company_data.get("employee_size") or "").isdigit() else None,
                domain=company_data.get("domain"),
            )
            if company_data.get("email"):
                try:
                    setattr(company_obj, "email", company_data.get("email"))
                except Exception:
                    pass

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

            stmt = select(Employee).where(
                (Employee.company_id == company_obj.id) & (func.lower(Employee.full_name) == full_name.lower())
            )
            existing_emp = session.exec(stmt).first()
            if existing_emp:
                if e.get("title") and existing_emp.title in (None, "Not found"):
                    existing_emp.title = e.get("title")
                if e.get("department") and existing_emp.department in (None, "Not found"):
                    existing_emp.department = e.get("department")
                if e.get("seniority") and existing_emp.seniority in (None, "Not found"):
                    existing_emp.seniority = e.get("seniority")
                if e.get("profile_url") and existing_emp.profile_url in (None, "Not found"):
                    existing_emp.profile_url = e.get("profile_url")
                if e.get("email") and getattr(existing_emp, "email", None) in (None, "Not found"):
                    try:
                        existing_emp.email = e.get("email")
                    except Exception:
                        pass
                session.add(existing_emp)
            else:
                new_emp_kwargs = {
                    "full_name": full_name,
                    "title": e.get("title") or "Not found",
                    "department": e.get("department") or "Not found",
                    "seniority": e.get("seniority") or "Not found",
                    "profile_url": e.get("profile_url") or "Not found",
                    "company_id": company_obj.id,
                }
                if e.get("email"):
                    new_emp_kwargs["email"] = e.get("email")
                new_emp = Employee(**new_emp_kwargs)
                session.add(new_emp)
        session.commit()
        session.refresh(company_obj)
        return company_obj


def generate_bulk_search_templates(company: str) -> List[str]:
    if not client:
        return []
    prompt = f'Generate 3 concise Google search queries to find verified employee emails for "{company}". Return JSON: {{"search_templates":["q1","q2","q3"]}}'
    resp = client.models.generate_content(model="gemini-2.5-flash", contents=prompt, config=types.GenerateContentConfig())
    parsed = safe_json_parse(getattr(resp, "text", str(resp)))
    return parsed.get("search_templates", []) if parsed else []

def fetch_bulk_emails(company: str, templates: List[str]) -> Dict[str, Any]:
    if not client:
        return {"company": {"name": company}, "employees": []}
    tool = types.Tool(google_search=types.GoogleSearch())
    cfg = types.GenerateContentConfig(tools=[tool])

    joined = "\n".join(templates)
    prompt = f"""
Using these queries:
{joined}
Find verified employees and their emails for '{company}'. Return JSON:
{{"company":{{"name":"{company}","domain":"string|null","industry":"string|null","employee_size":"integer|null"}}, "employees":[{{"full_name":"string","title":"string|null","department":"string|null","email":"string|null","profile_url":"string|null"}}]}}

    resp = client.models.generate_content(model="gemini-2.5-flash", contents=prompt, config=cfg)
    parsed = safe_json_parse(getattr(resp, "text", str(resp)))
    return parsed or {"company": {"name": company}, "employees": []}

st.set_page_config(page_title="Company Research", layout="wide")
st.title("🔎 Company Research Tool")

with st.form("company_form"):
    q = st.text_input("Enter company name", "")
    user_context = st.text_area("Additional context (optional)", placeholder="e.g., Electric vehicles company in California")
    st.form_submit_button("Search", type="primary")

if q:
    with Session(engine) as session:
        company = get_company_by_name(session, q)

    if company:
        needs_company_fields = not company.industry or not company.domain or (company.employee_size in (None, ""))
        with Session(engine) as session:
            has_any_employee = session.exec(select(Employee.id).where(Employee.company_id == company.id)).first() is not None

        if (needs_company_fields or not has_any_employee) and client:
            st.info("🔄 Enriching from Gemini…")
            try:
                data = fetch_from_gemini(q, user_context)
                company = upsert_company_and_employees(data, raw_user_input=q)
            except Exception as e:
                st.warning(f"Enrichment failed: {e}")
        st.success("✅ Found in Neon DB")
    else:
        if not client:
            st.error("Not in DB and GEMINI_API_KEY not configured")
            st.stop()
        st.info("🌐 Fetching from Gemini...")
        try:
            data = fetch_from_gemini(q, user_context)
            company = upsert_company_and_employees(data, raw_user_input=q)
        except Exception as e:
            st.error(f"Error fetching/saving: {e}")
            st.stop()

    with Session(engine) as session:
        employees = session.exec(select(Employee).where(Employee.company_id == company.id)).all()

    if (not employees or len(employees) == 0) and client:
        st.info("🔍 Attempting to find employees & emails...")
        try:
            templates = generate_bulk_search_templates(q)
            if templates:
                data = fetch_bulk_emails(q, templates)
                company = upsert_company_and_employees(data, raw_user_input=q)
                with Session(engine) as session:
                    employees = session.exec(select(Employee).where(Employee.company_id == company.id)).all()
                st.success("✅ Employee enrichment attempt finished")
        except Exception as e:
            st.warning(f"Email enrichment failed: {e}")

    st.subheader("🏢 Company Information")

    def compare_field(label, user_val, db_val):
        if user_val and str(user_val).strip():
            if str(user_val).strip().lower() != str(db_val or "").strip().lower():
                st.write(f"**{label}:** ❌ Mismatch | You entered: `{user_val}` | Best Result: `{db_val or 'Not found'}`")
            else:
                st.write(f"**{label}:** {db_val}")
        else:
            st.write(f"**{label}:** {db_val or 'Not found'}")

    compare_field("Name", q, company.name)
    st.write(f"**Industry:** {company.industry or 'Not found'}")
    st.write(f"**Employee size:** {company.employee_size or 'Not found'}")
    st.write(f"**Domain:** {company.domain or 'Not found'}")
    st.write(f"**Email:** {getattr(company, 'email', None) or 'Not found'}")

    st.subheader("👥 Employees")
    if employees:
        rows = []
        for e in employees:
            rows.append({
                "Name": e.full_name,
                "Title": e.title,
                "Department": e.department,
                "Seniority": e.seniority,
                "Email": getattr(e, "email", None) or "Not found",
                "Profile": e.profile_url or "Not found"
            })
        st.table(rows)
    else:
        st.info("No employees stored for this company.")
