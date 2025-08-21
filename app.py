import os
import json
import re
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
from sqlmodel import Session, select, create_engine, SQLModel
from sqlalchemy import func
from models import Company, Employee
import google.generativeai as genai
import streamlit as st
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL, echo=False)
SQLModel.metadata.create_all(engine)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
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
    # Exact match
    res = session.exec(select(Company).where(Company.name == name)).first()
    if res:
        return res
    # Normalized field 
    if has_name_normalized_field():
        norm = normalize_name(name)
        res = session.exec(select(Company).where(Company.name_normalized == norm)).first()
        if res:
            return res
    return session.exec(select(Company).where(func.lower(Company.name) == name.lower())).first()

def safe_json_parse(text: str) -> Optional[Dict[str, Any]]:
    """Extract valid JSON from Gemini response."""
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start:end+1])
            except Exception:
                return None
    return None

def parse_employee_list_from_text(text: str) -> List[Dict[str, Any]]:
    """Fallback parser if Gemini doesn’t return valid JSON."""
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
You are a professional data extraction agent. Your task is to provide a complete and accurate JSON object with company and employee data for a given query. 
**JSON Structure:**
{
  "company": {
    "name": "string",
    "industry": "string, multiple separated by comma",
    "employee_size": "string,",
    "domain": "string"
  },
  "employees": [
    {
      "full_name": "string",
      "title": "string",
      "department": "string",
      "seniority": "string, e.g., 'Entry', 'Mid', 'Senior', 'Manager', 'Director', 'VP', 'C-Level'",
      "profile_url": "string"
    }
  ]
}
**Instructions:**
- **Accuracy is important.** Use all available knowledge and data sources to ensure the information is correct. Google the company and its employees to verify details.
- If any data for a field is not available or cannot be verified, the value **must be `null`**. 
- If no employees can be found, the `employees` array must be an **empty list
- For the `seniority`, use one of the provided examples.

**Example: Amazon**
```json
{
  "company": {
    "name": "Amazon",
    "industry": "E-commerce, Technology, Retail, Cloud Computing, Logistics",
    "employee_size": "100000+",
    "domain": "amazon.com"
  },
  "employees": [
    {
      "full_name": "Andy Jassy",
      "title": "Chief Executive Officer",
      "department": "Executive",
      "seniority": "C-Level",
      "profile_url": "https://www.linkedin.com/in/andyjassy"
    },
    {
      "full_name": "Adam Selipsky",
      "title": "Chief Executive Officer, AWS",
      "department": "AWS",
      "seniority": "C-Level",
      "profile_url": "https://www.linkedin.com/in/adamselipsky"
    },
    {
      "full_name": "Doug Herrington",
      "title": "Chief Executive Officer, Worldwide Amazon Stores",
      "department": "Retail",
      "seniority": "C-Level",
      "profile_url": "https://www.linkedin.com/in/douglasherrington"
    }
  ]
} """

def fetch_from_gemini(company_query: str, context: str = "") -> Dict[str, Any]:
    if not client:
        raise RuntimeError("GEMINI_API_KEY not set; can't call Gemini.")
    query_text = f'Query: "{company_query}"'
    if context.strip():
        query_text += f'\nContext: "{context.strip()}"'
    resp = client.generate_content(GEMINI_PROMPT_COMPANY + "\n" + query_text)
    text = getattr(resp, "text", str(resp))
    parsed = safe_json_parse(text)
    if parsed:
        return parsed
    return {
        "company": {"name": company_query, "industry": None, "employee_size": None, "domain": None},
        "employees": parse_employee_list_from_text(text),
    }

def upsert_company_and_employees(data: Dict[str, Any]) -> Company:
    """Save Gemini result into Neon DB (Company + Employees)."""
    company_data = data.get("company") or {}
    employees = data.get("employees") or []
    company_name = company_data.get("name") or "Unknown"

    with Session(engine) as session:
        existing = get_company_by_name(session, company_name)
        if existing:
            # Update only missing fields
            if company_data.get("industry"): 
                existing.industry = company_data["industry"]
            if company_data.get("employee_size") not in (None, ""):
                try: 
                    existing.employee_size = int(company_data["employee_size"])
                except Exception:
                    pass
            if company_data.get("domain"): 
                existing.domain = company_data["domain"]
            if has_name_normalized_field(): 
                existing.name_normalized = normalize_name(company_name)
            session.add(existing)
            session.commit()
            session.refresh(existing)
            company_obj = existing
        else:
            # Insert new
            company_obj = Company(
                name=company_name,
                **({"name_normalized": normalize_name(company_name)} if has_name_normalized_field() else {}),
                industry=company_data.get("industry"),
                employee_size=int(company_data["employee_size"]) if str(company_data.get("employee_size") or "").isdigit() else None,
                domain=company_data.get("domain"),
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

            stmt = select(Employee).where(
                (Employee.company_id == company_obj.id) & 
                (func.lower(Employee.full_name) == full_name.lower())
            )
            existing_emp = session.exec(stmt).first()
            if existing_emp:
                # Only update if missing
                if e.get("title") and existing_emp.title in (None, "Not found"): 
                    existing_emp.title = e["title"]
                if e.get("department") and existing_emp.department in (None, "Not found"): 
                    existing_emp.department = e["department"]
                if e.get("seniority") and existing_emp.seniority in (None, "Not found"): 
                    existing_emp.seniority = e["seniority"]
                if e.get("profile_url") and existing_emp.profile_url in (None, "Not found"): 
                    existing_emp.profile_url = e["profile_url"]
                session.add(existing_emp)
            else:
                new_emp = Employee(
                    full_name=full_name,
                    title=e.get("title") or "Not found",
                    department=e.get("department") or "Not found",
                    seniority=e.get("seniority") or "Not found",
                    profile_url=e.get("profile_url") or "Not found",
                    company_id=company_obj.id,
                )
                session.add(new_emp)

        session.commit()
        session.refresh(company_obj)

st.set_page_config(page_title="Company Research", layout="wide")
st.title("🔎 Company Research Tool")

with st.form("company_form"):
    q = st.text_input("Enter company name", "")
    user_context = st.text_area("Additional context (optional)", placeholder="e.g., Electric vehicles company, not the investment firm")
    st.form_submit_button("Search", type="primary")
if q:
    with Session(engine) as session:
        company = get_company_by_name(session, q)

    if company:
        st.success("✅ Found in Neon DB (Gemini NOT called)")
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

    def display_field(label, val):
        st.write(f"**{label}:** {val or 'Not found'}")

    display_field("Name", company.name)
    display_field("Industry", company.industry)
    display_field("Employee size", company.employee_size)
    display_field("Domain", company.domain)

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
