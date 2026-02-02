
import json
import logging
import os
import re
from typing import Any

import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types
from sqlalchemy import func, text as sa_text
from sqlmodel import Session, SQLModel, create_engine, select

from models import Company, Employee

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# Configuration
DATABASE_URL = os.getenv("DATABASE_URL")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL not set in .env")

# Database setup
engine = create_engine(DATABASE_URL, echo=False, future=True)
SQLModel.metadata.create_all(engine)


def _add_email_column_if_missing() -> None:
    """Add email column to company table if it doesn't exist (handles SQLite and PostgreSQL)."""
    try:
        with engine.connect() as conn:
            # SQLite approach
            result = conn.execute(sa_text("PRAGMA table_info(company)"))
            columns = [row[1] for row in result.fetchall()]
            if "email" not in columns:
                conn.execute(
                    sa_text("ALTER TABLE company ADD COLUMN email VARCHAR(255);")
                )
                conn.commit()
    except Exception:
        # PostgreSQL approach
        try:
            with engine.connect() as conn:
                conn.execute(
                    sa_text(
                        "ALTER TABLE company ADD COLUMN IF NOT EXISTS email VARCHAR(255);"
                    )
                )
                conn.commit()
        except Exception:
            pass


_add_email_column_if_missing()

# Gemini client setup
gemini_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

# Constants
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


def normalize_name(s: str) -> str:
    """Normalize a name for comparison (lowercase, collapse whitespace)."""
    return " ".join(s.split()).strip().casefold() if s else ""


def get_company_by_name(session: Session, raw_user_input: str) -> Company | None:
    """Find a company by name using normalized and case-insensitive matching."""
    if not raw_user_input:
        return None

    raw_norm = raw_user_input.strip().lower()

    # Try normalized field first
    if hasattr(Company, "name_normalized"):
        company = session.exec(
            select(Company).where(Company.name_normalized == raw_norm)
        ).first()
        if company:
            return company

    # Exact name match
    company = session.exec(
        select(Company).where(Company.name == raw_user_input)
    ).first()
    if company:
        return company

    # Case-insensitive match
    return session.exec(
        select(Company).where(func.lower(Company.name) == raw_norm)
    ).first()


def strip_code_fences(text: str) -> str:
    """Remove markdown code fences from text."""
    if not text:
        return text
    text = re.sub(r"```(?:json|js|txt)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def safe_json_parse(text: str) -> dict[str, Any] | None:
    """Parse JSON from text, handling code fences and common formatting issues."""
    if not text:
        return None

    text = strip_code_fences(text)

    # Try direct parse first
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    # Try to extract JSON object from text
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
                except json.JSONDecodeError:
                    # Fix trailing commas
                    candidate_fixed = re.sub(r",\s*}", "}", candidate)
                    candidate_fixed = re.sub(r",\s*\]", "]", candidate_fixed)
                    try:
                        parsed = json.loads(candidate_fixed)
                        if isinstance(parsed, dict):
                            return parsed
                    except json.JSONDecodeError:
                        return None
    return None


def parse_employee_list_from_text(text: str) -> list[dict[str, Any]]:
    """Parse employee data from unstructured text as fallback."""
    employees: list[dict[str, Any]] = []
    email_pattern = re.compile(r"([a-zA-Z0-9.\-_+]+@[a-zA-Z0-9\-_]+\.[a-zA-Z0-9.\-_]+)")

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        parts = re.split(r"\s*\|\s*|\s*-\s*|\s*;\s*|\t|\s{2,}", line)
        emp: dict[str, Any] = {
            "full_name": parts[0].strip() if parts else None,
            "title": parts[1].strip() if len(parts) >= 2 else None,
            "department": parts[2].strip() if len(parts) >= 3 else None,
            "seniority": parts[3].strip() if len(parts) >= 4 else None,
            "profile_url": None,
            "email": None,
        }

        if len(parts) >= 5:
            maybe = parts[4].strip()
            if "@" in maybe:
                emp["email"] = maybe
            else:
                emp["profile_url"] = maybe

        # Extract email from line if not found
        if not emp["email"]:
            match = email_pattern.search(line)
            if match:
                emp["email"] = match.group(1)

        if emp["full_name"]:
            employees.append(emp)

    return employees


def fetch_from_gemini(company_query: str, context: str = "") -> dict[str, Any]:
    """Fetch company and employee data from Gemini AI with Google Search grounding."""
    if not gemini_client:
        raise RuntimeError("GEMINI_API_KEY not set")

    tool = types.Tool(google_search=types.GoogleSearch())
    cfg = types.GenerateContentConfig(tools=[tool])

    query_text = f'Query: "{company_query}"'
    if context and context.strip():
        query_text += f'\nContext: "{context.strip()}"'

    contents = GEMINI_PROMPT_COMPANY + "\n" + query_text
    resp = gemini_client.models.generate_content(
        model=GEMINI_MODEL, contents=contents, config=cfg
    )

    text = getattr(resp, "text", None) or str(resp)
    parsed = safe_json_parse(text)

    if parsed:
        return parsed

    # Fallback: try to parse employees from unstructured text
    return {
        "company": {
            "name": company_query,
            "industry": None,
            "employee_size": None,
            "domain": None,
            "email": None,
        },
        "employees": parse_employee_list_from_text(text),
    }


def _parse_employee_size(value: Any) -> int | None:
    """Safely parse employee size to integer."""
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (ValueError, TypeError):
        return None


def upsert_company_and_employees(
    data: dict[str, Any], raw_user_input: str = ""
) -> Company:
    """Insert or update company and employee records in the database."""
    company_data = data.get("company") or {}
    employees = data.get("employees") or []
    company_name = company_data.get("name") or raw_user_input or "Unknown"
    normalized_user_input = raw_user_input.strip().lower() if raw_user_input else None

    with Session(engine) as session:
        # Find existing company
        existing = _find_existing_company(
            session, normalized_user_input, company_data, company_name
        )

        if existing:
            company_obj = _update_company(
                session, existing, company_data, normalized_user_input
            )
        else:
            company_obj = _create_company(
                session, company_name, company_data, normalized_user_input
            )

        _upsert_employees(session, company_obj.id, employees)
        session.refresh(company_obj)
        return company_obj


def _find_existing_company(
    session: Session,
    normalized_input: str | None,
    company_data: dict,
    company_name: str,
) -> Company | None:
    """Find existing company by various matching strategies."""
    if normalized_input and hasattr(Company, "name_normalized"):
        existing = session.exec(
            select(Company).where(Company.name_normalized == normalized_input)
        ).first()
        if existing:
            return existing

    if company_data.get("name"):
        existing = session.exec(
            select(Company).where(Company.name == company_data["name"])
        ).first()
        if existing:
            return existing

    return session.exec(
        select(Company).where(func.lower(Company.name) == company_name.lower())
    ).first()


def _update_company(
    session: Session, company: Company, company_data: dict, normalized_input: str | None
) -> Company:
    """Update existing company with new data."""
    if company_data.get("industry"):
        company.industry = company_data["industry"]
    if company_data.get("employee_size") not in (None, ""):
        company.employee_size = _parse_employee_size(company_data["employee_size"])
    if company_data.get("domain"):
        company.domain = company_data["domain"]
    if company_data.get("email"):
        company.email = company_data["email"]
    if hasattr(Company, "name_normalized") and normalized_input:
        company.name_normalized = normalized_input

    session.add(company)
    session.commit()
    session.refresh(company)
    return company


def _create_company(
    session: Session,
    company_name: str,
    company_data: dict,
    normalized_input: str | None,
) -> Company:
    """Create a new company record."""
    kwargs: dict[str, Any] = {
        "name": company_name,
        "industry": company_data.get("industry"),
        "employee_size": _parse_employee_size(company_data.get("employee_size")),
        "domain": company_data.get("domain"),
        "email": company_data.get("email"),
    }

    if hasattr(Company, "name_normalized") and normalized_input:
        kwargs["name_normalized"] = normalized_input

    company_obj = Company(**kwargs)
    session.add(company_obj)
    session.commit()
    session.refresh(company_obj)
    return company_obj


def _upsert_employees(session: Session, company_id: int, employees: list[dict]) -> None:
    """Insert or update employee records for a company."""
    seen: set[str] = set()

    for emp_data in employees:
        full_name = (emp_data.get("full_name") or "").strip()
        if not full_name:
            continue

        key = normalize_name(full_name)
        if key in seen:
            continue
        seen.add(key)

        existing_emp = session.exec(
            select(Employee).where(
                (Employee.company_id == company_id)
                & (func.lower(Employee.full_name) == full_name.lower())
            )
        ).first()

        if existing_emp:
            _update_employee(existing_emp, emp_data)
            session.add(existing_emp)
        else:
            new_emp = Employee(
                full_name=full_name,
                title=emp_data.get("title") or "Not found",
                department=emp_data.get("department") or "Not found",
                seniority=emp_data.get("seniority") or "Not found",
                profile_url=emp_data.get("profile_url") or "Not found",
                email=emp_data.get("email"),
                company_id=company_id,
            )
            session.add(new_emp)

    session.commit()


def _update_employee(employee: Employee, data: dict) -> None:
    """Update employee fields if they have placeholder values."""
    placeholder_values = (None, "Not found")

    if data.get("title") and employee.title in placeholder_values:
        employee.title = data["title"]
    if data.get("department") and employee.department in placeholder_values:
        employee.department = data["department"]
    if data.get("seniority") and employee.seniority in placeholder_values:
        employee.seniority = data["seniority"]
    if data.get("profile_url") and employee.profile_url in placeholder_values:
        employee.profile_url = data["profile_url"]
    if data.get("email") and getattr(employee, "email", None) in placeholder_values:
        employee.email = data["email"]


def generate_bulk_search_templates(company: str) -> list[str]:
    """Generate search query templates for finding employee emails."""
    if not gemini_client:
        return []

    prompt = f'Generate 3 concise Google search queries to find verified employee emails for "{company}". Return JSON: {{"search_templates":["q1","q2","q3"]}}'
    resp = gemini_client.models.generate_content(
        model=GEMINI_MODEL, contents=prompt, config=types.GenerateContentConfig()
    )
    parsed = safe_json_parse(getattr(resp, "text", str(resp)))
    return parsed.get("search_templates", []) if parsed else []


def fetch_bulk_emails(company: str, templates: list[str]) -> dict[str, Any]:
    """Fetch employee emails using generated search templates."""
    if not gemini_client:
        return {"company": {"name": company}, "employees": []}

    tool = types.Tool(google_search=types.GoogleSearch())
    cfg = types.GenerateContentConfig(tools=[tool])

    joined = "\n".join(templates)
    prompt = f"""
Using these queries:
{joined}
Find verified employees and their emails for '{company}'. Return JSON:
{{"company":{{"name":"{company}","domain":"string|null","industry":"string|null","employee_size":"integer|null"}}, "employees":[{{"full_name":"string","title":"string|null","department":"string|null","email":"string|null","profile_url":"string|null"}}]}}
"""

    resp = gemini_client.models.generate_content(
        model=GEMINI_MODEL, contents=prompt, config=cfg
    )
    parsed = safe_json_parse(getattr(resp, "text", str(resp)))
    return parsed or {"company": {"name": company}, "employees": []}


# Streamlit UI
st.set_page_config(page_title="Company Research", layout="wide")
st.title("🔎 Company Research Tool")

with st.form("company_form"):
    query = st.text_input("Enter company name", "")
    user_context = st.text_area(
        "Additional context (optional)",
        placeholder="e.g., Electric vehicles company in California",
    )
    submitted = st.form_submit_button("Search", type="primary")

if query and submitted:
    with Session(engine) as session:
        company = get_company_by_name(session, query)

    if company:
        needs_enrichment = (
            not company.industry or not company.domain or company.employee_size is None
        )
        with Session(engine) as session:
            has_employees = (
                session.exec(
                    select(Employee.id).where(Employee.company_id == company.id)
                ).first()
                is not None
            )

        if (needs_enrichment or not has_employees) and gemini_client:
            st.info("🔄 Enriching from Gemini…")
            try:
                data = fetch_from_gemini(query, user_context)
                company = upsert_company_and_employees(data, raw_user_input=query)
            except Exception as e:
                st.warning(f"Enrichment failed: {e}")
                logger.exception("Enrichment failed")
        st.success("✅ Found in database")
    else:
        if not gemini_client:
            st.error("Company not in database and GEMINI_API_KEY not configured")
            st.stop()
        st.info("🌐 Fetching from Gemini...")
        try:
            data = fetch_from_gemini(query, user_context)
            company = upsert_company_and_employees(data, raw_user_input=query)
        except Exception as e:
            st.error(f"Error fetching/saving: {e}")
            logger.exception("Fetch failed")
            st.stop()

    # Load employees
    with Session(engine) as session:
        employees = session.exec(
            select(Employee).where(Employee.company_id == company.id)
        ).all()

    # Try bulk email enrichment if no employees found
    if not employees and gemini_client:
        st.info("🔍 Attempting to find employees & emails...")
        try:
            templates = generate_bulk_search_templates(query)
            if templates:
                data = fetch_bulk_emails(query, templates)
                company = upsert_company_and_employees(data, raw_user_input=query)
                with Session(engine) as session:
                    employees = session.exec(
                        select(Employee).where(Employee.company_id == company.id)
                    ).all()
                st.success("✅ Employee enrichment complete")
        except Exception as e:
            st.warning(f"Email enrichment failed: {e}")
            logger.exception("Email enrichment failed")

    # Display results
    st.subheader("🏢 Company Information")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Name:** {company.name}")
        st.write(f"**Industry:** {company.industry or 'Not found'}")
        st.write(f"**Employee Size:** {company.employee_size or 'Not found'}")
    with col2:
        st.write(f"**Domain:** {company.domain or 'Not found'}")
        st.write(f"**Email:** {getattr(company, 'email', None) or 'Not found'}")

    st.subheader("👥 Employees")
    if employees:
        rows = [
            {
                "Name": e.full_name,
                "Title": e.title or "Not found",
                "Department": e.department or "Not found",
                "Seniority": e.seniority or "Not found",
                "Email": getattr(e, "email", None) or "Not found",
                "Profile": e.profile_url or "Not found",
            }
            for e in employees
        ]
        st.table(rows)
    else:
        st.info("No employees found for this company.")
