
# backend.py
import os
import json
import re
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
from sqlmodel import Session, select, create_engine
from sqlalchemy import func
from sqlmodel import SQLModel
from models import Company, Employee

load_dotenv()

DB_FILE = os.getenv("DB_FILE", "company_research.db")
DATABASE_URL = f"sqlite:///{DB_FILE}"
engine = create_engine(DATABASE_URL, echo=False)

SQLModel.metadata.create_all(engine)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", None)

_HAS_GEMINI = False
client = None
tools = None
config = None

try:
    from google import genai
    from google.genai.types import GenerateContentConfig, GoogleSearch, Tool
    if GEMINI_API_KEY:
        client = genai.Client(api_key=GEMINI_API_KEY)
        tools = [Tool(google_search=GoogleSearch())]
        config = GenerateContentConfig(tools=tools)
        _HAS_GEMINI = True
except Exception as e:
    print(f"Error initializing Gemini client: {e}")

def normalize_name(s: Optional[str]) -> str:
    return " ".join(s.split()).strip().casefold() if s else ""

def has_name_normalized_field() -> bool:
    return hasattr(Company, "name_normalized")

def get_company_by_name_db(name: str) -> Optional[Company]:

    if not name:
        return None
    with Session(engine) as session:
        # exact match
        stmt = select(Company).where(Company.name == name)
        res = session.exec(stmt).first()
        if res:
            return res
        # normalized
        if has_name_normalized_field():
            norm = normalize_name(name)
            stmt = select(Company).where(Company.name_normalized == norm)
            res = session.exec(stmt).first()
            if res:
                return res
        # case-insensitive fallback
        stmt = select(Company).where(func.lower(Company.name) == name.lower())
        res = session.exec(stmt).first()
        return res

def _safe_json_parse(text: str) -> Optional[Dict[str, Any]]:
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

def _parse_employee_list_from_text(text: str) -> List[Dict[str, Any]]:
    employees: List[Dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith('#') or line.startswith('//'):
            continue
        parts = re.split(r"\s*\|\s*|\s*-\s*|\s*,\s*", line)
        emp = {"full_name": None, "title": None, "department": None, "seniority": None, "profile_url": None}
        if len(parts) >= 1: emp["full_name"] = parts[0].strip()
        if len(parts) >= 2: emp["title"] = parts[1].strip()
        if len(parts) >= 3: emp["department"] = parts[2].strip()
        if len(parts) >= 4: emp["seniority"] = parts[3].strip()
        if len(parts) >= 5: emp["profile_url"] = parts[4].strip()
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
Keep employees <=10. Only publicly listed information.
"""

def fetch_from_gemini(company_query: str) -> Dict[str, Any]:
    
    prompt = GEMINI_PROMPT_COMPANY + f'\nQuery: "{company_query}"'
    
    try:
        resp = client.models.generate_content(
            model="gemini-2.0-flash-exp", 
            contents=prompt, 
            config=config
        )
        text = getattr(resp, "text", str(resp))
        parsed = _safe_json_parse(text)
        if parsed:
            return parsed
    except Exception as e:
        print(f"Error calling Gemini API: {e}")
    
    return {
        "company": {"name": company_query, "industry": None, "employee_size": None, "domain": None},
        "employees": [],
    }

# Upsert function: stores company + employees into DB; returns a dict for UI use
def upsert_company_and_employees(data: Dict[str, Any]) -> Dict[str, Any]:
    company_data = data.get("company") or {}
    employees = data.get("employees") or []
    company_name = company_data.get("name") or "Unknown"
    industry = company_data.get("industry")
    employee_size = company_data.get("employee_size")
    domain = company_data.get("domain")

    with Session(engine) as session:
        existing = get_company_by_name_db(company_name)
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
                employee_size=int(employee_size) if isinstance(employee_size, (int, str)) and str(employee_size).strip().isdigit() else None,
                domain=domain or None,
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
                    company_id=company_obj.id,
                )
                session.add(new_emp)
        session.commit()
        session.refresh(company_obj)

        stored_emps = session.exec(select(Employee).where(Employee.company_id == company_obj.id)).all()
        return {
            "company": {
                "id": company_obj.id,
                "name": company_obj.name,
                "industry": company_obj.industry,
                "employee_size": company_obj.employee_size,
                "domain": company_obj.domain,
            },
            "employees": [
                {
                    "id": e.id,
                    "full_name": e.full_name,
                    "title": e.title,
                    "department": e.department,
                    "seniority": e.seniority,
                    "profile_url": e.profile_url,
                }
                for e in stored_emps
            ],
        }

def get_company_cached_or_fetch(name: str) -> Dict[str, Any]:

    found = get_company_by_name_db(name)
    if found:
        with Session(engine) as session:
            stored_emps = session.exec(select(Employee).where(Employee.company_id == found.id)).all()
            return {
                "source": "db",
                "company": {
                    "id": found.id,
                    "name": found.name,
                    "industry": found.industry,
                    "employee_size": found.employee_size,
                    "domain": found.domain,
                },
                "employees": [
                    {
                        "id": e.id,
                        "full_name": e.full_name,
                        "title": e.title,
                        "department": e.department,
                        "seniority": e.seniority,
                        "profile_url": e.profile_url,
                    }
                    for e in stored_emps
                ],
            }

    # not found in DB
    if not _HAS_GEMINI or not client:
        return {"source": "none", "message": "Not found in DB and Gemini is not available."}
    try:
        raw = fetch_from_gemini(name)
        stored = upsert_company_and_employees(raw)
        return {"source": "gemini", **stored}
    except Exception as e:
        return {"source": "error", "message": f"Error fetching from Gemini: {str(e)}"}