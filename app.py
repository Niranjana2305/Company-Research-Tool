# app.py
import reflex as rx
from typing import Dict, Any, List, Optional
from backend import get_company_cached_or_fetch, upsert_company_and_employees

class State(rx.State):
    company_name: str = ""
    industry_input: str = ""
    employee_size_input: str = ""
    domain_input: str = ""
    status: str = ""
    is_loading: bool = False

    company: Optional[Dict[str, Any]] = rx.field(default=None)
    employees: List[Dict[str, Any]] = rx.field(default_factory=list)

    def search(self):
        name = (self.company_name or "").strip()
        if not name:
            self.status = "Please enter a company name."
            self.company = None
            self.employees = []
            return
        self.is_loading = True
        self.status = "Searching..."
        yield

        result = get_company_cached_or_fetch(name)
        source = result.get("source")
        if source == "db":
            self.status = "Loaded from database (cached)."
        elif source == "gemini":
            self.status = "Fetched from Gemini AI & cached."
        elif source == "error":
            self.status = f"Error: {result.get('message')}"
        else:
            self.status = result.get("message", "No data found.")
        self.company = result.get("company")
        self.employees = result.get("employees", [])
        self.is_loading = False

    def add_manual_entry(self):
        name = (self.company_name or "").strip()
        if not name:
            self.status = "Please enter a company name to add manually."
            return
        manual_data = {
            "company": {
                "name": name,
                "industry": self.industry_input.strip() or None,
                "employee_size": int(self.employee_size_input) if self.employee_size_input.strip().isdigit() else None,
                "domain": self.domain_input.strip() or None,
            },
            "employees": []
        }
        try:
            stored = upsert_company_and_employees(manual_data)
            self.status = "Company added manually to database."
            self.company = stored.get("company")
            self.employees = stored.get("employees", [])
        except Exception as e:
            self.status = f"Error adding company: {str(e)}"

    def clear(self):
        self.company_name = ""
        self.industry_input = ""
        self.employee_size_input = ""
        self.domain_input = ""
        self.status = ""
        self.company = None
        self.employees = []
        self.is_loading = False

def company_card() -> rx.Component:
    return rx.vstack(
        rx.heading("Company Details", size="5", color="blue.600"),
        rx.box(
            rx.vstack(
                rx.hstack(rx.text("Name:", font_weight="600"), rx.text(State.company["name"])),
                rx.hstack(rx.text("Industry:", font_weight="600"),
                         rx.cond(State.company["industry"], rx.text(State.company["industry"]), rx.text("Not specified"))),
                rx.hstack(rx.text("Employee Size:", font_weight="600"),
                         rx.cond(State.company["employee_size"], rx.text(State.company["employee_size"]), rx.text("Not specified"))),
                rx.hstack(rx.text("Domain:", font_weight="600"),
                         rx.cond(State.company["domain"], rx.text(State.company["domain"]), rx.text("Not specified"))),
                spacing="2"
            ),
            padding="4", border="1px solid", border_color="gray.200", border_radius="lg", bg="gray.50"
        ),
        spacing="3"
    )

def employee_list() -> rx.Component:
    return rx.vstack(
        rx.heading("Employees", size="5", color="blue.600"),
        rx.cond(
            rx.len(State.employees) == 0,
            rx.text("No employees found.", color="gray.500", font_style="italic"),
            rx.vstack(
                rx.foreach(
                    State.employees,
                    lambda emp, i: rx.box(
                        rx.vstack(
                            rx.text(emp["full_name"], font_weight="600"),
                            rx.text(f"{emp.get('title', 'N/A')} — {emp.get('department', 'N/A')} — {emp.get('seniority', 'N/A')}"),
                            rx.cond(emp.get("profile_url"),
                                    rx.link("View Profile →", href=emp["profile_url"], is_external=True),
                                    rx.text("No profile URL", color="gray.400")),
                            spacing="2"
                        ),
                        padding="4", border="1px solid", border_color="gray.200",
                        border_radius="lg", bg="white", _hover={"bg": "gray.50"}
                    )
                ),
                spacing="3"
            )
        )
    )

@rx.page(route="/", title="Company Research Tool")
def index() -> rx.Component:
    return rx.center(
        rx.vstack(
            rx.heading("🏢 Company Research Tool", size="9", color="blue.600", text_align="center"),
            rx.input(value=State.company_name, placeholder="Enter company name", on_change=State.set_company_name),
            rx.hstack(
                rx.button(rx.cond(State.is_loading, rx.text("Searching..."), rx.text("🔍 Search")),
                          on_click=State.search, disabled=State.is_loading),
                rx.button("🗑️ Clear", on_click=State.clear),
                spacing="3"
            ),
            rx.details(
                rx.summary("➕ Add Manual Entry"),
                rx.vstack(
                    rx.input(value=State.industry_input, placeholder="Industry", on_change=State.set_industry_input),
                    rx.input(value=State.employee_size_input, placeholder="Employee count", on_change=State.set_employee_size_input),
                    rx.input(value=State.domain_input, placeholder="Website domain", on_change=State.set_domain_input),
                    rx.button("💾 Add to Database", on_click=State.add_manual_entry)
                )
            ),
            rx.cond(State.status, rx.text(State.status, color="blue.600")),
            rx.cond(State.company, rx.vstack(company_card(), employee_list())),
            spacing="4", width="100%", max_width="4xl", padding="6"
        )
    )

app = rx.App(State)
app.add_page(index)

if __name__ == "__main__":
    app.run()




