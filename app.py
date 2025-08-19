# app.py
import reflex as rx
import dataclasses
from typing import List, Optional, Dict, Any
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

        result = get_company_cached_or_fetch(name)

        if result.get("source") == "db":
            self.status = "Loaded from database (cached)."
            self.company = result.get("company")
            self.employees = result.get("employees", [])
        elif result.get("source") == "gemini":
            self.status = "Fetched from Gemini AI & cached to database."
            self.company = result.get("company")
            self.employees = result.get("employees", [])
        elif result.get("source") == "error":
            self.status = f"Error: {result.get('message', 'Unknown error occurred')}"
            self.company = None
            self.employees = []
        else:
            self.status = result.get("message", "Not found in database and Gemini AI not available.")
            self.company = None
            self.employees = []

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
            "employees": [],
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
                rx.hstack(
                    rx.text("Name:", font_weight="600", color="gray.700"),
                    rx.text(State.company["name"], color="gray.900"),
                    justify="space-between",
                    width="100%",
                ),
                rx.hstack(
                    rx.text("Industry:", font_weight="600", color="gray.700"),
                    rx.cond(
                        State.company["industry"],
                        rx.text(State.company["industry"], color="gray.900"),
                        rx.text("Not specified", color="gray.500", font_style="italic"),
                    ),
                    justify="space-between",
                    width="100%",
                ),
                rx.hstack(
                    rx.text("Employee Size:", font_weight="600", color="gray.700"),
                    rx.cond(
                        State.company["employee_size"],
                        rx.text(State.company["employee_size"], color="gray.900"),
                        rx.text("Not specified", color="gray.500", font_style="italic"),
                    ),
                    justify="space-between",
                    width="100%",
                ),
                rx.hstack(
                    rx.text("Domain:", font_weight="600", color="gray.700"),
                    rx.cond(
                        State.company["domain"],
                        rx.text(State.company["domain"], color="gray.900"),
                        rx.text("Not specified", color="gray.500", font_style="italic"),
                    ),
                    justify="space-between",
                    width="100%",
                ),
                spacing="3",
                align="start",
                width="100%",
            ),
            padding="4",
            border="1px solid",
            border_color="gray.200",
            border_radius="lg",
            bg="gray.50",
        ),
        spacing="3",
        align="start",
        width="100%",
    )


def employee_list() -> rx.Component:
    return rx.vstack(
        rx.heading("Employees", size="5", color="blue.600"),
        rx.cond(
            rx.len(State.employees) == 0,
            rx.box(
                rx.text("No employees found.", color="gray.500", font_style="italic"),
                padding="4",
                border="1px solid",
                border_color="gray.200",
                border_radius="lg",
                bg="gray.50",
                width="100%",
            ),
            rx.vstack(
                rx.foreach(
                    State.employees,
                    lambda emp, i: rx.box(
                        rx.vstack(
                            rx.text(emp["full_name"], font_weight="600", color="gray.900", size="4"),
                            rx.hstack(
                                rx.badge(emp.get("title", "Not found"), color_scheme="blue"),
                                rx.badge(emp.get("department", "Not found"), color_scheme="green"),
                                rx.badge(emp.get("seniority", "Not found"), color_scheme="purple"),
                                wrap="wrap",
                                spacing="2",
                            ),
                            rx.cond(
                                emp.get("profile_url") and emp.get("profile_url") != "Not found",
                                rx.link(
                                    "View Profile →",
                                    href=emp["profile_url"],
                                    is_external=True,
                                    color="blue.500",
                                    size="2",
                                ),
                                rx.text("No profile URL", color="gray.400", size="2"),
                            ),
                            align="start",
                            spacing="2",
                        ),
                        padding="4",
                        border="1px solid",
                        border_color="gray.200",
                        border_radius="lg",
                        bg="white",
                        width="100%",
                        _hover={"bg": "gray.50"},
                    ),
                ),
                spacing="3",
                width="100%",
            ),
        ),
        spacing="3",
        align="start",
        width="100%",
    )


#  Main Page 
@rx.page(route="/", title="Company Research Tool")
def index() -> rx.Component:
    return rx.center(
        rx.vstack(
            rx.heading("🏢 Company Research Tool", size="9", color="blue.600", text_align="center"),
            rx.text(
                "Search for companies and view their details and employees",
                color="gray.600",
                text_align="center",
                size="4",
            ),
            rx.divider(),
            # Search Section
            rx.vstack(
                rx.heading("Search Company", size="6", color="gray.800"),
                rx.input(
                    value=State.company_name,
                    placeholder="Enter company name (e.g., 'Apple', 'Google')",
                    size="3",
                    width="100%",
                    on_change=State.set_company_name,
                ),
                rx.hstack(
                    rx.button(
                        rx.cond(
                            State.is_loading,
                            rx.hstack(rx.spinner(size="2"), rx.text("Searching..."), spacing="2"),
                            rx.text("🔍 Search"),
                        ),
                        on_click=State.search,
                        color_scheme="blue",
                        size="3",
                        disabled=State.is_loading,
                        width="150px",
                    ),
                    rx.button(
                        "🗑️ Clear",
                        on_click=State.clear,
                        color_scheme="gray",
                        variant="outline",
                        size="3",
                    ),
                    spacing="3",
                    justify="center",
                    width="100%",
                ),
                spacing="4",
                align="center",
                width="100%",
            ),
            # Manual entry
            rx.details(
                rx.summary("➕ Add Manual Entry", cursor="pointer", color="blue.600"),
                rx.vstack(
                    rx.text("Add company information manually:", color="gray.600", size="2"),
                    rx.hstack(
                        rx.input(
                            value=State.industry_input,
                            placeholder="Industry (optional)",
                            on_change=State.set_industry_input,
                        ),
                        rx.input(
                            value=State.employee_size_input,
                            placeholder="Employee count (optional)",
                            on_change=State.set_employee_size_input,
                        ),
                        rx.input(
                            value=State.domain_input,
                            placeholder="Website domain (optional)",
                            on_change=State.set_domain_input,
                        ),
                        spacing="3",
                        width="100%",
                    ),
                    rx.button(
                        "💾 Add to Database",
                        on_click=State.add_manual_entry,
                        color_scheme="green",
                        size="2",
                    ),
                    spacing="3",
                    align="center",
                    width="100%",
                    padding="4",
                    border="1px solid",
                    border_color="gray.200",
                    border_radius="lg",
                    bg="gray.50",
                ),
                margin_top="4",
            ),
            # Status message
            rx.cond(
                State.status,
                rx.alert(
                    rx.alert_icon(),
                    rx.alert_title(State.status),
                    status=(
                        "error"
                        if "Error" in State.status
                        else "success"
                        if "Gemini" in State.status
                        else "info"
                    ),
                    width="100%",
                ),
            ),
            # Results section
            rx.cond(
                State.company,
                rx.vstack(rx.divider(), company_card(), employee_list(), spacing="6", width="100%"),
                rx.cond(
                    ~State.is_loading & (State.status == ""),
                    rx.box(
                        rx.text(
                            "👆 Enter a company name above to get started",
                            color="gray.500",
                            text_align="center",
                            size="4",
                        ),
                        padding="8",
                        width="100%",
                    ),
                ),
            ),
            spacing="6",
            align="center",
            width="100%",
            max_width="4xl",
            padding="6",
            margin="0 auto",
        ),
        width="100%",
        min_height="100vh",
        bg="gray.50",
        padding="4",
    )


app = rx.App(State)
app.lifespan_tasks = set() 

if __name__ == "__main__":
    app.run()
