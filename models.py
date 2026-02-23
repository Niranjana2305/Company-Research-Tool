from typing import Optional, List
from sqlmodel import SQLModel, Field, Relationship, UniqueConstraint

class Company(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    # index=True is great for the search queries you're running in app.py
    name: str = Field(index=True)
    name_normalized: str = Field(index=True, unique=True) 
    
    industry: Optional[str] = Field(default=None, nullable=True)
    employee_size: Optional[int] = Field(default=None, nullable=True)
    domain: Optional[str] = Field(default=None, nullable=True)
    email: Optional[str] = Field(default=None, nullable=True)
    
    # Relationship with cascade delete: if a company is deleted, so are its employees
    employees: List["Employee"] = Relationship(
        back_populates="company", 
        sa_relationship_kwargs={"cascade": "all, delete-orphan"}
    )

class Employee(SQLModel, table=True):
    # Adding a UniqueConstraint prevents duplicate entries for the same person at the same company
    __table_args__ = (
        UniqueConstraint("full_name", "company_id", name="unique_employee_per_company"),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    full_name: str = Field(index=True)
    title: Optional[str] = Field(default="Not found", nullable=True)
    department: Optional[str] = Field(default="Not found", nullable=True)
    seniority: Optional[str] = Field(default="Not found", nullable=True)
    profile_url: Optional[str] = Field(default=None, nullable=True)
    email: Optional[str] = Field(default=None, nullable=True)

    company_id: Optional[int] = Field(default=None, foreign_key="company.id", index=True)
    company: Optional[Company] = Relationship(back_populates="employees")
