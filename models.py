# models.py
from typing import Optional, List
from sqlmodel import SQLModel, Field, Relationship

class Company(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(index=True)
    name_normalized: str = Field(index=True)  # normalized for fast DB lookup
    industry: Optional[str] = Field(default=None, nullable=True)
    employee_size: Optional[int] = Field(default=None, nullable=True)
    domain: Optional[str] = Field(default=None, nullable=True)

    employees: List["Employee"] = Relationship(back_populates="company")

class Employee(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    full_name: str = Field(index=True)
    title: Optional[str] = Field(default=None, nullable=True)
    department: Optional[str] = Field(default=None, nullable=True)
    seniority: Optional[str] = Field(default=None, nullable=True)
    profile_url: Optional[str] = Field(default=None, nullable=True)
    company_id: Optional[int] = Field(default=None, foreign_key="company.id")
    company: Optional[Company] = Relationship(back_populates="employees")





