from typing import Optional
from sqlmodel import SQLModel, Field

class Company(SQLModel, table=True):
    __tablename__ = "company"
    __table_args__ = {"extend_existing": True}

    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(index=True)
    name_normalized: str = Field(index=True)
    industry: Optional[str] = Field(default=None, nullable=True)
    employee_size: Optional[int] = Field(default=None, nullable=True)
    domain: Optional[str] = Field(default=None, nullable=True)

class Employee(SQLModel, table=True):
    __tablename__ = "employee"
    __table_args__ = {"extend_existing": True}

    id: Optional[int] = Field(default=None, primary_key=True)
    full_name: str = Field(index=True)
    title: Optional[str] = Field(default=None, nullable=True)
    department: Optional[str] = Field(default=None, nullable=True)
    seniority: Optional[str] = Field(default=None, nullable=True)
    profile_url: Optional[str] = Field(default=None, nullable=True)
    company_id: Optional[int] = Field(default=None, foreign_key="company.id")










