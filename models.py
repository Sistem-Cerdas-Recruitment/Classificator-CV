from pydantic import BaseModel, Field
from typing import List, Optional, Any

class CVExtracted(BaseModel):
    name: str = Field(...)
    skills: List[str] = Field(...)
    links: List[str] = Field(...)
    experiences: List[dict[str, Any]] = Field(...)
    educations: List[dict[str, Any]] = Field(...)

class InsertedText(BaseModel):
    text: str

class CVToClassify(BaseModel):
    educations: List[dict[str, Any]]
    skills: List[str]
    experiences: List[dict[str, Any]]

class JobToClassify(BaseModel):
    minYoE: int
    jobDesc: str
    skills: List[str]
    role: str
    majors: List[str]


class JobAndCV(BaseModel):
    cv: CVToClassify
    job: JobToClassify

class ClassificationResult(BaseModel):
    score: float
    is_accepted: bool
class InsertedLink(BaseModel):
    link: str