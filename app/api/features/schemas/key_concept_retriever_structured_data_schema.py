from typing import List
from pydantic import BaseModel, Field

class StructuredDataStudyInputData(BaseModel):
    file_url: str = Field(..., description="URL of the file to be processed")
    file_type: str = Field(..., description="Type of the file (e.g., csv, json, xls, xlsx, xml)")
    lang: str = Field(..., description="Language of the document")

class ConceptDefinitionRelation(BaseModel):
    concept: str = Field(..., description="The main concept of the flashcard")
    definition: str = Field(..., description="The detailed definition of the concept")

class StructuredDataStudyOutputData(BaseModel):
    concepts: List[ConceptDefinitionRelation] = Field(..., description="List of concepts and their definitions")