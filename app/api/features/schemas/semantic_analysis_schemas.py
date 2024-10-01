from typing import List, Optional
from pydantic import BaseModel, Field

class SemanticAnalysisInputData(BaseModel):
    topic: str = Field(..., description="Topic for the Semantic Analysis")
    file_url: str = Field(..., description="URL of the file to be processed")
    file_type: str = Field(..., description="Type of the file (e.g., csv, json, xls, xlsx, xml)")
    lang: str = Field(..., description="Language of the document")

class Entity(BaseModel):
    text: str = Field(..., description="The entity found in the text.")
    type: str = Field(..., description="The type of entity (e.g., person, organization, location).")
    start_char: int = Field(..., description="Start character index of the entity in the input text.")
    end_char: int = Field(..., description="End character index of the entity in the input text.")
    confidence: Optional[float] = Field(None, description="Confidence score for the entity recognition.")

class Keyword(BaseModel):
    word: str = Field(..., description="A keyword extracted from the text.")
    importance: float = Field(..., description="Importance or relevance of the keyword in the text.")

class Sentiment(BaseModel):
    polarity: float = Field(..., description="Sentiment polarity, where -1 is very negative, 1 is very positive, and 0 is neutral.")
    subjectivity: float = Field(..., description="Subjectivity score, where 0 is very objective and 1 is very subjective.")

class SemanticAnalysisOutput(BaseModel):
    input_text: str = Field(..., description="The original text input for analysis.")
    entities: List[Entity] = Field(..., description="A list of named entities found in the text.")
    keywords: List[Keyword] = Field(..., description="A list of important keywords from the text.")
    sentiment: Sentiment = Field(..., description="Sentiment analysis result for the input text.")
    language: str = Field(..., description="Detected language of the input text.")
    confidence: float = Field(..., description="Confidence score of the overall semantic analysis.")
    summary: Optional[str] = Field(None, description="Optional summary of the input text.")