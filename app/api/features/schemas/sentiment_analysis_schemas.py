from typing import List, Optional
from pydantic import BaseModel, Field

class SentimentAnalysisInputData(BaseModel):
    topic: str = Field(..., description="The main topic")
    file_url: str = Field(..., description="URL of the file to be processed")
    file_type: str = Field(..., description="Type of the file (e.g., csv, json, xls, xlsx, xml)")
    lang: str = Field(..., description="Language of the document")

class SentimentScore(BaseModel):
    sentiment: str = Field(..., description="The detected sentiment (e.g., 'positive', 'negative', 'neutral')")
    score: float = Field(..., description="Confidence score for the detected sentiment")

class AspectSentiment(BaseModel):
    aspect: str = Field(..., description="The aspect or feature of the text (e.g., 'service', 'product quality')")
    sentiment: str = Field(..., description="Sentiment associated with the specific aspect (e.g., 'positive', 'negative', 'neutral')")
    score: float = Field(..., description="Confidence score for the sentiment related to the aspect")

class SentimentAnalysisInput(BaseModel):
    text_input: str = Field(..., description="The input text to analyze for sentiment")
    language: Optional[str] = Field(None, description="Language of the input text (default is 'en' for English)")

class SentimentAnalysisOutput(BaseModel):
    overall_sentiment: SentimentScore = Field(..., description="Overall sentiment and associated score for the entire input text")
    aspect_sentiments: List[AspectSentiment] = Field(None, description="List of sentiments associated with different aspects or features in the input text, if applicable")
    language: str = Field(..., description="Language of the analyzed text")