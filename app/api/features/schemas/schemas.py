from pydantic import BaseModel, Field

class VideoAnalysisRequestArgs(BaseModel):
    file_url: str = Field(..., description="The file's URL for retrieving the context")
    file_type: str = Field(..., description="The file's type")
    language: str = Field("en", description="Language of the video content for analysis, default is English")
    analysis_depth: str = Field("standard", description="Depth of analysis, options include 'standard', 'deep', or 'comprehensive'")
    additional_comments: str = Field("", description="Additional comments for a personalized key concepts retrieving process")
