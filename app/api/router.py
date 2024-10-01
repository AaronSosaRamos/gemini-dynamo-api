from app.api.features.dynamo_feature import (
    generate_concepts_from_img, 
    generate_flashcards, 
    get_summary, 
    summarize_transcript_youtube_url
)
from app.api.features.schemas.key_concept_retriever_structured_data_schema import StructuredDataStudyInputData
from app.api.features.schemas.schemas import VideoAnalysisRequestArgs
from fastapi import APIRouter, Depends
from app.api.features.schemas.semantic_analysis_schemas import SemanticAnalysisInputData
from app.api.features.schemas.topic_clustering_schemas import TopicClusteringInputData
from app.api.features.structured_data_study import run_chain
from app.api.features.semantic_analysis import run_chain as semantic_analysis_run_chain
from app.api.features.topic_clustering import run_chain as topic_clustering_run_chain
from app.api.logger import setup_logger
from app.api.auth.auth import key_check

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

logger = setup_logger(__name__)
router = APIRouter()

@router.get("/")
def read_root():
    return {"Hello": "World"}

@router.post("/retrieve-key-concepts")
async def submit_tool( data: VideoAnalysisRequestArgs, _ = Depends(key_check)):
    
    logger.info(f"File URL loaded: {data.file_url}")
    flashcards = []

    if data.file_type == "img":
        flashcards = generate_concepts_from_img(args=data)
    elif (data.file_type == 'youtube_url'):
        summary = summarize_transcript_youtube_url(youtube_url=data.file_url, verbose=True)
        flashcards = generate_flashcards(summary=summary, args=data, verbose=True)
    else:
        summary = get_summary(file_url=data.file_url, file_type=data.file_type, verbose=True)
        flashcards = generate_flashcards(summary=summary, args=data, verbose=True)

    return flashcards

@router.post("/structured-data-study")
async def submit_tool( data: StructuredDataStudyInputData, _ = Depends(key_check)):
    
    logger.info(f"File URL loaded: {data.file_url}")

    result = run_chain(data)

    return result

@router.post("/semantic-analysis")
async def submit_tool( data: SemanticAnalysisInputData, _ = Depends(key_check)):
    
    logger.info(f"File URL loaded: {data.file_url}")

    result = semantic_analysis_run_chain(data)

    return result

@router.post("/topic-clustering")
async def submit_tool( data: TopicClusteringInputData, _ = Depends(key_check)):
    
    logger.info(f"File URL loaded: {data.file_url}")

    result = topic_clustering_run_chain(data)

    return result