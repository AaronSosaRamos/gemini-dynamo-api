from typing import List, Dict, Optional
from pydantic import BaseModel, Field, conlist

class TopicClusteringInputData(BaseModel):
    topic: str = Field(..., description="The main topic")
    file_url: str = Field(..., description="URL of the file to be processed")
    file_type: str = Field(..., description="Type of the file (e.g., csv, json, xls, xlsx, xml)")
    lang: str = Field(..., description="Language of the document")

class Topic(BaseModel):
    topic_id: str = Field(..., description="Unique identifier for the topic")
    topic_name: str = Field(..., description="Human-readable name of the topic")
    keywords: List[str] = Field(..., description="List of keywords that define the topic")
    importance_score: float = Field(..., description="Score representing the importance of this topic in the clustering")
    description: Optional[str] = Field(None, description="Optional description of the topic")


class ClusterMetadata(BaseModel):
    num_clusters: int = Field(..., description="Total number of clusters generated")
    method: str = Field(..., description="Method used for clustering (e.g., K-means, LDA, etc.)")
    timestamp: Optional[str] = Field(None, description="Timestamp when the clustering process was completed")


class TopicCluster(BaseModel):
    cluster_id: str = Field(..., description="Unique identifier for the topic cluster")
    topics: List[Topic] = Field(..., description="List of topics within this cluster")
    central_topic: Optional[str] = Field(None, description="The central topic or most important topic within the cluster, if applicable")


class TopicClusteringOutput(BaseModel):
    clustering_algorithm: str = Field(..., description="Algorithm used to generate the clusters")
    clusters: conlist(TopicCluster) = Field(..., description="List of all topic clusters")
    metadata: ClusterMetadata = Field(..., description="Metadata regarding the clustering process")
