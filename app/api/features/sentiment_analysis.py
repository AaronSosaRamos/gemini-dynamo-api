from typing import List
from dotenv import find_dotenv, load_dotenv
from app.api.features.document_loaders import get_docs
from app.api.features.schemas.sentiment_analysis_schemas import (
   SentimentAnalysisInputData, 
   SentimentAnalysisOutput
)
from app.api.logger import setup_logger
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.storage import InMemoryStore
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain.retrievers import ParentDocumentRetriever
from langchain_core.output_parsers import JsonOutputParser
from langchain.prompts import PromptTemplate

logger = setup_logger(__name__)

load_dotenv(find_dotenv())

embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

llm_chat_openai = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

# This text splitter is used to create the parent documents - The big chunks
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)

# This text splitter is used to create the child documents - The small chunks
# It should create documents smaller than the parent
child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)

def build_big_chunks_retriever(documents: List[Document], embedding_model, store):
  vectorstore = Chroma.from_documents(documents, embedding_model)

  big_chunks_retriever = ParentDocumentRetriever(
      vectorstore=vectorstore,
      docstore=store,
      child_splitter=child_splitter,
      parent_splitter=parent_splitter,
  )

  return big_chunks_retriever

parser = JsonOutputParser(pydantic_object=SentimentAnalysisOutput)
format_instructions = parser.get_format_instructions()

prompt_template = """
Analyze the following document and perform sentiment analysis. Based on the document's content, select the most appropriate sentiment analysis method, identify the overall sentiment, and detect sentiment related to specific aspects or features for the topic.

Topic:
-----------------------------
{topic}

Context:
-----------------------------
{context}

Formatting:
-----------------------------
{format_instructions}

Ensure the following:
- Automatically select the sentiment analysis method based on the document.
- Provide metadata about the overall sentiment and confidence score.
- Identify and analyze sentiment for specific aspects or features, including their respective confidence scores.
- Highlight the overall sentiment for the entire text, as well as any significant sentiments tied to specific aspects.

You must respond in this language: {lang}
"""

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["topic", "context", "lang"],
    partial_variables={"format_instructions": format_instructions}
)

def compile_chain():
  chain = prompt | llm_chat_openai | parser
  return chain

def run_chain(input_data: SentimentAnalysisInputData):
    documents = get_docs(input_data.file_url, input_data.file_type, verbose=True)
    if not documents:
        logger.info("No documents loaded.")
        return

    store = InMemoryStore()

    big_chunks_retriever = build_big_chunks_retriever(documents, embedding_model, store)

    relevant_docs = big_chunks_retriever.invoke("Develop a sentiment analysis for: "+input_data.topic)

    context = "\n".join([doc.page_content for doc in relevant_docs])

    chain = compile_chain()

    output = chain.invoke({'topic': input_data.topic, 'context': context, 'lang': input_data.lang})

    logger.info(output)

    del big_chunks_retriever
    del store

    return output