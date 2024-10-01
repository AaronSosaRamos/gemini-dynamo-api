from typing import List
from dotenv import find_dotenv, load_dotenv
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from app.api.features.document_loaders import get_docs
from app.api.features.schemas.semantic_analysis_schemas import SemanticAnalysisInputData, SemanticAnalysisOutput
from app.api.logger import setup_logger
from langchain_core.output_parsers import JsonOutputParser
from langchain.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = setup_logger(__name__)

load_dotenv(find_dotenv())

embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

def build_vectorstore(documents: List[Document], embedding_model):
  bm25_retriever = BM25Retriever.from_documents(documents)
  bm25_retriever.k = 2

  faiss_vectorstore = FAISS.from_documents(documents, embedding_model)
  faiss_retriever = faiss_vectorstore.as_retriever(search_kwargs={"k": 2})

  ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, faiss_retriever],
                                       weights=[0.5, 0.5])

  return ensemble_retriever


parser = JsonOutputParser(pydantic_object=SemanticAnalysisOutput)
format_instructions = parser.get_format_instructions()

prompt_template = """
Perform a detailed semantic analysis of the following document. Extract and analyze key concepts, such as named entities, keywords, sentiment, and language. Ensure accuracy in entity recognition, keyword importance, and sentiment evaluation:

Topic: {topic}

Context:
-----------------------------
{context}

Formatting:
-----------------------------
{format_instructions}

Respond strictly according to the format instructions. You must respond in this language: {lang}
"""

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["topic", "context", "lang"],
    partial_variables={"format_instructions": format_instructions}
)

def return_chain():
  return prompt | llm | parser

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

def run_chain(input_data: SemanticAnalysisInputData):
    documents = get_docs(input_data.file_url, input_data.file_type, verbose=True)
    if not documents:
        logger.info("No documents loaded.")
        return

    split_docs = splitter.split_documents(documents)

    ensemble_retriever = build_vectorstore(split_docs, embedding_model)

    user_query = ""

    relevant_docs = ensemble_retriever.invoke(user_query)

    context = "\n".join([doc.page_content for doc in relevant_docs])

    chain = return_chain()

    output = chain.invoke({'topic': input_data.topic, 'context': context, 'lang': input_data.lang})

    logger.info(output)

    del ensemble_retriever

    return output