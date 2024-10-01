from typing import List
from dotenv import find_dotenv, load_dotenv
from app.api.features.document_loaders import get_docs
from app.api.features.schemas.topic_clustering_schemas import TopicClusteringInputData, TopicClusteringOutput
from app.api.logger import setup_logger
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_google_genai import GoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.chains import  HypotheticalDocumentEmbedder
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import JsonOutputParser
from langchain.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = setup_logger(__name__)

load_dotenv(find_dotenv())

embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

llm_chat_openai = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

llm_google_genai = GoogleGenerativeAI(model="gemini-1.5-pro")

hyde_prompt_template = """
Generate a hypothetical answer to the user's query.
Query: {query}
Hypothetical Answer:
"""

hyde_prompt = PromptTemplate(input_variables=["query"], template=hyde_prompt_template)

def compile_hyde_chain():
    hyde_chain = hyde_prompt | llm_google_genai
    return hyde_chain

def build_docsearch(documents: List[Document], embedding_model):
  embeddings = HypotheticalDocumentEmbedder(
    llm_chain=compile_hyde_chain(),
    base_embeddings=embedding_model
  )

  docsearch = Chroma.from_documents(documents, embeddings)

  return docsearch

parser = JsonOutputParser(pydantic_object=TopicClusteringOutput)
format_instructions = parser.get_format_instructions()

prompt_template = """
Analyze the following document and perform topic clustering. Based on the document's content, select the most appropriate clustering algorithm, identify relevant topics, group them into clusters, and determine the importance of each topic within its cluster.

Main Topic:
-----------------------------
{topic}

Context:
-----------------------------
{context}

Formatting:
-----------------------------
{format_instructions}

Ensure the following:
- Automatically select the clustering algorithm based on the document.
- Provide metadata about the number of clusters and the method used.
- Identify and cluster topics, including their importance score and relevant keywords.
- Highlight any central topic within each cluster if applicable.

You must respond in this language: {lang}
"""

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["topic", "context", "lang"],
    partial_variables={"format_instructions": format_instructions}
)

def compile_main_chain():
   chain = prompt | llm_chat_openai | parser
   return chain

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)


def run_chain(input_data: TopicClusteringInputData):
    documents = get_docs(input_data.file_url, input_data.file_type, verbose=True)
    if not documents:
        logger.info("No documents loaded.")
        return

    split_docs = splitter.split_documents(documents)

    docstore = build_docsearch(split_docs, embedding_model)

    user_query = "Develop a topic clustering based in this main topic: "+input_data.topic

    relevant_docs = docstore.similarity_search(user_query)

    context = "\n".join([doc.page_content for doc in relevant_docs])

    chain = compile_main_chain()

    output = chain.invoke({'topic': input_data.topic, 'context': context, 'lang': input_data.lang})

    logger.info(output)

    del docstore

    return output