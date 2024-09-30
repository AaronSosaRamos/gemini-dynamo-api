from dotenv import load_dotenv, find_dotenv
from app.api.features.schemas.key_concept_retriever_structured_data_schema import StructuredDataStudyInputData, StructuredDataStudyOutputData
from app.api.logger import setup_logger
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains.query_constructor.base import AttributeInfo
from typing import List
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import JsonOutputParser
from langchain.prompts import PromptTemplate
from langchain.retrievers.self_query.base import SelfQueryRetriever

from app.utils.key_concept_retriever_structured_data.document_loaders_sd import load_documents

logger = setup_logger(__name__)

load_dotenv(find_dotenv())

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

document_content_description = "Documents loaded from the file, containing data in the specified language for key concept retrieving."

metadata_field_info = [
    AttributeInfo(
        name="file_type",
        description="The type of the file (e.g., csv, json, xml, xls, xlsx).",
        type="string"
    ),
    AttributeInfo(
        name="processed_at",
        description="The timestamp when the document was processed, in ISO 8601 format.",
        type="string"
    ),
]

def build_vectorstore(documents: List[Document], embedding_model):
    vectorstore = Chroma.from_documents(documents, embedding_model)
    return vectorstore

parser = JsonOutputParser(pydantic_object=StructuredDataStudyOutputData)
format_instructions = parser.get_format_instructions()

prompt_template = """
Extract the key concepts from the following documents:

{context}

Formatting:
-----------------------------
{format_instructions}

Respond only according to the format instructions. You must respond in this language: {lang}
"""

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "lang"],
    partial_variables={"format_instructions": format_instructions}
)

def return_chain():
    return prompt | llm | parser

def run_chain(input_data: StructuredDataStudyInputData):
    documents = load_documents(input_data.file_url, input_data.file_type, verbose=True)
    if not documents:
        logger.info("No documents loaded.")
        return

    split_docs = splitter.split_documents(documents)

    vectorstore = build_vectorstore(split_docs, embedding_model)

    retriever = SelfQueryRetriever.from_llm(
        llm=llm,
        vectorstore=vectorstore,
        document_contents=document_content_description,
        metadata_field_info=metadata_field_info,
        verbose=True,
        enable_limit=True
    )

    user_query = "Extract key concepts from all the documents"
    relevant_docs = retriever.invoke(user_query)

    context = "\n".join([doc.page_content for doc in relevant_docs])

    chain = return_chain()

    output = chain.invoke({'context': context, "lang": input_data.lang})

    logger.info(output)

    del vectorstore
    del retriever

    return output