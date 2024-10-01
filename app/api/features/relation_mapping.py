from typing import List
from dotenv import find_dotenv, load_dotenv
from app.api.features.document_loaders import get_docs
from app.api.features.schemas.relation_mapping_schemas import RelationMappingInputData, RelationMappingOutput
from app.api.logger import setup_logger
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import JsonOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from operator import itemgetter
from langchain.prompts import (
  SystemMessagePromptTemplate, 
  HumanMessagePromptTemplate,
  ChatMessagePromptTemplate, 
  ChatPromptTemplate, 
  PromptTemplate
)
from langchain.load import dumps, loads
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = setup_logger(__name__)

load_dotenv(find_dotenv())

embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

llm_chat_openai = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

def build_retriever(documents: List[Document], embedding_model):
  vectorstore = Chroma.from_documents(documents, embedding_model)
  retriever = vectorstore.as_retriever(k=5)
  return retriever

parser = JsonOutputParser(pydantic_object=RelationMappingOutput)
format_instructions = parser.get_format_instructions()

prompt = ChatPromptTemplate(
    input_variables=['topic'],
    messages=[
        SystemMessagePromptTemplate(
            prompt=PromptTemplate(
                input_variables=[],
                template='You are an assistant specialized in performing multi-query operations based on a single main topic.'
            )
        ),
        HumanMessagePromptTemplate(
            prompt=PromptTemplate(
                input_variables=['topic'],
                template='Perform multi-query by generating multiple related queries for the following main topic: {topic} \n OUTPUT (4 queries):'
            )
        )
    ]
)

generate_queries = (
    prompt | llm_chat_openai | StrOutputParser() | (lambda x: x.split("\n"))
)

def reciprocal_rank_fusion(results: list[list], k=60):
    fused_scores = {}
    for docs in results:
        # Assumes the docs are returned in sorted order of relevance
        for rank, doc in enumerate(docs):
            doc_str = dumps(doc)
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            previous_score = fused_scores[doc_str]
            fused_scores[doc_str] += 1 / (rank + k)

    reranked_results = [
        (loads(doc), score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]
    return reranked_results

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

template = """
Based on the following relation mapping context, perform the necessary relation mapping operations related to the following topic.

Think step by step to ensure accuracy and completeness in the relation mapping process.

Main Topic: {topic}

Context:
{context}

Steps to follow:
1. Identify the entities involved in the relation mapping.
2. Analyze the relationships between entities, including cardinality and constraints.
3. Ensure the relation mappings align with the given context.
4. Verify the consistency and correctness of each mapping.
5. Finalize the relation mapping, considering any specific instructions.

Formatting:
-----------------------------
{format_instructions}

You must respond in this language: {lang}
"""

prompt_for_main_chain = PromptTemplate(
    template=template,
    input_variables=["topic", "context", "lang"],
    partial_variables={"format_instructions": format_instructions}
)

def run_chain(input_data: RelationMappingInputData):
    documents = get_docs(input_data.file_url, input_data.file_type, verbose=True)
    if not documents:
        logger.info("No documents loaded.")
        return

    split_docs = splitter.split_documents(documents)

    retriever = build_retriever(split_docs, embedding_model)

    ragfusion_chain = generate_queries | retriever.map() | reciprocal_rank_fusion

    full_rag_fusion_chain = (
      {
          "context": ragfusion_chain,
          "topic": RunnablePassthrough(),
          "lang": RunnablePassthrough()
      }
      | prompt_for_main_chain
      | llm_chat_openai
      | parser
    )

    output = full_rag_fusion_chain.invoke({'topic': input_data.topic, 'lang': input_data.lang})

    logger.info(output)

    del retriever

    return output