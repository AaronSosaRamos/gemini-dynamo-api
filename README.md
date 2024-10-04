# Gemini Dynamo API

**Gemini Dynamo API** is an advanced AI-powered backend service designed to retrieve, analyze, and process various types of content, delivering structured and insightful responses for concept retrieval, semantic analysis, topic clustering, and more. Built using state-of-the-art **Retrieval-Augmented Generation (RAG)** techniques, the API excels at retrieving and generating highly relevant and context-aware information.

Developed by **Wilfredo Aaron Sosa Ramos**, the API uses a wide range of **RAG techniques**, integrates multimodal capabilities, and provides detailed JSON-formatted responses based on few-shot learning approaches.

## Table of Contents

- [1. Features](#1-features)
- [2. Services Provided](#2-services-provided)
  - [2.1 Key Concepts Retriever](#21-key-concepts-retriever)
  - [2.2 Structured Data Study](#22-structured-data-study)
  - [2.3 Semantic Analysis](#23-semantic-analysis)
  - [2.4 Topic Clustering](#24-topic-clustering)
  - [2.5 Relation Mapping](#25-relation-mapping)
  - [2.6 Sentiment Analysis](#26-sentiment-analysis)
- [3. Advanced RAG Techniques](#3-advanced-rag-techniques)
- [4. Multimodal Capabilities](#4-multimodal-capabilities)
- [5. Technologies Used](#5-technologies-used)
- [6. Installation Guide](#6-installation-guide)
- [7. How to Use](#7-how-to-use)

---

## 1. Features

**Gemini Dynamo API** offers a wide range of advanced features, including:

- **Concept Retrieval Services**: Provides multiple services such as key concept retrieval, semantic analysis, structured data studies, and more.
- **Advanced RAG Pipelines**: Utilizes state-of-the-art RAG techniques, including **Self-Query Retrieval**, **Hybrid Search BM25**, **EnsembleRetriever**, **FAISS**, and more.
- **Few-Shot Learning**: Generates contextually accurate and relevant responses using a few-shot approach, tailored to the specific user query.
- **Multimodal Support**: Handles over 16 different file types (PDF, TXT, WORD, EXCEL, images, YouTube videos, etc.) to offer robust support for various data formats.
- **Structured JSON Output**: Delivers all responses in structured JSON format for seamless integration with other services and systems.

---

## 2. Services Provided

The **Gemini Dynamo API** offers six core services aimed at concept retrieval and analysis. These services ensure deep insights and accurate data representation in response to user queries.

### 2.1 Key Concepts Retriever

The **Key Concepts Retriever** service extracts the most important concepts from a given document or dataset. Features include:

- **Context-Aware Retrieval**: Uses RAG techniques to identify and retrieve key concepts related to the input.
- **Multimodal Input**: Accepts various formats such as PDFs, Word documents, and videos for extracting key concepts.
- **Few-Shot Accuracy**: Provides concept retrieval based on limited examples, ensuring relevant and concise results.

### 2.2 Structured Data Study

The **Structured Data Study** service processes structured data, enabling deeper analysis of relationships between data points. Features include:

- **Data Pattern Recognition**: Identifies trends and patterns in structured data such as CSV or Excel files.
- **Semantic Linking**: Establishes connections between disparate data fields, providing insightful analyses.

### 2.3 Semantic Analysis

The **Semantic Analysis** service focuses on understanding the meaning behind words and phrases within documents. Features include:

- **Meaning Extraction**: Uses advanced semantic analysis techniques to extract the true meaning behind content.
- **Contextual Understanding**: Ensures that the semantic relations are accurate within the context of the document.

### 2.4 Topic Clustering

The **Topic Clustering** service groups related topics together based on the content of the input data. Features include:

- **Automatic Topic Detection**: Clusters content into relevant topics using advanced clustering techniques.
- **Hierarchical Clustering**: Provides structured, hierarchical topic groupings for deeper analysis.

### 2.5 Relation Mapping

The **Relation Mapping** service identifies relationships between different data points, entities, or concepts. Features include:

- **Entity Recognition**: Recognizes key entities and maps out their relationships within the data.
- **Complex Relation Extraction**: Uses the RAG pipeline to establish multi-layered relationships across complex datasets.

### 2.6 Sentiment Analysis

The **Sentiment Analysis** service evaluates the sentiment of text, determining whether the overall tone is positive, negative, or neutral. Features include:

- **Tone Detection**: Analyzes the overall tone of the document or dataset.
- **Multimodal Sentiment**: Processes text, audio, and video to extract sentiment insights from multiple modalities.

---

## 3. Advanced RAG Techniques

**Gemini Dynamo API** integrates several cutting-edge RAG techniques to ensure accurate and contextually aware responses. These include:

- **Self-Query Retrieval**: Dynamically generates queries based on the user's input to refine the retrieval process.
- **Top-K Retrieval**: Retrieves the top K most relevant documents or concepts for further processing.
- **ChromaDB Integration**: Uses **ChromaDB** as the vector database to retrieve information efficiently.
- **Hybrid Search (BM25 + FAISS)**: Combines BM25 and FAISS for hybrid search capabilities, ensuring accurate retrieval of dense and sparse vectors.
- **EnsembleRetriever**: Merges the results of multiple retrievers for higher accuracy.
- **HyDE (Hypothetical Document Embeddings)**: Uses hypothetical document embeddings for better retrieval performance.
- **Multi-Query Retrieval**: Supports multi-query processing for complex retrieval tasks.
- **RAG-Fusion**: Combines retrieval-augmented generation with fusion techniques for improved answer generation.
- **CoT Prompting (Chain-of-Thought)**: Enhances LLM responses by using chain-of-thought prompting to improve reasoning tasks.
- **Parent-Document Retriever**: Retrieves the parent document for specific passages or pieces of information, ensuring context is maintained throughout the response.

These techniques ensure the API can handle a wide variety of retrieval and generation tasks with high precision.

---

## 4. Multimodal Capabilities

The **Gemini Dynamo API** supports a wide range of file types, making it a multimodal solution for concept retrieval and analysis. Supported file types include:

- **Documents**: PDF, TXT, WORD, EXCEL
- **Images**: PNG, JPG, GIF
- **Videos**: YouTube links, MP4
- **Data Files**: CSV, JSON
- **Other**: Markdown, LaTeX

This multimodal support allows users to integrate various content types into their retrieval processes, ensuring that the API can handle diverse data sources.

---

## 5. Technologies Used

The **Gemini Dynamo API** is built using a modern technology stack that ensures fast, accurate, and reliable performance:

- **Python**: The main programming language powering the backend.
- **FastAPI**: A modern, high-performance web framework for building APIs with Python.
- **LangChain**: Provides support for complex AI workflows and LLM-based applications.
- **Google Generative AI**: Powers the large language model responses, ensuring accurate and contextually relevant results.
- **ChromaDB**: A vector database that supports efficient retrieval in RAG pipelines.
- **RAG Techniques**: Advanced retrieval-augmented generation techniques like **Self-Query Retrieval**, **Top-K**, **FAISS**, and **RAG-Fusion**.

---

## 6. Installation Guide

Follow these steps to set up and run the **Gemini Dynamo API** locally:

1. **Clone the repository**:
   - Use the following command to clone the repository to your local machine:
     ```
     git clone https://github.com/yourusername/GeminiDynamoAPI.git
     ```

2. **Navigate to the project directory**:
   - Move into the project folder:
     ```
     cd GeminiDynamoAPI
     ```

3. **Set up a virtual environment** (optional but recommended):
   - Create and activate a virtual environment:
     ```
     python -m venv venv
     source venv/bin/activate
     ```

4. **Install dependencies**:
   - Install the required Python packages using pip:
     ```
     pip install -r requirements.txt
     ```

5. **Run the development server**:
   - Start the FastAPI server locally:
     ```
     uvicorn app.main:app --host 0.0.0.0 --port 8000
     ```

6. **Test the API**:
   - Visit `http://localhost:8000/docs` to view the interactive API documentation powered by Swagger UI.

---

## 7. How to Use

Once the **Gemini Dynamo API** is running, you can use the following services by sending requests to the available endpoints:

1. **Key Concepts Retriever**:
   - Input a document or dataset, and the API will extract the most important concepts, delivering the results in JSON format.

2. **Structured Data Study**:
   - Provide structured data, and the API will analyze it to reveal patterns, trends, and relationships.

3. **Semantic Analysis**:
   - Submit textual data, and the API will return a detailed analysis of the meaning behind the words and phrases.

4. **Topic Clustering**:
   - Input large datasets or documents, and the API will cluster topics into structured groups for easier analysis.

5. **Relation Mapping**:
   - Provide data or text to identify key relationships between entities, and the API will return a mapped representation in JSON.

6. **Sentiment Analysis**:
   - Submit textual, audio, or video data, and the API will evaluate the sentiment of the content, returning positive, negative, or neutral sentiment values.

All responses are formatted in JSON for easy integration into your applications.

---

With **Gemini Dynamo API**, users can leverage advanced RAG techniques to extract, analyze, and structure complex data with precision and efficiency. Whether you need concept retrieval, semantic analysis, or relation mapping, this API provides the tools to streamline your data workflows.
