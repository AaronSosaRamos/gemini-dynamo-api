from app.utils.key_concept_retriever_structured_data.file_handler_sd import FileHandler
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders import UnstructuredExcelLoader, UnstructuredXMLLoader
from langchain_community.document_loaders.json_loader import JSONLoader
from typing import List
from langchain.schema import Document

def load_csv_documents(csv_url: str, verbose=False):
    csv_loader = FileHandler(CSVLoader, "csv")
    docs = csv_loader.load(csv_url)

    if docs:
        if verbose:
            print(f"Found CSV file")
            print(f"Loaded {len(docs)} documents")
        return docs

def load_xls_documents(xls_url: str, verbose=False):
    xls_handler = FileHandler(UnstructuredExcelLoader, 'xls')
    docs = xls_handler.load(xls_url)
    if docs:
        if verbose:
            print(f"Found XLS file")
            print(f"Loaded {len(docs)} documents")
        return docs

def load_xlsx_documents(xlsx_url: str, verbose=False):
    xlsx_handler = FileHandler(UnstructuredExcelLoader, 'xlsx')
    docs = xlsx_handler.load(xlsx_url)
    if docs:
        if verbose:
            print(f"Found XLSX file")
            print(f"Loaded {len(docs)} documents")
        return docs

def load_xml_documents(xml_url: str, verbose=False):
    xml_handler = FileHandler(UnstructuredXMLLoader, 'xml')
    docs = xml_handler.load(xml_url)
    if docs:
        if verbose:
            print(f"Found XML file")
            print(f"Loaded {len(docs)} documents")
        return docs
def load_json_documents(json_url: str, verbose=False):
    json_handler = FileHandler(JSONLoader, 'json')
    docs = json_handler.load(json_url)
    if docs:
        if verbose:
            print(f"Found JSON file")
            print(f"Loaded {len(docs)} documents")
        return docs

def load_documents(file_url: str, file_type: str, verbose=False) -> List[Document]:
    if file_type.lower() == "csv":
        return load_csv_documents(file_url, verbose)
    elif file_type.lower() == "xls":
        return load_xls_documents(file_url, verbose)
    elif file_type.lower() == "xlsx":
        return load_xlsx_documents(file_url, verbose)
    elif file_type.lower() == "xml":
        return load_xml_documents(file_url, verbose)
    elif file_type.lower() == "json":
        return load_json_documents(file_url, verbose)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")