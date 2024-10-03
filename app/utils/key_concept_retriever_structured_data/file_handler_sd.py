from datetime import datetime
import os
import uuid
import requests
import tempfile
from langchain_community.document_loaders.json_loader import JSONLoader

from app.api.logger import setup_logger

logger = setup_logger(__name__)

class FileHandler:
    def __init__(self, file_loader, file_extension):
        self.file_loader = file_loader
        self.file_extension = file_extension

    def load(self, url):
        # Generate a unique filename with a UUID prefix
        unique_filename = f"{uuid.uuid4()}.{self.file_extension}"

        # Download the file from the URL and save it to a temporary file
        response = requests.get(url)
        response.raise_for_status()  # Ensure the request was successful

        with tempfile.NamedTemporaryFile(delete=False, suffix=unique_filename) as temp_file:
            temp_file.write(response.content)
            temp_file_path = temp_file.name

        # Use the file_loader to load the documents
        try:

            if(self.file_loader == JSONLoader):
              loader = self.file_loader(file_path=temp_file_path, jq_schema=".", text_content=False)
            else:
              loader = self.file_loader(file_path=temp_file_path)
        except Exception as e:
            logger.info(f"No such file found at {temp_file_path}")
            raise FileNotFoundError(f"No file found at {temp_file_path}") from e

        try:
            documents = loader.load()
            if documents:
              for doc in documents:
                  doc.metadata['file_type'] = self.file_extension
                  doc.metadata['processed_at'] = datetime.now().isoformat()
        except Exception as e:
            logger.info(f"File content might be private or unavailable or the URL is incorrect.")
            raise ValueError(f"No file content available at {temp_file_path}") from e

        # Remove the temporary file
        os.remove(temp_file_path)

        return documents