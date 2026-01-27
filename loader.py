import os
from langchain_community.document_loaders import UnstructuredFileLoader
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_document(file_path):
    """
    Loads and parses a document using Unstructured.io.
    Supports PDF, Image, DOCX, XLSX, PPTX.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    logger.info(f"Loading document: {file_path}")
    
    # UnstructuredFileLoader automatically detects file type
    # For OCR on images/PDFs, it uses 'ocr_languages' and 'strategy'
    loader = UnstructuredFileLoader(
        file_path,
        strategy="hi_res",  # 'hi_res' is better for OCR and layout preservation
        model_name="yolox", # Default model for layout analysis
    )
    
    docs = loader.load()
    
    # Combine content for simplicity or return list of docs (e.g. pages)
    full_text = "\n\n".join([doc.page_content for doc in docs])
    
    return full_text

if __name__ == "__main__":
    # Quick test
    import sys
    if len(sys.argv) > 1:
        text = load_document(sys.argv[1])
        print(f"--- Extracted Text (First 500 chars) ---")
        print(text[:500])
