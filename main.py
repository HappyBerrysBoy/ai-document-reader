import argparse
import sys
import os
from loader import load_document
from extractor import DocumentExtractor
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="On-premise Document Information Extractor")
    parser.add_argument("file", help="Path to the document (PDF, Image, DOCX, XLSX, PPTX)")
    parser.add_argument("query", help="What information do you want to extract or summarize?")
    parser.add_argument("--model", default="exaone3.5", help="Ollama model name (default: exaone3.5, 한글 추천)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.file):
        logger.error(f"Error: File '{args.file}' not found.")
        sys.exit(1)
        
    try:
        # 1. Load and parse document
        logger.info(f"Parsing {args.file}...")
        text = load_document(args.file)
        
        if not text.strip():
            logger.warning("No text extracted from the document.")
            # We could try a fallback or just inform the user
            
        # 2. Extract information using LLM
        logger.info(f"Processing query with model '{args.model}'...")
        extractor = DocumentExtractor(model_name=args.model)
        
        # Optional: Analyze query first (as suggested in architecture)
        # analysis = extractor.analyze_query(args.query, text)
        # logger.info(f"Query type analysis: {analysis}")
        
        result = extractor.extract_info(text, args.query)
        
        # 3. Output results
        print("\n" + "="*50)
        print("EXTRACTION RESULT")
        print("="*50)
        print(result)
        print("="*50 + "\n")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        if " Ollama " in str(e) or "connection" in str(e).lower():
            print("\nTIP: Make sure Ollama is running locally and the model is downloaded.")
            print(f"Run: ollama serve  (in a separate terminal)")
            print(f"Run: ollama pull {args.model}")
        sys.exit(1)

if __name__ == "__main__":
    main()
