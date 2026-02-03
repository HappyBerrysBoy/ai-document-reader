from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

class DocumentExtractor:
    def __init__(self, model_name="exaone3.5"):
        self.llm = OllamaLLM(model=model_name)
        
    def analyze_query(self, query, text_preview):
        """
        Optional: Uses LLM to decide how to process the information.
        Currently simple implementation.
        """
        router_prompt = PromptTemplate.from_template(
            """
            Analyze the following user query and a preview of the document content.
            Query: {query}
            Content Preview: {preview}
            
            Identify if the user wants:
            1. A summary
            2. Specific data extraction (e.g. dates, amounts, names)
            3. A question answered
            
            Respond only with the classification (SUMMARY, EXTRACTION, or Q&A).
            """
        )
        
        # We can implement more complex routing here if needed
        # For now, we'll just use this to refine the final prompt
        chain = router_prompt | self.llm | StrOutputParser()
        result = chain.invoke({"query": query, "preview": text_preview[:1000]})
        return result.strip()

    def extract_info(self, text, query):
        """
        Main extraction logic.
        """
        prompt = PromptTemplate.from_template(
            """
            You are a professional document assistant. 
            Based on the provided document content, please address the user's request.
            
            Document Content:
            {content}
            
            User Request: {query}
            
            Response:
            """
        )
        
        chain = prompt | self.llm | StrOutputParser()
        
        # If the text is extremely long, we might need a more sophisticated approach 
        # (like Map-Reduce or Refine), but for on-premise simple CLI, 
        # we'll start with a straightforward pass if it fits in context.
        return chain.invoke({"content": text, "query": query})

if __name__ == "__main__":
    # Test (requires Ollama running)
    extractor = DocumentExtractor()
    # text = "Sample text..."
    # print(extractor.extract_info(text, "Summarize this"))
