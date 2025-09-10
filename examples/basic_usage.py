#!/usr/bin/env python3
"""
Example usage of AvsarKendra AI components.

This script demonstrates how to use the main components of the AvsarKendra AI project:
- Document processing
- RAG pipeline
- Vector store
- LLM interaction
"""

import os
import sys

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.rag.rag_pipeline import RAGPipeline
from src.utils.document_processor import DocumentProcessor
from src.utils.text_processing import TextPreprocessor
from src.utils.logger import setup_logger

def main():
    """Run example usage of AvsarKendra AI components."""
    
    # Setup logging
    logger = setup_logger()
    logger.info("Starting AvsarKendra AI example")
    
    try:
        # Example 1: Text Processing
        print("=" * 50)
        print("Example 1: Text Processing")
        print("=" * 50)
        
        processor = TextPreprocessor()
        
        text = "Hello world! This is an example of text processing with NLP."
        cleaned = processor.preprocess(text)
        stats = processor.get_text_stats(text)
        keywords = processor.extract_keywords(text)
        
        print(f"Original text: {text}")
        print(f"Processed text: {cleaned}")
        print(f"Text statistics: {stats}")
        print(f"Keywords: {keywords}")
        
        # Example 2: Document Processing
        print("\n" + "=" * 50)
        print("Example 2: Document Processing")
        print("=" * 50)
        
        doc_processor = DocumentProcessor()
        supported_formats = doc_processor.get_supported_formats()
        print(f"Supported formats: {supported_formats}")
        
        # Example with a simple text document
        sample_text = """
        This is a sample document for testing.
        It contains multiple sentences and paragraphs.
        
        This is the second paragraph with more content.
        We can use this to test our document processing pipeline.
        """
        
        # Save sample text to a file
        sample_file_path = "/tmp/sample_document.txt"
        with open(sample_file_path, 'w') as f:
            f.write(sample_text)
        
        # Process the document
        result = doc_processor.process_document(sample_file_path)
        print(f"Document content: {result['content'][:100]}...")
        print(f"Document metadata: {result['metadata']}")
        
        # Clean up
        os.remove(sample_file_path)
        
        # Example 3: RAG Pipeline
        print("\n" + "=" * 50)
        print("Example 3: RAG Pipeline")
        print("=" * 50)
        
        # Note: This might take some time as it downloads models
        print("Initializing RAG pipeline (this may take a while for first run)...")
        
        try:
            rag = RAGPipeline()
            
            # Add some sample documents
            documents = [
                {
                    "content": "Python is a high-level programming language known for its simplicity and readability.",
                    "metadata": {"source": "python_intro", "topic": "programming"}
                },
                {
                    "content": "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience.",
                    "metadata": {"source": "ml_basics", "topic": "ai"}
                },
                {
                    "content": "Natural language processing (NLP) is a field of AI that focuses on the interaction between computers and human language.",
                    "metadata": {"source": "nlp_guide", "topic": "ai"}
                }
            ]
            
            # Add documents to the knowledge base
            for doc in documents:
                doc_id = rag.add_document(doc["content"], doc["metadata"])
                print(f"Added document with ID: {doc_id}")
            
            # Query the knowledge base
            queries = [
                "What is Python?",
                "Tell me about machine learning",
                "What is NLP?"
            ]
            
            for query in queries:
                print(f"\nQuery: {query}")
                response = rag.query(query)
                print(f"Answer: {response['answer']}")
                print(f"Sources: {len(response['sources'])}")
                print(f"Confidence: {response['confidence']:.2f}")
                
            # Get pipeline info
            info = rag.get_pipeline_info()
            print(f"\nRAG Pipeline Info: {info}")
            
        except Exception as e:
            logger.warning(f"RAG pipeline example failed (this is normal if models are not available): {str(e)}")
            print("RAG pipeline example skipped (models may not be available)")
        
        print("\n" + "=" * 50)
        print("Example completed successfully!")
        print("=" * 50)
        
    except Exception as e:
        logger.error(f"Error in example: {str(e)}")
        print(f"Error: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
