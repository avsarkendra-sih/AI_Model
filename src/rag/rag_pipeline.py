import os
import uuid
from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

from src.models.llm_model import EmbeddingModel
from src.vectordb.vector_store import VectorStore
from config.config import Config
from src.utils.logger import get_logger

class RAGPipeline:
    """Retrieval-Augmented Generation (RAG) pipeline."""
    
    def __init__(self):
        """Initialize the RAG pipeline."""
        self.logger = get_logger()
        self.config = Config()
        
        # Initialize components
        self.embedding_model = None
        self.vector_store = None
        self.text_splitter = None
        self.qa_chain = None
        
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize RAG components."""
        try:
            self.logger.info("Initializing RAG pipeline components")
            
            # Initialize embedding model
            self.embedding_model = EmbeddingModel()
            
            # Initialize text splitter
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.config.CHUNK_SIZE,
                chunk_overlap=self.config.CHUNK_OVERLAP,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )
            
            # Initialize vector store
            self.vector_store = VectorStore()
            
            # Initialize QA chain
            self._setup_qa_chain()
            
            self.logger.info("RAG pipeline initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing RAG pipeline: {str(e)}")
            raise
    
    def _setup_qa_chain(self):
        """Setup the QA chain."""
        try:
            # Create prompt template
            prompt_template = """Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context:
{context}

Question: {question}
Answer:"""
            
            prompt = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )
            
            # For now, we'll use a simple retrieval without LLM chain
            # In a full implementation, you would use LangChain's RetrievalQA
            self.prompt = prompt
            
            self.logger.info("QA chain setup completed")
            
        except Exception as e:
            self.logger.error(f"Error setting up QA chain: {str(e)}")
            raise
    
    def add_document(self, content: str, metadata: Dict[str, Any] = None) -> str:
        """Add a document to the knowledge base."""
        try:
            # Generate document ID
            doc_id = str(uuid.uuid4())
            
            # Prepare metadata
            if metadata is None:
                metadata = {}
            metadata['doc_id'] = doc_id
            
            # Split document into chunks
            documents = self.text_splitter.create_documents([content], [metadata])
            
            # Generate embeddings and store in vector database
            for i, doc in enumerate(documents):
                doc.metadata['chunk_id'] = f"{doc_id}_{i}"
                embedding = self.embedding_model.encode(doc.page_content)
                
                self.vector_store.add_document(
                    doc_id=doc.metadata['chunk_id'],
                    content=doc.page_content,
                    embedding=embedding,
                    metadata=doc.metadata
                )
            
            self.logger.info(f"Added document {doc_id} with {len(documents)} chunks")
            return doc_id
            
        except Exception as e:
            self.logger.error(f"Error adding document: {str(e)}")
            raise
    
    def query(self, question: str, top_k: int = None) -> Dict[str, Any]:
        """Query the knowledge base."""
        try:
            top_k = top_k or self.config.TOP_K_RESULTS
            
            # Generate query embedding
            query_embedding = self.embedding_model.encode(question)
            
            # Retrieve relevant documents
            retrieved_docs = self.vector_store.similarity_search(
                query_embedding=query_embedding,
                top_k=top_k
            )
            
            if not retrieved_docs:
                return {
                    "answer": "I don't have enough information to answer this question.",
                    "sources": [],
                    "confidence": 0.0
                }
            
            # Prepare context from retrieved documents
            context = "\n\n".join([doc['content'] for doc in retrieved_docs])
            
            # For now, return a simple response based on context
            # In a full implementation, you would use the LLM to generate the answer
            answer = self._generate_answer(question, context)
            
            # Prepare sources
            sources = [
                {
                    "content": doc['content'][:200] + "..." if len(doc['content']) > 200 else doc['content'],
                    "metadata": doc['metadata'],
                    "score": doc['score']
                }
                for doc in retrieved_docs
            ]
            
            return {
                "answer": answer,
                "sources": sources,
                "confidence": retrieved_docs[0]['score'] if retrieved_docs else 0.0
            }
            
        except Exception as e:
            self.logger.error(f"Error querying knowledge base: {str(e)}")
            raise
    
    def _generate_answer(self, question: str, context: str) -> str:
        """Generate answer based on context. Placeholder implementation."""
        # This is a simplified implementation
        # In a real scenario, you would use an LLM to generate the answer
        
        if not context.strip():
            return "I don't have enough information to answer this question."
        
        # Simple keyword-based answer generation (placeholder)
        context_lower = context.lower()
        question_lower = question.lower()
        
        # Find the most relevant sentence
        sentences = context.split('.')
        best_sentence = ""
        max_overlap = 0
        
        question_words = set(question_lower.split())
        
        for sentence in sentences:
            sentence_words = set(sentence.lower().split())
            overlap = len(question_words.intersection(sentence_words))
            
            if overlap > max_overlap:
                max_overlap = overlap
                best_sentence = sentence.strip()
        
        if best_sentence:
            return best_sentence
        else:
            return "Based on the available information: " + context[:200] + "..."
    
    def generate_embeddings(self, text: str):
        """Generate embeddings for text."""
        return self.embedding_model.encode(text)
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get information about the RAG pipeline."""
        return {
            "embedding_model": self.embedding_model.get_model_info() if self.embedding_model else None,
            "vector_store": self.vector_store.get_info() if self.vector_store else None,
            "chunk_size": self.config.CHUNK_SIZE,
            "chunk_overlap": self.config.CHUNK_OVERLAP,
            "top_k_results": self.config.TOP_K_RESULTS
        }
