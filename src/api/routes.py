from flask import Blueprint, request, jsonify
from src.models.llm_model import LLMModel
from src.rag.rag_pipeline import RAGPipeline
from src.utils.logger import get_logger
import traceback

# Create API blueprint
api_bp = Blueprint('api', __name__)
logger = get_logger()

# Initialize models (will be done lazily)
llm_model = None
rag_pipeline = None

def get_llm_model():
    """Get or initialize LLM model."""
    global llm_model
    if llm_model is None:
        llm_model = LLMModel()
    return llm_model

def get_rag_pipeline():
    """Get or initialize RAG pipeline."""
    global rag_pipeline
    if rag_pipeline is None:
        rag_pipeline = RAGPipeline()
    return rag_pipeline

@api_bp.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "service": "AvsarKendra AI API",
        "version": "0.1.0"
    })

@api_bp.route('/chat', methods=['POST'])
def chat():
    """Chat endpoint for general conversation."""
    try:
        data = request.get_json()
        
        if not data or 'message' not in data:
            return jsonify({"error": "Message is required"}), 400
        
        message = data['message']
        logger.info(f"Received chat message: {message}")
        
        # Get LLM model and generate response
        model = get_llm_model()
        response = model.generate_response(message)
        
        return jsonify({
            "response": response,
            "message": message
        })
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": "Internal server error"}), 500

@api_bp.route('/rag/query', methods=['POST'])
def rag_query():
    """RAG query endpoint for document-based Q&A."""
    try:
        data = request.get_json()
        
        if not data or 'query' not in data:
            return jsonify({"error": "Query is required"}), 400
        
        query = data['query']
        logger.info(f"Received RAG query: {query}")
        
        # Get RAG pipeline and process query
        rag = get_rag_pipeline()
        response = rag.query(query)
        
        return jsonify({
            "answer": response.get("answer", ""),
            "sources": response.get("sources", []),
            "query": query
        })
        
    except Exception as e:
        logger.error(f"Error in RAG query endpoint: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": "Internal server error"}), 500

@api_bp.route('/rag/add_document', methods=['POST'])
def add_document():
    """Add document to RAG knowledge base."""
    try:
        data = request.get_json()
        
        if not data or 'content' not in data:
            return jsonify({"error": "Document content is required"}), 400
        
        content = data['content']
        metadata = data.get('metadata', {})
        
        logger.info(f"Adding document to knowledge base")
        
        # Get RAG pipeline and add document
        rag = get_rag_pipeline()
        doc_id = rag.add_document(content, metadata)
        
        return jsonify({
            "message": "Document added successfully",
            "document_id": doc_id
        })
        
    except Exception as e:
        logger.error(f"Error adding document: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": "Internal server error"}), 500

@api_bp.route('/models/status', methods=['GET'])
def models_status():
    """Get status of loaded models."""
    try:
        status = {
            "llm_model_loaded": llm_model is not None,
            "rag_pipeline_loaded": rag_pipeline is not None
        }
        
        if llm_model:
            status["llm_model_info"] = llm_model.get_model_info()
        
        if rag_pipeline:
            status["rag_pipeline_info"] = rag_pipeline.get_pipeline_info()
        
        return jsonify(status)
        
    except Exception as e:
        logger.error(f"Error getting models status: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

@api_bp.route('/embeddings', methods=['POST'])
def generate_embeddings():
    """Generate embeddings for text."""
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({"error": "Text is required"}), 400
        
        text = data['text']
        
        # Get RAG pipeline for embeddings
        rag = get_rag_pipeline()
        embeddings = rag.generate_embeddings(text)
        
        return jsonify({
            "embeddings": embeddings.tolist(),
            "dimension": len(embeddings),
            "text": text
        })
        
    except Exception as e:
        logger.error(f"Error generating embeddings: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500
