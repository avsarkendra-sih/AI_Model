import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Configuration class for the application."""
    
    # Flask Configuration
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    FLASK_ENV = os.environ.get('FLASK_ENV') or 'development'
    
    # API Configuration
    API_TITLE = "AvsarKendra AI API"
    API_VERSION = "v1"
    
    # Database Configuration
    DATABASE_URL = os.environ.get('DATABASE_URL') or 'sqlite:///avsarkendra.db'
    
    # Vector Database Configuration
    CHROMA_PERSIST_DIRECTORY = os.environ.get('CHROMA_PERSIST_DIRECTORY') or './data/chroma_db'
    FAISS_INDEX_PATH = os.environ.get('FAISS_INDEX_PATH') or './data/faiss_index'
    PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
    PINECONE_ENVIRONMENT = os.environ.get('PINECONE_ENVIRONMENT')
    
    # HuggingFace Configuration
    HUGGINGFACE_API_TOKEN = os.environ.get('HUGGINGFACE_API_TOKEN')
    HUGGINGFACE_MODEL_NAME = os.environ.get('HUGGINGFACE_MODEL_NAME') or 'sentence-transformers/all-MiniLM-L6-v2'
    
    # OpenAI Configuration (if using OpenAI models)
    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
    
    # Model Configuration
    EMBEDDING_MODEL = os.environ.get('EMBEDDING_MODEL') or 'sentence-transformers/all-MiniLM-L6-v2'
    LLM_MODEL = os.environ.get('LLM_MODEL') or 'microsoft/DialoGPT-medium'
    MAX_TOKENS = int(os.environ.get('MAX_TOKENS', 512))
    TEMPERATURE = float(os.environ.get('TEMPERATURE', 0.7))
    
    # RAG Configuration
    CHUNK_SIZE = int(os.environ.get('CHUNK_SIZE', 1000))
    CHUNK_OVERLAP = int(os.environ.get('CHUNK_OVERLAP', 200))
    TOP_K_RESULTS = int(os.environ.get('TOP_K_RESULTS', 5))
    
    # File Upload Configuration
    UPLOAD_FOLDER = os.environ.get('UPLOAD_FOLDER') or './data/uploads'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx', 'csv', 'json'}
    
    # Logging Configuration
    LOG_LEVEL = os.environ.get('LOG_LEVEL') or 'INFO'
    LOG_FILE = os.environ.get('LOG_FILE') or './logs/app.log'

class DevelopmentConfig(Config):
    """Development configuration."""
    DEBUG = True
    TESTING = False

class ProductionConfig(Config):
    """Production configuration."""
    DEBUG = False
    TESTING = False

class TestingConfig(Config):
    """Testing configuration."""
    DEBUG = True
    TESTING = True
    DATABASE_URL = 'sqlite:///:memory:'

# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}
