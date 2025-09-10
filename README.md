# AvsarKendra AI

A comprehensive AI/ML GenAI project built with LangChain, HuggingFace, Flask, NLP, and RAG (Retrieval-Augmented Generation) capabilities. This project provides a complete solution for building intelligent applications with document processing, vector search, and conversational AI features.

## 🚀 Features

- **LangChain Integration**: Advanced text processing and chain operations
- **HuggingFace Models**: State-of-the-art LLMs and embedding models
- **RAG Pipeline**: Retrieval-Augmented Generation for knowledge-based Q&A
- **Vector Databases**: Support for ChromaDB, FAISS, Pinecone, and more
- **NLP Processing**: Advanced text preprocessing and analysis
- **Document Processing**: Support for PDF, DOCX, CSV, JSON, HTML, and web URLs
- **Flask REST API**: Complete web API for all functionalities
- **Modular Architecture**: Clean, scalable, and maintainable code structure

## 📋 Table of Contents

- [Installation](#installation)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [Development](#development)
- [Docker Support](#docker-support)
- [Contributing](#contributing)

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- pip
- Git

### Quick Start

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd AvsarKendra_AI
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install the package**
   ```bash
   pip install -e .
   ```

5. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

6. **Download required models and data**
   ```bash
   make setup-nltk
   make setup-spacy
   ```

7. **Run the application**
   ```bash
   python app.py
   ```

## 📁 Project Structure

```
AvsarKendra_AI/
├── src/
│   ├── __init__.py
│   ├── main.py                 # Main entry point
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py           # Flask API routes
│   ├── models/
│   │   ├── __init__.py
│   │   └── llm_model.py        # LLM and embedding models
│   ├── rag/
│   │   ├── __init__.py
│   │   └── rag_pipeline.py     # RAG implementation
│   ├── vectordb/
│   │   ├── __init__.py
│   │   └── vector_store.py     # Vector database interface
│   └── utils/
│       ├── __init__.py
│       ├── logger.py           # Logging utilities
│       ├── text_processing.py  # NLP utilities
│       └── document_processor.py # Document processing
├── config/
│   ├── __init__.py
│   └── config.py               # Configuration management
├── data/
│   ├── raw/                    # Raw data files
│   └── processed/              # Processed data files
├── tests/
│   └── test_main.py           # Test cases
├── logs/                       # Log files
├── app.py                      # Flask application
├── requirements.txt            # Python dependencies
├── setup.py                    # Package setup
├── Dockerfile                  # Docker configuration
├── docker-compose.yml          # Docker Compose setup
├── Makefile                    # Build automation
├── .env.example               # Environment variables template
├── .gitignore                 # Git ignore rules
└── README.md                  # This file
```

## ⚙️ Configuration

The project uses environment variables for configuration. Copy `.env.example` to `.env` and update the values:

### Key Configuration Options

- **Flask Settings**: SECRET_KEY, FLASK_ENV
- **Model Configuration**: EMBEDDING_MODEL, LLM_MODEL, MAX_TOKENS
- **Vector Database**: CHROMA_PERSIST_DIRECTORY, PINECONE_API_KEY
- **HuggingFace**: HUGGINGFACE_API_TOKEN, HUGGINGFACE_MODEL_NAME
- **RAG Settings**: CHUNK_SIZE, CHUNK_OVERLAP, TOP_K_RESULTS

## 🎯 Usage

### Basic Usage

1. **Start the Flask server**
   ```bash
   python app.py
   ```

2. **Access the API**
   - Health check: `GET http://localhost:5000/`
   - API documentation: `GET http://localhost:5000/api/health`

### Using the RAG Pipeline

```python
from src.rag.rag_pipeline import RAGPipeline

# Initialize RAG pipeline
rag = RAGPipeline()

# Add documents to knowledge base
doc_id = rag.add_document(
    content="Your document content here",
    metadata={"source": "example.pdf"}
)

# Query the knowledge base
response = rag.query("What is the main topic?")
print(response["answer"])
```

### Using Document Processor

```python
from src.utils.document_processor import DocumentProcessor

processor = DocumentProcessor()

# Process a PDF file
result = processor.process_document("path/to/document.pdf")
print(result["content"])
print(result["metadata"])
```

## 🔗 API Endpoints

### Core Endpoints

- `GET /` - Health check
- `GET /api/health` - API health status
- `POST /api/chat` - General chat interface
- `POST /api/rag/query` - RAG-based question answering
- `POST /api/rag/add_document` - Add documents to knowledge base
- `GET /api/models/status` - Model status information
- `POST /api/embeddings` - Generate text embeddings

### Example API Calls

**Chat with the model:**
```bash
curl -X POST http://localhost:5000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello, how are you?"}'
```

**Query the knowledge base:**
```bash
curl -X POST http://localhost:5000/api/rag/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is machine learning?"}'
```

**Add a document:**
```bash
curl -X POST http://localhost:5000/api/rag/add_document \
  -H "Content-Type: application/json" \
  -d '{"content": "Machine learning is...", "metadata": {"source": "textbook"}}'
```

## 🛠️ Development

### Setting up Development Environment

1. **Install development dependencies**
   ```bash
   make install-dev
   ```

2. **Set up pre-commit hooks**
   ```bash
   make setup-hooks
   ```

3. **Run tests**
   ```bash
   make test
   ```

4. **Code formatting**
   ```bash
   make format
   ```

5. **Linting**
   ```bash
   make lint
   ```

### Available Make Commands

- `make install` - Install production dependencies
- `make install-dev` - Install development dependencies
- `make test` - Run tests
- `make lint` - Run linting
- `make format` - Format code
- `make clean` - Clean build artifacts
- `make run` - Run the application
- `make docker-build` - Build Docker image
- `make docker-run` - Run Docker container

## 🐳 Docker Support

### Using Docker

1. **Build the image**
   ```bash
   docker build -t avsarkendra-ai .
   ```

2. **Run the container**
   ```bash
   docker run -p 5000:5000 avsarkendra-ai
   ```

### Using Docker Compose

1. **Start all services**
   ```bash
   docker-compose up -d
   ```

2. **View logs**
   ```bash
   docker-compose logs -f
   ```

3. **Stop services**
   ```bash
   docker-compose down
   ```

## 📚 Key Components

### LLM Models (`src/models/llm_model.py`)
- **LLMModel**: Wrapper for HuggingFace language models
- **EmbeddingModel**: Text embedding generation

### RAG Pipeline (`src/rag/rag_pipeline.py`)
- Document chunking and embedding
- Vector similarity search
- Context-aware answer generation

### Vector Store (`src/vectordb/vector_store.py`)
- Multi-backend support (ChromaDB, FAISS, simple store)
- Similarity search capabilities
- Document management

### Document Processing (`src/utils/document_processor.py`)
- Support for multiple file formats
- Metadata extraction
- Web URL processing

### Text Processing (`src/utils/text_processing.py`)
- NLP preprocessing pipeline
- Named entity recognition
- Keyword extraction

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run tests and linting
6. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:
- Create an issue on GitHub
- Check the documentation
- Review the example code

## 🚀 Next Steps

After setting up the project, you can:

1. **Customize the models** by updating the configuration
2. **Add your own documents** to the knowledge base
3. **Extend the API** with additional endpoints
4. **Integrate with frontend** applications
5. **Deploy to cloud** platforms

## 🔧 Troubleshooting

### Common Issues

1. **Import errors**: Make sure to install the package with `pip install -e .`
2. **Model download issues**: Check your internet connection and HuggingFace token
3. **Memory issues**: Consider using smaller models or increasing system memory
4. **NLTK/spaCy data**: Run `make setup-nltk` and `make setup-spacy`

For more detailed troubleshooting, check the logs in the `logs/` directory.