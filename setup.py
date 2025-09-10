from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Core dependencies from requirements.txt (excluding local/editable stuff)
requirements = [
    # Core ML/AI Libraries
    "transformers>=4.35.0",
    "huggingface-hub>=0.17.0",

    # LangChain and RAG
    "langchain>=0.1.0",
    "langchain-community>=0.0.10",
    "langchain-core>=0.1.0",
    "langchain-google-genai",
    "langchain-huggingface>=0.0.1",

    # Vector Databases
    "chromadb>=0.4.15",
    "faiss-cpu>=1.7.4",
    "pinecone-client>=2.2.4",
    "weaviate-client>=3.25.0",
    "qdrant-client>=1.6.0",

    # NLP Libraries
    "nltk>=3.8.1",
    "spacy>=3.7.0",
    "textblob>=0.17.1",
    "gensim>=4.3.2",
    "sentence-transformers>=2.2.2",

    # Web Framework
    "flask>=2.3.3",
    "flask-cors>=4.0.0",
    "flask-restful>=0.3.10",
    "gunicorn>=21.2.0",

    # Data Processing
    "pandas>=2.1.0",
    "numpy>=1.24.0",
    "scikit-learn>=1.3.0",
    "matplotlib>=3.7.0",
    "seaborn>=0.12.0",

    # Document Processing
    "pypdf2>=3.0.1",
    "python-docx>=0.8.11",
    "beautifulsoup4>=4.12.2",
    "requests>=2.31.0",

    # Environment and Configuration
    "python-dotenv>=1.0.0",
    "pydantic>=2.4.0",
    "pyyaml>=6.0.1",

    # Database
    "sqlalchemy>=2.0.0",

    # Utilities
    "python-multipart>=0.0.6",
    "uvicorn>=0.23.0",
]

setup(
    name="avsarkendra-ai",
    version="0.1.0",
    author="Satyam Kumar Jha",
    author_email="satyamjha4@gmail.com",
    description="AI/ML GenAI project with LangChain, HuggingFace, Flask, NLP, and RAG",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Satyamkumarjha4/AvsarKendra_AI",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.11",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
        ],
        "gpu": [
            # Users must install proper CUDA builds separately
            "torch>=2.0.0",
            "faiss-cpu>=1.7.4"
        ],
    },
    entry_points={
        "console_scripts": [
            "avsarkendra-ai=src.main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.txt", "*.md", "*.yml", "*.yaml"],
    },
)
