# import sys
# import os
# from src.api.routes import api_bp
# from src.models.llm_model import LLMModel
# from src.rag.rag_pipeline import RAGPipeline
# from src.utils.logger import setup_logger
# from config.config import Config

# def main():
#     """Main entry point for the application."""
#     logger = setup_logger()
    
#     logger.info("Starting AvsarKendra AI application")
    
#     # Initialize configuration
#     config = Config()
    
#     # You can add CLI functionality here
#     if len(sys.argv) > 1:
#         command = sys.argv[1]
        
#         if command == "test":
#             logger.info("Running tests...")
#             # Add test runner code here
            
#         elif command == "train":
#             logger.info("Starting training...")
#             # Add training code here
            
#         elif command == "serve":
#             logger.info("Starting server...")
#             from app import create_app
#             app = create_app()
#             app.run(host='0.0.0.0', port=5000, debug=True)
            
#         else:
#             logger.error(f"Unknown command: {command}")
#             print("Available commands: test, train, serve")
#     else:
#         print("AvsarKendra AI - Available commands: test, train, serve")

# if __name__ == "__main__":
#     main()
