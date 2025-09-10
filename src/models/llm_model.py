import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sentence_transformers import SentenceTransformer
from config.config import Config
from src.utils.logger import get_logger

class LLMModel:
    """Large Language Model wrapper for text generation."""
    
    def __init__(self, model_name=None):
        """Initialize the LLM model."""
        self.logger = get_logger()
        self.model_name = model_name or Config.LLM_MODEL
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        
        self._load_model()
    
    def _load_model(self):
        """Load the model and tokenizer."""
        try:
            self.logger.info(f"Loading LLM model: {self.model_name}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True
            )
            
            # Create pipeline
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            
            self.logger.info(f"Model loaded successfully on device: {self.device}")
            
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise
    
    def generate_response(self, prompt, max_length=None, temperature=None, top_p=0.9):
        """Generate response for given prompt."""
        try:
            max_length = max_length or Config.MAX_TOKENS
            temperature = temperature or Config.TEMPERATURE
            
            # Generate response
            response = self.pipeline(
                prompt,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                num_return_sequences=1
            )
            
            # Extract generated text
            generated_text = response[0]['generated_text']
            
            # Remove the input prompt from the response
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):].strip()
            
            return generated_text
            
        except Exception as e:
            self.logger.error(f"Error generating response: {str(e)}")
            return "Sorry, I encountered an error while generating a response."
    
    def get_model_info(self):
        """Get information about the loaded model."""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "model_loaded": self.model is not None,
            "tokenizer_loaded": self.tokenizer is not None
        }

class EmbeddingModel:
    """Embedding model for generating text embeddings."""
    
    def __init__(self, model_name=None):
        """Initialize the embedding model."""
        self.logger = get_logger()
        self.model_name = model_name or Config.EMBEDDING_MODEL
        self.model = None
        
        self._load_model()
    
    def _load_model(self):
        """Load the embedding model."""
        try:
            self.logger.info(f"Loading embedding model: {self.model_name}")
            
            self.model = SentenceTransformer(self.model_name)
            
            self.logger.info("Embedding model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading embedding model: {str(e)}")
            raise
    
    def encode(self, texts):
        """Encode texts into embeddings."""
        try:
            if isinstance(texts, str):
                texts = [texts]
            
            embeddings = self.model.encode(texts)
            return embeddings
            
        except Exception as e:
            self.logger.error(f"Error encoding texts: {str(e)}")
            raise
    
    def get_embedding_dimension(self):
        """Get the dimension of embeddings."""
        return self.model.get_sentence_embedding_dimension()
    
    def get_model_info(self):
        """Get information about the embedding model."""
        return {
            "model_name": self.model_name,
            "embedding_dimension": self.get_embedding_dimension() if self.model else None,
            "model_loaded": self.model is not None
        }
