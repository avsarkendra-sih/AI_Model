from typing import List, Dict, Any
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy

class TextPreprocessor:
    """Text preprocessing utilities for NLP tasks."""
    
    def __init__(self):
        """Initialize text preprocessor."""
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Download required NLTK data
        self._download_nltk_data()
        
        # Try to load spaCy model
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            self.nlp = None
    
    def _download_nltk_data(self):
        """Download required NLTK data."""
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
        except Exception:
            pass  # Ignore download errors
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        return word_tokenize(text)
    
    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """Remove stopwords from tokens."""
        return [token for token in tokens if token not in self.stop_words]
    
    def lemmatize(self, tokens: List[str]) -> List[str]:
        """Lemmatize tokens."""
        return [self.lemmatizer.lemmatize(token) for token in tokens]
    
    def preprocess(self, text: str, remove_stopwords: bool = True, lemmatize: bool = True) -> str:
        """Complete text preprocessing pipeline."""
        # Clean text
        text = self.clean_text(text)
        
        # Tokenize
        tokens = self.tokenize(text)
        
        # Remove stopwords
        if remove_stopwords:
            tokens = self.remove_stopwords(tokens)
        
        # Lemmatize
        if lemmatize:
            tokens = self.lemmatize(tokens)
        
        return ' '.join(tokens)
    
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract named entities using spaCy."""
        if self.nlp is None:
            return []
        
        doc = self.nlp(text)
        entities = []
        
        for ent in doc.ents:
            entities.append({
                'text': ent.text,
                'label': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char
            })
        
        return entities
    
    def extract_keywords(self, text: str, top_k: int = 10) -> List[str]:
        """Extract keywords using TF-IDF."""
        try:
            # Preprocess text
            processed_text = self.preprocess(text)
            
            # Create TF-IDF vectorizer
            vectorizer = TfidfVectorizer(max_features=top_k, ngram_range=(1, 2))
            
            # Fit and transform
            tfidf_matrix = vectorizer.fit_transform([processed_text])
            
            # Get feature names
            feature_names = vectorizer.get_feature_names_out()
            
            # Get scores
            scores = tfidf_matrix.toarray()[0]
            
            # Sort by score
            keyword_scores = list(zip(feature_names, scores))
            keyword_scores.sort(key=lambda x: x[1], reverse=True)
            
            return [keyword for keyword, score in keyword_scores if score > 0]
            
        except Exception:
            return []
    
    def sentence_split(self, text: str) -> List[str]:
        """Split text into sentences."""
        return sent_tokenize(text)
    
    def get_text_stats(self, text: str) -> Dict[str, int]:
        """Get basic text statistics."""
        sentences = self.sentence_split(text)
        words = self.tokenize(text.lower())
        
        return {
            'character_count': len(text),
            'word_count': len(words),
            'sentence_count': len(sentences),
            'average_words_per_sentence': len(words) / len(sentences) if sentences else 0,
            'unique_words': len(set(words))
        }
