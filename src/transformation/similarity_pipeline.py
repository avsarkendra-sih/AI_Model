"""
Main pipeline for calculating similarity scores between users and jobs.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from sklearn.metrics.pairwise import cosine_similarity
from utils.logger import BaseLogger
from transformation.text_embedder import TextEmbedder, TextEmbeddingManager
from transformation.skills_processor import SkillsProcessor
from transformation.categorical_encoder import PreferenceEncoder

class SimilarityCalculator:
    """Class for calculating various similarity scores."""
    
    def __init__(self, logger: Optional[BaseLogger] = None):
        """
        Initialize the similarity calculator.
        
        Args:
            logger: Logger instance
        """
        self.logger = logger or BaseLogger("SimilarityCalculator")
    
    def calculate_text_similarity(self, user_embedding: np.ndarray, 
                                job_embedding: np.ndarray) -> float:
        """
        Calculate text similarity between user and job embeddings.
        
        Args:
            user_embedding: User profile embedding
            job_embedding: Job description embedding
            
        Returns:
            Text similarity score (0-1)
        """
        if user_embedding is None or job_embedding is None:
            return 0.0
        
        if len(user_embedding.shape) == 1:
            user_embedding = user_embedding.reshape(1, -1)
        if len(job_embedding.shape) == 1:
            job_embedding = job_embedding.reshape(1, -1)
        
        try:
            similarity = cosine_similarity(user_embedding, job_embedding)[0][0]
            # Ensure the score is between 0 and 1
            return max(0.0, min(1.0, similarity))
        except Exception as e:
            self.logger.error(f"Error calculating text similarity: {str(e)}")
            return 0.0
    
    def calculate_skills_similarity(self, user_skills: List[str], job_skills: List[str]) -> float:
        """
        Calculate skills similarity between user and job.
        
        Args:
            user_skills: List of user skills
            job_skills: List of job skills
            
        Returns:
            Skills similarity score (0-1)
        """
        if not job_skills:
            return 1.0  # No skills required means perfect match
        
        skills_processor = SkillsProcessor(self.logger)
        result = skills_processor.process_skills_matching(user_skills, job_skills)
        
        return result['score']
    
    def calculate_preference_similarity(self, job_features: Dict[str, Any], 
                                      user_preferences: Dict[str, Any],
                                      preference_weights: Dict[str, float]) -> float:
        """
        Calculate preference similarity between job features and user preferences.
        
        Args:
            job_features: Dictionary of job features
            user_preferences: Dictionary of user preferences
            preference_weights: Dictionary of weights for each preference type
            
        Returns:
            Preference similarity score (0-1)
        """
        total_score = 0.0
        total_weight = 0.0
        
        encoder = PreferenceEncoder(self.logger)
        
        for feature, weight in preference_weights.items():
            if feature not in job_features or feature not in user_preferences:
                continue
            
            job_value = job_features[feature]
            user_preference = user_preferences[feature]
            
            # Determine feature type
            if feature in ['mode', 'job_type']:
                feature_type = 'categorical'
            elif feature == 'location':
                feature_type = 'multi_value'
            else:
                feature_type = 'binary'
            
            # Calculate feature score
            feature_score = encoder.calculate_preference_score(
                job_value, user_preference, feature_type
            )
            
            total_score += feature_score * weight
            total_weight += weight
        
        if total_weight == 0:
            return 0.0
        
        return total_score / total_weight

class SimilarityPipeline:
    """Main pipeline for calculating all similarity scores."""
    
    def __init__(self, embedder: TextEmbedder, logger: Optional[BaseLogger] = None):
        """
        Initialize the similarity pipeline.
        
        Args:
            embedder: Text embedder instance
            logger: Logger instance
        """
        self.logger = logger or BaseLogger("SimilarityPipeline")
        self.embedder = embedder
        self.embedding_manager = TextEmbeddingManager(embedder)
        self.similarity_calculator = SimilarityCalculator(logger)
        self.skills_processor = SkillsProcessor(logger)
    
    def calculate_all_similarities(self, user_profile: Dict[str, Any], 
                                 job_features: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate all similarity scores between user and job.
        
        Args:
            user_profile: Dictionary of user profile data
            job_features: Dictionary of job features
            
        Returns:
            Dictionary of similarity scores
        """
        # Calculate text similarity
        user_embedding = self.embedding_manager.get_embedding(user_profile.get('description', ''))
        job_embedding = self.embedding_manager.get_embedding(job_features.get('description', ''))
        text_similarity = self.similarity_calculator.calculate_text_similarity(
            user_embedding, job_embedding
        )
        
        # Calculate skills similarity
        user_skills = user_profile.get('skills', [])
        job_skills = job_features.get('skills', [])
        skills_similarity = self.similarity_calculator.calculate_skills_similarity(
            user_skills, job_skills
        )
        
        # Calculate preference similarity
        user_preferences = user_profile.get('preferences', {})
        preference_weights = {
            'job_type': 0.4,
            'location': 0.3,
            'mode': 0.3
        }
        preference_similarity = self.similarity_calculator.calculate_preference_similarity(
            job_features, user_preferences, preference_weights
        )
        
        return {
            'text_similarity': text_similarity,
            'skills_similarity': skills_similarity,
            'preference_similarity': preference_similarity
        }
    
    def calculate_composite_score(self, similarity_scores: Dict[str, float], 
                                weights: Dict[str, float]) -> float:
        """
        Calculate composite score from individual similarity scores.
        
        Args:
            similarity_scores: Dictionary of similarity scores
            weights: Dictionary of weights for each score type
            
        Returns:
            Composite score (0-1)
        """
        composite_score = 0.0
        total_weight = 0.0
        
        for score_type, weight in weights.items():
            if score_type in similarity_scores:
                composite_score += similarity_scores[score_type] * weight
                total_weight += weight
        
        if total_weight == 0:
            return 0.0
        
        return composite_score / total_weight
    
    def recommend_jobs(self, user_profile: Dict[str, Any], 
                      job_features_list: List[Dict[str, Any]], 
                      top_n: int = 5,
                      score_weights: Optional[Dict[str, float]] = None) -> List[Dict[str, Any]]:
        """
        Recommend top N jobs for a user.
        
        Args:
            user_profile: Dictionary of user profile data
            job_features_list: List of job features dictionaries
            top_n: Number of top jobs to recommend
            score_weights: Weights for different score types
            
        Returns:
            List of recommended jobs with scores
        """
        if score_weights is None:
            score_weights = {
                'text_similarity': 0.4,
                'skills_similarity': 0.3,
                'preference_similarity': 0.3
            }
        
        scored_jobs = []
        
        for job_features in job_features_list:
            # Calculate all similarity scores
            similarity_scores = self.calculate_all_similarities(user_profile, job_features)
            
            # Calculate composite score
            composite_score = self.calculate_composite_score(similarity_scores, score_weights)
            
            # Add to results
            scored_jobs.append({
                'job_id': job_features.get('job_id', ''),
                'composite_score': composite_score,
                **similarity_scores,
                'job_features': job_features
            })
        
        # Sort by composite score (descending)
        scored_jobs.sort(key=lambda x: x['composite_score'], reverse=True)
        
        # Return top N jobs
        return scored_jobs[:top_n]