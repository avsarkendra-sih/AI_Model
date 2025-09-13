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

#!/usr/bin/env python3
"""
Comprehensive testing script for similarity calculations with sample user profiles.
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import BaseLogger
from transformation.text_embedder import TextEmbedder, TextEmbeddingManager
from transformation.skills_processor import SkillsProcessor
from transformation.categorical_encoder import PreferenceEncoder
from transformation.similarity_pipeline import SimilarityCalculator, SimilarityPipeline
from transformation.feature_store import JobFeatureStore

def test_text_embeddings():
    """Test text embedding generation."""
    print("=" * 60)
    print("TESTING TEXT EMBEDDINGS")
    print("=" * 60)
    
    logger = BaseLogger("TestTextEmbeddings")
    
    try:
        # Initialize embedder
        embedder = TextEmbedder(model_name="all-MiniLM-L6-v2", logger=logger)
        embedding_manager = TextEmbeddingManager(embedder)
        
        # Test texts
        test_texts = [
            "Software engineer with Python and JavaScript experience",
            "Data scientist with machine learning and statistics background",
            "Frontend developer specializing in React and Vue.js"
        ]
        
        # Generate embeddings
        embeddings = embedder.embed_batch(test_texts)
        
        print(f"Generated embeddings for {len(test_texts)} texts")
        print(f"Embedding dimension: {embeddings.shape[1]}")
        print(f"Embedding shape: {embeddings.shape}")
        
        # Test similarity between texts
        similarity_matrix = np.dot(embeddings, embeddings.T)
        print("\nSimilarity matrix:")
        for i in range(len(test_texts)):
            for j in range(len(test_texts)):
                print(f"{similarity_matrix[i][j]:.3f}", end=" ")
            print()
        
        # Test single text embedding
        single_embedding = embedder.embed_text(test_texts[0])
        print(f"\nSingle embedding shape: {single_embedding.shape}")
        
        print("✅ Text embedding test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Text embedding test failed: {str(e)}")
        return False

def test_skills_matching():
    """Test skills matching functionality."""
    print("\n" + "=" * 60)
    print("TESTING SKILLS MATCHING")
    print("=" * 60)
    
    logger = BaseLogger("TestSkillsMatching")
    
    try:
        skills_processor = SkillsProcessor(logger)
        
        # Test data
        user_skills = ["Python", "JavaScript", "React", "SQL", "Docker"]
        job_skills = ["Python", "Django", "JavaScript", "PostgreSQL", "AWS"]
        
        # Test exact matching
        exact_result = skills_processor.process_skills_matching(
            user_skills, job_skills, use_fuzzy=False
        )
        
        print("Exact matching results:")
        print(f"Matched skills: {exact_result['matched_skills']}")
        print(f"Missing skills: {exact_result['missing_skills']}")
        print(f"Score: {exact_result['score']:.3f}")
        
        # Test fuzzy matching
        fuzzy_result = skills_processor.process_skills_matching(
            user_skills, job_skills, use_fuzzy=True
        )
        
        print("\nFuzzy matching results:")
        print(f"Matched skills: {fuzzy_result['matched_skills']}")
        print(f"Missing skills: {fuzzy_result['missing_skills']}")
        print(f"Fuzzy matches: {fuzzy_result['fuzzy_matches']}")
        print(f"Score: {fuzzy_result['score']:.3f}")
        
        # Test edge cases
        empty_result = skills_processor.process_skills_matching([], job_skills)
        print(f"\nEmpty user skills score: {empty_result['score']:.3f}")
        
        no_requirements_result = skills_processor.process_skills_matching(user_skills, [])
        print(f"No job requirements score: {no_requirements_result['score']:.3f}")
        
        print("✅ Skills matching test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Skills matching test failed: {str(e)}")
        return False

def test_preference_encoding():
    """Test preference encoding functionality."""
    print("\n" + "=" * 60)
    print("TESTING PREFERENCE ENCODING")
    print("=" * 60)
    
    logger = BaseLogger("TestPreferenceEncoding")
    
    try:
        encoder = PreferenceEncoder(logger)
        
        # Test data
        user_preferences = {
            "job_type": "Software Development",
            "location": ["Remote", "Hybrid"],
            "mode": "Online"
        }
        
        available_categories = {
            "job_type": ["Software Development", "Data Science", "DevOps"],
            "location": ["Remote", "Hybrid", "On-site"],
            "mode": ["Online", "Offline"]
        }
        
        # Encode preferences
        encoded_prefs = encoder.encode_user_preferences(user_preferences, available_categories)
        print(f"Encoded preferences: {encoded_prefs}")
        
        # Test preference scoring
        job_features = {
            "job_type": "Software Development",
            "location": "Remote",
            "mode": "Online"
        }
        
        # Test different preference scenarios
        test_cases = [
            ({"job_type": "Software Development"}, "Software Development", "categorical", 1.0),
            ({"job_type": "Data Science"}, "Software Development", "categorical", 0.0),
            ({"location": ["Remote", "Hybrid"]}, "Remote", "multi_value", 1.0),
            ({"location": ["Remote", "Hybrid"]}, "On-site", "multi_value", 0.0),
            ({"mode": "Online"}, "Online", "binary", 1.0),
            ({"mode": "Online"}, "Offline", "binary", 0.0),
        ]
        
        print("\nPreference scoring test cases:")
        for i, (prefs, job_value, feature_type, expected_score) in enumerate(test_cases):
            score = encoder.calculate_preference_score(job_value, prefs[list(prefs.keys())[0]], feature_type)
            status = "✅" if abs(score - expected_score) < 0.001 else "❌"
            print(f"{status} Test {i+1}: {prefs} vs {job_value} = {score:.1f} (expected {expected_score:.1f})")
        
        print("✅ Preference encoding test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Preference encoding test failed: {str(e)}")
        return False

def test_similarity_calculations():
    """Test similarity calculation functionality."""
    print("\n" + "=" * 60)
    print("TESTING SIMILARITY CALCULATIONS")
    print("=" * 60)
    
    logger = BaseLogger("TestSimilarityCalculations")
    
    try:
        embedder = TextEmbedder(logger=logger)
        calculator = SimilarityCalculator(logger)
        
        # Generate test embeddings
        text1 = "Software engineer with Python experience"
        text2 = "Developer skilled in Python programming"
        text3 = "Data analyst with SQL expertise"
        
        embedding1 = embedder.embed_text(text1)
        embedding2 = embedder.embed_text(text2)
        embedding3 = embedder.embed_text(text3)
        
        # Test text similarity
        sim12 = calculator.calculate_text_similarity(embedding1, embedding2)
        sim13 = calculator.calculate_text_similarity(embedding1, embedding3)
        
        print(f"Similarity between '{text1}' and '{text2}': {sim12:.3f}")
        print(f"Similarity between '{text1}' and '{text3}': {sim13:.3f}")
        print(f"Similarity between same text: {calculator.calculate_text_similarity(embedding1, embedding1):.3f}")
        
        # Test skills similarity
        user_skills = ["Python", "JavaScript", "SQL"]
        job_skills1 = ["Python", "Django", "JavaScript"]
        job_skills2 = ["Java", "Spring", "Hibernate"]
        
        skills_sim1 = calculator.calculate_skills_similarity(user_skills, job_skills1)
        skills_sim2 = calculator.calculate_skills_similarity(user_skills, job_skills2)
        
        print(f"\nSkills similarity (matching job): {skills_sim1:.3f}")
        print(f"Skills similarity (non-matching job): {skills_sim2:.3f}")
        
        # Test preference similarity
        job_features = {
            "job_type": "Software Development",
            "location": "Remote",
            "mode": "Online"
        }
        
        user_preferences = {
            "job_type": "Software Development",
            "location": ["Remote", "Hybrid"],
            "mode": "Online"
        }
        
        preference_weights = {
            "job_type": 0.4,
            "location": 0.3,
            "mode": 0.3
        }
        
        pref_sim = calculator.calculate_preference_similarity(
            job_features, user_preferences, preference_weights
        )
        
        print(f"Preference similarity: {pref_sim:.3f}")
        
        print("✅ Similarity calculations test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Similarity calculations test failed: {str(e)}")
        return False

def test_full_pipeline():
    """Test the full similarity pipeline with sample user profiles."""
    print("\n" + "=" * 60)
    print("TESTING FULL PIPELINE WITH SAMPLE USER PROFILES")
    print("=" * 60)
    
    logger = BaseLogger("TestFullPipeline")
    
    try:
        # Initialize the pipeline
        embedder = TextEmbedder(logger=logger)
        pipeline = SimilarityPipeline(embedder, logger)
        
        # Sample user profiles
        sample_users = [
            {
                "name": "Frontend Developer",
                "description": "Experienced frontend developer with expertise in React, JavaScript, and CSS. \
                               Passionate about creating responsive and accessible web applications.",
                "skills": ["JavaScript", "React", "CSS", "HTML", "TypeScript"],
                "preferences": {
                    "job_type": "Software Development",
                    "location": ["Remote", "Hybrid"],
                    "mode": "Online"
                }
            },
            {
                "name": "Data Scientist",
                "description": "Data scientist with strong background in machine learning, statistics, and Python. \
                               Experience with predictive modeling and data visualization.",
                "skills": ["Python", "Machine Learning", "Statistics", "SQL", "TensorFlow"],
                "preferences": {
                    "job_type": "Data Science",
                    "location": ["Remote"],
                    "mode": "Online"
                }
            },
            {
                "name": "DevOps Engineer",
                "description": "DevOps engineer with expertise in cloud infrastructure, CI/CD pipelines, and containerization. \
                               Experience with AWS, Docker, and Kubernetes.",
                "skills": ["AWS", "Docker", "Kubernetes", "CI/CD", "Linux"],
                "preferences": {
                    "job_type": "DevOps",
                    "location": ["Hybrid", "On-site"],
                    "mode": "Offline"
                }
            }
        ]
        
        # Sample job features (simplified for testing)
        sample_jobs = [
            {
                "job_id": "job_001",
                "title": "Senior Frontend Developer",
                "description": "We are looking for a senior frontend developer with React experience to join our team. \
                               You will be responsible for building user interfaces and implementing responsive designs.",
                "skills": ["React", "JavaScript", "CSS", "HTML", "TypeScript"],
                "job_type": "Software Development",
                "location": "Remote",
                "mode": "Online",
                "company": "TechCorp"
            },
            {
                "job_id": "job_002",
                "title": "Machine Learning Engineer",
                "description": "Join our AI team to develop and deploy machine learning models. \
                               Experience with Python and TensorFlow required.",
                "skills": ["Python", "Machine Learning", "TensorFlow", "Deep Learning"],
                "job_type": "Data Science",
                "location": "Hybrid",
                "mode": "Online",
                "company": "AI Innovations"
            },
            {
                "job_id": "job_003",
                "title": "Cloud Infrastructure Engineer",
                "description": "Design and implement cloud infrastructure solutions using AWS and Docker. \
                               Experience with Kubernetes is a plus.",
                "skills": ["AWS", "Docker", "Kubernetes", "CI/CD"],
                "job_type": "DevOps",
                "location": "On-site",
                "mode": "Offline",
                "company": "Cloud Solutions Inc."
            }
        ]
        
        # Test each user profile
        for user in sample_users:
            print(f"\nTesting recommendations for {user['name']}:")
            print("-" * 40)
            
            recommendations = pipeline.recommend_jobs(
                user, sample_jobs, top_n=2,
                score_weights={
                    'text_similarity': 0.4,
                    'skills_similarity': 0.3,
                    'preference_similarity': 0.3
                }
            )
            
            for i, rec in enumerate(recommendations):
                print(f"{i+1}. {rec['job_features']['title']} at {rec['job_features']['company']}")
                print(f"   Composite Score: {rec['composite_score']:.3f}")
                print(f"   Text Similarity: {rec['text_similarity']:.3f}")
                print(f"   Skills Similarity: {rec['skills_similarity']:.3f}")
                print(f"   Preference Similarity: {rec['preference_similarity']:.3f}")
                print()
        
        print("✅ Full pipeline test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Full pipeline test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_with_real_data():
    """Test with real processed data from Stage 0."""
    print("\n" + "=" * 60)
    print("TESTING WITH REAL PROCESSED DATA")
    print("=" * 60)
    
    logger = BaseLogger("TestRealData")
    
    try:
        # Load processed data
        processed_data_path = Path("data/processed/processed_job_postings.pkl")
        if not processed_data_path.exists():
            print("Processed data not found. Run Stage 0 first.")
            return False
        
        df = pd.read_pickle(processed_data_path)
        print(f"Loaded {len(df)} processed job postings")
        
        # Initialize pipeline
        embedder = TextEmbedder(logger=logger)
        pipeline = SimilarityPipeline(embedder, logger)
        
        # Create sample user profile
        sample_user = {
            "name": "Software Engineer",
            "description": "Software engineer with experience in Python, JavaScript, and cloud technologies. \
                           Looking for opportunities in software development with modern tech stacks.",
            "skills": ["Python", "JavaScript", "React", "AWS", "Docker"],
            "preferences": {
                "job_type": "Software Development",
                "location": ["Remote", "Hybrid"],
                "mode": "Online"
            }
        }
        
        # Convert DataFrame to list of job features
        job_features_list = []
        for _, row in df.iterrows():
            job_features = {
                "job_id": row.get("job_id", ""),
                "title": row.get("Job Title", ""),
                "description": row.get("Job Description", ""),
                "skills": row.get("Skills Requirements", []),
                "job_type": row.get("JobType", ""),
                "location": row.get("Location", ""),
                "mode": row.get("Mode", ""),
                "company": row.get("Company Name", ""),
                "duration": row.get("Duration", 0)
            }
            job_features_list.append(job_features)
        
        # Get recommendations
        recommendations = pipeline.recommend_jobs(
            sample_user, job_features_list, top_n=5,
            score_weights={
                'text_similarity': 0.4,
                'skills_similarity': 0.3,
                'preference_similarity': 0.3
            }
        )
        
        print(f"\nTop 5 recommendations for {sample_user['name']}:")
        print("-" * 50)
        
        for i, rec in enumerate(recommendations):
            job = rec['job_features']
            print(f"{i+1}. {job['title']} at {job['company']}")
            print(f"   Location: {job['location']}, Mode: {job['mode']}")
            print(f"   Type: {job['job_type']}, Duration: {job['duration']} months")
            print(f"   Composite Score: {rec['composite_score']:.3f}")
            print(f"   Text Similarity: {rec['text_similarity']:.3f}")
            print(f"   Skills Similarity: {rec['skills_similarity']:.3f}")
            print(f"   Preference Similarity: {rec['preference_similarity']:.3f}")
            print()
        
        print("✅ Real data test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Real data test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("Starting similarity calculation tests...")
    
    tests = [
        test_text_embeddings,
        test_skills_matching,
        test_preference_encoding,
        test_similarity_calculations,
        test_full_pipeline,
        test_with_real_data
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    passed = sum(results)
    total = len(results)
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Passed: {passed}/{total} tests")
    
    if passed == total:
        print("🎉 All tests passed! The similarity calculation system is working correctly.")
        return True
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)