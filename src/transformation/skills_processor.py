"""
Skills processing for matching and scoring.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Set, Tuple, Optional
import re
from rapidfuzz import process, fuzz
from utils.logger import BaseLogger

class SkillsNormalizer:
    """Class for normalizing skills across different representations."""
    
    def __init__(self, normalization_rules: Optional[Dict[str, str]] = None):
        """
        Initialize the skills normalizer.
        
        Args:
            normalization_rules: Dictionary of normalization rules
        """
        self.normalization_rules = normalization_rules or self._get_default_rules()
        self.skill_synonyms = self._get_skill_synonyms()
    
    def _get_default_rules(self) -> Dict[str, str]:
        """Get default normalization rules."""
        return {
            'javascript': 'JavaScript',
            'python': 'Python',
            'java': 'Java',
            'c++': 'C++',
            'c#': 'C#',
            'react': 'React',
            'react.js': 'React',
            'reactjs': 'React',
            'node': 'Node.js',
            'nodejs': 'Node.js',
            'sql': 'SQL',
            'nosql': 'NoSQL',
            'mongodb': 'MongoDB',
            'aws': 'AWS',
            'azure': 'Azure',
            'gcp': 'GCP',
            'docker': 'Docker',
            'kubernetes': 'Kubernetes',
            'jenkins': 'Jenkins',
            'git': 'Git',
            'ci/cd': 'CI/CD',
            'ml': 'Machine Learning',
            'ai': 'Artificial Intelligence',
            'nlp': 'Natural Language Processing',
            'cv': 'Computer Vision',
            'ds': 'Data Science'
        }
    
    def _get_skill_synonyms(self) -> Dict[str, List[str]]:
        """Get skill synonyms for fuzzy matching."""
        return {
            'JavaScript': ['js', 'javascript', 'es6', 'es2015'],
            'Python': ['python', 'py'],
            'React': ['react', 'react.js', 'reactjs'],
            'Node.js': ['node', 'nodejs'],
            'SQL': ['sql', 'mysql', 'postgresql', 'postgres'],
            'AWS': ['aws', 'amazon web services'],
            'Docker': ['docker', 'docker container'],
            'Kubernetes': ['kubernetes', 'k8s'],
            'Machine Learning': ['ml', 'machine learning'],
            'Data Science': ['ds', 'data science']
        }
    
    def normalize_skill(self, skill: str) -> str:
        """
        Normalize a skill name.
        
        Args:
            skill: Skill to normalize
            
        Returns:
            Normalized skill name
        """
        if not skill or not isinstance(skill, str):
            return ""
        
        # Convert to lowercase and strip whitespace
        skill = skill.lower().strip()
        
        # Apply normalization rules
        if skill in self.normalization_rules:
            return self.normalization_rules[skill]
        
        # Capitalize if no specific rule exists
        return skill.capitalize()
    
    def normalize_skills(self, skills: List[str]) -> List[str]:
        """
        Normalize a list of skills.
        
        Args:
            skills: List of skills to normalize
            
        Returns:
            List of normalized skills
        """
        if not skills:
            return []
        
        normalized = [self.normalize_skill(skill) for skill in skills]
        # Remove duplicates and empty strings
        return list(set([skill for skill in normalized if skill]))

class SkillsMatcher:
    """Class for matching skills between jobs and users."""
    
    def __init__(self, normalizer: SkillsNormalizer, logger: Optional[BaseLogger] = None):
        """
        Initialize the skills matcher.
        
        Args:
            normalizer: Skills normalizer instance
            logger: Logger instance
        """
        self.normalizer = normalizer
        self.logger = logger or BaseLogger("SkillsMatcher")
    
    def exact_match(self, user_skills: List[str], job_skills: List[str]) -> Tuple[List[str], List[str]]:
        """
        Find exact matches between user skills and job skills.
        
        Args:
            user_skills: List of user skills
            job_skills: List of job skills
            
        Returns:
            Tuple of (matched_skills, missing_skills)
        """
        if not user_skills or not job_skills:
            return [], job_skills if job_skills else []
        
        # Normalize both sets of skills
        norm_user_skills = set(self.normalizer.normalize_skills(user_skills))
        norm_job_skills = set(self.normalizer.normalize_skills(job_skills))
        
        # Find matches
        matched_skills = list(norm_user_skills.intersection(norm_job_skills))
        missing_skills = list(norm_job_skills.difference(norm_user_skills))
        
        return matched_skills, missing_skills
    
    def fuzzy_match(self, user_skills: List[str], job_skills: List[str], 
                   threshold: float = 80.0) -> Tuple[List[str], List[str], List[Tuple[str, str, float]]]:
        """
        Find fuzzy matches between user skills and job skills.
        
        Args:
            user_skills: List of user skills
            job_skills: List of job skills
            threshold: Similarity threshold for matching (0-100)
            
        Returns:
            Tuple of (matched_skills, missing_skills, fuzzy_matches)
        """
        if not user_skills or not job_skills:
            return [], job_skills if job_skills else [], []
        
        # Normalize skills
        norm_user_skills = self.normalizer.normalize_skills(user_skills)
        norm_job_skills = self.normalizer.normalize_skills(job_skills)
        
        matched_skills = []
        missing_skills = norm_job_skills.copy()
        fuzzy_matches = []  # List of (user_skill, job_skill, similarity)
        
        for job_skill in norm_job_skills:
            # Try exact match first
            if job_skill in norm_user_skills:
                matched_skills.append(job_skill)
                if job_skill in missing_skills:
                    missing_skills.remove(job_skill)
                continue
            
            # Try fuzzy match - FIXED: Handle the return value correctly
            result = process.extractOne(
                job_skill, norm_user_skills, scorer=fuzz.token_set_ratio
            )
            
            # Extract the match and score from the result
            if result:
                best_match, score, index = result  # Unpack all three values
                
                if score >= threshold:
                    matched_skills.append(job_skill)
                    fuzzy_matches.append((best_match, job_skill, score))
                    if job_skill in missing_skills:
                        missing_skills.remove(job_skill)
        
        return matched_skills, missing_skills, fuzzy_matches
    
    def calculate_skills_score(self, user_skills: List[str], job_skills: List[str], 
                              use_fuzzy: bool = True, fuzzy_threshold: float = 80.0) -> float:
        """
        Calculate skills matching score between user and job.
        
        Args:
            user_skills: List of user skills
            job_skills: List of job skills
            use_fuzzy: Whether to use fuzzy matching
            fuzzy_threshold: Threshold for fuzzy matching
            
        Returns:
            Skills matching score (0-1)
        """
        if not job_skills:
            return 1.0  # No skills required means perfect match
        
        if use_fuzzy:
            matched_skills, missing_skills, _ = self.fuzzy_match(
                user_skills, job_skills, fuzzy_threshold
            )
        else:
            matched_skills, missing_skills = self.exact_match(user_skills, job_skills)
        
        return len(matched_skills) / len(job_skills)

class SkillsProcessor:
    """Main skills processor class that combines normalization and matching."""
    
    def __init__(self, logger: Optional[BaseLogger] = None):
        """
        Initialize the skills processor.
        
        Args:
            logger: Logger instance
        """
        self.logger = logger or BaseLogger("SkillsProcessor")
        self.normalizer = SkillsNormalizer()
        self.matcher = SkillsMatcher(self.normalizer, self.logger)
    
    def process_skills_matching(self, user_skills: List[str], job_skills: List[str], 
                               use_fuzzy: bool = True) -> Dict[str, any]:
        """
        Process skills matching between user and job.
        
        Args:
            user_skills: List of user skills
            job_skills: List of job skills
            use_fuzzy: Whether to use fuzzy matching
            
        Returns:
            Dictionary with matching results
        """
        if use_fuzzy:
            matched_skills, missing_skills, fuzzy_matches = self.matcher.fuzzy_match(
                user_skills, job_skills
            )
        else:
            matched_skills, missing_skills = self.matcher.exact_match(user_skills, job_skills)
            fuzzy_matches = []
        
        score = self.matcher.calculate_skills_score(user_skills, job_skills, use_fuzzy)
        
        return {
            "matched_skills": matched_skills,
            "missing_skills": missing_skills,
            "fuzzy_matches": fuzzy_matches,
            "score": score,
            "required_skills_count": len(job_skills),
            "matched_skills_count": len(matched_skills)
        }