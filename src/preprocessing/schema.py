"""Data schema definitions and validation."""

from pydantic import BaseModel, Field, field_validator
from typing import List

class JobPostingSchema(BaseModel):
    """Schema for validating job posting data."""
    job_id: str = Field(..., pattern=r'^[a-f0-9]{8}$')
    company: str = Field(..., min_length=1)
    title: str = Field(..., min_length=1)
    skills: List[str]
    location: str
    mode: str
    description: str
    special_requirements: str = Field("")
    duration: int = Field(..., gt=0)  # Changed to required field
    job_type: str = Field(..., min_length=1)  # Changed to required field
    
    @field_validator('skills', mode='before')
    def parse_skills_string(cls, v):
        """Convert string representation of list to actual list."""
        if isinstance(v, str):
            # Remove brackets and quotes, then split
            v = v.strip('[]').replace("'", "").replace('"', '').split(', ')
            # Remove empty strings
            v = [skill.strip() for skill in v if skill.strip()]
        return v
    
    @field_validator('mode')
    def validate_mode(cls, v):
        """Validate work mode is one of the expected values."""
        valid_modes = ['Hybrid', 'Offline', 'Online']
        if v not in valid_modes:
            # Try to standardize
            v_lower = v.lower()
            if 'hybrid' in v_lower:
                return 'Hybrid'
            elif any(word in v_lower for word in ['office', 'onsite', 'on-site', 'offline']):
                return 'Offline'
            elif any(word in v_lower for word in ['online', 'remote', 'work from home', 'wfh']):
                return 'Online'
            else:
                raise ValueError(f"Invalid mode: {v}. Expected one of {valid_modes}")
        return v

if __name__ == "__main__":
    # Example usage
    example_data = {
        "job_id": "1a2b3c4d",
        "company": "TCS",
        "title": "Software Engineer",
        "skills": "['Python', 'Django', 'REST']",
        "location": "Delhi, India",
        "mode": "Online",
        "description": "Develop and maintain web applications.",
        "duration": 12,
        "job_type": "SDE"
    }
    
    job_posting = JobPostingSchema(**example_data)
    print(job_posting)