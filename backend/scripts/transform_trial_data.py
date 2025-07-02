#!/usr/bin/env python3
"""
Transform Function for Trial Data (Task 5)

This module implements the transformation component of the ETL pipeline
that parses JSON responses, extracts structured metadata, and generates vector embeddings.
"""

import json
import logging
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


def initialize_sentence_transformer(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    """
    Initialize the SentenceTransformer model for generating embeddings.
    
    Args:
        model_name: Name of the SentenceTransformer model to use
        
    Returns:
        SentenceTransformer model instance
    """
    logger.info(f"Loading SentenceTransformer model: {model_name}")
    try:
        model = SentenceTransformer(model_name)
        logger.info("SentenceTransformer model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Failed to load SentenceTransformer model: {e}")
        raise


def extract_trial_metadata(trial: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract structured metadata from a trial JSON object.
    
    Args:
        trial: Raw trial data from NCI API
        
    Returns:
        Dictionary of structured metadata
    """
    # Helper function to safely get nested values
    def safe_get(obj, *keys, default=""):
        for key in keys:
            if isinstance(obj, dict) and key in obj:
                obj = obj[key]
            else:
                return default
        return obj if obj is not None else default
    
    # Helper function to convert lists to JSON strings
    def list_to_json(lst):
        if isinstance(lst, list):
            return json.dumps(lst) if lst else ""
        return str(lst) if lst else ""
    
    # Extract eligibility criteria
    eligibility = trial.get("eligibility", {})
    eligibility_text = eligibility.get("criteria", {})
    if isinstance(eligibility_text, dict):
        eligibility_text = eligibility_text.get("textblock", "")
    
    # Extract age restrictions
    min_age = safe_get(eligibility, "minimum_age")
    max_age = safe_get(eligibility, "maximum_age")
    
    # Extract phase information
    phase_obj = trial.get("phase", {})
    phase = phase_obj.get("phase", "") if isinstance(phase_obj, dict) else str(phase_obj)
    
    # Extract sponsor information
    sponsor_obj = trial.get("lead_sponsor", {})
    sponsor = sponsor_obj.get("name", "") if isinstance(sponsor_obj, dict) else str(sponsor_obj)
    
    # Extract enrollment count
    enrollment_obj = trial.get("enrollment", {})
    enrollment_count = enrollment_obj.get("value", 0) if isinstance(enrollment_obj, dict) else 0
    
    metadata = {
        "nct_id": trial.get("nct_id", ""),
        "title": trial.get("brief_title", ""),
        "status": trial.get("current_trial_status", ""),
        "phase": phase,
        "study_type": trial.get("study_type", ""),
        "conditions": list_to_json(trial.get("diseases", [])),
        "locations": list_to_json(trial.get("sites", [])),
        "last_updated_date": trial.get("last_update_posted", ""),
        "brief_summary": trial.get("brief_summary", ""),
        "detailed_description": trial.get("detailed_description", ""),
        "eligibility_criteria": eligibility_text,
        "primary_purpose": safe_get(trial, "primary_purpose", "phase"),
        "intervention_type": list_to_json(trial.get("interventions", [])),
        "minimum_age": min_age,
        "maximum_age": max_age,
        "gender": safe_get(eligibility, "gender"),
        "healthy_volunteers": safe_get(eligibility, "healthy_volunteers"),
        "enrollment_count": enrollment_count,
        "sponsor": sponsor,
        "collaborators": list_to_json(trial.get("collaborators", [])),
        "keywords": list_to_json(trial.get("keywords", [])),
        "mesh_terms": list_to_json(trial.get("condition_mesh", [])),
        "arm_groups": list_to_json(trial.get("arm_groups", [])),
        "outcomes": list_to_json(trial.get("primary_outcomes", []) + trial.get("secondary_outcomes", [])),
        "inclusion_criteria": "",  # Will be parsed from eligibility_criteria if needed
        "exclusion_criteria": ""   # Will be parsed from eligibility_criteria if needed
    }
    
    # Ensure nct_id is not empty
    if not metadata["nct_id"]:
        raise ValueError("Trial missing required NCT ID")
    
    return metadata


def generate_embedding(text: str, model: SentenceTransformer) -> List[float]:
    """
    Generate vector embedding for given text using SentenceTransformer.
    
    Args:
        text: Text to generate embedding for
        model: Initialized SentenceTransformer model
        
    Returns:
        List of float values representing the embedding
    """
    if not text or not text.strip():
        # Use a default text for empty content
        text = "No content available"
    
    try:
        embedding = model.encode(text.strip()).tolist()
        return embedding
    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        raise


def create_embedding_text(metadata: Dict[str, Any]) -> str:
    """
    Create text for embedding generation by combining relevant fields.
    
    Args:
        metadata: Extracted metadata dictionary
        
    Returns:
        Combined text string for embedding
    """
    # Prioritize eligibility criteria, then summary, then title
    text_components = []
    
    # Primary: eligibility criteria
    if metadata.get("eligibility_criteria", "").strip():
        text_components.append(metadata["eligibility_criteria"].strip())
    
    # Secondary: brief summary
    if metadata.get("brief_summary", "").strip():
        text_components.append(metadata["brief_summary"].strip())
    
    # Tertiary: title
    if metadata.get("title", "").strip():
        text_components.append(metadata["title"].strip())
    
    # Additional context: conditions and interventions
    if metadata.get("conditions", "").strip():
        conditions = json.loads(metadata["conditions"]) if metadata["conditions"] else []
        if conditions:
            text_components.append(f"Conditions: {', '.join([str(c) for c in conditions])}")
    
    if metadata.get("intervention_type", "").strip():
        interventions = json.loads(metadata["intervention_type"]) if metadata["intervention_type"] else []
        if interventions:
            text_components.append(f"Interventions: {', '.join([str(i) for i in interventions])}")
    
    # Combine all components
    combined_text = " | ".join(text_components)
    
    # Limit text length for embedding (models typically have token limits)
    max_length = 5000
    if len(combined_text) > max_length:
        combined_text = combined_text[:max_length]
        logger.debug(f"Truncated embedding text to {max_length} characters")
    
    return combined_text if combined_text.strip() else "No content available"


def transform_trial_data(
    trials: List[Dict[str, Any]], 
    model: SentenceTransformer
) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    """
    Transform trial data by extracting metadata and generating embeddings.
    
    Args:
        trials: List of raw trial data from API
        model: Initialized SentenceTransformer model
        
    Returns:
        List of tuples containing (metadata_dict, vector_data_dict)
    """
    logger.info(f"Starting transformation of {len(trials)} trials...")
    
    transformed_data = []
    skipped_count = 0
    
    for i, trial in enumerate(trials):
        try:
            # Extract structured metadata
            metadata = extract_trial_metadata(trial)
            
            # Create text for embedding generation
            embedding_text = create_embedding_text(metadata)
            
            if not embedding_text.strip() or embedding_text == "No content available":
                logger.warning(f"Trial {metadata['nct_id']} has no meaningful text for embedding")
                # Still continue, but with minimal text
            
            # Generate embedding
            embedding = generate_embedding(embedding_text, model)
            
            # Create vector data structure
            vector_data = {
                "nct_id": metadata["nct_id"],
                "embedding": embedding,
                "criteria_text": embedding_text[:5000]  # Limit text length for storage
            }
            
            transformed_data.append((metadata, vector_data))
            
            # Log progress every 100 trials
            if (i + 1) % 100 == 0:
                logger.info(f"Transformed {i + 1}/{len(trials)} trials...")
                
        except Exception as e:
            logger.error(f"Error transforming trial {trial.get('nct_id', 'unknown')}: {str(e)}")
            skipped_count += 1
            continue
    
    logger.info(f"Transformation completed. Successfully processed: {len(transformed_data)}, Skipped: {skipped_count}")
    return transformed_data


def validate_transformed_data(transformed_data: List[Tuple[Dict[str, Any], Dict[str, Any]]]) -> bool:
    """
    Validate the transformed data to ensure it meets requirements.
    
    Args:
        transformed_data: List of (metadata, vector_data) tuples
        
    Returns:
        True if validation passes, False otherwise
    """
    logger.info("Validating transformed data...")
    
    if not transformed_data:
        logger.error("No transformed data to validate")
        return False
    
    errors = []
    
    for i, (metadata, vector_data) in enumerate(transformed_data[:10]):  # Check first 10 records
        # Check metadata
        if not metadata.get("nct_id"):
            errors.append(f"Record {i}: Missing NCT ID")
        
        if not metadata.get("title"):
            errors.append(f"Record {i}: Missing title")
        
        # Check vector data
        if not vector_data.get("nct_id"):
            errors.append(f"Record {i}: Vector data missing NCT ID")
        
        if not vector_data.get("embedding"):
            errors.append(f"Record {i}: Missing embedding")
        
        embedding = vector_data.get("embedding", [])
        if not isinstance(embedding, list) or len(embedding) == 0:
            errors.append(f"Record {i}: Invalid embedding format")
        
        # Check NCT ID consistency
        if metadata.get("nct_id") != vector_data.get("nct_id"):
            errors.append(f"Record {i}: NCT ID mismatch between metadata and vector data")
    
    if errors:
        logger.error(f"Validation failed with {len(errors)} errors:")
        for error in errors[:5]:  # Show first 5 errors
            logger.error(f"  - {error}")
        return False
    
    logger.info("Validation passed successfully")
    return True


if __name__ == "__main__":
    # Test the transformation functions
    logging.basicConfig(level=logging.INFO)
    
    # Sample trial data for testing
    sample_trial = {
        "nct_id": "NCT12345678",
        "brief_title": "Test Clinical Trial",
        "current_trial_status": "Recruiting",
        "phase": {"phase": "Phase 2"},
        "study_type": "Interventional",
        "diseases": [{"name": "Cancer"}],
        "brief_summary": "This is a test trial for cancer treatment",
        "eligibility": {
            "criteria": {
                "textblock": "Inclusion: Adults 18+. Exclusion: Pregnant women."
            }
        }
    }
    
    # Test metadata extraction
    try:
        metadata = extract_trial_metadata(sample_trial)
        print(f"Extracted metadata for trial: {metadata['nct_id']}")
        
        # Test embedding generation
        model = initialize_sentence_transformer()
        embedding_text = create_embedding_text(metadata)
        embedding = generate_embedding(embedding_text, model)
        
        print(f"Generated embedding with {len(embedding)} dimensions")
        print(f"Embedding text: {embedding_text[:100]}...")
        
    except Exception as e:
        print(f"Test failed: {e}") 