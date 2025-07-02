#!/usr/bin/env python3
"""
Transform Function for ClinicalTrials.gov API Data

This module implements the transformation component of the ETL pipeline
that parses ClinicalTrials.gov API responses and generates vector embeddings.
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
        logger.info("✅ SentenceTransformer model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"❌ Failed to load SentenceTransformer model: {e}")
        raise


def extract_trial_metadata(trial: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract structured metadata from a ClinicalTrials.gov trial object.
    
    Args:
        trial: Raw trial data from ClinicalTrials.gov API
        
    Returns:
        Dictionary containing structured trial metadata
    """
    protocol = trial.get('protocolSection', {})
    
    # Identification module
    identification = protocol.get('identificationModule', {})
    nct_id = identification.get('nctId', '')
    brief_title = identification.get('briefTitle', '')
    official_title = identification.get('officialTitle', '')
    
    # Use brief title as primary, fall back to official title
    title = brief_title or official_title
    
    # Status module
    status_module = protocol.get('statusModule', {})
    overall_status = status_module.get('overallStatus', '')
    last_update_date = status_module.get('lastUpdateSubmitDate', '')
    
    # Design module
    design_module = protocol.get('designModule', {})
    study_type = design_module.get('studyType', '')
    phases = design_module.get('phases', [])
    phase = phases[0] if phases else ''
    
    # Conditions module
    conditions_module = protocol.get('conditionsModule', {})
    conditions = conditions_module.get('conditions', [])
    conditions_str = '; '.join(conditions) if conditions else ''
    
    # Eligibility module
    eligibility_module = protocol.get('eligibilityModule', {})
    eligibility_criteria = eligibility_module.get('eligibilityCriteria', '')
    
    # Contacts and locations module
    contacts_locations = protocol.get('contactsLocationsModule', {})
    locations = contacts_locations.get('locations', [])
    location_names = []
    for location in locations:
        facility = location.get('facility', '')
        city = location.get('city', '')
        state = location.get('state', '')
        if facility:
            location_str = facility
            if city and state:
                location_str += f" ({city}, {state})"
            elif city:
                location_str += f" ({city})"
            location_names.append(location_str)
    
    locations_str = '; '.join(location_names[:5])  # Limit to first 5 locations
    
    # Description module
    description_module = protocol.get('descriptionModule', {})
    brief_summary = description_module.get('briefSummary', '')
    detailed_description = description_module.get('detailedDescription', '')
    
    # Sponsor/collaborators module
    sponsor_module = protocol.get('sponsorCollaboratorsModule', {})
    lead_sponsor = sponsor_module.get('leadSponsor', {})
    sponsor_name = lead_sponsor.get('name', '')
    
    # Derived section (additional info)
    derived_section = trial.get('derivedSection', {})
    misc_info = derived_section.get('miscInfoModule', {})
    
    # Enrollment info
    enrollment_info = design_module.get('enrollmentInfo', {})
    enrollment_count = enrollment_info.get('count', 0)
    
    return {
        'nct_id': nct_id,
        'title': title,
        'status': overall_status,
        'phase': phase,
        'study_type': study_type,
        'conditions': conditions_str,
        'locations': locations_str,
        'brief_summary': brief_summary,
        'detailed_description': detailed_description,
        'eligibility_criteria': eligibility_criteria,
        'sponsor': sponsor_name,
        'enrollment_count': enrollment_count,
        'last_updated_date': last_update_date
    }


def create_embedding_text(metadata: Dict[str, Any]) -> str:
    """
    Create optimized text for embedding generation.
    
    Args:
        metadata: Structured trial metadata
        
    Returns:
        Text string optimized for vector embedding
    """
    # Prioritize eligibility criteria and brief summary for embeddings
    text_parts = []
    
    # Add title for context
    if metadata.get('title'):
        text_parts.append(f"Title: {metadata['title']}")
    
    # Add conditions
    if metadata.get('conditions'):
        text_parts.append(f"Conditions: {metadata['conditions']}")
    
    # Add brief summary
    if metadata.get('brief_summary'):
        text_parts.append(f"Summary: {metadata['brief_summary']}")
    
    # Add eligibility criteria (most important for matching)
    if metadata.get('eligibility_criteria'):
        text_parts.append(f"Eligibility: {metadata['eligibility_criteria']}")
    
    # Add phase and status for additional context
    if metadata.get('phase'):
        text_parts.append(f"Phase: {metadata['phase']}")
    
    if metadata.get('status'):
        text_parts.append(f"Status: {metadata['status']}")
    
    # Join all parts
    embedding_text = " | ".join(text_parts)
    
    # Truncate if too long (typical models have token limits)
    if len(embedding_text) > 8000:  # Conservative limit
        embedding_text = embedding_text[:8000] + "..."
    
    return embedding_text


def transform_trial_data(
    trials: List[Dict[str, Any]], 
    model: SentenceTransformer
) -> List[Tuple[Dict[str, Any], List[float]]]:
    """
    Transform raw trial data into structured metadata and vector embeddings.
    
    Args:
        trials: List of raw trial data from ClinicalTrials.gov API
        model: SentenceTransformer model for generating embeddings
        
    Returns:
        List of tuples containing (metadata_dict, embedding_vector)
    """
    logger.info(f"Transforming {len(trials)} trials...")
    
    transformed_data = []
    failed_count = 0
    
    for i, trial in enumerate(trials):
        try:
            # Extract structured metadata
            metadata = extract_trial_metadata(trial)
            
            # Validate essential fields
            if not metadata.get('nct_id'):
                logger.warning(f"Trial {i} missing NCT ID, skipping")
                failed_count += 1
                continue
            
            if not metadata.get('title'):
                logger.warning(f"Trial {metadata['nct_id']} missing title, skipping")
                failed_count += 1
                continue
            
            # Create text for embedding
            embedding_text = create_embedding_text(metadata)
            
            if not embedding_text.strip():
                logger.warning(f"Trial {metadata['nct_id']} has no meaningful text for embedding, skipping")
                failed_count += 1
                continue
            
            # Generate embedding
            embedding = model.encode(embedding_text).tolist()
            
            # Add to results
            transformed_data.append((metadata, embedding))
            
            if (i + 1) % 50 == 0:
                logger.info(f"Processed {i + 1}/{len(trials)} trials...")
                
        except Exception as e:
            failed_count += 1
            trial_id = trial.get('protocolSection', {}).get('identificationModule', {}).get('nctId', f'trial_{i}')
            logger.error(f"Error transforming trial {trial_id}: {e}")
            continue
    
    success_count = len(transformed_data)
    logger.info(f"Transformation completed: {success_count} successful, {failed_count} failed")
    
    return transformed_data


def validate_transformed_data(transformed_data: List[Tuple[Dict[str, Any], List[float]]]) -> bool:
    """
    Validate the transformed data structure and content.
    
    Args:
        transformed_data: List of (metadata, embedding) tuples
        
    Returns:
        True if validation passes, False otherwise
    """
    if not transformed_data:
        logger.error("No transformed data to validate")
        return False
    
    required_metadata_fields = ['nct_id', 'title', 'status', 'eligibility_criteria']
    
    for i, (metadata, embedding) in enumerate(transformed_data[:5]):  # Check first 5
        # Validate metadata
        for field in required_metadata_fields:
            if field not in metadata:
                logger.error(f"Transformed data {i} missing metadata field: {field}")
                return False
        
        # Validate embedding
        if not isinstance(embedding, list):
            logger.error(f"Transformed data {i} embedding is not a list")
            return False
        
        if len(embedding) == 0:
            logger.error(f"Transformed data {i} has empty embedding")
            return False
        
        # Check that all embedding values are numbers
        if not all(isinstance(x, (int, float)) for x in embedding):
            logger.error(f"Transformed data {i} embedding contains non-numeric values")
            return False
    
    logger.info(f"✅ Validation passed for {len(transformed_data)} transformed trials")
    return True


if __name__ == "__main__":
    # Test the transformation function
    logging.basicConfig(level=logging.INFO)
    
    # Import the extraction function to get test data
    from extract_clinicaltrials_gov import extract_trials_from_clinicaltrials_gov, test_clinicaltrials_gov_connection
    
    # Test with a small sample
    if test_clinicaltrials_gov_connection():
        print("Extracting test data...")
        test_trials = extract_trials_from_clinicaltrials_gov(page_size=3, limit=3)
        
        if test_trials:
            print("Initializing transformer...")
            model = initialize_sentence_transformer()
            
            print("Transforming data...")
            transformed = transform_trial_data(test_trials, model)
            
            if validate_transformed_data(transformed):
                print(f"✅ Successfully transformed {len(transformed)} trials")
                
                # Print sample metadata
                if transformed:
                    sample_metadata, sample_embedding = transformed[0]
                    print(f"\nSample trial: {sample_metadata['nct_id']}")
                    print(f"Title: {sample_metadata['title']}")
                    print(f"Embedding dimensions: {len(sample_embedding)}")
            else:
                print("❌ Transformation validation failed")
        else:
            print("❌ No test data extracted")
    else:
        print("❌ API connection test failed") 