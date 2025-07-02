#!/usr/bin/env python3
"""
Extract Function for NCI API (Task 4)

This module implements the extraction component of the ETL pipeline
that fetches clinical trial data from the NCI API with proper pagination and rate limiting.
"""

import logging
import requests
import time
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


def extract_trials_from_nci_api(
    page_size: int = 50, 
    limit: Optional[int] = None,
    base_url: str = "https://clinicaltrialsapi.cancer.gov/api/v2/trials"
) -> List[Dict[str, Any]]:
    """
    Extract clinical trial data from the NCI API with pagination and rate limiting.
    
    Args:
        page_size: Number of trials to fetch per API call
        limit: Optional limit on total trials to fetch (for testing)
        base_url: NCI API base URL
        
    Returns:
        List of trial dictionaries
        
    Raises:
        Exception: If API request fails or pagination fails
    """
    logger.info("Starting extraction from NCI API...")
    
    all_trials = []
    from_index = 0
    total_trials = None
    consecutive_errors = 0
    max_consecutive_errors = 5
    
    while total_trials is None or from_index < total_trials:
        # Check if we've hit our limit
        if limit and len(all_trials) >= limit:
            logger.info(f"Reached limit of {limit} trials, stopping extraction")
            break
            
        try:
            logger.info(f"Fetching trials from index {from_index} (page size: {page_size})")
            
            # Make API request with comprehensive field inclusion
            response = requests.get(
                base_url,
                params={
                    "size": page_size, 
                    "from": from_index,
                    "include": [
                        "nct_id", "brief_title", "current_trial_status", 
                        "phase", "study_type", "diseases", "sites", 
                        "brief_summary", "detailed_description", 
                        "eligibility", "primary_purpose", "interventions",
                        "minimum_age", "maximum_age", "gender", 
                        "healthy_volunteers", "enrollment", "lead_sponsor",
                        "collaborators", "keywords", "condition_mesh",
                        "arm_groups", "primary_outcomes", "secondary_outcomes"
                    ]
                },
                timeout=30
            )
            
            if response.status_code != 200:
                logger.error(f"API request failed with status {response.status_code}: {response.text}")
                consecutive_errors += 1
                
                if consecutive_errors >= max_consecutive_errors:
                    raise Exception(f"Too many consecutive API errors ({consecutive_errors})")
                
                # Exponential backoff
                wait_time = min(2 ** consecutive_errors, 60)
                logger.info(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
                continue
            
            # Reset error counter on success
            consecutive_errors = 0
            
            # Parse response
            data = response.json()
            
            if total_trials is None:
                total_trials = data.get("total", 0)
                logger.info(f"Total trials available: {total_trials}")
                
                if limit:
                    total_trials = min(total_trials, limit)
                    logger.info(f"Limited to {total_trials} trials for this run")
            
            # Extract trials from response
            trials = data.get("trials", [])
            all_trials.extend(trials)
            
            logger.info(f"Fetched {len(trials)} trials. Total fetched so far: {len(all_trials)}")
            
            # Update pagination
            from_index += page_size
            
            # Polite delay between API calls (0.5 seconds as specified in PRD)
            time.sleep(0.5)
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error during API request: {e}")
            consecutive_errors += 1
            
            if consecutive_errors >= max_consecutive_errors:
                raise Exception(f"Too many consecutive network errors: {e}")
            
            wait_time = min(2 ** consecutive_errors, 60)
            logger.info(f"Waiting {wait_time} seconds before retry...")
            time.sleep(wait_time)
            
        except Exception as e:
            logger.error(f"Unexpected error during extraction: {e}")
            raise
    
    logger.info(f"Extraction completed. Total trials extracted: {len(all_trials)}")
    return all_trials


def test_nci_api_connection(base_url: str = "https://clinicaltrialsapi.cancer.gov/api/v2/trials") -> bool:
    """
    Test the NCI API connection by fetching a small sample.
    
    Args:
        base_url: NCI API base URL
        
    Returns:
        True if connection successful, False otherwise
    """
    try:
        logger.info("Testing NCI API connection...")
        
        response = requests.get(
            base_url,
            params={"size": 1, "from": 0},
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            total = data.get("total", 0)
            logger.info(f"NCI API connection successful. Total trials available: {total}")
            return True
        else:
            logger.error(f"NCI API connection failed with status {response.status_code}")
            return False
            
    except Exception as e:
        logger.error(f"NCI API connection test failed: {e}")
        return False


if __name__ == "__main__":
    # Test the extraction function
    logging.basicConfig(level=logging.INFO)
    
    # Test connection
    if test_nci_api_connection():
        # Test extraction with a small sample
        trials = extract_trials_from_nci_api(page_size=5, limit=5)
        print(f"Successfully extracted {len(trials)} trials")
        
        if trials:
            print(f"Sample trial ID: {trials[0].get('nct_id', 'N/A')}")
    else:
        print("NCI API connection test failed") 