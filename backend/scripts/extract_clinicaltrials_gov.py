#!/usr/bin/env python3
"""
Extract Function for ClinicalTrials.gov API

This module implements the extraction component of the ETL pipeline
that fetches clinical trial data from the official ClinicalTrials.gov API.
"""

import logging
import requests
import time
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


def test_clinicaltrials_gov_connection(base_url: str = "https://clinicaltrials.gov/api/v2/studies") -> bool:
    """
    Test connectivity to the ClinicalTrials.gov API.
    
    Args:
        base_url: Base URL for the ClinicalTrials.gov API
        
    Returns:
        True if connection successful, False otherwise
    """
    try:
        logger.info("Testing ClinicalTrials.gov API connection...")
        
        response = requests.get(
            f"{base_url}?pageSize=1&format=json",
            timeout=10,
            headers={
                'User-Agent': 'Clinical-Trials-ETL-Pipeline/1.0 (Research Purpose)',
                'Accept': 'application/json'
            }
        )
        
        if response.status_code == 200:
            data = response.json()
            if 'studies' in data:
                logger.info("✅ ClinicalTrials.gov API connection successful")
                return True
            else:
                logger.error("❌ Unexpected response format from API")
                return False
        else:
            logger.error(f"❌ API returned status {response.status_code}: {response.text}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error testing API connection: {e}")
        return False


def extract_trials_from_clinicaltrials_gov(
    page_size: int = 100, 
    limit: Optional[int] = None,
    base_url: str = "https://clinicaltrials.gov/api/v2/studies",
    condition: str = "cancer"
) -> List[Dict[str, Any]]:
    """
    Extract clinical trial data from ClinicalTrials.gov API with pagination.
    
    Args:
        page_size: Number of trials to fetch per API call
        limit: Optional limit on total trials to fetch (for testing)
        base_url: Base URL for the ClinicalTrials.gov API
        condition: Condition to search for (default: cancer)
        
    Returns:
        List of trial dictionaries
    """
    logger.info("Starting extraction from ClinicalTrials.gov API...")
    
    trials = []
    next_page_token = None
    consecutive_errors = 0
    max_consecutive_errors = 5
    
    # Request headers
    headers = {
        'User-Agent': 'Clinical-Trials-ETL-Pipeline/1.0 (Research Purpose)',
        'Accept': 'application/json'
    }
    
    while True:
        try:
            # Build URL with parameters
            params = {
                'pageSize': min(page_size, 1000),  # API max is 1000
                'format': 'json',
                'query.cond': condition  # Search for condition
            }
            
            if next_page_token:
                params['pageToken'] = next_page_token
            
            logger.info(f"Fetching page with {params['pageSize']} trials (token: {next_page_token or 'first page'})")
            
            # Make API request
            response = requests.get(
                base_url,
                params=params,
                headers=headers,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                page_trials = data.get('studies', [])
                
                if not page_trials:
                    logger.info("No more trials found, ending extraction")
                    break
                
                trials.extend(page_trials)
                consecutive_errors = 0  # Reset error counter on success
                
                logger.info(f"✅ Fetched {len(page_trials)} trials. Total: {len(trials)}")
                
                # Check if we've hit our limit
                if limit and len(trials) >= limit:
                    trials = trials[:limit]
                    logger.info(f"Reached limit of {limit} trials")
                    break
                
                # Get next page token
                next_page_token = data.get('nextPageToken')
                if not next_page_token:
                    logger.info("No more pages available")
                    break
                
                # Rate limiting - be polite to the API
                time.sleep(0.5)
                
            else:
                consecutive_errors += 1
                logger.error(f"API request failed with status {response.status_code}: {response.text}")
                
                if consecutive_errors >= max_consecutive_errors:
                    logger.error(f"Too many consecutive API errors ({consecutive_errors})")
                    raise Exception(f"Too many consecutive API errors ({consecutive_errors})")
                
                # Exponential backoff
                wait_time = 2 ** consecutive_errors
                logger.info(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
                
        except Exception as e:
            consecutive_errors += 1
            logger.error(f"Error during API request: {e}")
            
            if consecutive_errors >= max_consecutive_errors:
                logger.error(f"Unexpected error during extraction: {e}")
                raise
            
            # Wait before retry
            wait_time = 2 ** consecutive_errors
            logger.info(f"Waiting {wait_time} seconds before retry...")
            time.sleep(wait_time)
    
    logger.info(f"Extraction completed. Total trials fetched: {len(trials)}")
    return trials


def validate_trial_data(trials: List[Dict[str, Any]]) -> bool:
    """
    Validate that the extracted trial data has the expected structure.
    
    Args:
        trials: List of trial dictionaries
        
    Returns:
        True if data is valid, False otherwise
    """
    if not trials:
        logger.error("No trials to validate")
        return False
    
    required_fields = ['protocolSection']
    
    for i, trial in enumerate(trials[:5]):  # Check first 5 trials
        for field in required_fields:
            if field not in trial:
                logger.error(f"Trial {i} missing required field: {field}")
                return False
        
        protocol = trial.get('protocolSection', {})
        identification = protocol.get('identificationModule', {})
        
        if not identification.get('nctId'):
            logger.error(f"Trial {i} missing NCT ID")
            return False
    
    logger.info(f"✅ Validation passed for {len(trials)} trials")
    return True


if __name__ == "__main__":
    # Test the extraction function
    logging.basicConfig(level=logging.INFO)
    
    # Test connection
    if test_clinicaltrials_gov_connection():
        # Extract a small sample
        test_trials = extract_trials_from_clinicaltrials_gov(page_size=5, limit=5)
        
        if validate_trial_data(test_trials):
            print(f"✅ Successfully extracted and validated {len(test_trials)} trials")
            
            # Print first trial for inspection
            if test_trials:
                first_trial = test_trials[0]
                protocol = first_trial.get('protocolSection', {})
                identification = protocol.get('identificationModule', {})
                
                print(f"\nFirst trial: {identification.get('nctId', 'NO_ID')}")
                print(f"Title: {identification.get('briefTitle', 'NO_TITLE')}")
                print(f"Status: {protocol.get('statusModule', {}).get('overallStatus', 'NO_STATUS')}")
        else:
            print("❌ Data validation failed")
    else:
        print("❌ API connection test failed") 