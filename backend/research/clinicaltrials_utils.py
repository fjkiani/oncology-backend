import requests
import asyncio
import logging
from typing import Dict, Any, List, Optional, Iterator
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ClinicalTrials.gov API base URL
API_BASE_URL = "https://clinicaltrials.gov/api/v2/studies"

def parse_study(study: Dict[str, Any]) -> Dict[str, Any]:
    """
    Helper function to extract relevant fields from a study object.
    Enhanced to include trial phase and eligibility criteria.
    """
    protocol = study.get('protocolSection', {})
    
    # Identifiers
    ids = protocol.get('identificationModule', {})
    nct_id = ids.get('nctId', 'N/A')
    brief_title = ids.get('briefTitle', 'No Title Available')
    
    # Status and Phase
    status_module = protocol.get('statusModule', {})
    overall_status = status_module.get('overallStatus', 'Unknown')
    
    design_module = protocol.get('designModule', {})
    phases = design_module.get('phases', ['Not Applicable']) # It's a list

    # Description
    desc = protocol.get('descriptionModule', {})
    brief_summary = desc.get('briefSummary', 'No Summary Available')

    # Conditions
    cond_module = protocol.get('conditionsModule', {})
    conditions = cond_module.get('conditions', [])
    
    # Interventions
    arms_interv_module = protocol.get('armsInterventionsModule', {})
    interventions = [
        interv.get('name', 'Unknown Intervention') 
        for interv in arms_interv_module.get('interventions', [])
    ]
        
    # Eligibility
    eligibility_module = protocol.get('eligibilityModule', {})
    eligibility_criteria = eligibility_module.get('eligibilityCriteria', 'No Criteria Provided')

    return {
        "id": nct_id,
        "title": brief_title,
        "status": overall_status,
        "phases": phases,
        "summary": brief_summary,
        "conditions": conditions,
        "interventions": interventions,
        "eligibility_criteria": eligibility_criteria,
        "source": "ClinicalTrials.gov"
    }

async def fetch_all_trials_generator(
    criteria: Dict[str, Any], 
    page_size: int = 100,
    max_pages: Optional[int] = None
) -> Iterator[List[Dict[str, Any]]]:
    """
    A generator that fetches all clinical trials from ClinicalTrials.gov page by page.

    This is memory-efficient as it yields one page of parsed studies at a time
    instead of loading all studies into memory.

    Args:
        criteria: A dictionary containing search parameters for the API.
        page_size: The number of results to fetch per API call. Max is 1000.
        max_pages: A limit on the number of pages to fetch, for testing purposes.
                   If None, it will fetch all available pages.

    Yields:
        A list of parsed study dictionaries for each page.
    """
    page_token: Optional[str] = None
    pages_fetched = 0

    logger.info(f"Starting trial fetch with criteria: {criteria}, page size: {page_size}")

    while True:
        params = {
            "format": "json",
            "countTotal": "true",
            "pageSize": page_size,
            **criteria  # Unpack criteria into params
        }
        
        if page_token:
            params["pageToken"] = page_token
                
        logger.info(f"Fetching page {pages_fetched + 1}...")
        
        try:
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(
                None, 
                lambda: requests.get(API_BASE_URL, params=params, timeout=30)
            )
            response.raise_for_status()
            data = response.json()
            
            if pages_fetched == 0:
                total_count = data.get('totalCount', 0)
                logger.info(f"API reports a total of {total_count} studies available.")
            
            studies = data.get('studies', [])
            if not studies:
                logger.info("No more studies found on this page. Ending fetch.")
                break

            parsed_studies = [parse_study(study) for study in studies]
            yield parsed_studies
            
            page_token = data.get('nextPageToken')
            if not page_token:
                logger.info("No 'nextPageToken' found. This was the last page.")
                break

            pages_fetched += 1
            if max_pages is not None and pages_fetched >= max_pages:
                logger.info(f"Reached max_pages limit of {max_pages}. Stopping fetch.")
                break

            # Per project rules, be polite to the API server.
            time.sleep(0.5)
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed on page {pages_fetched + 1}: {e}")
            logger.info("Stopping fetch due to API error.")
            break
        except Exception as e:
            logger.error(f"An unexpected error occurred on page {pages_fetched + 1}: {e}")
            break

# Example Usage (for testing this module directly)
if __name__ == '__main__':
    async def test_generator_search():
        # A broad query to fetch many results
        test_criteria = {'query.cond': 'cancer'}
        
        print(f"\n--- Testing fetch_all_trials_generator with a 2-page limit ---")
        
        fetched_trials = []
        pages_count = 0
        async for page_of_trials in fetch_all_trials_generator(test_criteria, page_size=5, max_pages=2):
            pages_count += 1
            print(f"--- Received Page {pages_count} with {len(page_of_trials)} trials ---")
            fetched_trials.extend(page_of_trials)
        
        print(f"\n--- Generator Test Complete ---")
        print(f"Total trials fetched: {len(fetched_trials)} across {pages_count} pages.")
        
        if fetched_trials:
            print("\n--- Verifying content of the first fetched trial ---")
            first_trial = fetched_trials[0]
            print(f"  ID: {first_trial.get('id')}")
            print(f"  Title: {first_trial.get('title')}")
            print(f"  Status: {first_trial.get('status')}")
            print(f"  Phases: {first_trial.get('phases')}")
            print(f"  Eligibility Criteria (first 100 chars): {first_trial.get('eligibility_criteria', '')[:100]}...")
        else:
            print("\n--- No results were returned from the generator ---")
            
    asyncio.run(test_generator_search())