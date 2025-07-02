import requests
import asyncio
from typing import Dict, Any, List

# ClinicalTrials.gov API base URL
API_BASE_URL = "https://clinicaltrials.gov/api/v2/studies"

def parse_study(study: Dict[str, Any]) -> Dict[str, Any]:
    """Helper function to extract relevant fields from a study object."""
    protocol = study.get('protocolSection', {})
    
    # Identifiers
    ids = protocol.get('identificationModule', {})
    nct_id = ids.get('nctId', 'N/A')
    
    # Status
    status = protocol.get('statusModule', {})
    overall_status = status.get('overallStatus', 'Unknown')
    
    # Design
    design = protocol.get('designModule', {})
    phases = design.get('phases', ['N/A']) # It's a list
    
    # Description
    desc = protocol.get('descriptionModule', {})
    brief_title = desc.get('briefTitle', 'No Title Available')
    brief_summary = desc.get('briefSummary', 'No Summary Available')

    # Eligibility
    eligibility = protocol.get('eligibilityModule', {})
    eligibility_criteria_text = eligibility.get('eligibilityCriteria', '')

    # Conditions
    cond = protocol.get('conditionsModule', {})
    conditions = cond.get('conditions', [])
    
    # Interventions
    arms_interv = protocol.get('armsInterventionsModule', {})
    interventions = []
    for intervention in arms_interv.get('interventions', []):
        interventions.append(intervention.get('name', 'Unknown Intervention'))
        
    return {
        "id": nct_id,
        "title": brief_title,
        "status": overall_status,
        "phases": phases,
        "summary": brief_summary,
        "conditions": conditions,
        "interventions": interventions,
        "source_url": f"https://clinicaltrials.gov/study/{nct_id}" if nct_id != 'N/A' else None,
        "eligibility_criteria": eligibility_criteria_text
    }

async def search_clinical_trials(criteria: Dict[str, Any], max_results: int = 10) -> List[Dict[str, Any]]:
    """Searches ClinicalTrials.gov based on provided criteria.

    Args:
        criteria: A dictionary containing search parameters. 
                  Example: {'query.cond': 'glioblastoma', 'query.intr': 'CAR-T'}
                  See API documentation for available fields: 
                  https://clinicaltrials.gov/data-api/api
        max_results: Maximum number of studies to return.

    Returns:
        A list of dictionaries, each containing details of a clinical trial.
        Returns an empty list on error.
    """
    results = []
    params = {
        "format": "json",
        "countTotal": "true", # Get total count for potential pagination later
        "pageSize": max_results
    }
    
    # Add criteria to params - API uses dot notation in query params
    for key, value in criteria.items():
        if value: # Only add criteria if they have a value
            params[key] = value
            
    print(f"Searching ClinicalTrials.gov with params: {params}")
    
    try:
        # Use asyncio.to_thread for non-blocking IO with requests
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None, 
            lambda: requests.get(API_BASE_URL, params=params, timeout=20) # Increased timeout
        )
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
        data = response.json()
        
        total_count = data.get('totalCount', 0)
        print(f"ClinicalTrials.gov API found {total_count} total studies.")
        
        studies = data.get('studies', [])
        if studies:
            print(f"Parsing top {len(studies)} studies...")
            for study in studies:
                try:
                    parsed = parse_study(study)
                    results.append(parsed)
                except Exception as e:
                     print(f"Error parsing study (NCT ID maybe?): {e} - Skipping study.")
                     continue
            print(f"Successfully parsed {len(results)} clinical trials.")
        else:
            print("No studies found in the API response.")
            
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from ClinicalTrials.gov API: {e}")
        return []
    except Exception as e:
        print(f"An unexpected error occurred during ClinicalTrials.gov search: {e}")
        return []
        
    return results

async def fetch_all_trials_generator(criteria: Dict[str, Any], page_size: int = 100, max_pages: int = None):
    """
    Asynchronously generates pages of clinical trials from the API.
    Handles pagination to fetch all results.
    """
    params = {
        "format": "json",
        "pageSize": page_size,
        "countTotal": "true"
    }
    params.update(criteria)
    
    page_count = 0
    next_page_token = None

    while True:
        if next_page_token:
            params["pageToken"] = next_page_token
        
        try:
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(
                None,
                lambda: requests.get(API_BASE_URL, params=params, timeout=30)
            )
            response.raise_for_status()
            data = response.json()
            
            studies = data.get('studies', [])
            if not studies:
                print("No more studies found. Ending fetch.")
                break
                
            parsed_studies = [parse_study(s) for s in studies]
            yield parsed_studies # Yield a page of parsed trials
            
            page_count += 1
            if max_pages and page_count >= max_pages:
                print(f"Reached max page limit of {max_pages}.")
                break

            next_page_token = data.get('nextPageToken')
            if not next_page_token:
                print("No next page token. All trials fetched.")
                break
                
        except requests.exceptions.RequestException as e:
            print(f"API request failed: {e}")
            break
        except Exception as e:
            print(f"An unexpected error occurred during trial fetching: {e}")
            break

# Example Usage (for testing this module directly)
if __name__ == '__main__':
    async def test_search():
        test_criteria = {
            'query.cond': 'Non-small cell lung cancer', 
            'query.intr': 'osimertinib', 
            'query.term': 'phase 3' # Example using term for phase
        }
        print(f"--- Testing ClinicalTrials.gov Search with criteria: {test_criteria} ---")
        search_results = await search_clinical_trials(test_criteria, max_results=5)
        
        if search_results:
            print(f"\n--- Found {len(search_results)} Results ---")
            for i, result in enumerate(search_results):
                print(f"\nResult {i+1}:")
                print(f"  NCT ID: {result['id']}")
                print(f"  Title: {result['title']}")
                print(f"  Status: {result['status']}")
                print(f"  Conditions: {result['conditions']}")
                print(f"  Interventions: {result['interventions']}")
                print(f"  Summary: {result['summary'][:200]}...")
        else:
            print("\n--- No results returned from search_clinical_trials ---")
            
    asyncio.run(test_search())