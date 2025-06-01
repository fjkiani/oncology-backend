from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
import sqlite3
import os
import logging

# Import the utility function
from .pubmed_utils import search_pubmed

# Import the agent
try:
    from backend.agents.genomic_analyst_agent import GenomicAnalystAgent, GenomicAnalysisResult
except ImportError:
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    try:
        from agents.genomic_analyst_agent import GenomicAnalystAgent, GenomicAnalysisResult
    except ImportError:
         logging.error("Failed to import GenomicAnalystAgent from expected paths.")
         # Define dummy classes if import fails during runtime
         class GenomicAnalystAgent:
             async def run(self, *args, **kwargs):
                 return {"status": "ERROR", "evidence": "GenomicAnalystAgent not loaded.", "simulated_vep_details": []}
         
         class GenomicAnalysisResult:
             """Dummy class for when import fails"""
             def __init__(self, **kwargs):
                 for key, value in kwargs.items():
                     setattr(self, key, value)
             
             def model_dump(self):
                 return vars(self)

# --- Mock Data & DB Path (Potentially duplicate or refactor later) ---
# This is not ideal, refactor to use dependency injection or shared config later
# Assuming main.py defines these, we might need to replicate or import carefully
# For now, let's define them here, assuming structure relative to this file's potential location
PROJECT_ROOT_FROM_ROUTER = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PATIENT_MUTATIONS_DB_PATH = os.path.join(PROJECT_ROOT_FROM_ROUTER, 'backend', 'data', 'patient_data.db')

# --- IMPORT mock_patient_data_dict ---
# This assumes backend.data is in sys.path or PYTHONPATH
# Adjust import if running router in a different context
try:
    from backend.data.mock_patient_data_dict import mock_patient_data_dict
    logging.info("Successfully imported mock_patient_data_dict in research/router.py")
except ImportError:
    logging.error("Failed to import mock_patient_data_dict in research/router.py. Mutation analysis might fail or use empty data.")
    mock_patient_data_dict = {} # Fallback to prevent crash, but operations will likely fail

# Need access to mock_patient_data_dict, maybe import from main? Or redefine subset?
# Importing directly from main can cause circular dependency issues.
# Let's assume for now we only need mutations, fetched from DB.
# If base patient data is needed by agent later, this needs rethinking.
# --- End Mock Data & DB Path ---

router = APIRouter()

# --- Pydantic Model for Mutation Analysis Request --- 
class MutationAnalysisRequest(BaseModel):
    patient_id: str = Field(..., description="ID of the patient whose mutations are relevant.")
    prompt: str = Field(..., description="The genomic criterion or question to analyze.")
    intent: str = Field(..., description="The intent for the agent, e.g., 'analyze_genomic_criterion'.")
    target_mutation_ids: Optional[List[str]] = Field(default=None, description="Optional list of specific mutation identifiers (e.g., from patient's list) to focus analysis on.")

@router.post("/pubmed/search", response_model=List[Dict[str, Any]])
async def search_pubmed_endpoint(payload: Dict[str, Any] = Body(...)):
    """ Endpoint to search PubMed. """
    query = payload.get('query')
    max_results = payload.get('max_results', 10) # Default to 10 results

    if not query:
        raise HTTPException(status_code=400, detail="Missing 'query' in request body")
    
    print(f"Received PubMed search request for query: {query}")
    try:
        results = await search_pubmed(query, max_results=max_results)
        return results
    except Exception as e:
        # Log the exception details on the server
        print(f"Error calling PubMed search utility: {e}") 
        # Return a generic error to the client
        raise HTTPException(status_code=500, detail="Internal server error during PubMed search.")

@router.post("/clinicaltrials/search", response_model=List[Dict[str, Any]])
async def search_clinical_trials_endpoint(payload: Dict[str, Any] = Body(...)):
    """ Endpoint to search ClinicalTrials.gov. """
    criteria = payload.get('criteria')
    if not criteria or not isinstance(criteria, dict):
        raise HTTPException(status_code=400, detail="Missing or invalid 'criteria' object in request body")

    print(f"Received ClinicalTrials search request with criteria: {criteria}")
    # TODO: Implement actual call to clinicaltrials_utils.search_clinical_trials
    
    # Placeholder response
    mock_results = [
        { "id": "NCT12345", "title": "Mock Clinical Trial 1", "summary": "Mock summary for criteria: " + str(criteria), "source": "ClinicalTrials.gov" },
        { "id": "NCT67890", "title": "Mock Clinical Trial 2", "summary": "Another mock trial summary...", "source": "ClinicalTrials.gov" }
    ]
    return mock_results

# --- New Mutation Analysis Endpoint --- 
@router.post("/mutation-analysis", response_model=Dict[str, Any])
async def analyze_mutations(request: MutationAnalysisRequest):
    """ 
    Analyzes a genomic query against a patient's mutations using the 
    GenomicAnalystAgent.
    """
    logging.info(f"Received mutation analysis request for patient: {request.patient_id}, prompt: '{request.prompt[:50]}...', intent: {request.intent}")

    patient_id_upper = request.patient_id.upper() # Ensure uppercase for dict lookup

    # --- Fetch patient data (including mutations) from mock_patient_data_dict ---
    patient_info_from_mock = mock_patient_data_dict.get(patient_id_upper)

    if not patient_info_from_mock:
        logging.error(f"Patient {patient_id_upper} not found in mock_patient_data_dict.")
        raise HTTPException(status_code=404, detail=f"Patient {patient_id_upper} not found in mock data.")

    mutations = patient_info_from_mock.get("mutations", [])
    logging.info(f"Using {len(mutations)} mutations from mock_patient_data_dict for patient {request.patient_id}")

    # The direct DB fetching block is now removed/commented out.
    # --- Fetch patient mutations from DB --- 
    # Simplified logic, similar to main.py, but without fetching mock_patient_data_dict
    # mutations = [] # This line and subsequent DB logic is replaced by mock data fetching above.
    # conn = None
    # try:
    #     logging.debug(f"Connecting to mutations DB at: {PATIENT_MUTATIONS_DB_PATH}")
    #     if not os.path.exists(PATIENT_MUTATIONS_DB_PATH):
    #         logging.warning(f"Mutations database not found at {PATIENT_MUTATIONS_DB_PATH} for patient {patient_id}. Returning empty mutation list.")
    #     else:
    #         conn = sqlite3.connect(PATIENT_MUTATIONS_DB_PATH)
    #         conn.row_factory = sqlite3.Row
    #         cursor = conn.cursor()
    #         # Assuming the DB contains necessary fields like hugo_gene_symbol, protein_change, variant_type
    #         cursor.execute("SELECT * FROM mutations WHERE patient_id = ?", (patient_id,))
    #         rows = cursor.fetchall()
    #         mutations = [dict(row) for row in rows]
    #         logging.info(f"Fetched {len(mutations)} mutations from DB for patient {patient_id}")
            
    # except sqlite3.Error as e:
    #     logging.error(f"Database error fetching mutations for {patient_id}: {e}", exc_info=True)
    #     raise HTTPException(status_code=500, detail=f"Database error fetching mutations.")
    # except Exception as e:
    #     logging.error(f"Unexpected error fetching mutations for {patient_id}: {e}", exc_info=True)
    #     raise HTTPException(status_code=500, detail=f"Unexpected error fetching mutations.")
    # finally:
    #     if conn:
    #         conn.close()
            
    # --- Instantiate and Run Agent --- 
    try:
        agent = GenomicAnalystAgent()
        logging.debug(f"Running GenomicAnalystAgent for patient {request.patient_id} with prompt: {request.prompt}")
        
        # Construct patient_data for the agent's run method
        # Use the full patient_info_from_mock if available, otherwise construct minimally.
        patient_data_for_agent = patient_info_from_mock if patient_info_from_mock else {
            "patientId": request.patient_id, # Fallback if full mock data wasn't found but we proceed
            "mutations": mutations, 
            "demographics": {
                "first_name": "Patient", 
                "last_name": request.patient_id 
            }
        }
        # Ensure mutations are correctly placed if patient_info_from_mock was used directly
        if "mutations" not in patient_data_for_agent or not patient_data_for_agent["mutations"]:
            patient_data_for_agent["mutations"] = mutations

        prompt_details_for_agent = {
            "prompt": request.prompt,
            "intent": request.intent,
            "patient_id": request.patient_id # Though patientId is also in patient_data, agent might look here too
            # "entities": {} # If we were parsing entities from the prompt
        }

        result = await agent.run(
            # genomic_query=genomic_query, 
            # patient_id=request.patient_id, 
            # patient_mutations=mutations
            patient_data=patient_data_for_agent,
            prompt_details=prompt_details_for_agent
        )
        logging.info(f"GenomicAnalystAgent finished for patient {request.patient_id}. Status: {result.status if isinstance(result, GenomicAnalysisResult) else result.get('status')}")
        
        # Ensure the result is a dictionary for the FastAPI response
        if isinstance(result, GenomicAnalysisResult):
            # Convert Pydantic model to dict if necessary, or ensure it's serializable
            # For now, assuming GenomicAnalysisResult has a .dict() method or is directly serializable
            # If GenomicAnalysisResult is a Pydantic model, it should serialize automatically.
            # If it's a plain class, you might need a to_dict method or similar.
            # Let's assume it's compatible or FastAPI handles it.
            # For safety, explicitly convert to dict if it's a known Pydantic model type
             response_data = result.model_dump() if hasattr(result, 'model_dump') else result
        else:
             response_data = result # Assuming it's already a dict

        return response_data
    except Exception as e:
        logging.error(f"Error running GenomicAnalystAgent for patient {request.patient_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error during genomic analysis: {e}")