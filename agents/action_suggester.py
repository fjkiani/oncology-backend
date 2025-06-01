import re
import logging
from typing import List, Dict, Optional, Literal

# Define action types
ActionType = Literal["TASK", "LAB_ORDER_SUGGESTION", "PATIENT_MESSAGE_SUGGESTION", "CHART_REVIEW_SUGGESTION", "UNKNOWN"]

# Basic keywords for categorization
# Expand these lists based on observed LLM outputs
LAB_KEYWORDS = ['lab', 'level', 'count', 'value', 'assay', 'status', 'test', 'profile', 'marker', 'blood', 'serum', 'plasma', 'urine', 'biopsy result', 'mutation', 'gene', 'genomic', 'molecular']
MEASUREMENT_KEYWORDS = ['bmi', 'weight', 'height', 'bp', 'blood pressure', 'ecog', 'kps', 'performance status', 'qtc', 'lvef', 'ejection fraction', 'measurement']
PATIENT_QUERY_KEYWORDS = ['history of', 'use of', 'prior', 'concomitant medication', 'willingness', 'able to', 'plan to', 'pregnancy', 'breastfeeding', 'lifestyle', 'dietary', 'smoking', 'alcohol', 'preference']
CHART_REVIEW_KEYWORDS = ['report', 'note', 'scan', 'imaging', 'pathology', 'surgical', 'procedure', 'consult', 'record', 'dated', 'finding', 'status post']

def analyze_unclear_criterion(criterion: str, missing_info: Optional[str]) -> ActionType:
    """Analyzes unclear criterion text to suggest an action category."""
    text_to_analyze = f"{criterion or ''} {missing_info or ''}".lower()
    
    # Prioritize more specific keywords
    if any(keyword in text_to_analyze for keyword in LAB_KEYWORDS):
        # Check if it might actually be chart review for existing results
        if any(chart_kw in text_to_analyze for chart_kw in CHART_REVIEW_KEYWORDS):
            return "CHART_REVIEW_SUGGESTION"
        return "LAB_ORDER_SUGGESTION" # Suggest ordering if result isn't just needed from chart
        
    if any(keyword in text_to_analyze for keyword in MEASUREMENT_KEYWORDS):
         # Check if it might be chart review for existing results
        if any(chart_kw in text_to_analyze for chart_kw in CHART_REVIEW_KEYWORDS):
            return "CHART_REVIEW_SUGGESTION"
        return "TASK" # Task to obtain measurement

    if any(keyword in text_to_analyze for keyword in PATIENT_QUERY_KEYWORDS):
        return "PATIENT_MESSAGE_SUGGESTION"
        
    if any(keyword in text_to_analyze for keyword in CHART_REVIEW_KEYWORDS):
        return "CHART_REVIEW_SUGGESTION"

    # Default / Fallback
    logging.warning(f"Could not categorize unclear criterion: '{criterion}' / '{missing_info}'")
    return "UNKNOWN" # Needs manual review

def draft_action_for_category(category: ActionType, criterion: str, missing_info: Optional[str], patient_context: Dict) -> Optional[Dict]:
    """Drafts a specific action text based on the category."""
    patient_id = patient_context.get('patientId', '[PatientID]')
    patient_name = patient_context.get('demographics', {}).get('name', 'the patient')
    criterion_short = criterion[:100] + '...' if len(criterion) > 100 else criterion
    missing_info_text = missing_info or "information related to criterion"

    draft = None
    if category == "TASK":
        draft = {
            "action_type": "TASK",
            "draft_text": f"Obtain/Document '{missing_info_text}' for {patient_name} ({patient_id}) regarding trial criterion: '{criterion_short}'",
            "suggestion": f"Obtain/Document {missing_info_text}"
        }
    elif category == "LAB_ORDER_SUGGESTION":
         draft = {
            "action_type": "TASK", # Treat as task initially - review chart first
            "draft_text": f"Review patient chart / Consider ordering lab/test for '{missing_info_text}' for {patient_name} ({patient_id}) regarding trial criterion: '{criterion_short}'",
             "suggestion": f"Review chart / Consider lab for {missing_info_text}"
        }
    elif category == "PATIENT_MESSAGE_SUGGESTION":
         draft = {
            "action_type": "PATIENT_MESSAGE_SUGGESTION",
            "draft_text": f"Dear {patient_name}, regarding your potential eligibility for a clinical trial, could you please provide information about '{missing_info_text}'? Thank you.",
            "suggestion": f"Draft message to patient asking about {missing_info_text}"
        }
    elif category == "CHART_REVIEW_SUGGESTION":
         draft = {
            "action_type": "TASK",
            "draft_text": f"Review chart (e.g., notes, pathology, imaging reports) for '{missing_info_text}' for {patient_name} ({patient_id}) regarding trial criterion: '{criterion_short}'",
            "suggestion": f"Review chart for {missing_info_text}"
        }
    # Handle UNKNOWN category if needed, or just return None
    # else: 
    #     draft = {"action_type": "TASK", "draft_text": f"Follow up required for unclear trial criterion: {criterion_short}"}

    return draft

def get_action_suggestions_for_trial(eligibility_assessment: Dict, patient_context: Dict) -> List[Dict]:
    """Gets action suggestions for all unclear criteria in an assessment."""
    suggestions = []
    unclear_criteria = eligibility_assessment.get('unclear_criteria', [])
    
    if not isinstance(unclear_criteria, list):
        logging.error("Invalid format for unclear_criteria, expected a list.")
        return []
        
    for item in unclear_criteria:
        if isinstance(item, dict):
            criterion = item.get('criterion')
            missing_info = item.get('missing_info')
            if criterion:
                category = analyze_unclear_criterion(criterion, missing_info)
                action_draft = draft_action_for_category(category, criterion, missing_info, patient_context)
                if action_draft:
                    # Add original criterion info for context
                    action_draft['criterion'] = criterion
                    action_draft['missing_info'] = missing_info
                    suggestions.append(action_draft)
            else:
                 logging.warning("Skipping unclear item with no 'criterion' key.")
        else:
             logging.warning(f"Skipping invalid item in unclear_criteria: {item}")
             
    return suggestions 