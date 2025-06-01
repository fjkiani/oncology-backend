from typing import Dict, Any, List, Optional
from .agent_interface import AgentInterface

class LabOrderAgent(AgentInterface):
    def __init__(self):
        self.name = "LabOrderAgent"
        self.description = "Handles drafting and managing lab orders."

    async def run(self, patient_data: Dict[str, Any], prompt_details: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes a request to draft a lab order.
        For demo, it will return a mock lab order structure or a confirmation.
        `prompt_details` is expected to contain a 'prompt' field with the user's request.
        """
        user_prompt = prompt_details.get("prompt", "Draft a standard lab order.")
        patient_name = patient_data.get("demographics", {}).get("name", "Unknown Patient")
        diagnosis = patient_data.get("diagnosis", {}).get("primary", "uunspecified condition")

        # Mocking a lab order draft
        mock_lab_order = {
            "patient_name": patient_name,
            "diagnosis_context": diagnosis,
            "order_request_details": user_prompt,
            "panels_ordered": [
                {"panel_name": "Complete Blood Count (CBC) with Differential", "reason": f"Monitor {diagnosis}"},
                {"panel_name": "Comprehensive Metabolic Panel (CMP)", "reason": f"Monitor {diagnosis}"},
                {"panel_name": "Tumor Markers (e.g., LDH, specific markers if known)", "reason": f"Monitor {diagnosis}"}
            ],
            "status": "draft",
            "notes": "Further customization may be required by the ordering physician.",
            "summary": f"Drafted lab order for {patient_name} to monitor {diagnosis} based on prompt: '{user_prompt[:50]}...'"
        }

        return {
            "status": "success",
            "lab_order_details": mock_lab_order,
            "summary": mock_lab_order["summary"]
        } 