"""
Agent responsible for drafting referral letters using an LLM.
"""

import json
import os
from datetime import datetime
from typing import Any, Dict

import google.generativeai as genai

# Import the base class
from backend.core.agent_interface import AgentInterface

class ReferralAgent(AgentInterface):
    """ Drafts referral letters using Gemini based on context and user request. """

    def __init__(self):
        """ Initialize the referral agent and Gemini model. """
        self.api_key = os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("ReferralAgent requires GOOGLE_API_KEY environment variable.")
        
        # Configure Gemini client (assuming genai.configure was called globally)
        try:
            # Using flash for potentially faster drafting tasks
            self.model = genai.GenerativeModel('gemini-1.5-flash') 
            print("ReferralAgent Initialized with Gemini Client.")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Gemini client for ReferralAgent: {e}")
        
        print("ReferralAgent Initialization complete.")

    @property
    def name(self) -> str:
        return "referral_drafter"

    @property
    def description(self) -> str:
        return "Handles drafting comprehensive referral letters using an LLM, incorporating relevant patient details."

    async def run(self, patient_data: Dict[str, Any] = None, prompt_details: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Drafts a referral letter using an LLM.
        """
        print(f"ReferralAgent running.")
        entities = prompt_details.get("entities", {}) if prompt_details else {}
        original_prompt = prompt_details.get("prompt", "") if prompt_details else ""
        
        # Extract patient details
        demographics = patient_data.get("demographics", {}) if patient_data else {}
        patient_name = demographics.get("name", "Patient")

        # Extract referral details
        recipient_specialty = entities.get("recipient_specialty", entities.get("recipient", "Specialist"))
        reason = entities.get("reason_for_referral", entities.get("specific_condition", "evaluation"))
        urgency = entities.get("urgency", "Routine")

        try:
            # --- Draft Referral Letter using LLM --- 
            drafted_content_raw = await self._call_llm_for_referral_draft(
                patient_data=patient_data,
                recipient_specialty=recipient_specialty,
                reason_for_referral=reason,
                urgency=urgency,
                original_request=original_prompt
            )

            # --- Robust JSON Extraction and Parsing --- 
            subject = f"Referral Request: {patient_name} to {recipient_specialty}" # Default subject
            body = "Could not extract referral letter body from LLM response." # Default body
            extracted_json = None
            
            json_start = drafted_content_raw.find('{')
            json_end = drafted_content_raw.rfind('}')

            if json_start != -1 and json_end != -1 and json_end > json_start:
                json_str = drafted_content_raw[json_start:json_end+1]
                print(f"[ReferralAgent] Extracted JSON String: {json_str}")
                try:
                    extracted_json = json.loads(json_str)
                except json.JSONDecodeError as e:
                    print(f"[ReferralAgent] Failed to parse extracted JSON block: {e}")
                    body = drafted_content_raw # Use raw content as fallback body
            else:
                print("[ReferralAgent] Could not find JSON block in LLM response. Using raw text as body.")
                body = drafted_content_raw # Use raw content as fallback body

            # If JSON parsed, extract subject and body
            if extracted_json:
                 subject = extracted_json.get("subject", subject)
                 body = extracted_json.get("body", body)
            # --------------------------------------------

            import asyncio
            await asyncio.sleep(0.1)
            
            return {
                "status": "success", 
                "output": {
                    "recipient_specialty": recipient_specialty,
                    "subject": subject,
                    "referral_letter_draft": body,
                    "needs_review": True # Always flag for review
                },
                "summary": f"Successfully drafted referral letter to {recipient_specialty} using LLM."
            }

        except Exception as e:
            print(f"Error during referral agent execution: {e}")
            return {
                "status": "failure", "output": None,
                "summary": f"Failed to draft referral letter: {e}", "error_message": str(e)
            }

    def _generate_referral_drafting_prompt(self, patient_data: dict, recipient_specialty: str, reason_for_referral: str, urgency: str, original_request: str) -> str:
        """ Creates the prompt for the LLM to draft the referral letter. """
        # Basic patient info for quick reference
        demographics = patient_data.get("demographics", {})
        patient_name = demographics.get("name", "Patient")
        patient_dob = demographics.get("dob", "N/A")
        primary_diagnosis = patient_data.get("diagnosis", {}).get("primary", "N/A")
        
        # Selectively include relevant context (can be expanded)
        relevant_context = {
            "demographics": demographics,
            "diagnosis": patient_data.get("diagnosis"),
            "medicalHistory": patient_data.get("medicalHistory"),
            "currentMedications": patient_data.get("currentMedications"),
            "allergies": patient_data.get("allergies"),
            "recentLabs_summary": "See full record for details" if patient_data.get("recentLabs") else None, # Avoid flooding prompt
            "imagingStudies_summary": "See full record for details" if patient_data.get("imagingStudies") else None,
            "notes_summary": "See full record for details" if patient_data.get("notes") else None,
        }
        # Filter out None values
        relevant_context = {k: v for k, v in relevant_context.items() if v is not None}
        context_json = json.dumps(relevant_context, indent=2)

        prompt = f"""
You are a helpful clinical assistant AI specializing in oncology. Your task is to draft a professional and comprehensive referral letter.

**Referral Details:**
*   **Recipient Specialty:** {recipient_specialty}
*   **Reason for Referral:** {reason_for_referral}
*   **Urgency:** {urgency}
*   **Original User Request:** `{original_request}`

**Patient Information Summary:**
{context_json}
(Full patient record available separately)

**Instructions:**
1.  Draft a standard clinical referral letter addressed to the {recipient_specialty} team.
2.  Include key patient demographics (Name, DOB).
3.  Clearly state the primary diagnosis and the specific reason for referral.
4.  Briefly summarize relevant clinical information from the provided Patient Information Summary that supports the reason for referral. Focus on pertinent history, medications, or findings.
5.  Maintain a professional tone.
6.  Conclude appropriately, requesting evaluation and offering collaboration.
7.  Respond ONLY with a JSON object containing two keys:
    -   "subject": A concise subject line (e.g., "Referral Request: [Patient Name] for [Reason]").
    -   "body": The full drafted referral letter body (string, including date, salutation, content, closing).

**Example JSON Response:**
```json
{{
  "subject": "Referral Request: Jane Doe for Cardiology Evaluation",
  "body": "Date: [Current Date]\n\nRE: Referral for Jane Doe (DOB: 1965-03-15)\n\nDear Cardiology Team,\n\nI am referring Jane Doe, diagnosed with Stage III Invasive Ductal Carcinoma, for evaluation due to [Specific Reason derived from context/prompt, e.g., recent ECG changes].\n\nRelevant history includes hypertension and type 2 diabetes. Current pertinent medications include Letrozole. [Add any other key relevant details from the summary context].\n\nPlease evaluate Ms. Doe at your earliest convenience. Relevant records can be provided upon request.\n\nThank you for your expertise and collaboration.\n\nSincerely,\n\n[Referring Clinician Name/AI CoPilot System]"
}}
```

**JSON Response:**
"""
        return prompt

    async def _call_llm_for_referral_draft(self, patient_data: dict, recipient_specialty: str, reason_for_referral: str, urgency: str, original_request: str) -> str:
        """ Calls the Gemini API to draft the referral letter. """
        if not self.model:
            raise RuntimeError("ReferralAgent: Gemini model not initialized.")

        prompt = self._generate_referral_drafting_prompt(patient_data, recipient_specialty, reason_for_referral, urgency, original_request)

        print("Sending referral drafting prompt to Gemini...")
        try:
            response = await self.model.generate_content_async(prompt)
            drafted_text = response.text.strip() # Basic strip is enough here
            print(f"Received raw drafted referral content from Gemini: {drafted_text}")
            return drafted_text
        except Exception as e:
            print(f"Error during Gemini API call for referral drafting: {e}")
            raise e # Re-raise to be caught by run method

# Example Usage (for testing)
# if __name__ == '__main__':
#     async def main():
#         agent = ReferralAgent()
#         mock_context = {
#             "patient_data": {
#                 "demographics": {"name": "Jane Doe", "dob": "1965-03-15"},
#                 "diagnosis": {"primary": "Stage III IDC"}
#             }
#         }
#         mock_kwargs = {
#             "prompt": "Draft referral to Cardiology for cardiac evaluation",
#             "entities": {"recipient_specialty": "Cardiology", "reason_for_referral": "cardiac evaluation"}
#         }
#         result = await agent.run(mock_context, **mock_kwargs)
#         print("Referral Draft Result:", json.dumps(result, indent=2))
#     import asyncio
#     asyncio.run(main()) 