"""
Agent responsible for identifying and suggesting management for side effects.
"""

import json
import os
from typing import Any, Dict, Optional

# Import the base class
from backend.core.agent_interface import AgentInterface

# Placeholder: Could use LLM or structured database later
# import google.generativeai as genai 

# Mock data for common side effects (simplified)
MOCK_SIDE_EFFECT_DB = {
    "letrozole": ["Hot flashes", "Joint pain", "Fatigue"],
    "metformin": ["Diarrhea", "Nausea", "Gas"],
    "lisinopril": ["Dry cough", "Dizziness"],
    "chemotherapy": ["Nausea", "Fatigue", "Hair loss", "Low blood counts"], # Generic category
    "immunotherapy": ["Fatigue", "Rash", "Diarrhea", "Colitis"] # Generic category
}
MOCK_MANAGEMENT_TIPS = {
    "nausea": "Consider anti-nausea medication (e.g., Zofran). Stay hydrated. Eat small, frequent meals.",
    "fatigue": "Prioritize rest. Gentle exercise as tolerated. Ensure adequate nutrition and hydration.",
    "diarrhea": "Stay hydrated (water, broth, electrolytes). Avoid high-fiber, greasy foods. Consider loperamide if severe.",
    "joint pain": "Over-the-counter pain relievers (consult physician first). Gentle stretching.",
    "rash": "Keep skin moisturized. Avoid harsh soaps. Antihistamines or topical steroids may help (consult physician).",
    "dry cough": "Stay hydrated. Lozenges. Discuss with physician if persistent."
}

# Mock data for medication interactions (simplified)
MOCK_INTERACTION_DB = {
    ("letrozole", "tamoxifen"): "Concurrent use not recommended. Both are hormonal therapies that may interfere with each other's efficacy.",
    ("metformin", "contrast_agents"): "Temporarily discontinue metformin before and after imaging studies with contrast agents to reduce risk of lactic acidosis.",
    ("metformin", "alcohol"): "Avoid excessive alcohol use with metformin as it increases risk of lactic acidosis.",
    ("nsaids", "anticoagulants"): "Increased risk of bleeding when NSAIDs are taken with anticoagulants.",
    ("nsaids", "ssris"): "Increased risk of GI bleeding when NSAIDs are taken with SSRIs.",
    ("warfarin", "antibiotics"): "Many antibiotics can potentiate the effect of warfarin, increasing INR and bleeding risk."
}

# Function to check if a medication belongs to a class
def med_belongs_to_class(med_name, class_name):
    med_lower = med_name.lower()
    if class_name == "nsaids":
        return any(nsaid in med_lower for nsaid in ["ibuprofen", "naproxen", "aspirin", "celecoxib", "diclofenac"])
    elif class_name == "anticoagulants":
        return any(anticoag in med_lower for anticoag in ["warfarin", "heparin", "apixaban", "rivaroxaban", "dabigatran", "edoxaban"])
    elif class_name == "ssris":
        return any(ssri in med_lower for ssri in ["fluoxetine", "sertraline", "paroxetine", "citalopram", "escitalopram"])
    elif class_name == "antibiotics":
        return any(abx in med_lower for abx in ["ciprofloxacin", "amoxicillin", "azithromycin", "doxycycline", "clindamycin"])
    return False

class SideEffectAgent(AgentInterface):
    """ Identifies potential side effects and suggests management strategies. """

    def __init__(self):
        """ Initialize the side effect agent. """
        # Placeholder: Could initialize LLM or database connection
        print("SideEffectAgent Initialized.")

    @property
    def name(self) -> str:
        return "side_effect_manager"

    @property
    def description(self) -> str:
        return "Identifies potential medication/treatment side effects and suggests management tips."

    async def run(self, patient_data: Dict[str, Any] = None, prompt_details: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Identifies potential side effects, interactions, or provides management tips.

        Args:
            patient_data: Dictionary containing patient information.
            prompt_details: Dictionary containing intent and entities from the orchestrator.

        Returns:
            A dictionary with status and relevant side effect information.
        """
        print(f"SideEffectAgent running with intent: {prompt_details.get('intent', 'unknown')}")
        
        # Handle both old and new signature formats
        if not patient_data and "context" in locals():
            patient_data = locals()["context"].get("patient_data", {})
        
        entities = prompt_details.get("entities", {}) if prompt_details else {}
        prompt = prompt_details.get("prompt", "") if prompt_details else ""
        
        # --- Identify potential topic --- 
        target_med = entities.get("medication_name")
        target_symptom = entities.get("symptom", entities.get("specific_condition"))
        target_treatment = entities.get("treatment_type")
        check_interactions = "interactions" in prompt.lower() or "drug interaction" in prompt.lower()
        
        potential_side_effects = []
        management_tips = []
        interactions = []

        # --- Scenario 4: User asks about medication interactions ---
        if check_interactions:
            print("Checking medication interactions")
            meds = patient_data.get("currentMedications", [])
            med_names = [med.get("name", "").lower() for med in meds if med.get("name")]
            
            # Check all pairs of medications
            for i in range(len(med_names)):
                for j in range(i+1, len(med_names)):
                    med1, med2 = med_names[i], med_names[j]
                    # Check direct interactions
                    if (med1, med2) in MOCK_INTERACTION_DB:
                        interactions.append({
                            "medication1": med1,
                            "medication2": med2,
                            "description": MOCK_INTERACTION_DB[(med1, med2)]
                        })
                    elif (med2, med1) in MOCK_INTERACTION_DB:
                        interactions.append({
                            "medication1": med1,
                            "medication2": med2,
                            "description": MOCK_INTERACTION_DB[(med2, med1)]
                        })
                    
                    # Check class-based interactions
                    for class1, class2 in MOCK_INTERACTION_DB.keys():
                        if type(class1) == str and type(class2) == str:  # Ensure we're dealing with string class names
                            if med_belongs_to_class(med1, class1) and med_belongs_to_class(med2, class2):
                                interactions.append({
                                    "medication1": med1,
                                    "medication2": med2,
                                    "description": MOCK_INTERACTION_DB[(class1, class2)]
                                })
                            elif med_belongs_to_class(med1, class2) and med_belongs_to_class(med2, class1):
                                interactions.append({
                                    "medication1": med1,
                                    "medication2": med2,
                                    "description": MOCK_INTERACTION_DB[(class1, class2)]
                                })
            
            if interactions:
                summary = "Potential medication interactions detected: "
                for idx, interaction in enumerate(interactions):
                    if idx > 0:
                        summary += "; "
                    summary += f"{interaction['medication1'].capitalize()} + {interaction['medication2'].capitalize()}"
            else:
                summary = "No significant medication interactions identified in the current regimen."
        
        # --- Scenario 1: User asks about side effects of a specific med ---
        elif target_med:
            med_lower = target_med.lower()
            print(f"Checking side effects for medication: {target_med}")
            current_effects = MOCK_SIDE_EFFECT_DB.get(med_lower, []) # Get direct effects

            # If no direct effects, check for known class representatives or keywords in the name
            if not current_effects:
                if med_lower == "pembrolizumab": # Specific check for Pembrolizumab
                    print(f"No direct entry for {target_med}, recognized as Immunotherapy.")
                    current_effects.extend(MOCK_SIDE_EFFECT_DB.get("immunotherapy", []))
                # Example for a chemo drug - extend this list or use a mapping
                elif med_lower == "fluorouracil": # Example chemo drug without direct entry
                    print(f"No direct entry for {target_med}, recognized as Chemotherapy.")
                    current_effects.extend(MOCK_SIDE_EFFECT_DB.get("chemotherapy", []))
                # Fallback: Original keyword check if it's a general category name asked directly
                elif "chemo" in med_lower: # e.g., user asks "side effects of chemo"
                     current_effects.extend(MOCK_SIDE_EFFECT_DB.get("chemotherapy", []))
                elif "immuno" in med_lower: # e.g., user asks "side effects of immunotherapy"
                     current_effects.extend(MOCK_SIDE_EFFECT_DB.get("immunotherapy", []))
            else:
                # If direct effects were found, still check if the name also implies a general category
                # This handles cases where a specific drug might have its own list AND belong to a category
                if "chemo" in med_lower:
                     current_effects.extend(MOCK_SIDE_EFFECT_DB.get("chemotherapy", []))
                if "immuno" in med_lower: # e.g. if a drug was "SuperImmunoDrug"
                     current_effects.extend(MOCK_SIDE_EFFECT_DB.get("immunotherapy", []))

            potential_side_effects = list(set(current_effects)) # Unique list
            
            summary = f"Potential side effects for {target_med}: {', '.join(potential_side_effects) if potential_side_effects else 'None listed in mock DB.'}"

        # Scenario 2: User asks for management of a specific symptom
        elif target_symptom:
            symptom_lower = target_symptom.lower()
            print(f"Checking management tips for symptom: {target_symptom}")
            tip = MOCK_MANAGEMENT_TIPS.get(symptom_lower)
            if tip:
                management_tips.append({ "symptom": target_symptom, "tip": tip })
            summary = f"Management tips for {target_symptom}: {tip if tip else 'No specific tips in mock DB.'}"
            
        # Scenario 3: User asks generally about side effects for the patient
        else:
            print("Checking potential side effects based on patient's current medications.")
            meds = patient_data.get("currentMedications", [])
            temp_effects_list = [] # Use a temporary list to gather all effects

            for med_entry in meds:
                med_name_original = med_entry.get("name", "")
                med_name_lower = med_name_original.lower()

                if med_name_lower:
                    # 1. Check for direct match in MOCK_SIDE_EFFECT_DB
                    direct_effects = MOCK_SIDE_EFFECT_DB.get(med_name_lower, [])
                    if direct_effects:
                        temp_effects_list.extend(direct_effects)

                    # 2. Check for generic categories by substring
                    if "chemo" in med_name_lower:
                        chemo_effects = MOCK_SIDE_EFFECT_DB.get("chemotherapy", [])
                        if chemo_effects:
                            temp_effects_list.extend(chemo_effects)
                    
                    if "immuno" in med_name_lower:
                        immuno_effects = MOCK_SIDE_EFFECT_DB.get("immunotherapy", [])
                        if immuno_effects:
                            temp_effects_list.extend(immuno_effects)
            
            potential_side_effects = list(set(temp_effects_list)) # Make unique

            if potential_side_effects:
                summary = f"Potential side effects based on current meds: {', '.join(potential_side_effects)}."
            else:
                summary = "No common side effects identified from the mock database for the patient's current medications."

        # Simulate async work
        import asyncio
        await asyncio.sleep(0.1)

        # --- Return Result --- 
        return {
            "status": "success", 
            "output": {
                "target_medication": target_med,
                "target_symptom": target_symptom,
                "potential_side_effects": potential_side_effects, # List of strings
                "management_tips": management_tips, # List of {symptom, tip} dicts
                "interactions": interactions # List of {medication1, medication2, description} dicts
            },
            "summary": summary
        }

# Example Usage (for testing)
# if __name__ == '__main__':
#     async def main():
#         agent = SideEffectAgent()
#         ctx = {"patient_data": {"currentMedications": [{"name": "Letrozole"}, {"name": "Metformin"}]}}
#         
#         # Test 1: General check
#         kw1 = {"prompt": "Any side effects to watch for?"}
#         res1 = await agent.run(ctx, **kw1)
#         print("Result 1 (General):", json.dumps(res1, indent=2))
#         
#         # Test 2: Specific med
#         kw2 = {"prompt": "What are side effects of Letrozole?", "entities": {"medication_name": "Letrozole"}}
#         res2 = await agent.run(ctx, **kw2)
#         print("Result 2 (Specific Med):", json.dumps(res2, indent=2))
#         
#         # Test 3: Specific symptom
#         kw3 = {"prompt": "How to manage nausea?", "entities": {"symptom": "nausea"}}
#         res3 = await agent.run(ctx, **kw3)
#         print("Result 3 (Specific Symptom):", json.dumps(res3, indent=2))
#         
#     import asyncio
#     asyncio.run(main()) 