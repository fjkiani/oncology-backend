import logging
from typing import Dict, Any, List, Optional
from .data_analysis_agent import DataAnalysisAgent
from .clinical_trial_agent import ClinicalTrialAgent
# Assuming access to an LLM client (e.g., Gemini)
# from ..llm.gemini_client import GeminiClient 

logger = logging.getLogger(__name__)

# Placeholder for the actual LLM client initialization
# gemini_client = GeminiClient()

DEFAULT_SYNTHESIS_PROMPT = """
You are an AI assistant designed to help a consulting physician synthesize information received at the start of an e-consultation. 
Your goal is to provide the consultant with a clear summary and actionable next steps.

You will receive the following inputs:
1.  **Initial Consultation Context:** Provided by the initiating physician, including their original note, the reason for the consult, and potentially some included data snippets.
2.  **AI Analysis of Initiator's Note:** A preliminary AI analysis of the initiating physician's note, highlighting key points and potential questions.
3.  **Deep Analysis Results:** Results from a deep analysis of the patient's data to answer key questions.
4.  **Clinical Trial Analysis:** Results from an analysis of relevant clinical trials for the patient.

Based on ALL the provided inputs, perform the following tasks from the perspective of the *consulting* physician receiving this information:

1.  **Synthesize Key Information:** Briefly summarize the patient's situation, the reason for the consult, and key findings from the initiator's note and AI analysis. Mention if deep analysis or trial searches were also conducted.
2.  **Identify Key Questions & Present Answers:**
    *   Based on the initiator's note, its AI analysis, and the overall context, list 3-5 key clinical questions the consulting physician should consider.
    *   For each question, provide an answer based on the "Deep Analysis Results". If the deep analysis couldn't answer a question, state that clearly.
3.  **Present Clinical Trial Findings:**
    *   Summarize the findings from the "Clinical Trial Analysis".
    *   List any trials deemed potentially relevant or those for which eligibility couldn't be assessed, along with brief reasons.
4.  **Propose Actionable Next Steps for Consultant:** Suggest concrete actions the consultant should take next (e.g., review specific data, formulate an opinion, prepare for chat discussion, specific questions to ask the initiator if info is still missing AFTER deep analysis).

**Output Format:**
Provide the output as a single block of markdown text. Use headings, bullet points, and bold text for clarity.
Example:

**Synthesized Consultation Plan:**
The initiating physician (Dr. [Initiator Name]) requests a consult for [Patient ID] regarding [Consult Reason]. Initiator note analysis highlighted [key points from note analysis].
The initial AI analysis of the initiator's note and a deep dive into the patient's record have been performed. Additionally, a search for relevant clinical trials was conducted.

**Key Questions and Answers:**
**Deep Dive Summary (Patient Data):**
[Summary from deep analysis of patient data, or "Error - Patient data missing." if applicable]

**Q: [Question 1]?**
**A:** [Answer from deep analysis, or "Error analyzing question - Patient data missing/insufficient." if applicable]

**Q: [Question 2]?**
**A:** [Answer from deep analysis]

...

**Clinical Trial Analysis:**
[Summary of clinical trial findings. e.g., "Found X potentially relevant trials. Y trials need more data for assessment."]
- **[Trial ID 1 (NCT...)]**: [Status/Reason, e.g., Potentially Eligible - BRAF V600E matched, or Needs Review - ECOG status unclear]
- **[Trial ID 2 (NCT...)]**: [Status/Reason]

**Proposed Consultant Actions:**
*   Review the comprehensive analysis including patient data insights and clinical trial findings.
*   Consider the [specific data snippet, e.g., Diagnosis data snippet] carefully.
*   Draft initial thoughts on the patient's situation, incorporating all analyses.
*   Prepare to discuss any remaining questions or unclear trial eligibilities in the chat. 
*   Verify specific trial details (e.g., NCT IDs mentioned) if proceeding with trial consideration.

**Note:** This synthesis includes a deep analysis of the patient's data and clinical trial information. For an even more detailed analysis or to explore specific trials further, you can use the dedicated tools or click the "Deep Analysis" button if more focused options are available.
"""

class ConsultationSynthesizerAgent:
    """
    Agent responsible for synthesizing consultation context for the receiving consultant.
    """
    def __init__(self, llm_client=None):
        self.data_analysis_agent = DataAnalysisAgent()
        self.clinical_trial_agent = ClinicalTrialAgent()
        # In a real implementation, llm_client would be injected
        # self.llm_client = llm_client or gemini_client

    async def _extract_key_questions(self, initiator_note_analysis: str, patient_data: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        Extracts key questions from the initiator's note analysis or generates default ones.
        Optionally uses patient_data for context if available.
        """
        logger.debug(f"Extracting key questions. Patient data available: {patient_data is not None}")
        # For now, using a simplified approach. This could involve another LLM call.
        questions = []
        if "question:" in initiator_note_analysis.lower():
            lines = initiator_note_analysis.split('\n')
            for line in lines:
                if line.lower().startswith("question:") or line.lower().startswith("- question:"):
                    questions.append(line.split(":", 1)[1].strip())
        
        if not questions: # If no questions extracted, generate some generic ones
            logger.info("No specific questions in initiator note analysis, generating generic ones.")
            # Add some context from patient_data if available for more relevant generic questions
            primary_diagnosis = patient_data.get('diagnosis', {}).get('primary', 'the patient\'s condition') if patient_data else 'the patient\'s condition'
            
            questions.extend([
                f"What is the optimal management strategy for {primary_diagnosis} at this stage?",
                f"What are the key prognostic factors for {primary_diagnosis}?",
                f"What monitoring is required for {primary_diagnosis} and potential complications?",
                "Are there relevant clinical trials to consider?"
            ])
        return questions[:5] # Limit to 5 questions

    async def _get_deep_analysis_for_questions(self, questions: List[str], patient_data: Optional[Dict[str, Any]]) -> str:
        """Get deep analysis answers for the extracted questions."""
        analysis_results = []
        
        if not patient_data or not isinstance(patient_data, dict) or not patient_data:
            logger.warning("Deep analysis attempted with no or invalid patient data.")
            analysis_results.append("**Deep Dive Summary (Patient Data):** Error - Patient data missing.")
            for i, q_text in enumerate(questions):
                analysis_results.append(f"**Q: * {q_text.strip()}**")
                analysis_results.append(f"**A:** Error analyzing question - Patient data missing.")
            return "\n\n".join(analysis_results)

        logger.info(f"Performing deep dive summary for patient. Patient data keys: {list(patient_data.keys())}")
        try:
            deep_dive_result = await self.data_analysis_agent.run(
                patient_data=patient_data,
                prompt_details={"intent": "summarize_deep_dive"}
            )
            if deep_dive_result.get("status") == "success":
                summary_text = deep_dive_result.get("output", {}).get("summary_text", "No summary available from deep dive.")
                # Include conceptual agent insights if present
                deep_dive_sections = deep_dive_result.get("output", {}).get("deep_dive_sections", [])
                conceptual_insights = []
                for section in deep_dive_sections:
                    source = section.get("source", "Unknown Source")
                    if "Conceptual" in source or "Mock" in source: # Check for conceptual/mock agent insights
                        topic = section.get("topic", "Insight")
                        elaboration = section.get("elaboration", "No details.")
                        conceptual_insights.append(f"  - *Insight from {source.replace('Conceptual','').replace('Agent','').replace('Mock','').strip()}:* {topic} - {elaboration[:150]}...") # Truncate for brevity
                
                full_summary_text = summary_text
                if conceptual_insights:
                    full_summary_text += "\n  **Key Conceptual Agent Insights Included:**\n" + "\n".join(conceptual_insights)

                analysis_results.append(f"**Deep Dive Summary (Patient Data):**\n{full_summary_text}")

            else:
                error_msg = deep_dive_result.get("error", "Unknown error during deep dive.")
                logger.error(f"Deep dive summarization failed: {error_msg}")
                analysis_results.append(f"**Deep Dive Summary (Patient Data):** Error - {error_msg}")
        except Exception as e:
            logger.exception("Exception during deep dive summarization call:")
            analysis_results.append(f"**Deep Dive Summary (Patient Data):** Error - Exception during analysis: {str(e)}")

        for i, q_text in enumerate(questions):
            analysis_results.append(f"**Q: * {q_text.strip()}**")
            try:
                logger.info(f"Answering question '{q_text}' using DataAnalysisAgent. Patient data keys: {list(patient_data.keys())}")
                answer_result = await self.data_analysis_agent.run(
                    patient_data=patient_data,
                    prompt_details={"intent": "answer_question", "prompt": q_text}
                )
                if answer_result.get("status") == "success":
                    answer = answer_result.get("output", {}).get("answer_text", "No specific answer found.")
                    analysis_results.append(f"**A:** {answer}")
                else:
                    error_msg = answer_result.get("error", "Unknown error answering question.")
                    logger.error(f"Failed to answer question '{q_text}': {error_msg}")
                    analysis_results.append(f"**A:** Error analyzing question - {error_msg}")
            except Exception as e:
                logger.exception(f"Exception while answering question '{q_text}':")
                analysis_results.append(f"**A:** Error analyzing question - Exception: {str(e)}")
        
        return "\n\n".join(analysis_results)

    async def _get_clinical_trial_analysis(self, patient_data_for_trials: Optional[Dict[str, Any]], synthesis_context: str) -> str:
        """Gets clinical trial analysis using the ClinicalTrialAgent."""
        if not patient_data_for_trials or not isinstance(patient_data_for_trials, dict) or not patient_data_for_trials:
            logger.warning("Clinical trial analysis attempted with no or invalid patient_data_for_trials.")
            return "**Clinical Trial Analysis:**\nError - Patient data missing for trial search."

        logger.info(f"Performing clinical trial search. Patient data keys: {list(patient_data_for_trials.keys())}")
        trial_query = f"Find relevant clinical trials for a patient with {patient_data_for_trials.get('diagnosis', {}).get('primary', 'cancer')} considering their profile. Context: {synthesis_context}"
        
        try:
            trial_result = await self.clinical_trial_agent.run(
                patient_data=patient_data_for_trials,
                prompt_details={"query": trial_query, "task": "match_patient_to_trials"} # task might be specific to how CT agent parses
            )
            
            if trial_result.get("status") == "success":
                trials = trial_result.get("output", {}).get("trials_data", [])
                if not trials:
                    return "**Clinical Trial Analysis:**\nNo relevant clinical trials found based on the initial search."

                analysis_str = "**Clinical Trial Analysis:**\n**Relevant Clinical Trials Found:**\n"
                count = 0
                for trial in trials:
                    assessment = trial.get('assessment', {})
                    llm_verdict = assessment.get('llm_verdict', 'Not Assessed')
                    llm_reason = assessment.get('llm_reason', 'No LLM Task' if llm_verdict == 'Not Assessed' else 'N/A')
                    
                    title = trial.get('brief_title', trial.get('official_title', 'Unknown Title'))
                    nct_id = trial.get('nct_id', 'N/A')
                    
                    # Check if summary is needed (only if not explicitly provided by LLM assessment)
                    summary_to_display = trial.get('summary', 'Assessment not performed.')
                    if llm_verdict != 'Not Assessed' and llm_reason != 'No LLM Task':
                        summary_to_display = llm_reason # Use LLM reason as the summary if available
                    
                    analysis_str += f"- **{llm_verdict} ({nct_id})**: {title}. Summary: {summary_to_display}\n"
                    count += 1
                    if count >= 3 and len(trials) > 3: # Show first 3 in detail
                        analysis_str += f"...and {len(trials) - 3} more trials found.\n"
                        break
                return analysis_str
            else:
                error_msg = trial_result.get("error", "Unknown error during clinical trial search.")
                logger.error(f"Clinical trial search failed: {error_msg}")
                return f"**Clinical Trial Analysis:**\nError - {error_msg}"
        except Exception as e:
            logger.exception("Exception during clinical trial agent call:")
            return f"**Clinical Trial Analysis:**\nError - Exception during trial search: {str(e)}"

    async def run(self, initial_context: Dict[str, Any], initiator_note_analysis: str, patient_data_for_synthesis: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        logger.info("--- ConsultationSynthesizerAgent RUN START ---")
        logger.info(f"ARGUMENT initial_context IS_NONE: {initial_context is None}")
        if initial_context:
            logger.info(f"ARGUMENT initial_context IS DICT: {isinstance(initial_context, dict)}")
            if isinstance(initial_context, dict):
                logger.info(f"ARGUMENT initial_context HAS 'patient_data' KEY: {'patient_data' in initial_context}")
                if 'patient_data' in initial_context:
                    patient_data_in_initial_context = initial_context['patient_data']
                    logger.info(f"ARGUMENT initial_context['patient_data'] IS_NONE: {patient_data_in_initial_context is None}")
                    if patient_data_in_initial_context is not None:
                        logger.info(f"ARGUMENT initial_context['patient_data'] TYPE: {type(patient_data_in_initial_context)}")
                        logger.info(f"ARGUMENT initial_context['patient_data'] IS DICT: {isinstance(patient_data_in_initial_context, dict)}")
                        logger.info(f"ARGUMENT initial_context['patient_data'] IS EMPTY DICT: {patient_data_in_initial_context == {}}")
                        if isinstance(patient_data_in_initial_context, dict):
                             logger.info(f"ARGUMENT initial_context['patient_data'] KEYS: {list(patient_data_in_initial_context.keys())}")


        logger.info(f"ARGUMENT patient_data_for_synthesis IS_NONE: {patient_data_for_synthesis is None}")
        if patient_data_for_synthesis is not None:
            logger.info(f"ARGUMENT patient_data_for_synthesis TYPE: {type(patient_data_for_synthesis)}")
            logger.info(f"ARGUMENT patient_data_for_synthesis IS DICT: {isinstance(patient_data_for_synthesis, dict)}")
            logger.info(f"ARGUMENT patient_data_for_synthesis IS EMPTY DICT: {patient_data_for_synthesis == {}}")
            if isinstance(patient_data_for_synthesis, dict):
                logger.info(f"ARGUMENT patient_data_for_synthesis KEYS: {list(patient_data_for_synthesis.keys())}")
        
        patient_data_to_use = None
        # Prioritize patient_data_for_synthesis if it's a non-empty dictionary
        if patient_data_for_synthesis and isinstance(patient_data_for_synthesis, dict) and patient_data_for_synthesis: # Check if dict is not empty
            logger.info("CONDITION MET: Using patient_data_for_synthesis provided directly as it's a non-empty dict.")
            patient_data_to_use = patient_data_for_synthesis
        # Fallback to initial_context['patient_data'] if it's a non-empty dictionary
        elif initial_context and isinstance(initial_context, dict) and \
             'patient_data' in initial_context and \
             initial_context['patient_data'] and isinstance(initial_context['patient_data'], dict) and initial_context['patient_data']: # Check if dict is not empty
            logger.info("CONDITION MET: Extracting patient_data from initial_context as it's a non-empty dict.")
            patient_data_to_use = initial_context.get('patient_data')
        else:
            logger.error("CONDITION NOT MET: Patient data is effectively missing or empty in both patient_data_for_synthesis and initial_context. Deep analysis will fail.")
            # patient_data_to_use remains None

        logger.info(f"FINAL patient_data_to_use IS_NONE: {patient_data_to_use is None}")
        if patient_data_to_use is not None:
            logger.info(f"FINAL patient_data_to_use TYPE: {type(patient_data_to_use)}")
            if isinstance(patient_data_to_use, dict):
                logger.info(f"FINAL patient_data_to_use (first 5 keys): {list(patient_data_to_use.keys())[:5]}")
            else:
                logger.warning("FINAL patient_data_to_use is not a dict.")
        else:
            logger.warning("FINAL patient_data_to_use is None. Proceeding with error messages for data-dependent sections.")

        # Simplified patient identifier for the initial part of the synthesis
        patient_id_for_summary = "PATUNKNOWN"
        initiator_name = "Dr. A" # Placeholder, should ideally come from context
        consult_reason = "Consult on Patient Case" # Placeholder

        if initial_context and isinstance(initial_context, dict):
            initiator_name = initial_context.get('initiator', {}).get('name', initiator_name)
            consult_reason = initial_context.get('initialTrigger', {}).get('description', consult_reason)
            if patient_data_to_use and isinstance(patient_data_to_use, dict): # Check if it's a dict and not None
                patient_id_for_summary = patient_data_to_use.get('patientId', 'PATUNKNOWN')
                if not patient_id_for_summary or patient_id_for_summary == 'PATUNKNOWN':
                     logger.warning(f"patientId not found or is 'PATUNKNOWN' in patient_data_to_use. Keys: {list(patient_data_to_use.keys()) if patient_data_to_use else 'None'}")
            else:
                logger.warning("Cannot get patientId for summary because patient_data_to_use is None or not a dict.")
        else:
            logger.warning("initial_context is None or not a dict, using default initiator_name and consult_reason.")


        synthesis_intro = (
            f"The initiating physician ({initiator_name}) requests a consult for {patient_id_for_summary} "
            f"regarding {consult_reason}. Initiator note analysis highlighted key concerns.\n\n"
            f"The initial AI analysis of the initiator's note and a deep dive into the patient's record have been performed. "
            f"Additionally, a search for relevant clinical trials was conducted."
        )
        
        key_questions = await self._extract_key_questions(initiator_note_analysis, patient_data_to_use)
        
        # Get deep analysis (will handle None patient_data internally)
        deep_analysis_str = await self._get_deep_analysis_for_questions(key_questions, patient_data_to_use)
        
        # Get clinical trial analysis (will handle None patient_data internally)
        # For trial agent context, we can pass a summary of the main reason for consult
        synthesis_context_for_trials = f"Consult for {patient_id_for_summary} regarding {consult_reason}. Key points from initiator: {initiator_note_analysis[:200]}..."
        clinical_trial_analysis_str = await self._get_clinical_trial_analysis(patient_data_to_use, synthesis_context_for_trials)

        # Construct the full synthesis
        full_synthesis = f"{synthesis_intro}\n\n**Key Questions and Answers:**\n{deep_analysis_str}\n\n{clinical_trial_analysis_str}\n\n**Proposed Consultant Actions:**\n*   Review the comprehensive analysis including patient data insights and clinical trial findings.\n*   Consider the Diagnosis data snippet carefully.\n*   Draft initial thoughts on the patient's situation, incorporating all analyses.\n*   Prepare to discuss any remaining questions or unclear trial eligibilities in the chat. \n*   Verify specific trial details (e.g., NCT IDs mentioned) if proceeding with trial consideration.\n\n**Note:** This synthesis includes a deep analysis of the patient's data and clinical trial information. For an even more detailed analysis or to explore specific trials further, you can use the dedicated tools or click the \"Deep Analysis\" button if more focused options are available."
        
        # Using the DEFAULT_SYNTHESIS_PROMPT structure, but filling it programmatically for now
        # In future, might send all collected strings (synthesis_intro, deep_analysis_str, etc.) to an LLM
        # with the DEFAULT_SYNTHESIS_PROMPT as a guide for final formatting.
        # For now, we are manually assembling it.

        logger.info("Successfully generated consultation synthesis content string.")
        return {"status": "success", "result": {"text": full_synthesis}, "type": "consultation_synthesis_result"}

# Example usage (for testing purposes)
if __name__ == '__main__':
    import asyncio
    import json
    # Mock data
    mock_context = {
        "initialTrigger": {"description": "Review complex melanoma case"},
        "initiatorNote": "Patient progressed on BRAF/MEK, now stable on Pembro. Has BRAF, EGFR, TP53. Any trials?",
        "useAI": True,
        "relatedInfo": {"Diagnosis": {"primary": "Metastatic Melanoma", "stage": "IV"}},
        "consultFocusStatement": "Focus on long-term management and trial options.",
        "noteInitiatorId": "dr_a",
        "initiator": {"id": "dr_a", "name": "Adams"},
        "patientId": "PAT12345",
        "patient_data": { # Crucial for trial agent and deep dive
             "patientId": "PAT12345",
             "demographics": {"name": "Test Patient", "dob": "1970-01-01"},
             "diagnosis": {"primary": "Metastatic Melanoma", "stage": "IV", "histology": "Cutaneous Melanoma"},
             "biomarkers": [{"marker": "BRAF", "status": "V600E Mutation Detected"}, {"marker": "EGFR", "status": "Not Amplified"}, {"marker": "TP53", "status": "Mutation Detected"}],
             "currentMedications": [{"name": "Pembrolizumab", "dosage": "200mg", "frequency": "Q3W"}],
             "prior_treatments": [{"name": "Dabrafenib+Trametinib", "reason_stopped": "Progression"}],
             "notes": [{"type": "Progress Note", "date": "2023-10-15", "text": "Patient stable on Pembrolizumab. Discussed potential future options including clinical trials."}]
        }
    }
    mock_analysis = """
**Analysis:**
**1. Reasoning:** Pt has multiple mutations, resistance noted, progressing on current lines. Initiator asks about trials.
**2. Questions:** What are suitable clinical trials for this patient? What is the significance of the TP53 mutation in this context? Long-term plan?
**3. Concerns:** Resistance, limited standard options.
**4. Summary:** Complex melanoma case, needs new treatment plan, trial exploration is key.
"""
    
    async def test_run():
        agent = ConsultationSynthesizerAgent()
        result = await agent.run(mock_context, mock_analysis)
        print(json.dumps(result, indent=2))

    asyncio.run(test_run()) 