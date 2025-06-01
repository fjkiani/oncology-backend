"""
Agent responsible for analyzing patient data, generating summaries, and extracting insights.
"""

import google.generativeai as genai
import os
import json
from datetime import datetime
from typing import Any, Dict, Union

# Import the base agent interface and the LLM client
from backend.core.agent_interface import AgentInterface
from backend.core.llm_clients import GeminiClient

# Placeholder for Gemini/LangChain integration
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser

# Define supported tasks
SUMMARIZE = "summarize"
SUMMARIZE_DEEP_DIVE = "summarize_deep_dive" # New intent for two-stage summary

# --- Search Logic Configuration (Copied from EligibilityDeepDiveAgent - consider refactoring to shared utils later) ---
KNOWN_PGP_INHIBITORS = ['nelfinavir', 'indinavir', 'saquinavir', 'ritonavir', 'ketoconazole', 'itraconazole']
KNOWN_AZOLE_ANTIFUNGALS = ['itraconazole', 'ketoconazole', 'voriconazole', 'fluconazole', 'posaconazole']

SEARCH_TARGETS_PATIENT_SUMMARY = [
    {
        "id": "ecog_ps",
        "topic_keywords": ["performance status", "ecog", "karnofsky"], # Keywords to match a deep dive topic
        "search_fields": ["notes"],
        "search_type": "keyword_sentence",
        "keywords_for_data": ["ecog", "ps", "performance status", "kps", "karnofsky", "ambulatory", "bedridden", r"ecog\\s*\\d", r"kps\\s*\\d+"]
    },
    {
        "id": "platelet",
        "topic_keywords": ["platelet", "thrombocyte"],
        "search_fields": ["recentLabs"],
        "search_type": "lab_component",
        "lab_test_keywords": ["platelet", "plt"]
    },
    {
        "id": "anc",
        "topic_keywords": ["anc", "neutrophil"],
        "search_fields": ["recentLabs"],
        "search_type": "lab_component",
        "lab_test_keywords": ["anc", "absolute neutrophil", "neutrophils abs"]
    },
    {
        "id": "hemoglobin",
        "topic_keywords": ["hemoglobin", "hgb", "hb", "anemia"],
        "search_fields": ["recentLabs"],
        "search_type": "lab_component",
        "lab_test_keywords": ["hemoglobin", "hgb", "hb"]
    },
    {
        "id": "creatinine_renal",
        "topic_keywords": ["creatinine", "renal function", "kidney function", "crcl"],
        "search_fields": ["recentLabs", "notes"], # Check notes for mentions too
        "search_type": "compound", # Uses lab for values, notes for context
        "lab_test_keywords": ["creatinine", "crcl", "creat"],
        "notes_keywords_for_data": ["renal", "kidney", "crcl", "gfr", "dialysis", "nephropathy"]
    },
    {
        "id": "bilirubin_liver",
        "topic_keywords": ["bilirubin", "liver function", "hepatic function", "ast", "alt"],
        "search_fields": ["recentLabs", "notes"],
        "search_type": "compound",
        "lab_test_keywords": ["bilirubin", "bili", "total bilirubin", "alt", "sgpt", "ast", "sgot"],
        "notes_keywords_for_data": ["liver", "hepatic", "jaundice", "cirrhosis", "hepatitis"]
    },
    {
        "id": "diagnosis_staging",
        "topic_keywords": ["diagnosis", "staging", "cancer details"],
        "search_fields": ["diagnosis", "notes"],
        "search_type": "keyword_sentence", # Pull from diagnosis structure and notes
        "keywords_for_data": ["diagnosis", "stage", "grade", "histology", "metastatic", "primary site", "biopsy"]
    },
    {
        "id": "comorbidities_management",
        "topic_keywords": ["comorbidities", "medical history", "active problems"],
        "search_fields": ["medicalHistory", "notes"],
        "search_type": "keyword_sentence",
        "keywords_for_data": ["history of", "diagnosed with", "manages", "controlled with", "status post", "active problem"] # generic terms
    },
    {
        "id": "medications_concerns",
        "topic_keywords": ["medication", "drug", "pharmacy", "adherence", "interaction"],
        "search_fields": ["currentMedications", "notes"],
        "search_type": "compound", # List meds, check notes for adherence/issues
        "notes_keywords_for_data": ["adherence", "compliance", "side effect", "interaction", "unable to take", "missed dose"]
    },
    {
        "id": "recent_encounters_progress",
        "topic_keywords": ["recent visit", "encounter", "progress note", "follow-up"],
        "search_fields": ["notes"],
        "search_type": "keyword_sentence", # Summarize recent notes
        "keywords_for_data": ["follow-up", "assessment", "plan", "discussed", "impression", "findings", "interval history"]
    }
    # Add more targets specific to patient summary deep dive
]

DEFAULT_DEEP_DIVE_TOPICS = [
    "Patient's Primary Diagnosis and Staging Details",
    "Key Comorbidities and Their Current Management",
    "Recent Significant Lab Results and Trends (e.g. Creatinine, Hemoglobin, ANC, Platelets, Liver Enzymes)",
    "Current Medications and Potential Adherence/Interaction Concerns",
    "Summary of Recent Clinical Encounters and Progress"
]
# --- End Search Logic Configuration ---

class DataAnalysisAgent(AgentInterface):
    """ Analyzes clinical data and generates summaries or insights using Gemini. """

    def __init__(self):
        """ Initialize the agent and configure the Gemini client. """
        self.api_key = os.getenv("GOOGLE_API_KEY")
        
        # Robust check for API Key after dotenv load
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set. "
                             "Ensure it is defined in the .env file in the 'backend' directory.")

        # Configure the Gemini client
        try:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('gemini-1.5-flash')
            print(f"DataAnalysisAgent Initialized with Gemini Client (Model: {self.model.model_name}).")
        except Exception as e:
             print(f"Error configuring Gemini client: {e}")
             # Optionally re-raise or handle more gracefully depending on requirements
             raise RuntimeError(f"Failed to initialize Gemini client: {e}")
        
        # Placeholder for the LLM setup (Implementation in step 2b)
        # self.llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=self.api_key)
        
        # Placeholder for prompt templates
        # self.summary_prompt_template = self._create_summary_prompt()
        
        print("DataAnalysisAgent Initialization complete.") # Updated log message

    @property
    def name(self) -> str:
        return "data_analyzer"

    @property
    def description(self) -> str:
        return "Analyzes patient data using Gemini to generate clinical summaries, identify key findings, or answer specific questions about the data."

    async def run(self, patient_data: Dict[str, Any] = None, prompt_details: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Runs the data analysis based on the patient data and prompt details.
        Can perform a simple summary or a two-stage summary with a deep dive.

        Args:
            patient_data: Dictionary containing patient data.
            prompt_details: Dictionary containing intent and entities from the orchestrator.

        Returns:
            A dictionary with status and the generated output (summary or answer).
        """
        print(f"DataAnalysisAgent running with task: {prompt_details.get('intent') if prompt_details else None}")
        
        if not patient_data:
             return {
                "status": "failure",
                "output": None,
                "summary": "Failed: Patient data missing.",
                "error_message": "Patient data missing."
            }
            
        intent = prompt_details.get("intent", "summarize") if prompt_details else "summarize"
        original_prompt = prompt_details.get("prompt", "") if prompt_details else ""
        
        # --- Task mapping ---
        task_mapping = {
            "summarize": "summarize",
            "summarize_patient_record": "summarize", # Existing intent from UI
            SUMMARIZE_DEEP_DIVE: "summarize_deep_dive", # New intent for deep dive
            "answer_question": "answer_question",
            "data_query": "answer_question"
        }
        task = task_mapping.get(intent, "summarize")
        # --- End Task mapping ---

        if task == "summarize" or task == "summarize_deep_dive":
            if not self.model:
                print("Gemini model not available, using placeholder summary.")
                placeholder_summary = self._generate_placeholder_summary(patient_data)
                return {
                    "status": "success",
                    "output": {"summary_text": placeholder_summary, "deep_dive_sections": [] if task == "summarize_deep_dive" else None},
                    "summary": "Generated placeholder summary (Gemini unavailable)."
                }
            try:
                initial_summary_text = await self._call_llm_for_summary(patient_data)
                
                if task == "summarize_deep_dive":
                    print("Performing deep dive summarization...")
                    # For now, use predefined topics. Could be dynamically suggested by initial summary later.
                    deep_dive_topics = DEFAULT_DEEP_DIVE_TOPICS
                    
                    # Prepare patient data snippet for deep dive (can be less comprehensive than full data)
                    # For simplicity, reusing a similar snippet logic as EligibilityDeepDiveAgent
                    patient_data_snippet = {
                        "patientId": patient_data.get("patientId"),
                        "diagnosis": patient_data.get("diagnosis"),
                        "medicalHistory": patient_data.get("medicalHistory", [])[:5],
                        "currentMedications": patient_data.get("currentMedications", [])[:5],
                        "allergies": patient_data.get("allergies", []),
                        "recentLabs": patient_data.get("recentLabs", []), # Pass all labs for deep dive
                        "notes": patient_data.get("notes", []), # Pass all notes
                        "mutations": patient_data.get("mutations", [])
                    }
                    patient_data_snippet = {k: v for k, v in patient_data_snippet.items() if v}

                    deep_dive_results = await self._perform_deep_summary_dive(
                        topics=deep_dive_topics,
                        full_patient_data=patient_data, # For internal search
                        patient_data_snippet_for_llm=patient_data_snippet # For LLM prompt
                    )
                    return {
                        "status": "success",
                        "output": {"summary_text": initial_summary_text, "deep_dive_sections": deep_dive_results},
                        "summary": "Successfully generated initial summary and deep dive sections using Gemini."
                    }
                else: # Just a standard summary
                    return {
                        "status": "success",
                        "output": {"summary_text": initial_summary_text},
                        "summary": "Successfully generated summary using Gemini."
                    }

            except Exception as e:
                 print(f"Error calling Gemini for summary/deep_dive: {e}")
                 return {
                    "status": "failure",
                    "output": None,
                    "summary": f"Failed to generate summary/deep_dive via Gemini: {e}",
                    "error_message": str(e)
                }
        elif task == "answer_question":
            user_prompt = original_prompt
            if not user_prompt:
                 return {
                    "status": "failure", "output": None,
                    "summary": "Failed: No question prompt provided for answer_question task.", "error_message": "Missing prompt for question."
                }
            
            if not self.model:
                # Fallback to placeholder if API key/model is missing
                print("Gemini model not available, using placeholder answer.")
                return {
                    "status": "requires_review",
                    "output": {"answer_text": f"[Placeholder Answer for: \"{user_prompt}\"] (Gemini unavailable)"},
                    "summary": "Generated placeholder answer (Gemini unavailable)."
                }
            try:
                # Call the actual LLM for question answering
                answer_text = await self._call_llm_for_question(patient_data, user_prompt)
                return {
                    "status": "success", # Or potentially 'requires_review' depending on confidence?
                    "output": {"answer_text": answer_text},
                    "summary": "Successfully generated answer using Gemini."
                }
            except Exception as e:
                 print(f"Error calling Gemini for question answering: {e}")
                 return {
                    "status": "failure", "output": None,
                    "summary": f"Failed to answer question via Gemini: {e}", "error_message": str(e)
                }
        else:
            return {
                "status": "failure",
                "output": None,
                "summary": f"Unsupported task: {task}",
                "error_message": f"DataAnalysisAgent does not support task: {task}"
            }
            
    def _generate_summary_prompt(self, patient_data: dict) -> str:
        """ Constructs the detailed prompt for clinical summarization. """
        try:
            dob = datetime.strptime(patient_data.get("demographics", {}).get("dob", ""), "%Y-%m-%d")
            today = datetime.today()
            age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
            age_str = f"{age} years old"
        except ValueError:
            age_str = "Age N/A"

        prompt = f"""
Act as a medical professional reviewing a patient's electronic health record.
Based *only* on the following structured patient data, generate a concise clinical summary suitable for a quick overview during a clinical encounter.

Patient Data:
{json.dumps(patient_data, indent=2)}

Instructions for the summary:
1. Start with a brief statement including patient name, age ({age_str}), and primary diagnosis with status.
2. Briefly mention relevant active comorbidities from medical history.
3. Summarize key findings from the most recent progress notes.
4. Highlight any critical or significantly abnormal recent lab results (mention specific values and flags).
5. Summarize key findings and recommendations from recent imaging reports.
6. Note any significant events or concerning trends reported in patient-generated health data, if available.
7. Conclude with the patient's overall current status or immediate next steps if clearly stated in the notes (e.g., awaiting biopsy, scheduled for cycle 2).
8. Keep the summary concise, objective, and focused on clinically relevant information.
9. Do not infer information not present in the provided data.

Clinical Summary:
"""
        return prompt

    async def _call_llm_for_summary(self, patient_data: dict) -> str:
        """ Calls the Gemini API to generate the summary. """
        if not self.model:
            raise RuntimeError("Gemini model not initialized. Check API key.")
            
        prompt = self._generate_summary_prompt(patient_data)
        
        print("Sending summarization prompt to Gemini...")
        try:
            # Use generate_content_async for async FastAPI
            response = await self.model.generate_content_async(prompt)
            summary = response.text
            print("Received summary from Gemini.")
            return summary
        except Exception as e:
            print(f"Error during Gemini API call: {e}")
            # Re-raise the exception to be caught by the run method
            raise e

    def _generate_placeholder_summary(self, patient_data: dict) -> str:
        """ Creates a simple placeholder summary string. """
        name = patient_data.get("demographics", {}).get("name", "N/A")
        diagnosis = patient_data.get("diagnosis", {}).get("primary", "N/A")
        return f"Placeholder Summary for {name}. Diagnosis: {diagnosis}. Analysis complete (simulation)."

    def _generate_question_prompt(self, patient_data: dict, question: str) -> str:
        """ Constructs the prompt for answering specific questions based on patient data. """
        prompt = f"""
Act as a clinical data assistant. Based *only* on the provided Patient Data JSON object, answer the user's question accurately and concisely.
If the answer cannot be found directly within the provided data, state that the information is not available in the record.
Do not infer information or make assumptions beyond what is present in the data.

Patient Data:
{json.dumps(patient_data, indent=2)}

User Question: {question}

Answer:
"""
        return prompt

    async def _call_llm_for_question(self, patient_data: dict, question: str) -> str:
        """ Calls the Gemini API to answer a specific question based on patient data. """
        if not self.model:
            raise RuntimeError("Gemini model not initialized. Check API key.")
            
        prompt = self._generate_question_prompt(patient_data, question)
        
        print("Sending question prompt to Gemini...")
        try:
            response = await self.model.generate_content_async(prompt)
            answer = response.text
            print(f"Received answer from Gemini for question '{question}'.")
            return answer
        except Exception as e:
            print(f"Error during Gemini API call for question answering: {e}")
            raise e

    # --- Future methods for LLM interaction ---
    # def _create_summary_prompt(self):
    #     # Define the LangChain prompt template here
    #     pass
    
    # async def _call_llm_for_summary(self, patient_data_str: str):
    #     # chain = self.summary_prompt_template | self.llm | StrOutputParser()
    #     # result = await chain.ainvoke({"patient_data": patient_data_str})
    #     # return result
    #     pass

    # --- Methods for Deep Dive Summarization ---
    async def _perform_deep_summary_dive(self, topics: list[str], full_patient_data: dict, patient_data_snippet_for_llm: dict) -> list[dict]:
        """
        Performs a deep dive analysis for a list of topics using the patient data.
        """
        import asyncio # Ensure asyncio is imported

        dive_results = []
        if not self.model:
            # Return placeholder if model isn't available
            return [{"topic": topic, "elaboration": "[Deep Dive Elaboration Placeholder - Gemini unavailable]", "evidence_snippets": []} for topic in topics]

        tasks = []
        for topic in topics:
            tasks.append(self._analyze_single_topic_for_deep_dive_async(
                topic=topic,
                full_patient_data=full_patient_data,
                patient_data_snippet_for_llm=patient_data_snippet_for_llm
            ))
        
        if tasks:
            try:
                # Using asyncio.gather for concurrent execution
                topic_analysis_results = await asyncio.gather(*tasks)
                dive_results = [res for res in topic_analysis_results if res] 
            except Exception as e:
                print(f"Error during concurrent deep dive topic analysis: {e}")
                # Populate with error state for each original topic
                dive_results = [
                    {"topic": topic, "elaboration": f"Error during deep dive analysis: {e}", "evidence_snippets": [], "error": True}
                    for topic in topics
                ]
        
        # --- Add Conceptual Mock Agent Insights for the "Holistic Healing Demo" ---
        patient_id_for_demo = full_patient_data.get("patientId")
        conceptual_insights = []

        # For the demo, we'll focus on PAT_ALPHA_001 (or PAT12345 if that's the active demo ID)
        # Let's assume PAT_ALPHA_001 is the new holistic demo patient ID.
        # If you are testing with PAT12345, you can change this condition.
        if patient_id_for_demo == "PAT_ALPHA_001" or patient_id_for_demo == "PAT12345": 
            print(f"DataAnalysisAgent: Adding conceptual mock insights for patient {patient_id_for_demo}")
            
            comp_therapy_summary = self._get_conceptual_comparative_therapy_summary(full_patient_data)
            if comp_therapy_summary:
                conceptual_insights.append({
                    "topic": "Conceptual Insight: Comparative Therapy Options (Mock)",
                    "elaboration": comp_therapy_summary,
                    "evidence_snippets": [], "error": False, "source": "ComparativeTherapyAgentMock"
                })

            crispr_summary = self._get_conceptual_crispr_summary(full_patient_data)
            if crispr_summary:
                conceptual_insights.append({
                    "topic": "Conceptual Insight: CRISPR Research Perspective (Mock)",
                    "elaboration": crispr_summary,
                    "evidence_snippets": [], "error": False, "source": "CRISPRAgentMock"
                })

            integrative_summary = self._get_conceptual_integrative_medicine_summary(full_patient_data)
            if integrative_summary:
                conceptual_insights.append({
                    "topic": "Conceptual Insight: Integrative Medicine Considerations (Mock)",
                    "elaboration": integrative_summary,
                    "evidence_snippets": [], "error": False, "source": "IntegrativeMedicineAgentMock"
                })
        
        if conceptual_insights:
            dive_results.extend(conceptual_insights)
            # The main run method should also be aware of these to potentially pass to UI
            # For now, they are part of dive_results which is fine.

        return dive_results

    async def _analyze_single_topic_for_deep_dive_async(self, topic: str, full_patient_data: dict, patient_data_snippet_for_llm: dict) -> dict:
        """
        Analyzes a single topic for deep dive using LLM and internal data search.
        """
        import re # Ensure re is imported
        result = {
            "topic": topic,
            "elaboration": "Could not generate elaboration.",
            "evidence_snippets": [], # Snippets from internal search used for LLM context
            "error": False,
            "source": "DataAnalysisAgent_LLM" # Added source for LLM generated topics
        }

        # 1. Internal Data Search (similar to EligibilityDeepDiveAgent)
        found_context_for_llm = []
        search_target_matched_topic = None

        for target_config in SEARCH_TARGETS_PATIENT_SUMMARY:
            topic_keywords = target_config.get("topic_keywords", [])
            if any(kw.lower() in topic.lower() for kw in topic_keywords):
                search_target_matched_topic = target_config
                break
        
        if search_target_matched_topic:
            search_type = search_target_matched_topic.get("search_type")
            search_fields = search_target_matched_topic.get("search_fields", [])
            
            # Using full_patient_data for the search
            for field_name in search_fields:
                field_data_from_full_patient_record = full_patient_data.get(field_name)
                if field_data_from_full_patient_record is None:
                    continue

                try:
                    if search_type == "lab_component":
                        lab_keywords = search_target_matched_topic.get("lab_test_keywords", [])
                        found_context_for_llm.extend(self._search_lab_component(field_data_from_full_patient_record, lab_keywords))
                    elif search_type == "keyword_sentence" or search_type == "compound": # For compound, notes search part
                        keywords_for_data = search_target_matched_topic.get("keywords_for_data", search_target_matched_topic.get("notes_keywords_for_data", []))
                        # Assuming keyword_sentence search for notes in compound type
                        if isinstance(field_data_from_full_patient_record, list) and all(isinstance(item, dict) for item in field_data_from_full_patient_record): # Likely notes
                             found_context_for_llm.extend(self._search_keyword_sentence(field_data_from_full_patient_record, keywords_for_data))
                        elif isinstance(field_data_from_full_patient_record, dict) and field_name == "diagnosis": # Special handling for diagnosis object
                            # Convert diagnosis object into a searchable string or list of strings
                            diag_text_list = [f"{k}: {v}" for k,v in field_data_from_full_patient_record.items() if isinstance(v, str)]
                            diag_notes_equiv = [{"text": ". ".join(diag_text_list)}] # Make it look like a note
                            found_context_for_llm.extend(self._search_keyword_sentence(diag_notes_equiv, keywords_for_data))

                    if search_type == "compound" and field_name == "recentLabs": # For compound, lab search part
                        lab_keywords = search_target_matched_topic.get("lab_test_keywords", [])
                        found_context_for_llm.extend(self._search_lab_component(field_data_from_full_patient_record, lab_keywords))
                    
                    if search_type == "compound" and field_name == "currentMedications":
                        # Simply list medications, specific checks are harder without LLM here
                         if isinstance(field_data_from_full_patient_record, list):
                            for med_entry in field_data_from_full_patient_record[:5]: # Limit shown meds
                                if isinstance(med_entry, dict) and "name" in med_entry:
                                     found_context_for_llm.append({
                                         "source": "Current Medications (List)",
                                         "context": f"{med_entry.get('name')} {med_entry.get('dosage','')} {med_entry.get('frequency','')}"
                                     })


                except Exception as search_ex:
                    print(f"Error during internal search for deep dive topic '{topic}', target '{search_target_matched_topic.get('id')}': {search_ex}")
            
            # Store objects with both snippet and source
            result["evidence_snippets"] = [
                {"snippet": item.get("context"), "source": item.get("source")} 
                for item in found_context_for_llm[:3] # Limit to top 3 for prompt brevity
            ]

        # 2. LLM Call for Elaboration
        try:
            # Prepare evidence for the prompt, now including sources if available
            context_for_prompt_str = ""
            if result["evidence_snippets"]:
                formatted_snippets = []
                for ev in result["evidence_snippets"]:
                    source_text = f"(Source: {ev.get('source', 'Unknown')})" if ev.get('source') else ""
                    formatted_snippets.append(f"- {ev.get('snippet', '')} {source_text}")
                context_for_prompt_str = "\\n".join(formatted_snippets)
            else:
                context_for_prompt_str = "No specific snippets found by internal search. Use general patient data."

            prompt = f"""
As a clinical expert, provide a detailed elaboration on the following topic concerning a patient.
Base your elaboration ONLY on the provided Patient Data Snippet and any relevant Context Snippets found by an internal search.

Topic to Elaborate: {topic}

Patient Data Snippet (General Overview):
```json
{json.dumps(patient_data_snippet_for_llm, indent=2, default=str)}
```

Context Snippets (Potentially relevant details found by internal search for the topic):
{context_for_prompt_str}

Instructions:
- Focus on providing a comprehensive, insightful, and clinically relevant elaboration on the Topic.
- Synthesize information from both the general Patient Data Snippet and the specific Context Snippets if provided.
- If Context Snippets are available, prioritize them for details related to the topic.
- Maintain a professional, analytical tone.
- If the provided data is insufficient to elaborate meaningfully on the topic, clearly state that.
- The elaboration should be a narrative paragraph or a few paragraphs. Do not use lists unless appropriate for the content (e.g. listing medications).
- Do NOT repeat the topic in your answer. Just provide the elaboration.
- Do NOT include a sign-off or greeting.

Detailed Elaboration:
"""

            if not self.model:
                raise ValueError("LLM Client (self.model) not initialized in DataAnalysisAgent")

            response = await self.model.generate_content_async(prompt) # Add appropriate generation_config and safety_settings if needed
            
            elaboration_text = ""
            if response.parts:
                elaboration_text = response.parts[0].text
            elif hasattr(response, 'text'):
                elaboration_text = response.text
            else:
                elaboration_text = "Error: LLM response structure invalid or missing text for deep dive topic."
            
            result["elaboration"] = elaboration_text.strip()

        except Exception as e:
            print(f"Error during LLM call for deep dive topic '{topic}': {e}")
            result["elaboration"] = f"Error generating elaboration for this topic: {e}"
            result["error"] = True
        
        return result

    # --- Internal Search Helper Methods (Copied from EligibilityDeepDiveAgent - consider refactoring) ---
    def _search_lab_component(self, labs_data: list[dict], lab_keywords: list[str]) -> list[dict]:
        """Searches lab data for specific components."""
        import re # Ensure re is imported
        findings = []
        if not labs_data or not isinstance(labs_data, list):
            return findings
            
        keywords_lower = [k.lower() for k in lab_keywords]
        
        for lab_panel in labs_data:
            if not isinstance(lab_panel, dict): continue
            panel_name = lab_panel.get('panelName', 'Unknown Panel')
            lab_date_str = lab_panel.get('resultDate', lab_panel.get('orderDate', '?'))
            # Try to parse date for recency sorting later if needed, but keep original str for display
            # For now, just use the string.
            
            for component in lab_panel.get('components', []):
                if not isinstance(component, dict): continue
                test_name = component.get('test', '')
                if not test_name or not isinstance(test_name, str): continue
                
                test_name_lower = test_name.lower()
                for keyword in keywords_lower:
                    if keyword in test_name_lower:
                        findings.append({
                            "source": f"Lab Panel '{panel_name}' ({lab_date_str})",
                            "context": f"{test_name}: {component.get('value', 'N/A')} {component.get('unit', '')} (Ref: {component.get('refRange', '?')}, Flag: {component.get('flag', 'N/A')})",
                            "match": keyword,
                            "raw_component": component # For potential structured use later
                        })
                        break 
        return findings

    def _search_keyword_sentence(self, notes_data: list[dict], keywords: list[str]) -> list[dict]:
        """Searches notes for sentences containing specific keywords or regex patterns."""
        import re # Ensure re is imported
        findings = []
        if not notes_data or not isinstance(notes_data, list):
            return findings

        keywords_lower = [k.lower() for k in keywords if isinstance(k, str)] 
        regex_patterns = [re.compile(p, re.IGNORECASE) for p in keywords if not isinstance(p, str) and isinstance(p, str)] # Ensure p is str

        for note in notes_data: # Assuming notes_data is a list of dicts, each with a 'text' field
            if not isinstance(note, dict): continue
            note_text_content = note.get('text', note.get('note_text', '')) # Adapt to potential key names
            if not note_text_content or not isinstance(note_text_content, str): continue
            
            note_source_name = f"Note ({note.get('date', note.get('encounterDate', '?'))} by {note.get('provider', note.get('author', '?'))})"
            
            sentences = [s.strip() for s in re.split(r'(?<=[.!?\\n])\\s+', note_text_content) if s and s.strip()]
            for sentence in sentences:
                sentence_lower = sentence.lower()
                found_match_keyword = None
                # Check basic keywords
                for keyword in keywords_lower:
                    if keyword in sentence_lower:
                        found_match_keyword = keyword
                        break
                # Check regex patterns if keyword not found
                if not found_match_keyword:
                    for pattern in regex_patterns:
                        if pattern.search(sentence): 
                            found_match_keyword = pattern.pattern 
                            break
                            
                if found_match_keyword:
                    findings.append({
                        "source": note_source_name,
                        "context": sentence,
                        "match": found_match_keyword
                    })
        return findings
    # --- End Search Helpers ---

    # --- Placeholder Methods for Conceptual Mock Agent Summaries (Task 2.2.1) ---
    def _get_conceptual_comparative_therapy_summary(self, patient_data: Dict[str, Any]) -> Union[str, None]:
        """ Returns a mock summary for the Comparative Therapy Agent. """
        # Basic check for demo patient / condition
        diagnosis_info = patient_data.get("diagnosis", {}).get("primary", "").lower()
        patient_id = patient_data.get("patientId")

        if patient_id == "PAT_ALPHA_001" or patient_id == "PAT12345": # Specific to our demo patient
            if "nsclc" in diagnosis_info and "egfr" in str(patient_data.get("mutations", [])).lower():
                 return "For EGFR Exon 19 deletion positive NSCLC, targeted therapies like Osimertinib are a standard first-line consideration, often demonstrating superior efficacy and a different side effect profile compared to traditional chemotherapy. Clinical trials continue to explore optimal sequencing and combination strategies."
        return None

    def _get_conceptual_crispr_summary(self, patient_data: Dict[str, Any]) -> Union[str, None]:
        """ Returns a mock summary for the CRISPR Agent. """
        patient_id = patient_data.get("patientId")
        if patient_id == "PAT_ALPHA_001" or patient_id == "PAT12345": # Specific to our demo patient
            if "egfr" in str(patient_data.get("mutations", [])).lower(): # Check if there's any EGFR mention in mutations
                return "Conceptual Note: CRISPR-Cas9 gene editing research for somatic mutations, including EGFR variants, is an active area of pre-clinical investigation. Currently, this remains experimental and is not a clinical treatment option for this patient. Approved targeted therapies are the standard."
        return None

    def _get_conceptual_integrative_medicine_summary(self, patient_data: Dict[str, Any]) -> Union[str, None]:
        """ Returns a mock summary for the Integrative Medicine Agent. """
        # For this demo, let's make it always return for PAT_ALPHA_001 / PAT12345 if they have notes indicating anxiety, or just generally for the demo patient.
        patient_id = patient_data.get("patientId")
        notes_text_lower = "".join([note.get("text", "").lower() for note in patient_data.get("notes", [])])

        if patient_id == "PAT_ALPHA_001" or patient_id == "PAT12345":
            if "anxiety" in notes_text_lower or "stress" in notes_text_lower or "distress" in notes_text_lower : # Check for anxiety in notes
                return "Integrative Support: For patients like this managing NSCLC and expressed anxiety, incorporating integrative approaches could be beneficial. This may include mindfulness techniques for stress reduction, acupuncture for potential symptom relief (e.g., nausea, pain if they arise), and personalized nutritional counseling to support overall well-being during treatment."
            # Fallback for demo patient even if no anxiety explicit in simple mock notes
            return "Integrative Support: For patients undergoing cancer treatment, integrative approaches such as mindfulness, acupuncture for symptom management, and nutritional counseling can be beneficial for overall well-being and coping with treatment-related stress. These should be discussed with the care team."
        return None
    # --- End Conceptual Mock Placeholders ---

# Example usage (for testing purposes)
# if __name__ == '__main__':
# import asyncio
# async def main_test():
# agent = DataAnalysisAgent()
# # ... (rest of the test setup) ...
#         result = await agent.run(
#             patient_data=mock_patient_data_for_deep_dive, # Define this mock data
#             prompt_details={"intent": SUMMARIZE_DEEP_DIVE}
# )
# print(json.dumps(result, indent=2))
# asyncio.run(main_test()) 