import os
from typing import Dict, Any
# from core.llm_clients import GeminiClient
from backend.core.llm_clients import GeminiClient

class PatientEducationDraftAgent:
    """
    Agent responsible for drafting patient-friendly explanations.
    """
    def __init__(self):
        """
        Initialize the agent, loading the LLM client.
        """
        self.llm_client = GeminiClient()
        print("PatientEducationDraftAgent initialized with GeminiClient.")

    async def run(self, topic: str, context: Dict[str, Any] = None) -> str:
        """
        Executes the patient education drafting task.

        Args:
            topic: The specific medical topic to explain.
            context: Optional additional context (e.g., recent chat messages, patient details).

        Returns:
            A formatted string containing the draft explanation or an error message.
        """
        print(f"Running PatientEducationDraftAgent for topic: {topic}")

        try:
            # 1. TODO: Extract/prepare relevant context if provided
            context_str = f"Context: {str(context)}" if context else "No additional context provided."

            # 2. Build the prompt for the LLM
            prompt = self._build_draft_prompt(topic, context_str)

            # 3. Call the LLM to get the draft explanation
            print(f"Sending prompt to LLM:\n{prompt}") # Log the prompt for debugging
            draft_text = await self.llm_client.generate(prompt)

            # 4. Format the output clearly as a draft
            formatted_result = self._format_result(draft_text, topic)

            return formatted_result

        except Exception as e:
            print(f"Error in PatientEducationDraftAgent: {e}")
            return f"Error generating patient education draft: {e}"

    def _build_draft_prompt(self, topic: str, context_str: str) -> str:
        """Helper function to build the LLM prompt for drafting patient education.

        Args:
            topic: The core topic to explain.
            context_str: A string representation of any relevant context.

        Returns:
            The prompt string for the LLM.
        """
        # TODO: Refine prompt engineering for better tone, reading level, structure.
        prompt = (
            f"You are an empathetic medical assistant writing for a patient. Your task is to draft an explanation about the following topic.\n"
            f"**Topic:** {topic}\n\n"
            f"**Relevant Context:** {context_str}\n\n"
            f"**Instructions:**\n"
            f"- Explain the topic clearly and simply, using plain language (aim for 8th-grade reading level).\n"
            f"- Avoid complex medical jargon where possible, or explain it briefly if necessary.\n"
            f"- Structure the explanation logically (e.g., What it is, Why it's important/relevant, What to expect/do).\n"
            f"- Use a supportive and encouraging tone.\n"
            f"- IMPORTANT: This is a DRAFT for a clinician to review. Do not add any sign-off like 'Sincerely' or 'Your Doctor'.\n"
            f"- Do NOT include phrases like 'This draft is for...' or 'Here is the draft...'. Just provide the explanation text itself.\n"
            f"\n**DRAFT Explanation:**\n"
        )
        return prompt

    def _format_result(self, draft_text: str, topic: str) -> str:
        """Helper function to format the final output string, clearly marking it as a draft."""
        header = f"**--- DRAFT: Patient Explanation (Review Required) ---**\n\n**Subject:** Understanding {topic}\n\n"
        footer = "\n\n**--- END DRAFT ---**"
        return header + draft_text + footer

# Example usage (for testing purposes)
if __name__ == '__main__':
    import asyncio

    async def main():
        agent = PatientEducationDraftAgent()
        result = await agent.run(
            topic="Paclitaxel/Carboplatin Chemotherapy Side Effects", 
            context={"Patient Age": 65, "Diagnosis": "Stage III NSCLC", "Note": "Patient asking about managing nausea."}
        )
        print("\n--- Agent Result ---")
        print(result)

    asyncio.run(main()) 