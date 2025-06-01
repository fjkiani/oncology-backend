import os
from typing import Dict, Any
# Import necessary LLM or data source utilities later
# from ..core.llm_clients import GeminiClient # Old relative import
# from core.llm_clients import GeminiClient # Corrected absolute import
from backend.core.llm_clients import GeminiClient # Fixed absolute import with backend prefix

class ComparativeTherapyAgent:
    """
    Agent responsible for comparing treatment therapies based on criteria.
    """
    def __init__(self):
        """
        Initialize the agent, potentially loading LLM clients or data connectors.
        """
        # Initialize LLM client (e.g., Gemini)
        self.llm_client = GeminiClient() 
        print("ComparativeTherapyAgent initialized with GeminiClient.")

    async def run(self, patient_id: str, therapy_a: str, therapy_b: str, focus_criteria: list[str], context: Dict[str, Any] = None) -> str:
        """
        Executes the therapy comparison task.

        Args:
            patient_id: The ID of the patient for context.
            therapy_a: Name or description of the first therapy.
            therapy_b: Name or description of the second therapy.
            focus_criteria: List of criteria to focus the comparison on (e.g., ["side_effects", "efficacy"]).
            context: Optional additional context (e.g., recent chat messages).

        Returns:
            A formatted string containing the comparison results or an error message.
        """
        print(f"Running ComparativeTherapyAgent for patient {patient_id}")
        print(f"Comparing: '{therapy_a}' vs '{therapy_b}'")
        print(f"Focusing on: {focus_criteria}")

        try:
            # 1. TODO: Fetch relevant patient data (mock or real) using patient_id
            patient_data = f"Patient {patient_id} relevant context placeholder." # Placeholder

            # 2. TODO: Construct a detailed prompt for the LLM
            prompt = self._build_comparison_prompt(patient_data, therapy_a, therapy_b, focus_criteria)
            
            # 3. Call the LLM to get the comparison
            print(f"Sending prompt to LLM:\n{prompt}") # Log the prompt for debugging
            comparison_text = await self.llm_client.generate(prompt)
            # comparison_text = f"Placeholder comparison for {therapy_a} vs {therapy_b} focusing on {', '.join(focus_criteria)}.\nDisclaimer: AI-generated placeholder. Requires clinical verification." # Placeholder removed

            # 4. Format the output
            formatted_result = self._format_result(comparison_text, therapy_a, therapy_b)

            return formatted_result

        except Exception as e:
            print(f"Error in ComparativeTherapyAgent: {e}")
            return f"Error generating therapy comparison: {e}"

    def _build_comparison_prompt(self, patient_data: str, therapy_a: str, therapy_b: str, focus_criteria: list[str]) -> str:
        """Helper function to build the LLM prompt."""
        # TODO: Implement robust prompt engineering
        criteria_str = ", ".join(focus_criteria)
        prompt = (
            f"Given the following patient context: {patient_data}\n\n"
            f"Provide a structured comparison between '{therapy_a}' and '{therapy_b}'.\n"
            f"Focus specifically on these criteria: {criteria_str}.\n"
            f"Present the comparison clearly. If possible from your knowledge base, mention relative differences or key points for each criterion.\n"
            f"Include a disclaimer that this is AI-generated and requires clinical verification."
        )
        return prompt

    def _format_result(self, comparison_text: str, therapy_a: str, therapy_b: str) -> str:
        """Helper function to format the final output string."""
        # TODO: Implement better formatting if needed (e.g., Markdown)
        header = f"**Therapy Comparison: {therapy_a} vs. {therapy_b}**\n\n"
        # Disclaimer is expected to be part of the LLM response based on the prompt
        return header + comparison_text

# Example usage (for testing purposes)
if __name__ == '__main__':
    import asyncio

    async def main():
        agent = ComparativeTherapyAgent()
        result = await agent.run(
            patient_id="PAT123", 
            therapy_a="Paclitaxel/Carboplatin", 
            therapy_b="Pemetrexed/Cisplatin", 
            focus_criteria=["common side effects", "efficacy in NSCLC Adenocarcinoma"]
        )
        print("\n--- Agent Result ---")
        print(result)

    asyncio.run(main()) 