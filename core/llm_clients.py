import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
# Assumes .env file is at the project root (two levels up from core)
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '..', '.env')
load_dotenv(dotenv_path=dotenv_path)

class GeminiClient:
    """Client for interacting with the Google Gemini API."""

    def __init__(self, model_name="gemini-1.5-flash"):
        """Initializes the Gemini client.

        Args:
            model_name (str): The name of the Gemini model to use.
                              Defaults to "gemini-1.5-flash".
        """
        self.api_key = os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set.")
        
        try:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(model_name)
            print(f"GeminiClient initialized with model: {model_name}")
        except Exception as e:
            print(f"Error initializing Gemini model: {e}")
            raise RuntimeError(f"Failed to initialize Gemini model ({model_name}): {e}") from e

    async def generate(self, prompt: str) -> str:
        """Generates text using the configured Gemini model.

        Args:
            prompt: The input prompt string.

        Returns:
            The generated text response.
        
        Raises:
            Exception: If the API call fails.
        """
        try:
            # print(f"GeminiClient sending prompt:\n---\n{prompt}\n---") # Optional debug logging
            response = await self.model.generate_content_async(prompt)
            # print(f"GeminiClient received response parts: {response.parts}") # Optional debug logging
            # Handle potential lack of text in response parts gracefully
            if response.parts:
                # Assuming the first part contains the primary text response
                # This might need adjustment based on specific model behavior or safety settings
                return response.text # Use response.text for convenience
            else:
                # Handle cases where response might be blocked or empty
                block_reason = response.prompt_feedback.block_reason if response.prompt_feedback else 'unknown'
                print(f"Warning: Gemini response was empty or blocked. Reason: {block_reason}")
                return f"[AI response blocked or empty - Reason: {block_reason}]"
        except Exception as e:
            print(f"Error during Gemini API call: {e}")
            # Re-raise or handle more specifically as needed
            raise Exception(f"Gemini API generation failed: {e}") from e

# Example usage (for testing purposes - requires running from a context where imports work)
# if __name__ == '__main__':
#     import asyncio
# 
#     async def test_client():
#         try:
#             client = GeminiClient()
#             result = await client.generate("Explain the concept of a Large Language Model simply.")
#             print("\n--- Client Result ---")
#             print(result)
#         except Exception as e:
#             print(f"Test failed: {e}")
# 
#     asyncio.run(test_client()) 