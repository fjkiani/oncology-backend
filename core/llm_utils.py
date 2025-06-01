"""Utility functions for direct interaction with LLMs."""

import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env')) 
# Assumes .env is in the parent 'backend' directory

# --- Gemini Client Initialization --- 
# Initialize client globally or within the function (consider implications)
# Global initialization is generally more efficient if the function is called often.
API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL_NAME = 'gemini-1.5-flash' # Or fetch from env vars

GEMINI_MODEL = None
if API_KEY:
    try:
        genai.configure(api_key=API_KEY)
        GEMINI_MODEL = genai.GenerativeModel(GEMINI_MODEL_NAME)
        print(f"llm_utils: Gemini Client Initialized (Model: {GEMINI_MODEL_NAME})")
    except Exception as e:
        print(f"llm_utils: Error configuring Gemini client: {e}")
        GEMINI_MODEL = None # Ensure model is None if init fails
else:
    print("llm_utils: GOOGLE_API_KEY not found. Gemini client not initialized.")

# Safety settings (adjust as needed)
SAFETY_SETTINGS = { 
    # Defaults are usually reasonable, but customize if blocking is too aggressive/lenient
    # See https://ai.google.dev/docs/safety_setting_gemini
}

async def get_llm_text_response(prompt: str) -> str:
    """
    Sends a prompt directly to the configured Gemini model and returns the text response.

    Args:
        prompt: The text prompt to send to the LLM.

    Returns:
        The generated text response from the LLM, or an empty string if an error occurs or no model is available.
    """
    if not GEMINI_MODEL:
        print("llm_utils.get_llm_text_response: Gemini model not available.")
        return "" # Return empty string or raise an error?

    print(f"llm_utils: Sending prompt to Gemini (first 100 chars): {prompt[:100]}...")
    try:
        # Use generate_content_async for async FastAPI
        response = await GEMINI_MODEL.generate_content_async(
            prompt,
            safety_settings=SAFETY_SETTINGS
            # generation_config can be added here if needed
        )
        
        # Check for safety blocks or empty responses
        if not response.candidates:
             print(f"llm_utils: Gemini response blocked or empty. Prompt: {prompt[:100]}... Response: {response}")
             # Try to get block reason if available
             block_reason = response.prompt_feedback.block_reason if response.prompt_feedback else 'Unknown'
             return f"(AI response blocked due to: {block_reason})"
        
        # Access the text content safely
        llm_text = response.text
        print(f"llm_utils: Received response from Gemini.")
        return llm_text
        
    except Exception as e:
        print(f"llm_utils: Error during Gemini API call: {e}", exc_info=True)
        return f"(Error during AI generation: {e})" # Return error message instead of empty string?

# Example usage (for potential testing)
# import asyncio
# async def test():
#     res = await get_llm_text_response("What is the capital of France?")
#     print("Test Result:", res)
# if __name__ == "__main__":
#      asyncio.run(test()) 