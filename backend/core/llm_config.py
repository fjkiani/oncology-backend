import os
from dotenv import load_dotenv
from typing import Optional, Dict, Any

# Load environment variables
load_dotenv()

class LLMConfig:
    def __init__(self):
        self.provider = os.getenv("LLM_PROVIDER", "GEMINI")
        self.config: Dict[str, Any] = {}
        
        if self.provider == "LITELLM":
            self.config = {
                "base_url": os.getenv("LITELLM_BASE_URL"),
                "api_key": os.getenv("LITELLM_API_KEY"),
                "model": "anthropic/claude-3-5-sonnet",  # definitive test model
            }
        elif self.provider == "GEMINI":
            self.config = {
                "api_key": os.getenv("GOOGLE_API_KEY"),
                "model": "gemini-1.5-flash"
            }
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")

    def get_client(self):
        if self.provider == "LITELLM":
            from litellm import completion
            return completion
        elif self.provider == "GEMINI":
            import google.generativeai as genai
            genai.configure(api_key=self.config["api_key"])
            return genai
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")

# Create a singleton instance
llm_config = LLMConfig() 