from backend.core.llm_utils import get_llm_text_response, get_llm_chat_response
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_text_response():
    print("\nTesting text response...")
    prompt = "What is the capital of France?"
    try:
        response = get_llm_text_response(prompt)
        print(f"Provider: {os.getenv('LLM_PROVIDER')}")
        print(f"Prompt: {prompt}")
        print(f"Response: {response}")
    except Exception as e:
        print(f"Error in text response: {str(e)}")

def test_chat_response():
    print("\nTesting chat response...")
    messages = [
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm doing well, thank you! How can I help you today?"},
        {"role": "user", "content": "What's the weather like?"}
    ]
    try:
        response = get_llm_chat_response(messages)
        print(f"Provider: {os.getenv('LLM_PROVIDER')}")
        print(f"Last message: {messages[-1]['content']}")
        print(f"Response: {response}")
    except Exception as e:
        print(f"Error in chat response: {str(e)}")

if __name__ == "__main__":
    print("Testing LLM functionality...")
    test_text_response()
    test_chat_response() 