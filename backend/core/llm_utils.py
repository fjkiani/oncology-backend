from typing import Optional, Dict, Any
from .llm_config import llm_config

def get_llm_text_response(prompt: str, context: Optional[Dict[str, Any]] = None) -> str:
    """
    Get a text response from the configured LLM provider.
    
    Args:
        prompt (str): The prompt to send to the LLM
        context (Optional[Dict[str, Any]]): Additional context for the LLM
        
    Returns:
        str: The LLM's response
    """
    client = llm_config.get_client()
    
    if llm_config.provider == "LITELLM":
        response = client(
            model=llm_config.config["model"],
            messages=[{"role": "user", "content": prompt}],
            api_base=llm_config.config["base_url"],
            api_key=llm_config.config["api_key"]
        )
        return response.choices[0].message.content
    
    elif llm_config.provider == "GEMINI":
        model = client.GenerativeModel(llm_config.config["model"])
        response = model.generate_content(prompt)
        return response.text
    
    else:
        raise ValueError(f"Unsupported LLM provider: {llm_config.provider}")

def get_llm_chat_response(messages: list, context: Optional[Dict[str, Any]] = None) -> str:
    """
    Get a chat response from the configured LLM provider.
    
    Args:
        messages (list): List of message dictionaries with 'role' and 'content'
        context (Optional[Dict[str, Any]]): Additional context for the LLM
        
    Returns:
        str: The LLM's response
    """
    client = llm_config.get_client()
    
    if llm_config.provider == "LITELLM":
        response = client(
            model=llm_config.config["model"],
            messages=messages,
            api_base=llm_config.config["base_url"],
            api_key=llm_config.config["api_key"]
        )
        return response.choices[0].message.content
    
    elif llm_config.provider == "GEMINI":
        model = client.GenerativeModel(llm_config.config["model"])
        # Convert messages to Gemini format
        gemini_messages = []
        for msg in messages:
            if msg["role"] == "user":
                gemini_messages.append({"role": "user", "parts": [msg["content"]]})
            elif msg["role"] == "assistant":
                gemini_messages.append({"role": "model", "parts": [msg["content"]]})
        
        chat = model.start_chat(history=gemini_messages[:-1])
        response = chat.send_message(messages[-1]["content"])
        return response.text
    
    else:
        raise ValueError(f"Unsupported LLM provider: {llm_config.provider}") 