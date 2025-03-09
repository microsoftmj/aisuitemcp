"""
Reasoning utilities for AISuite MCP.

This module provides utilities for leveraging enhanced reasoning capabilities
of LLMs like Claude 3.7 Sonnet.
"""

import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger("aisuite_mcp.utils.reasoning")

def get_reasoning_system_prompt(config: Dict[str, Any]) -> str:
    """
    Generate a system prompt that encourages step-by-step reasoning for Claude 3.7 Sonnet.
    
    Args:
        config: Configuration dictionary with reasoning settings
        
    Returns:
        System prompt string that promotes reasoning
    """
    if not config.get("ANTHROPIC_REASONING_ENABLED", False):
        return ""
    
    # Create a system prompt that encourages detailed reasoning
    reasoning_prompt = """
    When reviewing content, please use explicit step-by-step reasoning before reaching conclusions.

    Your reasoning process should:
    1. Break down the problem or evaluation into clear components
    2. Consider multiple perspectives and possibilities
    3. Evaluate evidence systematically
    4. Identify assumptions being made
    5. Acknowledge areas of uncertainty
    6. Explain your chain of reasoning that leads to your conclusions
    
    Structure your thinking in a way that demonstrates your reasoning process transparently.
    """
    
    return reasoning_prompt.strip()

def get_anthropic_parameters(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get the appropriate parameters for an Anthropic API call based on config.
    
    Args:
        config: Configuration dictionary with reasoning settings
        
    Returns:
        Dictionary of parameters to add to Anthropic API calls
    """
    params = {}
    
    # Only apply if reasoning is enabled
    if not config.get("ANTHROPIC_REASONING_ENABLED", False):
        return params
    
    # Add temperature setting if configured
    if "ANTHROPIC_TEMPERATURE" in config:
        params["temperature"] = config["ANTHROPIC_TEMPERATURE"]
        
    return params

def apply_reasoning_to_messages(
    messages: List[Dict[str, str]], 
    config: Dict[str, Any],
    is_anthropic_model: bool = False
) -> List[Dict[str, str]]:
    """
    Apply reasoning prompts to message list based on configuration.
    
    Args:
        messages: List of chat messages
        config: Configuration dictionary with reasoning settings
        is_anthropic_model: Whether the model is from Anthropic
        
    Returns:
        Updated list of messages with reasoning prompts if appropriate
    """
    if not config.get("ANTHROPIC_REASONING_ENABLED", False) or not is_anthropic_model:
        return messages
    
    # Create a copy of the messages to avoid modifying the original
    updated_messages = messages.copy()
    
    # Look for a system message to enhance
    for i, message in enumerate(updated_messages):
        if message["role"] == "system":
            # Append reasoning prompt to existing system message
            updated_messages[i]["content"] = message["content"] + "\n\n" + get_reasoning_system_prompt(config)
            return updated_messages
    
    # If no system message was found, add one at the beginning
    updated_messages.insert(0, {
        "role": "system",
        "content": get_reasoning_system_prompt(config)
    })
    
    return updated_messages

def is_anthropic_model(model_name: str) -> bool:
    """
    Check if a model name refers to an Anthropic Claude model.
    
    Args:
        model_name: The name of the model to check
        
    Returns:
        True if it's an Anthropic model, False otherwise
    """
    if ':' in model_name:
        provider, model = model_name.split(':', 1)
        return provider.lower() == 'anthropic'
    return False
