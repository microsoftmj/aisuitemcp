"""
Configuration utilities for AISuite MCP.

This module handles loading environment variables and configuration settings
from .env files using python-dotenv.
"""

import os
import logging
from typing import Dict, Any, Optional

# Import dotenv conditionally since it's an optional dependency
try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False

logger = logging.getLogger("aisuite_mcp.utils.config")

# Default configuration values
DEFAULT_CONFIG = {
    "DEFAULT_GENERATOR_MODEL": "openai:gpt-4o",
    "DEFAULT_REVIEWER_MODEL": "openai:gpt-4o",
    "LOG_LEVEL": "INFO",
    "ENABLE_SEARCH_TOOL": "false",
    "REQUEST_TIMEOUT": "60",
    "MAX_RETRIES": "3",
}


def load_env_config(env_file: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from environment variables, optionally using a .env file.
    
    Args:
        env_file: Path to .env file (optional)
        
    Returns:
        Dictionary of configuration values
    """
    if DOTENV_AVAILABLE:
        # Load from .env file if available and exists
        if env_file and os.path.exists(env_file):
            load_dotenv(env_file)
            logger.info(f"Loaded environment variables from {env_file}")
        else:
            # Try to load from default locations
            load_dotenv()
            logger.debug("Attempted to load environment variables from default locations")
    else:
        logger.warning("python-dotenv not installed. Install with 'pip install python-dotenv' to use .env files.")
    
    # Build configuration dictionary from environment variables
    config = DEFAULT_CONFIG.copy()
    
    # Update with environment variables
    for key in [
        "OPENAI_API_KEY", 
        "OPENAI_ORGANIZATION",
        "ANTHROPIC_API_KEY",
        "GOOGLE_API_KEY",
        "MISTRAL_API_KEY",
        "COHERE_API_KEY",
        "HUGGINGFACE_API_KEY",
        "XAI_API_KEY",
        "XAI_API_BASE",
        "LOG_LEVEL",
        "ENABLE_SEARCH_TOOL",
        "DEFAULT_GENERATOR_MODEL",
        "DEFAULT_REVIEWER_MODEL",
        "REQUEST_TIMEOUT",
        "MAX_RETRIES",
    ]:
        if key in os.environ:
            config[key] = os.environ[key]
    
    # Convert boolean and numeric values
    if config.get("ENABLE_SEARCH_TOOL", "").lower() in ["true", "1", "yes"]:
        config["ENABLE_SEARCH_TOOL"] = True
    else:
        config["ENABLE_SEARCH_TOOL"] = False
    
    if "REQUEST_TIMEOUT" in config:
        try:
            config["REQUEST_TIMEOUT"] = int(config["REQUEST_TIMEOUT"])
        except ValueError:
            config["REQUEST_TIMEOUT"] = 60
    
    if "MAX_RETRIES" in config:
        try:
            config["MAX_RETRIES"] = int(config["MAX_RETRIES"])
        except ValueError:
            config["MAX_RETRIES"] = 3
    
    return config


def get_provider_configs(config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Extract provider configurations from the loaded config.
    
    Args:
        config: Configuration dictionary from load_env_config
        
    Returns:
        Dictionary of provider configurations
    """
    provider_configs = {}
    
    # OpenAI
    if "OPENAI_API_KEY" in config:
        openai_config = {"api_key": config["OPENAI_API_KEY"]}
        if "OPENAI_ORGANIZATION" in config:
            openai_config["organization"] = config["OPENAI_ORGANIZATION"]
        provider_configs["openai"] = openai_config
    
    # Anthropic
    if "ANTHROPIC_API_KEY" in config:
        provider_configs["anthropic"] = {"api_key": config["ANTHROPIC_API_KEY"]}
    
    # Google
    if "GOOGLE_API_KEY" in config:
        provider_configs["google"] = {"api_key": config["GOOGLE_API_KEY"]}
    
    # Mistral
    if "MISTRAL_API_KEY" in config:
        provider_configs["mistral"] = {"api_key": config["MISTRAL_API_KEY"]}
    
    # Cohere
    if "COHERE_API_KEY" in config:
        provider_configs["cohere"] = {"api_key": config["COHERE_API_KEY"]}
    
    # Hugging Face
    if "HUGGINGFACE_API_KEY" in config:
        provider_configs["huggingface"] = {"api_key": config["HUGGINGFACE_API_KEY"]}
    
    # xAI (Grok)
    if "XAI_API_KEY" in config:
        xai_config = {"api_key": config["XAI_API_KEY"]}
        if "XAI_API_BASE" in config:
            xai_config["api_base"] = config["XAI_API_BASE"]
        provider_configs["xai"] = xai_config
    
    return provider_configs
