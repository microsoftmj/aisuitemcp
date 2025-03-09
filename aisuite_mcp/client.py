"""
Main client interface for AISuite MCP.

This module provides the primary client interface for interacting with
the AISuite MCP platform for AI-to-AI peer review.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union

import aisuite
from aisuite import Client as AISuiteClient

from .models import ReviewSpec, ReviewResult, FormatSpec, ReviewType
from .orchestrator.orchestrator import MCPOrchestrator
from .format.parser import FormatParser
from .review.manager import ReviewManager
from .tools.manager import ToolsManager


class MCPClient:
    """
    Client for AISuite MCP.
    
    This client provides methods for creating peer-reviewed completions
    using the AISuite MCP framework.
    """
    
    def __init__(
        self,
        provider_configs: Dict[str, Dict[str, Any]] = None,
        aisuite_client: Optional[AISuiteClient] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the MCP Client.
        
        Args:
            provider_configs: Configuration for different LLM providers
            aisuite_client: An existing AISuite client (optional)
            logger: Logger instance
        """
        self.logger = logger or logging.getLogger("aisuite_mcp")
        
        # Set up AISuite client
        if aisuite_client:
            self.client = aisuite_client
        else:
            self.client = AISuiteClient(provider_configs or {})
        
        # Initialize components
        self.format_parser = FormatParser()
        self.tools_manager = ToolsManager()
        self.review_manager = ReviewManager(
            aisuite_client=self.client,
            format_parser=self.format_parser,
            tools_manager=self.tools_manager,
            logger=self.logger,
        )
        
        # Create orchestrator
        self.orchestrator = MCPOrchestrator(
            aisuite_client=self.client,
            format_parser=self.format_parser,
            review_manager=self.review_manager,
            tools_manager=self.tools_manager,
            logger=self.logger,
        )
    
    def create_reviewed_completion(
        self,
        prompt: str,
        review_spec: Union[ReviewSpec, Dict[str, Any]],
        system_prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
    ) -> ReviewResult:
        """
        Create a peer-reviewed completion.
        
        Args:
            prompt: The user's prompt/query
            review_spec: Specification for how the review should be conducted
            system_prompt: System prompt for the generator model
            messages: Previous messages in the conversation (optional)
            
        Returns:
            ReviewResult containing the initial response, review feedback, and final answer
        """
        # Convert dict to ReviewSpec if needed
        if isinstance(review_spec, dict):
            review_spec = ReviewSpec(**review_spec)
        
        # Prepare messages
        if messages is None:
            messages = []
        
        if prompt:
            # Add the user prompt if provided
            messages.append({"role": "user", "content": prompt})
        
        # Delegate to orchestrator
        result = self.orchestrator.process_completion(
            messages=messages,
            review_spec=review_spec,
            system_prompt=system_prompt,
        )
        
        return result
    
    async def create_reviewed_completion_async(
        self,
        prompt: str,
        review_spec: Union[ReviewSpec, Dict[str, Any]],
        system_prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
    ) -> ReviewResult:
        """
        Create a peer-reviewed completion asynchronously.
        
        Args:
            prompt: The user's prompt/query
            review_spec: Specification for how the review should be conducted
            system_prompt: System prompt for the generator model
            messages: Previous messages in the conversation (optional)
            
        Returns:
            ReviewResult containing the initial response, review feedback, and final answer
        """
        # Convert dict to ReviewSpec if needed
        if isinstance(review_spec, dict):
            review_spec = ReviewSpec(**review_spec)
        
        # Prepare messages
        if messages is None:
            messages = []
        
        if prompt:
            # Add the user prompt if provided
            messages.append({"role": "user", "content": prompt})
        
        # Delegate to orchestrator async method
        result = await self.orchestrator.process_completion_async(
            messages=messages,
            review_spec=review_spec,
            system_prompt=system_prompt,
        )
        
        return result
    
    def configure(self, provider_configs: Dict[str, Dict[str, Any]]) -> None:
        """
        Configure the client with provider configurations.
        
        Args:
            provider_configs: Configuration for different LLM providers
        """
        self.client.configure(provider_configs)
    
    def register_tool(self, tool_func: callable) -> None:
        """
        Register a tool function to be used during the review process.
        
        Args:
            tool_func: Callable function to register as a tool
        """
        self.tools_manager.register_tool(tool_func)
