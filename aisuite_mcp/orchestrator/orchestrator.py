"""
MCP Orchestrator for AISuite MCP.

This module provides the central orchestrator that coordinates
interactions between different components of the MCP system.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union

from ..models import ReviewSpec, ReviewResult, FormatSpec, ReviewType
from ..format.parser import FormatParser
from ..review.manager import ReviewManager
from ..tools.manager import ToolsManager


class MCPOrchestrator:
    """
    Orchestrator for AISuite MCP.
    
    This class coordinates the interactions between different components
    of the MCP system, such as the format parser, review manager, and tools manager.
    """
    
    def __init__(
        self,
        aisuite_client,  # AISuiteClient
        format_parser: FormatParser,
        review_manager: ReviewManager,
        tools_manager: ToolsManager,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the MCP orchestrator.
        
        Args:
            aisuite_client: AISuite client instance
            format_parser: Format parser instance
            review_manager: Review manager instance
            tools_manager: Tools manager instance
            logger: Logger instance
        """
        self.client = aisuite_client
        self.format_parser = format_parser
        self.review_manager = review_manager
        self.tools_manager = tools_manager
        self.logger = logger or logging.getLogger("aisuite_mcp.orchestrator")
    
    def process_completion(
        self,
        messages: List[Dict[str, str]],
        review_spec: ReviewSpec,
        system_prompt: Optional[str] = None,
    ) -> ReviewResult:
        """
        Process a completion request with peer review.
        
        Args:
            messages: List of conversation messages
            review_spec: Specification for how the review should be conducted
            system_prompt: Custom system prompt (optional)
            
        Returns:
            Review result with initial response, review feedback, and final answer
        """
        self.logger.info("Processing completion request")
        
        # Configure system prompt
        if system_prompt is not None or review_spec.system_prompt_override is not None:
            # Use the provided system prompt or the override from the review spec
            system_prompt_content = system_prompt or review_spec.system_prompt_override
            
            # Check if there's already a system message
            system_message_exists = False
            for message in messages:
                if message.get("role") == "system":
                    message["content"] = system_prompt_content
                    system_message_exists = True
                    break
            
            # If no system message exists, add one
            if not system_message_exists:
                messages = [{"role": "system", "content": system_prompt_content}] + messages
        
        # Add format instructions to the messages if needed
        format_instructions = self.format_parser.format_to_prompt_instructions(
            review_spec.response_format
        )
        
        # Modify the last user message to include format instructions
        for i in range(len(messages) - 1, -1, -1):
            if messages[i]["role"] == "user":
                messages[i]["content"] += f"\n\n{format_instructions}"
                break
        
        # Configure tools if specified
        tools = None
        max_turns = None
        if review_spec.tools:
            tools = self.tools_manager.get_aisuite_tools(review_spec.tools)
            max_turns = 5  # Allow up to 5 tool calls by default
        
        # Get the initial response from the generator model
        self.logger.info(f"Calling generator model: {review_spec.generator_model}")
        
        # Call with or without tools/max_turns depending on whether tools are specified
        if tools is not None:
            response = self.client.chat.completions.create(
                model=review_spec.generator_model,
                messages=messages,
                tools=tools,
                max_turns=max_turns,
            )
        else:
            response = self.client.chat.completions.create(
                model=review_spec.generator_model,
                messages=messages,
            )
        
        initial_response = response.choices[0].message.content
        self.logger.debug(f"Initial response: {initial_response}")
        
        # Perform review if needed
        if review_spec.reviewer_model == review_spec.generator_model and review_spec.review_type == ReviewType.STANDARD:
            # If using the same model for generation and review with standard review,
            # we can optimize by skipping the review step
            self.logger.info("Skipping review step (same model for generation and review)")
            
            # Format the final answer
            final_answer = self.format_parser.ensure_format_compliance(
                initial_response, review_spec.response_format
            )
            
            # Create the review result
            result = ReviewResult(
                initial_response=initial_response,
                review_feedback=None,
                final_answer=final_answer,
                debug_info={
                    "review_skipped": True,
                    "reason": "Same model for generation and review"
                },
            )
        else:
            # Perform the review
            result = self.review_manager.perform_review(
                messages, review_spec, initial_response, system_prompt
            )
        
        return result
    
    async def process_completion_async(
        self,
        messages: List[Dict[str, str]],
        review_spec: ReviewSpec,
        system_prompt: Optional[str] = None,
    ) -> ReviewResult:
        """
        Process a completion request with peer review asynchronously.
        
        Args:
            messages: List of conversation messages
            review_spec: Specification for how the review should be conducted
            system_prompt: Custom system prompt (optional)
            
        Returns:
            Review result with initial response, review feedback, and final answer
        """
        self.logger.info("Processing async completion request")
        
        # Configure system prompt
        if system_prompt is not None or review_spec.system_prompt_override is not None:
            # Use the provided system prompt or the override from the review spec
            system_prompt_content = system_prompt or review_spec.system_prompt_override
            
            # Check if there's already a system message
            system_message_exists = False
            for message in messages:
                if message.get("role") == "system":
                    message["content"] = system_prompt_content
                    system_message_exists = True
                    break
            
            # If no system message exists, add one
            if not system_message_exists:
                messages = [{"role": "system", "content": system_prompt_content}] + messages
        
        # Add format instructions to the messages if needed
        format_instructions = self.format_parser.format_to_prompt_instructions(
            review_spec.response_format
        )
        
        # Modify the last user message to include format instructions
        for i in range(len(messages) - 1, -1, -1):
            if messages[i]["role"] == "user":
                messages[i]["content"] += f"\n\n{format_instructions}"
                break
        
        # Configure tools if specified
        tools = None
        max_turns = None
        if review_spec.tools:
            tools = self.tools_manager.get_aisuite_tools(review_spec.tools)
            max_turns = 5  # Allow up to 5 tool calls by default
        
        # Get the initial response from the generator model asynchronously
        self.logger.info(f"Calling generator model: {review_spec.generator_model}")
        
        # Since AISuite doesn't have async methods yet, run in thread
        loop = asyncio.get_event_loop()
        
        if tools is not None:
            response = await loop.run_in_executor(
                None,
                lambda: self.client.chat.completions.create(
                    model=review_spec.generator_model,
                    messages=messages,
                    tools=tools,
                    max_turns=max_turns,
                )
            )
        else:
            response = await loop.run_in_executor(
                None,
                lambda: self.client.chat.completions.create(
                    model=review_spec.generator_model,
                    messages=messages,
                )
            )
        
        initial_response = response.choices[0].message.content
        self.logger.debug(f"Initial response: {initial_response}")
        
        # Perform review if needed
        if review_spec.reviewer_model == review_spec.generator_model and review_spec.review_type == ReviewType.STANDARD:
            # If using the same model for generation and review with standard review,
            # we can optimize by skipping the review step
            self.logger.info("Skipping review step (same model for generation and review)")
            
            # Format the final answer
            final_answer = self.format_parser.ensure_format_compliance(
                initial_response, review_spec.response_format
            )
            
            # Create the review result
            result = ReviewResult(
                initial_response=initial_response,
                review_feedback=None,
                final_answer=final_answer,
                debug_info={
                    "review_skipped": True,
                    "reason": "Same model for generation and review"
                },
            )
        else:
            # Perform the review asynchronously
            result = await self.review_manager.perform_review_async(
                messages, review_spec, initial_response, system_prompt
            )
        
        return result
