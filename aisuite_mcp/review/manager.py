"""
Review Manager for AISuite MCP.

This module handles different AI-to-AI review strategies.
"""

import asyncio
import logging
import json
from typing import Dict, List, Optional, Any, Union, Tuple

from ..models import ReviewSpec, ReviewType, ReviewResult, FormatSpec
from ..format.parser import FormatParser
from ..tools.manager import ToolsManager


class ReviewManager:
    """
    Manager for AI-to-AI review processes.
    
    This class implements different review strategies and coordinates
    the interaction between generator and reviewer AIs.
    """
    
    def __init__(
        self,
        aisuite_client,  # AISuiteClient
        format_parser: FormatParser,
        tools_manager: ToolsManager,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the review manager.
        
        Args:
            aisuite_client: AISuite client instance
            format_parser: Format parser instance
            tools_manager: Tools manager instance
            logger: Logger instance
        """
        self.client = aisuite_client
        self.format_parser = format_parser
        self.tools_manager = tools_manager
        self.logger = logger or logging.getLogger("aisuite_mcp.review")

    def perform_review(
        self,
        messages: List[Dict[str, str]],
        review_spec: ReviewSpec,
        initial_response: Any,
        system_prompt: Optional[str] = None,
    ) -> ReviewResult:
        """
        Perform a review of the initial response based on the review strategy.
        
        Args:
            messages: List of conversation messages
            review_spec: Specification for how the review should be conducted
            initial_response: The initial response from the generator model
            system_prompt: Custom system prompt (optional)
            
        Returns:
            Review result with initial response, review feedback, and final answer
        """
        # Choose the appropriate review strategy based on the review type
        if review_spec.review_type == ReviewType.STANDARD:
            return self._standard_review(
                messages, review_spec, initial_response, system_prompt
            )
        elif review_spec.review_type == ReviewType.DEBATE:
            return self._debate_review(
                messages, review_spec, initial_response, system_prompt
            )
        elif review_spec.review_type == ReviewType.SELF_CONSISTENCY:
            return self._self_consistency_review(
                messages, review_spec, initial_response, system_prompt
            )
        else:
            raise ValueError(f"Unsupported review type: {review_spec.review_type}")
    
    async def perform_review_async(
        self,
        messages: List[Dict[str, str]],
        review_spec: ReviewSpec,
        initial_response: Any,
        system_prompt: Optional[str] = None,
    ) -> ReviewResult:
        """
        Perform a review of the initial response asynchronously.
        
        Args:
            messages: List of conversation messages
            review_spec: Specification for how the review should be conducted
            initial_response: The initial response from the generator model
            system_prompt: Custom system prompt (optional)
            
        Returns:
            Review result with initial response, review feedback, and final answer
        """
        # Choose the appropriate review strategy based on the review type
        if review_spec.review_type == ReviewType.STANDARD:
            return await self._standard_review_async(
                messages, review_spec, initial_response, system_prompt
            )
        elif review_spec.review_type == ReviewType.DEBATE:
            return await self._debate_review_async(
                messages, review_spec, initial_response, system_prompt
            )
        elif review_spec.review_type == ReviewType.SELF_CONSISTENCY:
            return await self._self_consistency_review_async(
                messages, review_spec, initial_response, system_prompt
            )
        else:
            raise ValueError(f"Unsupported review type: {review_spec.review_type}")
    
    def _standard_review(
        self,
        messages: List[Dict[str, str]],
        review_spec: ReviewSpec,
        initial_response: Any,
        system_prompt: Optional[str] = None,
    ) -> ReviewResult:
        """
        Perform a standard review where one model reviews another's response.
        
        Args:
            messages: List of conversation messages
            review_spec: Specification for how the review should be conducted
            initial_response: The initial response from the generator model
            system_prompt: Custom system prompt (optional)
            
        Returns:
            Review result with initial response, review feedback, and final answer
        """
        self.logger.info("Performing standard review")
        
        # Extract the original query from messages
        original_query = self._extract_original_query(messages)
        
        # Create the review prompt
        reviewer_prompt = self._create_reviewer_prompt(
            original_query, initial_response, review_spec
        )
        
        # Set up reviewer messages
        reviewer_messages = []
        
        # Add system prompt if provided or use default
        if review_spec.reviewer_prompt_override:
            reviewer_messages.append({
                "role": "system",
                "content": review_spec.reviewer_prompt_override
            })
        else:
            reviewer_messages.append({
                "role": "system",
                "content": self._default_reviewer_system_prompt()
            })
        
        # Add the review prompt
        reviewer_messages.append({
            "role": "user",
            "content": reviewer_prompt
        })
        
        # Configure tools if specified
        tools = None
        max_turns = None
        if review_spec.tools:
            tools = self.tools_manager.get_aisuite_tools(review_spec.tools)
            max_turns = 5  # Allow up to 5 tool calls by default
        
        # Get the reviewer's feedback
        self.logger.info(f"Calling reviewer model: {review_spec.reviewer_model}")
        
        # Call with or without tools/max_turns depending on whether tools are specified
        if tools is not None:
            review_response = self.client.chat.completions.create(
                model=review_spec.reviewer_model,
                messages=reviewer_messages,
                tools=tools,
                max_turns=max_turns,
            )
        else:
            review_response = self.client.chat.completions.create(
                model=review_spec.reviewer_model,
                messages=reviewer_messages,
            )
        
        review_feedback = review_response.choices[0].message.content
        self.logger.debug(f"Review feedback: {review_feedback}")
        
        # Format the final answer based on the review
        final_answer = self._format_final_answer(
            initial_response, review_feedback, review_spec.response_format
        )
        
        # Create the review result
        result = ReviewResult(
            initial_response=initial_response,
            review_feedback=review_feedback,
            final_answer=final_answer,
            debug_info={
                "reviewer_prompt": reviewer_prompt,
                "reviewer_model": review_spec.reviewer_model,
            },
        )
        
        return result
    
    async def _standard_review_async(
        self,
        messages: List[Dict[str, str]],
        review_spec: ReviewSpec,
        initial_response: Any,
        system_prompt: Optional[str] = None,
    ) -> ReviewResult:
        """
        Perform a standard review asynchronously.
        
        This is currently a synchronous implementation that runs in a thread,
        but could be updated to use async AISuite clients in the future.
        """
        # Run the synchronous version in a thread pool
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            self._standard_review,
            messages,
            review_spec,
            initial_response,
            system_prompt,
        )
        return result
    
    def _debate_review(
        self,
        messages: List[Dict[str, str]],
        review_spec: ReviewSpec,
        initial_response: Any,
        system_prompt: Optional[str] = None,
    ) -> ReviewResult:
        """
        Perform a debate-style review where models engage in back-and-forth.
        
        Args:
            messages: List of conversation messages
            review_spec: Specification for how the review should be conducted
            initial_response: The initial response from the generator model
            system_prompt: Custom system prompt (optional)
            
        Returns:
            Review result with initial response, debate transcript, and final answer
        """
        self.logger.info("Performing debate review")
        
        # Extract the original query from messages
        original_query = self._extract_original_query(messages)
        
        # Initialize debate participants
        generator_model = review_spec.generator_model
        reviewer_model = review_spec.reviewer_model
        
        # Initialize debate transcript
        debate_transcript = []
        
        # Add the initial response to the transcript
        debate_transcript.append({
            "role": "generator",
            "model": generator_model,
            "content": initial_response
        })
        
        # Set up debate messages
        debate_messages = []
        
        # Add system prompt for the reviewer
        if review_spec.reviewer_prompt_override:
            debate_messages.append({
                "role": "system",
                "content": review_spec.reviewer_prompt_override
            })
        else:
            debate_messages.append({
                "role": "system",
                "content": self._default_debate_system_prompt()
            })
        
        # Create the initial debate prompt for the reviewer
        debate_prompt = self._create_debate_prompt(
            original_query, initial_response, review_spec
        )
        
        # Add the debate prompt
        debate_messages.append({
            "role": "user",
            "content": debate_prompt
        })
        
        # Configure tools if specified
        tools = None
        max_turns = None
        if review_spec.tools:
            tools = self.tools_manager.get_aisuite_tools(review_spec.tools)
            max_turns = 3  # Allow up to 3 tool calls per debate turn
        
        # Perform the debate for the specified number of turns
        max_debate_turns = review_spec.max_debate_turns or 3
        
        for turn in range(max_debate_turns - 1):  # -1 because we already have initial response
            # Alternate between reviewer and generator
            if turn % 2 == 0:
                # Reviewer's turn
                self.logger.info(f"Debate turn {turn+1}: Reviewer ({reviewer_model})")
                current_model = reviewer_model
                current_role = "reviewer"
            else:
                # Generator's turn
                self.logger.info(f"Debate turn {turn+1}: Generator ({generator_model})")
                current_model = generator_model
                current_role = "generator"
            
            # Get the next turn in the debate
            response = self.client.chat.completions.create(
                model=current_model,
                messages=debate_messages,
                tools=tools,
                max_turns=max_turns,
            )
            
            turn_content = response.choices[0].message.content
            
            # Add the response to the debate transcript
            debate_transcript.append({
                "role": current_role,
                "model": current_model,
                "content": turn_content
            })
            
            # Add the response to the debate messages
            debate_messages.append({
                "role": "assistant",
                "content": turn_content
            })
            
            # Add a prompt for the next turn
            if turn < max_debate_turns - 2:  # Don't add prompt for the last turn
                debate_messages.append({
                    "role": "user",
                    "content": f"Continue the debate by {'addressing the points raised' if turn % 2 == 0 else 'critiquing the response'}."
                })
        
        # Generate a final answer from the debate
        judge_system_prompt = """You are a judge evaluating a debate between two AI models.
        Your task is to determine which arguments are most compelling and accurate.
        Synthesize the strongest points from both sides into a final answer that best addresses the original query."""
        
        judge_messages = [
            {"role": "system", "content": judge_system_prompt},
            {"role": "user", "content": self._create_judge_prompt(original_query, debate_transcript, review_spec)}
        ]
        
        # Get the judge's decision
        self.logger.info("Getting judge's decision")
        judge_response = self.client.chat.completions.create(
            model=reviewer_model,  # Use reviewer model as judge
            messages=judge_messages
        )
        
        final_answer = judge_response.choices[0].message.content
        
        # Format the final answer
        final_answer = self.format_parser.ensure_format_compliance(
            final_answer, review_spec.response_format
        )
        
        # Create the review result
        result = ReviewResult(
            initial_response=initial_response,
            review_feedback=debate_transcript,
            final_answer=final_answer,
            debug_info={
                "debate_transcript": debate_transcript,
                "generator_model": generator_model,
                "reviewer_model": reviewer_model,
                "max_debate_turns": max_debate_turns,
            },
        )
        
        return result
    
    async def _debate_review_async(
        self,
        messages: List[Dict[str, str]],
        review_spec: ReviewSpec,
        initial_response: Any,
        system_prompt: Optional[str] = None,
    ) -> ReviewResult:
        """
        Perform a debate-style review asynchronously.
        
        This is currently a synchronous implementation that runs in a thread,
        but could be updated to use async AISuite clients in the future.
        """
        # Run the synchronous version in a thread pool
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            self._debate_review,
            messages,
            review_spec,
            initial_response,
            system_prompt,
        )
        return result
    
    def _self_consistency_review(
        self,
        messages: List[Dict[str, str]],
        review_spec: ReviewSpec,
        initial_response: Any,
        system_prompt: Optional[str] = None,
    ) -> ReviewResult:
        """
        Perform a self-consistency review by generating multiple paths.
        
        Args:
            messages: List of conversation messages
            review_spec: Specification for how the review should be conducted
            initial_response: The initial response from the generator model
            system_prompt: Custom system prompt (optional)
            
        Returns:
            Review result with multiple responses and consensus answer
        """
        self.logger.info("Performing self-consistency review")
        
        # We already have one response, generate additional samples
        consistency_samples = review_spec.consistency_samples or 3
        samples = [initial_response]
        
        # Create system prompt to encourage diversity
        system_message = None
        for msg in messages:
            if msg["role"] == "system":
                system_message = msg
                break
        
        if system_message:
            # Update existing system message
            original_content = system_message["content"]
            system_message["content"] = original_content + "\nGenerate a different reasoning path than previous attempts."
        else:
            # Add new system message
            messages = [{"role": "system", "content": "Generate a different reasoning path than previous attempts."}] + messages
        
        # Configure tools if specified
        tools = None
        max_turns = None
        if review_spec.tools:
            tools = self.tools_manager.get_aisuite_tools(review_spec.tools)
            max_turns = 5  # Allow up to 5 tool calls by default
        
        # Generate additional samples
        for i in range(consistency_samples - 1):
            self.logger.info(f"Generating self-consistency sample {i+2}/{consistency_samples}")
            
            # Get another response using the same messages
            response = self.client.chat.completions.create(
                model=review_spec.generator_model,
                messages=messages,
                tools=tools,
                max_turns=max_turns,
            )
            
            sample_content = response.choices[0].message.content
            samples.append(sample_content)
        
        # Create a prompt for the reviewer to analyze the samples
        original_query = self._extract_original_query(messages)
        consistency_prompt = self._create_consistency_prompt(
            original_query, samples, review_spec
        )
        
        # Set up reviewer messages
        reviewer_messages = []
        
        # Add system prompt for the reviewer
        if review_spec.reviewer_prompt_override:
            reviewer_messages.append({
                "role": "system",
                "content": review_spec.reviewer_prompt_override
            })
        else:
            reviewer_messages.append({
                "role": "system",
                "content": self._default_consistency_system_prompt()
            })
        
        # Add the consistency prompt
        reviewer_messages.append({
            "role": "user",
            "content": consistency_prompt
        })
        
        # Get the reviewer's analysis
        self.logger.info(f"Analyzing self-consistency samples with {review_spec.reviewer_model}")
        review_response = self.client.chat.completions.create(
            model=review_spec.reviewer_model,
            messages=reviewer_messages,
            tools=tools,
            max_turns=max_turns,
        )
        
        consistency_analysis = review_response.choices[0].message.content
        
        # Format the final answer
        final_answer = self.format_parser.ensure_format_compliance(
            consistency_analysis, review_spec.response_format
        )
        
        # Create the review result
        result = ReviewResult(
            initial_response=samples[0],
            review_feedback={
                "samples": samples,
                "analysis": consistency_analysis
            },
            final_answer=final_answer,
            debug_info={
                "consistency_samples": consistency_samples,
                "generator_model": review_spec.generator_model,
                "reviewer_model": review_spec.reviewer_model,
            },
        )
        
        return result
    
    async def _self_consistency_review_async(
        self,
        messages: List[Dict[str, str]],
        review_spec: ReviewSpec,
        initial_response: Any,
        system_prompt: Optional[str] = None,
    ) -> ReviewResult:
        """
        Perform a self-consistency review asynchronously.
        """
        self.logger.info("Performing async self-consistency review")
        
        # Extract the original query from messages
        original_query = self._extract_original_query(messages)
        
        # We already have one response, need to generate additional samples
        consistency_samples = review_spec.consistency_samples or 3
        samples = [initial_response]
        
        # Create system prompt to encourage diversity
        system_message = None
        for msg in messages:
            if msg["role"] == "system":
                system_message = msg
                break
        
        if system_message:
            # Update existing system message
            original_content = system_message["content"]
            system_message["content"] = original_content + "\nGenerate a different reasoning path than previous attempts."
        else:
            # Add new system message
            messages = [{"role": "system", "content": "Generate a different reasoning path than previous attempts."}] + messages
        
        # Configure tools if specified
        tools = None
        max_turns = None
        if review_spec.tools:
            tools = self.tools_manager.get_aisuite_tools(review_spec.tools)
            max_turns = 5  # Allow up to 5 tool calls by default
        
        # Generate additional samples in parallel
        tasks = []
        for i in range(consistency_samples - 1):
            self.logger.info(f"Generating async self-consistency sample {i+2}/{consistency_samples}")
            
            # Create a task to generate another sample
            task = asyncio.ensure_future(
                self._generate_sample_async(
                    messages, review_spec.generator_model, tools, max_turns
                )
            )
            tasks.append(task)
        
        # Wait for all samples to be generated
        sample_results = await asyncio.gather(*tasks)
        samples.extend(sample_results)
        
        # Create a prompt for the reviewer to analyze the samples
        consistency_prompt = self._create_consistency_prompt(
            original_query, samples, review_spec
        )
        
        # Set up reviewer messages
        reviewer_messages = []
        
        # Add system prompt for the reviewer
        if review_spec.reviewer_prompt_override:
            reviewer_messages.append({
                "role": "system",
                "content": review_spec.reviewer_prompt_override
            })
        else:
            reviewer_messages.append({
                "role": "system",
                "content": self._default_consistency_system_prompt()
            })
        
        # Add the consistency prompt
        reviewer_messages.append({
            "role": "user",
            "content": consistency_prompt
        })
        
        # Get the reviewer's analysis
        self.logger.info(f"Analyzing self-consistency samples with {review_spec.reviewer_model}")
        
        # Call the reviewer synchronously (AISuite doesn't support async yet)
        loop = asyncio.get_event_loop()
        review_response = await loop.run_in_executor(
            None,
            lambda: self.client.chat.completions.create(
                model=review_spec.reviewer_model,
                messages=reviewer_messages,
                tools=tools,
                max_turns=max_turns,
            )
        )
        
        consistency_analysis = review_response.choices[0].message.content
        
        # Format the final answer
        final_answer = self.format_parser.ensure_format_compliance(
            consistency_analysis, review_spec.response_format
        )
        
        # Create the review result
        result = ReviewResult(
            initial_response=samples[0],
            review_feedback={
                "samples": samples,
                "analysis": consistency_analysis
            },
            final_answer=final_answer,
            debug_info={
                "consistency_samples": consistency_samples,
                "generator_model": review_spec.generator_model,
                "reviewer_model": review_spec.reviewer_model,
            },
        )
        
        return result
    
    async def _generate_sample_async(self, messages, model, tools, max_turns):
        """Generate a single sample asynchronously."""
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self.client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools,
                max_turns=max_turns,
            )
        )
        return response.choices[0].message.content
    
    def _extract_original_query(self, messages: List[Dict[str, str]]) -> str:
        """Extract the original user query from messages."""
        # Look for the most recent user message
        for message in reversed(messages):
            if message["role"] == "user":
                return message["content"]
        
        # If no user message found, return empty string
        return ""
    
    def _create_reviewer_prompt(
        self, query: str, initial_response: str, review_spec: ReviewSpec
    ) -> str:
        """Create a prompt for the reviewer model."""
        # Get formatting instructions
        format_instructions = self.format_parser.format_to_prompt_instructions(
            review_spec.response_format
        )
        
        # Construct criteria string
        criteria_items = [f"- {criterion}" for criterion in review_spec.review_criteria]
        criteria_str = "\n".join(criteria_items)
        
        # Build the review prompt
        prompt = f"""
        I need you to review another AI's response to the following query:
        
        QUERY:
        {query}
        
        AI'S RESPONSE:
        {initial_response}
        
        Please review this response based on the following criteria:
        {criteria_str}
        
        Provide a thorough analysis of the response, identify any issues or errors,
        and then provide an improved version of the answer that addresses any problems.
        
        {format_instructions}
        """
        
        return prompt.strip()
    
    def _create_debate_prompt(
        self, query: str, initial_response: str, review_spec: ReviewSpec
    ) -> str:
        """Create a prompt for the debate review."""
        # Get formatting instructions
        format_instructions = self.format_parser.format_to_prompt_instructions(
            review_spec.response_format
        )
        
        # Construct criteria string
        criteria_items = [f"- {criterion}" for criterion in review_spec.review_criteria]
        criteria_str = "\n".join(criteria_items)
        
        # Build the debate prompt
        prompt = f"""
        I need you to critically evaluate the following response to a query.
        Your role is to point out flaws, inaccuracies, or areas of improvement
        in the response.
        
        ORIGINAL QUERY:
        {query}
        
        RESPONSE TO EVALUATE:
        {initial_response}
        
        Please critique this response based on these criteria:
        {criteria_str}
        
        Focus on identifying factual errors, logical flaws, omissions, or biases.
        Provide counter-arguments when appropriate and suggest a better approach
        to answering the query.
        
        {format_instructions}
        """
        
        return prompt.strip()
    
    def _create_judge_prompt(
        self, query: str, debate_transcript: List[Dict[str, Any]], review_spec: ReviewSpec
    ) -> str:
        """Create a prompt for the debate judge."""
        # Get formatting instructions
        format_instructions = self.format_parser.format_to_prompt_instructions(
            review_spec.response_format
        )
        
        # Format the debate transcript
        transcript_str = ""
        for i, entry in enumerate(debate_transcript):
            role = entry["role"]
            content = entry["content"]
            transcript_str += f"\n\n{role.upper()}:\n{content}"
        
        # Build the judge prompt
        prompt = f"""
        You are evaluating a debate between two AI systems on the following query:
        
        QUERY:
        {query}
        
        DEBATE TRANSCRIPT:
        {transcript_str}
        
        After careful consideration of all arguments presented in this debate,
        provide a final answer to the original query. Your answer should:
        
        1. Incorporate the strongest points from both sides
        2. Correct any factual errors identified
        3. Present a comprehensive and balanced view
        4. Explicitly address areas of disagreement by providing the best supported conclusion
        
        {format_instructions}
        """
        
        return prompt.strip()
    
    def _create_consistency_prompt(
        self, query: str, samples: List[str], review_spec: ReviewSpec
    ) -> str:
        """Create a prompt for the self-consistency reviewer."""
        # Get formatting instructions
        format_instructions = self.format_parser.format_to_prompt_instructions(
            review_spec.response_format
        )
        
        # Format the samples
        samples_str = ""
        for i, sample in enumerate(samples):
            samples_str += f"\n\nSAMPLE {i+1}:\n{sample}"
        
        # Build the consistency prompt
        prompt = f"""
        I need you to analyze multiple responses to the following query:
        
        QUERY:
        {query}
        
        The following are different responses to this query:
        {samples_str}
        
        Please analyze these responses and:
        1. Identify common themes and conclusions across the samples
        2. Note any contradictions or inconsistencies between them
        3. Evaluate which reasoning paths are most sound and convincing
        4. Synthesize a final answer that represents the most reliable information
           from all samples, preferring conclusions that appear consistently
        
        {format_instructions}
        """
        
        return prompt.strip()
    
    def _default_reviewer_system_prompt(self) -> str:
        """Get the default system prompt for the reviewer model."""
        return """You are an expert reviewer AI with strong critical thinking skills. 
        Your task is to evaluate another AI's response to a query, identify any issues, 
        and provide an improved version. Focus on accuracy, coherence, comprehensiveness, 
        and addressing the specific question. Be thorough and constructive in your evaluation."""
    
    def _default_debate_system_prompt(self) -> str:
        """Get the default system prompt for the debate review."""
        return """You are participating in a reasoned debate with another AI system.
        Your goal is to critically evaluate arguments, identify flaws in reasoning, 
        point out factual errors, and present counter-arguments when appropriate.
        Focus on the strength of arguments rather than rhetorical techniques.
        Be fair and thorough in your analysis."""
    
    def _default_consistency_system_prompt(self) -> str:
        """Get the default system prompt for the consistency reviewer."""
        return """You are an expert evaluator analyzing multiple AI-generated responses
        to the same query. Your task is to identify common themes and conclusions across samples,
        note contradictions, evaluate the soundness of different reasoning paths, and synthesize
        a final answer that represents the most reliable information. Focus on conclusions that
        appear consistently across multiple samples."""
    
    def _format_final_answer(
        self, initial_response: str, review_feedback: str, format_spec: FormatSpec
    ) -> Any:
        """
        Format the final answer based on the review feedback and format specification.
        
        In the standard review process, this extracts the final answer from the review feedback.
        """
        # For standard review, the review feedback should contain the final answer
        
        # Attempt to extract the final answer from the review feedback
        # Common patterns: "Final Answer:", "Improved Answer:", "Revised Answer:", etc.
        final_answer_patterns = [
            "Final Answer:", "Improved Answer:", "Revised Answer:",
            "Final Response:", "Improved Response:", "Revised Response:",
            "My Answer:", "My Response:", "Corrected Answer:",
            "Final version:", "Improved version:", "Here is the improved answer:",
            "## Final Answer", "# Final Answer", 
            "## Improved Answer", "# Improved Answer"
        ]
        
        for pattern in final_answer_patterns:
            if pattern in review_feedback:
                # Extract the content after the pattern
                final_answer = review_feedback.split(pattern, 1)[1].strip()
                break
        else:
            # If no explicit final answer section, use the entire review feedback
            final_answer = review_feedback
        
        # Ensure format compliance
        formatted_answer = self.format_parser.ensure_format_compliance(
            final_answer, format_spec
        )
        
        return formatted_answer
