"""
Core models for AISuite MCP.

This module defines the data structures and types used throughout AISuite MCP.
"""

from enum import Enum, auto
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field


class ReviewType(str, Enum):
    """Types of AI-to-AI review processes."""
    
    STANDARD = "standard"  # Basic one-model-reviews-another approach
    DEBATE = "debate"  # Two models debate with multiple turns
    SELF_CONSISTENCY = "self_consistency"  # Multiple reasoning paths


class ReviewCriteria(str, Enum):
    """Common criteria for reviewing AI responses."""
    
    FACTUAL_ACCURACY = "factual_accuracy"
    REASONING = "reasoning"
    COMPLETENESS = "completeness"
    COHERENCE = "coherence"
    RELEVANCE = "relevance"
    BIAS = "bias"
    SAFETY = "safety"
    CREATIVITY = "creativity"
    CONCISENESS = "conciseness"
    CODE_QUALITY = "code_quality"
    
    @classmethod
    def all(cls) -> List[str]:
        """Return all available criteria as strings."""
        return [e.value for e in cls]


class ResponseFormat(str, Enum):
    """Output format types."""
    
    JSON = "json"
    MARKDOWN = "markdown"
    TEXT = "text"


class FormatSpec(BaseModel):
    """
    Specification for how responses should be formatted.
    
    This allows the originating AI to specify its desired response structure.
    """
    
    structure: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Format structure of the response"
    )
    sections: Optional[List[str]] = Field(
        default=None,
        description="Sections to include in the response"
    )
    example: Optional[str] = Field(
        default=None,
        description="Example of the desired format"
    )
    custom_instructions: Optional[str] = Field(
        default=None,
        description="Any custom instructions for formatting the response"
    )


class ReviewSpec(BaseModel):
    """
    Specification for how the AI-to-AI review should be conducted.
    
    This defines the models to use, review method, criteria, and response format.
    """
    
    generator_model: str = Field(
        description="Model to use for generating the initial response (e.g., 'openai:gpt-4')"
    )
    reviewer_model: str = Field(
        description="Model to use for reviewing the response (e.g., 'anthropic:claude-3-opus')"
    )
    review_type: ReviewType = Field(
        default=ReviewType.STANDARD,
        description="Type of review to perform"
    )
    review_criteria: List[str] = Field(
        default_factory=lambda: [
            ReviewCriteria.FACTUAL_ACCURACY.value,
            ReviewCriteria.REASONING.value,
            ReviewCriteria.COMPLETENESS.value
        ],
        description="Criteria to use for the review"
    )
    response_format: FormatSpec = Field(
        default_factory=FormatSpec,
        description="Format specification for the response"
    )
    tools: Optional[List[str]] = Field(
        default=None,
        description="Tools to make available during the review process"
    )
    max_debate_turns: Optional[int] = Field(
        default=3,
        description="Maximum number of turns for debate mode"
    )
    consistency_samples: Optional[int] = Field(
        default=3,
        description="Number of samples to generate for self-consistency checks"
    )
    system_prompt_override: Optional[str] = Field(
        default=None,
        description="Custom system prompt to use for the generator model"
    )
    reviewer_prompt_override: Optional[str] = Field(
        default=None,
        description="Custom system prompt to use for the reviewer model"
    )


class ReviewResult(BaseModel):
    """Results from an AI-to-AI review process."""
    
    initial_response: Any = Field(
        description="The initial response from the generator model"
    )
    review_feedback: Optional[Any] = Field(
        default=None,
        description="Feedback from the reviewer model"
    )
    final_answer: Any = Field(
        description="The final answer after the review process"
    )
    debug_info: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Debug information about the review process"
    )
    metrics: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Metrics about the review process (e.g., confidence scores)"
    )
