"""
Basic usage examples for AISuite MCP.

This script demonstrates the basic usage of AISuite MCP for AI-to-AI peer review.
"""

import os
import logging
from pprint import pprint

from aisuite_mcp import (
    MCPClient, 
    ReviewSpec, 
    FormatSpec, 
    ResponseFormat, 
    ReviewType, 
    ReviewCriteria
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("aisuite_mcp_example")

# Set API keys from environment variables
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")

# Check if API keys are set
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable not set")

# Initialize the MCP client
client = MCPClient(
    provider_configs={
        "openai": {"api_key": OPENAI_API_KEY},
    }
)

# Add Anthropic if available
if ANTHROPIC_API_KEY:
    client.configure({"anthropic": {"api_key": ANTHROPIC_API_KEY}})


def example_standard_review():
    """
    Example of standard review where one model reviews another's response.
    """
    print("\n=== Example: Standard Review ===\n")
    
    # Define review specification
    review_spec = ReviewSpec(
        generator_model="openai:gpt-3.5-turbo",
        reviewer_model="openai:gpt-4",
        review_type=ReviewType.STANDARD,
        review_criteria=[
            ReviewCriteria.FACTUAL_ACCURACY.value,
            ReviewCriteria.REASONING.value,
            ReviewCriteria.COMPLETENESS.value,
        ],
        response_format=FormatSpec(
            structure=ResponseFormat.MARKDOWN,
            sections=["Analysis", "Answer"],
        )
    )
    
    # Get peer-reviewed response
    result = client.create_reviewed_completion(
        prompt="Explain the implications of quantum computing on cryptography",
        review_spec=review_spec,
    )
    
    print("Initial response:")
    print("----------------")
    print(result.initial_response)
    print("\nReview feedback:")
    print("---------------")
    print(result.review_feedback)
    print("\nFinal answer:")
    print("-------------")
    print(result.final_answer)


def example_format_control():
    """
    Example showing control over response formats.
    """
    print("\n=== Example: Format Control ===\n")
    
    # JSON format
    json_format = FormatSpec(
        structure=ResponseFormat.JSON,
        sections=["reasoning", "answer", "confidence"],
        example='{ "reasoning": "Step-by-step analysis...", "answer": "Final conclusion", "confidence": 0.95 }'
    )
    
    review_spec = ReviewSpec(
        generator_model="openai:gpt-3.5-turbo",
        reviewer_model="openai:gpt-4",
        response_format=json_format
    )
    
    # Get peer-reviewed response with JSON format
    result = client.create_reviewed_completion(
        prompt="What is the capital of France and what is its population?",
        review_spec=review_spec,
    )
    
    print("JSON Formatted Answer:")
    print("---------------------")
    print(result.final_answer)
    
    # Markdown format
    markdown_format = FormatSpec(
        structure=ResponseFormat.MARKDOWN,
        sections=["Background", "Analysis", "Conclusion"],
    )
    
    review_spec.response_format = markdown_format
    
    # Get peer-reviewed response with Markdown format
    result = client.create_reviewed_completion(
        prompt="What is the capital of France and what is its population?",
        review_spec=review_spec,
    )
    
    print("\nMarkdown Formatted Answer:")
    print("-------------------------")
    print(result.final_answer)


def example_debate_review():
    """
    Example of debate-style review where models engage in back-and-forth.
    """
    print("\n=== Example: Debate Review ===\n")
    
    # Define review specification for debate
    review_spec = ReviewSpec(
        generator_model="openai:gpt-3.5-turbo",
        reviewer_model="openai:gpt-4",
        review_type=ReviewType.DEBATE,
        max_debate_turns=3,
        response_format=FormatSpec(
            structure=ResponseFormat.MARKDOWN,
            sections=["Analysis", "Conclusion"],
        )
    )
    
    # Get peer-reviewed response with debate
    result = client.create_reviewed_completion(
        prompt="Is artificial general intelligence (AGI) likely to be achieved in the next decade?",
        review_spec=review_spec,
    )
    
    print("Debate Transcript:")
    print("-----------------")
    for entry in result.review_feedback:
        print(f"\n{entry['role'].upper()} ({entry['model']}):")
        print(entry['content'])
    
    print("\nFinal Conclusion:")
    print("----------------")
    print(result.final_answer)


def example_self_consistency():
    """
    Example of self-consistency review generating multiple reasoning paths.
    """
    print("\n=== Example: Self-Consistency Review ===\n")
    
    # Define review specification for self-consistency
    review_spec = ReviewSpec(
        generator_model="openai:gpt-3.5-turbo",
        reviewer_model="openai:gpt-4",
        review_type=ReviewType.SELF_CONSISTENCY,
        consistency_samples=3,
        response_format=FormatSpec(
            structure=ResponseFormat.MARKDOWN,
            sections=["Common Themes", "Inconsistencies", "Final Answer"],
        )
    )
    
    # Get peer-reviewed response with self-consistency
    result = client.create_reviewed_completion(
        prompt="A train travels from New York to Boston at 60 mph and returns at 40 mph. If the total travel time is 6 hours, what is the distance between New York and Boston?",
        review_spec=review_spec,
    )
    
    print("Multiple Reasoning Paths:")
    print("-----------------------")
    for i, sample in enumerate(result.review_feedback["samples"]):
        print(f"\nSample {i+1}:")
        print(sample)
    
    print("\nAnalysis of Samples:")
    print("------------------")
    print(result.review_feedback["analysis"])
    
    print("\nConsensus Answer:")
    print("----------------")
    print(result.final_answer)


def example_tool_usage():
    """
    Example of using tools during the review process.
    """
    print("\n=== Example: Tool Usage ===\n")
    
    # Define a simple calculator tool
    def calculate(expression: str) -> str:
        """
        Calculate the result of a mathematical expression.
        
        Args:
            expression: Mathematical expression to evaluate
            
        Returns:
            Result of the calculation
        """
        try:
            result = eval(expression, {"__builtins__": {}})
            return f"Result of {expression} = {result}"
        except Exception as e:
            return f"Error evaluating expression: {str(e)}"
    
    # Register the tool
    client.register_tool(calculate)
    
    # Define review specification with tools
    review_spec = ReviewSpec(
        generator_model="openai:gpt-4",
        reviewer_model="openai:gpt-4",
        tools=["calculate"],
        response_format=FormatSpec(
            structure=ResponseFormat.MARKDOWN,
            sections=["Reasoning", "Calculations", "Answer"],
        )
    )
    
    # Get peer-reviewed response with tools
    result = client.create_reviewed_completion(
        prompt="If I invest $1000 at a 5% annual interest rate compounded monthly, how much will I have after 10 years?",
        review_spec=review_spec,
    )
    
    print("Final answer with tool usage:")
    print("---------------------------")
    print(result.final_answer)


async def example_async_usage():
    """
    Example of asynchronous usage.
    """
    print("\n=== Example: Asynchronous Usage ===\n")
    
    # Define review specification
    review_spec = ReviewSpec(
        generator_model="openai:gpt-3.5-turbo",
        reviewer_model="openai:gpt-4",
    )
    
    # Get peer-reviewed response asynchronously
    result = await client.create_reviewed_completion_async(
        prompt="Summarize the key advancements in artificial intelligence in the past year",
        review_spec=review_spec,
    )
    
    print("Async Final Answer:")
    print("------------------")
    print(result.final_answer)


if __name__ == "__main__":
    # Run examples
    example_standard_review()
    example_format_control()
    example_debate_review()
    example_self_consistency()
    example_tool_usage()
    
    # To run the async example, you would need to use an async runner
    # import asyncio
    # asyncio.run(example_async_usage())
