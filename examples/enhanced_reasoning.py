"""
Example of using AISuite MCP with Claude 3.7 Sonnet's enhanced reasoning capabilities.

This example demonstrates how to leverage the extended reasoning features
of Claude 3.7 Sonnet in peer review scenarios.
"""

import logging
import os
from aisuite_mcp import MCPClient
from aisuite_mcp.models import ReviewSpec, ReviewType, FormatSpec, ResponseFormat

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("aisuite_mcp.examples")

def main():
    """
    Run the enhanced reasoning example.
    """
    # Initialize the client with only the providers we need
    # This will automatically load from .env file
    client = MCPClient(provider_configs={
        "openai": {"api_key": os.environ.get("OPENAI_API_KEY")},
        "anthropic": {"api_key": os.environ.get("ANTHROPIC_API_KEY")}
    })
    
    # Create a challenging prompt that requires deep reasoning
    prompt = """
    Consider the trolley problem:
    You see a runaway trolley moving toward five tied-up people lying on the tracks. 
    You are standing beside a lever that controls a switch. If you pull the lever, 
    the trolley will be redirected onto a side track, and the five people on the 
    main track will be saved. However, there is a single person lying on the side track.
    
    What is the ethical choice? What moral frameworks would support different decisions?
    Provide a comprehensive analysis of this scenario.
    """
    
    # Define review criteria focused on reasoning quality
    review_criteria = [
        "Depth of reasoning",
        "Consideration of multiple perspectives",
        "Logical consistency",
        "Identification of assumptions",
        "Clarity of ethical principles invoked",
        "Comprehensiveness of analysis",
    ]
    
    # Set up review specification with Claude 3.7 Sonnet as reviewer
    review_spec = ReviewSpec(
        generator_model="openai:gpt-4o",
        reviewer_model="anthropic:claude-3-7-sonnet-20250219",
        review_type=ReviewType.STANDARD,
        review_criteria=review_criteria,
        response_format=FormatSpec(structure=ResponseFormat.MARKDOWN),
    )
    
    # Run the peer review
    logger.info("Creating peer-reviewed completion with enhanced reasoning...")
    result = client.create_reviewed_completion(
        prompt=prompt,
        review_spec=review_spec,
    )
    
    # Display results
    print("\n=== INITIAL RESPONSE ===\n")
    print(result.initial_response)
    
    print("\n=== REVIEW FEEDBACK ===\n")
    print(result.review_feedback)
    
    print("\n=== FINAL ANSWER ===\n")
    print(result.final_answer)

if __name__ == "__main__":
    main()
