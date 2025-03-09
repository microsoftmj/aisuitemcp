"""
Test implementation of AISuite MCP.

This script performs a basic test of the AISuite MCP implementation
to verify that it works end-to-end with AISuite.
"""

import os
import logging
import sys

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from aisuite_mcp import (
    MCPClient, 
    ReviewSpec, 
    FormatSpec, 
    ResponseFormat, 
    ReviewType
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("aisuite_mcp_test")

# Check if API key is set
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("Error: OPENAI_API_KEY environment variable not set")
    print("Please set your OpenAI API key with: export OPENAI_API_KEY='your-api-key'")
    sys.exit(1)

def test_basic_review():
    """Test basic review functionality"""
    print("\n===== Testing Basic Review Functionality =====\n")
    
    # Initialize client with OpenAI
    client = MCPClient(
        provider_configs={
            "openai": {"api_key": OPENAI_API_KEY},
        }
    )
    
    # Define a simple review specification
    review_spec = ReviewSpec(
        generator_model="openai:gpt-3.5-turbo",
        reviewer_model="openai:gpt-3.5-turbo",
        review_type=ReviewType.STANDARD,
        response_format=FormatSpec(
            structure=ResponseFormat.MARKDOWN,
            sections=["Answer"],
        )
    )
    
    # Test with a simple prompt
    prompt = "What is 2 + 2?"
    
    print(f"Testing with prompt: '{prompt}'")
    print(f"Generator model: {review_spec.generator_model}")
    print(f"Reviewer model: {review_spec.reviewer_model}")
    print(f"Review type: {review_spec.review_type}")
    print(f"Format: {review_spec.response_format.structure}")
    
    try:
        # Get peer-reviewed response
        result = client.create_reviewed_completion(
            prompt=prompt,
            review_spec=review_spec,
        )
        
        print("\n--- Initial Response ---")
        print(result.initial_response)
        
        print("\n--- Review Feedback ---")
        print(result.review_feedback)
        
        print("\n--- Final Answer ---")
        print(result.final_answer)
        
        print("\n✅ Test completed successfully!")
        return True
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        return False

def main():
    """Run tests and report results"""
    print("Starting AISuite MCP implementation test...")
    
    # Run tests
    basic_test_success = test_basic_review()
    
    # Report results
    print("\n===== Test Results =====")
    print(f"Basic Review Test: {'✅ PASSED' if basic_test_success else '❌ FAILED'}")
    
    if basic_test_success:
        print("\nAISuite MCP implementation is working correctly!")
    else:
        print("\nAISuite MCP implementation has issues that need to be fixed.")
    
if __name__ == "__main__":
    main()
