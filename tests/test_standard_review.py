"""
Test for standard AI-to-AI peer review functionality.

This script tests the basic flow of the standard review process
where one model generates a response and another reviews it.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# Add parent directory to path to allow importing aisuite_mcp
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from aisuite_mcp import (
    MCPClient, 
    ReviewSpec, 
    FormatSpec, 
    ResponseFormat, 
    ReviewType,
    ReviewResult
)


class TestStandardReview(unittest.TestCase):
    """Test cases for standard AI-to-AI peer review."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock AISuite client
        self.mock_aisuite_client = MagicMock()
        
        # Set up mock response for the generator model
        generator_response = MagicMock()
        generator_response.choices = [MagicMock()]
        generator_response.choices[0].message = MagicMock()
        generator_response.choices[0].message.content = "This is the initial response from the generator model."
        
        # Set up mock response for the reviewer model
        reviewer_response = MagicMock()
        reviewer_response.choices = [MagicMock()]
        reviewer_response.choices[0].message = MagicMock()
        reviewer_response.choices[0].message.content = """
        Review of the initial response:
        
        The initial response is correct but could be more detailed.
        
        Improved Answer:
        This is the improved response from the reviewer model.
        """
        
        # Configure the mock client to return the mock responses
        self.mock_aisuite_client.chat.completions.create = MagicMock(
            side_effect=[generator_response, reviewer_response]
        )
        
        # Create the MCP client with the mock AISuite client
        with patch('aisuite_mcp.client.AISuiteClient', return_value=self.mock_aisuite_client):
            self.client = MCPClient()
        
        # Replace the real AISuite client with the mock
        self.client.client = self.mock_aisuite_client
    
    def test_standard_review(self):
        """Test the standard review process."""
        # Define review specification
        review_spec = ReviewSpec(
            generator_model="openai:gpt-3.5-turbo",
            reviewer_model="openai:gpt-4",
            review_type=ReviewType.STANDARD,
            response_format=FormatSpec(
                structure=ResponseFormat.MARKDOWN,
                sections=["Analysis", "Answer"],
            )
        )
        
        # Get peer-reviewed response
        result = self.client.create_reviewed_completion(
            prompt="Test prompt",
            review_spec=review_spec,
        )
        
        # Verify the result
        self.assertIsInstance(result, ReviewResult)
        self.assertEqual(
            result.initial_response, 
            "This is the initial response from the generator model."
        )
        self.assertIn("Review of the initial response", result.review_feedback)
        self.assertIn("This is the improved response", result.final_answer)
        
        # Verify that AISuite client was called with the correct parameters
        call_args_list = self.mock_aisuite_client.chat.completions.create.call_args_list
        
        # Verify generator call
        generator_call = call_args_list[0]
        self.assertEqual(generator_call[1]["model"], "openai:gpt-3.5-turbo")
        self.assertIn({"role": "user", "content": "Test prompt"}, generator_call[1]["messages"])
        
        # Verify reviewer call
        reviewer_call = call_args_list[1]
        self.assertEqual(reviewer_call[1]["model"], "openai:gpt-4")
        self.assertIn("system", reviewer_call[1]["messages"][0]["role"])
        self.assertIn("user", reviewer_call[1]["messages"][1]["role"])
        self.assertIn("This is the initial response", reviewer_call[1]["messages"][1]["content"])


if __name__ == '__main__':
    unittest.main()
