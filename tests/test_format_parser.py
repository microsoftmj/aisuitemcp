"""
Test for the format parser functionality.

This script tests the format parser's ability to generate prompt instructions
and ensure compliance with specified format requirements.
"""

import os
import sys
import unittest

# Add parent directory to path to allow importing aisuite_mcp
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from aisuite_mcp import FormatSpec, ResponseFormat
from aisuite_mcp.format.parser import FormatParser


class TestFormatParser(unittest.TestCase):
    """Test cases for format parser functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.parser = FormatParser()
    
    def test_format_to_prompt_instructions(self):
        """Test generating prompt instructions from format specifications."""
        # Test JSON format
        json_format = FormatSpec(
            structure=ResponseFormat.JSON,
            sections=["reasoning", "answer", "confidence"],
            example='{ "reasoning": "Step-by-step analysis", "answer": "42", "confidence": 0.95 }'
        )
        
        json_instructions = self.parser.format_to_prompt_instructions(json_format)
        
        self.assertIn("Format your response as JSON", json_instructions)
        self.assertIn("reasoning, answer, confidence", json_instructions)
        self.assertIn("Step-by-step analysis", json_instructions)
        
        # Test Markdown format
        markdown_format = FormatSpec(
            structure=ResponseFormat.MARKDOWN,
            sections=["Background", "Analysis", "Conclusion"],
        )
        
        markdown_instructions = self.parser.format_to_prompt_instructions(markdown_format)
        
        self.assertIn("Format your response as MARKDOWN", markdown_instructions)
        self.assertIn("Background, Analysis, Conclusion", markdown_instructions)
        self.assertIn("headers", markdown_instructions)
        
        # Test with custom instructions
        custom_format = FormatSpec(
            structure=ResponseFormat.TEXT,
            custom_instructions="Use bullet points for lists and number all steps."
        )
        
        custom_instructions = self.parser.format_to_prompt_instructions(custom_format)
        
        self.assertIn("Format your response as TEXT", custom_instructions)
        self.assertIn("Use bullet points for lists", custom_instructions)
    
    def test_ensure_format_compliance_json(self):
        """Test ensuring compliance with JSON format."""
        json_format = FormatSpec(
            structure=ResponseFormat.JSON,
            sections=["reasoning", "answer"]
        )
        
        # Test with valid JSON
        valid_json = '{"reasoning": "The calculation is simple", "answer": 42}'
        result = self.parser.ensure_format_compliance(valid_json, json_format)
        
        self.assertIsInstance(result, str)
        self.assertIn("reasoning", result)
        self.assertIn("answer", result)
        
        # Test with missing section
        incomplete_json = '{"reasoning": "The calculation is simple"}'
        result = self.parser.ensure_format_compliance(incomplete_json, json_format)
        
        self.assertIn("answer", result)
        self.assertIn("No answer provided", result)
        
        # Test with non-JSON
        non_json = "The answer is 42. Reasoning: It's the meaning of life."
        result = self.parser.ensure_format_compliance(non_json, json_format)
        
        self.assertIn("reasoning", result)
        self.assertIn("answer", result)
    
    def test_ensure_format_compliance_markdown(self):
        """Test ensuring compliance with Markdown format."""
        markdown_format = FormatSpec(
            structure=ResponseFormat.MARKDOWN,
            sections=["Analysis", "Conclusion"]
        )
        
        # Test with proper markdown
        valid_markdown = """
        ## Analysis
        
        This is an analysis.
        
        ## Conclusion
        
        This is a conclusion.
        """
        
        result = self.parser.ensure_format_compliance(valid_markdown, markdown_format)
        self.assertIn("## Analysis", result)
        self.assertIn("## Conclusion", result)
        
        # Test with missing section
        incomplete_markdown = """
        ## Analysis
        
        This is an analysis.
        """
        
        result = self.parser.ensure_format_compliance(incomplete_markdown, markdown_format)
        self.assertIn("## Analysis", result)
        self.assertIn("## Conclusion", result)
        self.assertIn("No Conclusion provided", result)
    
    def test_extract_section(self):
        """Test extracting sections from unstructured content."""
        content = """
        ## Introduction
        
        This is an introduction.
        
        ## Analysis
        
        This is an analysis.
        
        ## Conclusion
        
        This is a conclusion.
        """
        
        intro = self.parser._extract_section(content, "Introduction")
        self.assertEqual(intro.strip(), "This is an introduction.")
        
        analysis = self.parser._extract_section(content, "Analysis")
        self.assertEqual(analysis.strip(), "This is an analysis.")
        
        conclusion = self.parser._extract_section(content, "Conclusion")
        self.assertEqual(conclusion.strip(), "This is a conclusion.")
        
        # Test case insensitivity
        analysis_lower = self.parser._extract_section(content, "analysis")
        self.assertEqual(analysis_lower.strip(), "This is an analysis.")
        
        # Test non-existent section
        nonexistent = self.parser._extract_section(content, "NonExistent")
        self.assertIsNone(nonexistent)


if __name__ == '__main__':
    unittest.main()
