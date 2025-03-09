"""
Format Parser for AISuite MCP.

This module handles parsing and enforcement of response format specifications.
"""

import json
from typing import Dict, List, Optional, Any, Union

from ..models import FormatSpec, ResponseFormat


class FormatParser:
    """
    Parser for response format specifications.
    
    This class handles parsing format specifications and generating
    appropriate prompts to ensure responses adhere to the specified format.
    """
    
    def __init__(self):
        """Initialize the format parser."""
        pass
    
    def format_to_prompt_instructions(self, format_spec: FormatSpec) -> str:
        """
        Convert a format specification to prompt instructions.
        
        Args:
            format_spec: Specification for how responses should be formatted
            
        Returns:
            String containing instructions for the LLM to follow the format
        """
        structure = format_spec.structure.value
        instructions = f"Format your response as {structure.upper()}."
        
        if format_spec.sections:
            if structure == ResponseFormat.JSON.value:
                instructions += f" Include the following sections as JSON keys: {', '.join(format_spec.sections)}."
            elif structure == ResponseFormat.MARKDOWN.value:
                instructions += f" Include the following sections as headers: {', '.join(format_spec.sections)}."
            else:  # Text format
                instructions += f" Include the following sections clearly labeled: {', '.join(format_spec.sections)}."
        
        if format_spec.example:
            instructions += f" For example: {format_spec.example}"
        
        if format_spec.custom_instructions:
            instructions += f" {format_spec.custom_instructions}"
        
        return instructions
    
    def ensure_format_compliance(self, content: str, format_spec: FormatSpec) -> str:
        """
        Attempt to ensure the content complies with the format specification.
        
        Args:
            content: The response content to format
            format_spec: Specification for how responses should be formatted
            
        Returns:
            Formatted content that complies with the specification
        """
        # For JSON format, attempt to parse and validate sections
        if format_spec.structure == ResponseFormat.JSON:
            try:
                # Try to parse as JSON
                data = json.loads(content)
                
                # Check if all required sections are present
                if format_spec.sections:
                    missing_sections = [
                        section for section in format_spec.sections
                        if section not in data
                    ]
                    
                    if missing_sections:
                        # If sections are missing, add them with placeholder values
                        for section in missing_sections:
                            data[section] = f"[No {section} provided]"
                
                # Convert back to JSON string
                return json.dumps(data, indent=2)
            except json.JSONDecodeError:
                # If not valid JSON, try to extract content and format properly
                if format_spec.sections:
                    result = {}
                    
                    for section in format_spec.sections:
                        section_content = self._extract_section(content, section)
                        result[section] = section_content or f"[No {section} provided]"
                    
                    return json.dumps(result, indent=2)
                else:
                    # Just wrap in a generic JSON object if we can't determine sections
                    return json.dumps({"content": content}, indent=2)
        
        # For Markdown format, ensure proper headers
        elif format_spec.structure == ResponseFormat.MARKDOWN:
            if format_spec.sections:
                # Check if the markdown already has the required sections
                lines = content.split("\n")
                existing_sections = [
                    line.strip("#").strip().lower() 
                    for line in lines 
                    if line.strip().startswith("#")
                ]
                
                if not all(section.lower() in existing_sections for section in format_spec.sections):
                    # If sections are missing, try to format properly
                    formatted_parts = []
                    remaining_content = content
                    
                    for section in format_spec.sections:
                        section_content = self._extract_section(remaining_content, section)
                        
                        if section_content:
                            formatted_parts.append(f"## {section}\n\n{section_content}\n")
                            # Remove the extracted section from remaining content
                            remaining_content = remaining_content.replace(section_content, "")
                        else:
                            formatted_parts.append(f"## {section}\n\n[No {section} provided]\n")
                    
                    # If there's content left that wasn't associated with a section, add it to the end
                    remaining_content = remaining_content.strip()
                    if remaining_content:
                        formatted_parts.append(f"## Additional Information\n\n{remaining_content}\n")
                    
                    return "\n".join(formatted_parts)
            
            # If no specific formatting needed, return as is
            return content
        
        # For plain text, just ensure sections are labeled if specified
        elif format_spec.structure == ResponseFormat.TEXT:
            if format_spec.sections:
                # Check if the text already has the required sections
                formatted_parts = []
                
                for section in format_spec.sections:
                    section_content = self._extract_section(content, section)
                    
                    if section_content:
                        formatted_parts.append(f"{section}:\n{section_content}\n")
                    else:
                        formatted_parts.append(f"{section}:\n[No {section} provided]\n")
                
                return "\n".join(formatted_parts)
            
            # If no specific formatting needed, return as is
            return content
        
        # Default: return content as is
        return content
    
    def _extract_section(self, content: str, section_name: str) -> Optional[str]:
        """
        Attempt to extract a section from unstructured content.
        
        Args:
            content: The content to extract from
            section_name: Name of the section to extract
            
        Returns:
            Extracted section content or None if not found
        """
        # Try common patterns for section headers
        patterns = [
            f"## {section_name}",
            f"#{section_name}",
            f"{section_name}:",
            f'"{section_name}":'
        ]
        
        for pattern in patterns:
            pattern_lower = pattern.lower()
            if pattern_lower in content.lower():
                # Find the start of the section
                start_idx = content.lower().find(pattern_lower)
                start_idx = content.find("\n", start_idx)
                if start_idx == -1:
                    start_idx = content.find(":", content.lower().find(pattern_lower))
                    if start_idx == -1:
                        continue
                    start_idx += 1
                else:
                    start_idx += 1
                
                # Find the end of the section (next section or end of content)
                end_idx = len(content)
                for next_pattern in patterns:
                    next_idx = content.lower().find(next_pattern.lower(), start_idx)
                    if next_idx != -1 and next_idx < end_idx:
                        end_idx = next_idx
                
                # Extract the section content
                section_content = content[start_idx:end_idx].strip()
                return section_content
        
        # Try to find the section name in the content and extract surrounding text
        section_idx = content.lower().find(section_name.lower())
        if section_idx != -1:
            # Find start and end based on surrounding whitespace/punctuation
            start_idx = content.rfind("\n", 0, section_idx)
            if start_idx == -1:
                start_idx = 0
            else:
                start_idx += 1
            
            end_idx = content.find("\n\n", section_idx)
            if end_idx == -1:
                end_idx = len(content)
            
            section_content = content[start_idx:end_idx].strip()
            return section_content
        
        return None

    def extract_final_answer(self, content: str, format_spec: FormatSpec) -> Any:
        """
        Extract the final answer from the content based on the format specification.
        
        Args:
            content: The content to extract from
            format_spec: Specification for how responses are formatted
            
        Returns:
            Extracted final answer, format depends on the response format
        """
        if format_spec.structure == ResponseFormat.JSON:
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                # If not valid JSON, return the raw content
                return content
        
        # For markdown and text, return the content as is
        return content
