"""
AISuite MCP: Multi-Component Platform for AI-to-AI peer review

This package extends AISuite to enable AI-to-AI peer review through
structured prompt-response frameworks.
"""

from .client import MCPClient
from .models import ReviewSpec, FormatSpec, ResponseFormat, ReviewType, ReviewCriteria

__version__ = "0.1.0"

__all__ = [
    "MCPClient",
    "ReviewSpec",
    "FormatSpec",
    "ResponseFormat",
    "ReviewType",
    "ReviewCriteria",
]
