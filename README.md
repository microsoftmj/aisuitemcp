# AISuite MCP: Multi-Component Platform for AI-to-AI Peer Review

AISuite MCP extends the [AISuite](https://github.com/andrewyng/aisuite) library to enable AI-to-AI peer review through structured prompt-response frameworks. This platform facilitates advanced reasoning techniques such as chain-of-thought, self-consistency, and debate-style AI interactions, while allowing the originating AI to specify its desired response format.

## Features

- **AI-to-AI Peer Review**: Have one AI generate a response and another AI critique and improve it
- **Flexible Review Mechanisms**: Support for standard review, debate mode, and self-consistency checks
- **Format Specification**: Originating AI can specify desired response formats and structures
- **Tool Integration**: Transparent access to tools for verification and enhanced reasoning
- **Asynchronous Processing**: Support for both synchronous and asynchronous workflows

## Installation

### From PyPI

```bash
pip install aisuite-mcp
```

### From Source

```bash
# Clone the repository
git clone https://github.com/microsoftmj/aisuitemcp.git
cd aisuitemcp

# Install in development mode
pip install -e .
```

## Prerequisites

This package requires:
- Python 3.9 or later
- AISuite 0.1.5 or later
- Valid API keys for LLM providers (OpenAI, Anthropic, etc.)

## Quick Start

```python
from aisuite_mcp import MCPClient, ReviewSpec, ReviewType

# Initialize the client
mcp_client = MCPClient(
    provider_configs={
        "openai": {"api_key": "your_openai_key"},
        "anthropic": {"api_key": "your_anthropic_key"}
    }
)

# Define review specification
review_spec = ReviewSpec(
    generator_model="openai:gpt-4",
    reviewer_model="openai:gpt-4",
    review_type=ReviewType.STANDARD,
    review_criteria=["factual_accuracy", "reasoning", "completeness"],
    response_format={
        "structure": "markdown",
        "sections": ["reasoning", "answer", "confidence"]
    }
)

# Get peer-reviewed response
response = mcp_client.create_reviewed_completion(
    prompt="Explain the implications of quantum computing on cryptography",
    review_spec=review_spec
)

print(response.final_answer)
```

## Architecture

AISuite MCP is built as a wrapper around AISuite, leveraging its unified interface to multiple LLM providers while adding multi-component orchestration capabilities. The architecture consists of:

- **MCP Orchestrator**: Coordinates the workflow between different AI models and components
- **Format Parser**: Handles parsing and enforcement of response format specifications
- **Review Manager**: Implements various review strategies and processes  
- **Tools Manager**: Manages tool registration, execution, and result integration

## Format Specification

The format specification allows the originating AI to define how it wants responses structured:

```python
from aisuite_mcp import FormatSpec, ResponseFormat

format_spec = FormatSpec(
    structure=ResponseFormat.JSON,
    sections=["reasoning", "answer", "confidence"],
    example='{ "reasoning": "...", "answer": "...", "confidence": 0.95 }'
)
```

## Review Mechanisms

AISuite MCP supports multiple review mechanisms:

- **Standard Review**: One AI generates, another reviews based on criteria
- **Debate Mode**: Two AIs debate a question with multiple turns
- **Self-Consistency**: Generate multiple reasoning paths and determine consensus

## Tool Integration

You can register custom tools that will be available during the review process:

```python
def search(query: str) -> str:
    """Search for information on a topic."""
    # Implement search functionality
    return f"Results for: {query}"

# Register the tool
mcp_client.register_tool(search)

# Use in review specification
review_spec = ReviewSpec(
    generator_model="openai:gpt-4",
    reviewer_model="openai:gpt-4",
    tools=["search"]  # Make search tool available
)
```

## Examples

See the `examples/` directory for more detailed usage examples:
- `basic_usage.py`: Demonstrates core functionality
- `test_implementation.py`: Simple test script for verification

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT
