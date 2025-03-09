"""
Tools Manager for AISuite MCP.

This module handles tool registration and execution during the review process.
"""

import inspect
import logging
import json
from typing import Dict, List, Optional, Any, Callable, Type, Set
from pydantic import BaseModel, create_model, Field, ValidationError

class Tools:
    """
    Simplified implementation of AISuite's Tools class.
    
    This class handles tool registration and execution for use with AISuite.
    """
    
    def __init__(self, tools: list[Callable] = None):
        self._tools = {}
        if tools:
            for tool in tools:
                self._add_tool(tool)
    
    def _add_tool(self, func: Callable):
        """Register a tool function with metadata."""
        tool_spec, param_model = self.__infer_from_signature(func)
        
        self._tools[func.__name__] = {
            "function": func,
            "param_model": param_model,
            "spec": tool_spec,
        }
    
    def tools(self, format="openai") -> list:
        """Return tools in the specified format (default OpenAI)."""
        if format == "openai":
            return [{"type": "function", "function": tool["spec"]} for tool in self._tools.values()]
        return [tool["spec"] for tool in self._tools.values()]
    
    def __infer_from_signature(self, func: Callable):
        """Infer parameters and model from function signature."""
        signature = inspect.signature(func)
        fields = {}
        required_fields = []
        
        # Get docstring
        docstring = inspect.getdoc(func) or ""
        
        for param_name, param in signature.parameters.items():
            # Check if type annotation is missing
            if param.annotation == inspect._empty:
                raise TypeError(
                    f"Parameter '{param_name}' in function '{func.__name__}' must have a type annotation."
                )
            
            # Determine field type and optionality
            param_type = param.annotation
            description = ""
            
            if param.default == inspect._empty:
                fields[param_name] = (param_type, Field(..., description=description))
                required_fields.append(param_name)
            else:
                fields[param_name] = (
                    param_type,
                    Field(default=param.default, description=description),
                )
        
        # Create a Pydantic model based on the function signature
        param_model = create_model(f"{func.__name__.capitalize()}Params", **fields)
        
        # Create a tool spec for OpenAI function calling format
        tool_spec = {
            "name": func.__name__,
            "description": docstring,
            "parameters": {
                "type": "object",
                "properties": {
                    name: {
                        "type": self.__type_to_json_type(param.annotation),
                        "description": "",
                    }
                    for name, param in signature.parameters.items()
                },
                "required": required_fields,
            },
        }
        
        return tool_spec, param_model
    
    def __type_to_json_type(self, typ):
        """Convert Python type to JSON Schema type."""
        type_mapping = {
            str: "string",
            int: "integer",
            float: "number",
            bool: "boolean",
            list: "array",
            dict: "object",
        }
        return type_mapping.get(typ, "string")
    
    def execute(self, tool_calls):
        """Execute registered tools."""
        results = []
        messages = []
        
        # Handle single tool call or list of tool calls
        if not isinstance(tool_calls, list):
            tool_calls = [tool_calls]
        
        for tool_call in tool_calls:
            # Handle both dictionary and object-style tool calls
            if isinstance(tool_call, dict):
                tool_name = tool_call["function"]["name"]
                arguments = tool_call["function"]["arguments"]
                tool_call_id = tool_call.get("id", "tool_call_0")
            else:
                tool_name = tool_call.function.name
                arguments = tool_call.function.arguments
                tool_call_id = getattr(tool_call, "id", "tool_call_0")
            
            # Ensure arguments is a dict
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            
            if tool_name not in self._tools:
                raise ValueError(f"Tool '{tool_name}' not registered.")
            
            tool = self._tools[tool_name]
            tool_func = tool["function"]
            param_model = tool["param_model"]
            
            # Validate and parse the arguments with Pydantic
            try:
                validated_args = param_model(**arguments)
                result = tool_func(**validated_args.model_dump())
                results.append(result)
                messages.append({
                    "role": "tool",
                    "name": tool_name,
                    "content": json.dumps(result),
                    "tool_call_id": tool_call_id,
                })
            except ValidationError as e:
                raise ValueError(f"Error in tool '{tool_name}' parameters: {e}")
        
        return results, messages
    
    def execute_tool(self, tool_calls):
        """Execute tools and return results and messages as expected by AISuite.
        
        This matches the interface of AISuite's Tools.execute_tool method.
        """
        return self.execute(tool_calls)

class ToolsManager:
    """
    Manager for MCP tools.
    
    This class handles tool registration and makes tools available
    for use during the AI generation and review process.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the tools manager.
        
        Args:
            logger: Logger instance
        """
        self.logger = logger or logging.getLogger("aisuite_mcp.tools")
        self._tools: Dict[str, Callable] = {}
        self._aisuite_tools = Tools()
    
    def register_tool(self, tool_func: Callable) -> None:
        """
        Register a tool function to be used during the review process.
        
        Args:
            tool_func: Callable function to register as a tool
        """
        # Validate that the function has type annotations
        signature = inspect.signature(tool_func)
        for param_name, param in signature.parameters.items():
            if param.annotation == inspect._empty:
                raise TypeError(
                    f"Parameter '{param_name}' in tool function '{tool_func.__name__}' "
                    f"must have a type annotation."
                )
        
        # Register the tool
        self._tools[tool_func.__name__] = tool_func
        
        # Register with AISuite tools
        self._aisuite_tools._add_tool(tool_func)
        
        self.logger.info(f"Registered tool: {tool_func.__name__}")
    
    def get_tool(self, tool_name: str) -> Optional[Callable]:
        """
        Get a registered tool function by name.
        
        Args:
            tool_name: Name of the tool to retrieve
            
        Returns:
            The tool function if found, None otherwise
        """
        return self._tools.get(tool_name)
    
    def get_tool_names(self) -> List[str]:
        """
        Get the names of all registered tools.
        
        Returns:
            List of tool names
        """
        return list(self._tools.keys())
    
    def get_tool_descriptions(self) -> List[Dict[str, Any]]:
        """
        Get descriptions of all registered tools.
        
        Returns:
            List of tool descriptions in OpenAI format
        """
        tool_specs = self._aisuite_tools.tools("openai")
        return tool_specs
    
    def get_aisuite_tools(self, tool_names: Optional[List[str]] = None) -> Tools:
        """
        Get a Tools object with the specified tools.
        
        Args:
            tool_names: List of tool names to include (or None for all)
            
        Returns:
            Tools object with the specified tools
        """
        if tool_names is None:
            # Return all tools
            return self._aisuite_tools
        
        # Filter to only the requested tools
        filtered_tools = Tools()
        for name in tool_names:
            tool = self.get_tool(name)
            if tool:
                filtered_tools._add_tool(tool)
        
        return filtered_tools
    
    def execute_tool(self, tool_name: str, **kwargs) -> Any:
        """
        Execute a registered tool function.
        
        Args:
            tool_name: Name of the tool to execute
            **kwargs: Arguments to pass to the tool function
            
        Returns:
            Result of the tool execution
            
        Raises:
            ValueError: If the tool is not registered
        """
        tool = self.get_tool(tool_name)
        if not tool:
            raise ValueError(f"Tool '{tool_name}' not registered.")
        
        try:
            self.logger.debug(f"Executing tool {tool_name} with args: {kwargs}")
            result = tool(**kwargs)
            self.logger.debug(f"Tool {tool_name} result: {result}")
            return result
        except Exception as e:
            self.logger.error(f"Error executing tool {tool_name}: {e}")
            raise
