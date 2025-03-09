"""
Setup script for AISuite MCP.

This script is used to install the AISuite MCP package.
"""

from setuptools import setup, find_packages

setup(
    name="aisuite_mcp",
    version="0.1.0",
    description="Multi-Component Platform for AI-to-AI peer review built on AISuite",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="AI Suite MCP Team",
    license="MIT",
    packages=find_packages(),
    install_requires=[
        "aisuite>=0.1.5",
        "pydantic>=2.0.0",
        "aiohttp>=3.8.0",
        "typing-extensions>=4.0.0",
    ],
    python_requires=">=3.9",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
