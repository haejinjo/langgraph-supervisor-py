# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

ALWAYS activate the virtual environment before running ANY Python commands.
ALWAYS use python3.10 to start the venv and run Python scripts 

## IMPORTANT: Environment Setup

```bash
# Create virtual environment if it doesn't exist. 
python3.10 -m venv .venv

# ALWAYS activate the virtual environment before any pip or python command
source .venv/bin/activate

# Install the package in development mode
pip install -e .

# Install development dependencies
pip install pytest>=8.0.0 ruff>=0.9.4 mypy>=1.8.0 pytest-socket>=0.7.0 types-setuptools>=69.0.0

# Run tests
pytest

# Run a specific test file
pytest tests/test_supervisor.py

# Run a specific test function
pytest tests/test_supervisor.py::test_supervisor_basic_workflow

# Run linting
ruff check .

# Run type checking
mypy .
```

## Project Architecture

LangGraph Supervisor is a Python library for creating hierarchical multi-agent systems using LangGraph. The architecture centers around a supervisor agent that coordinates multiple specialized agents.

### Core Components

1. **Supervisor Module** (`supervisor.py`): 
   - Contains the main `create_supervisor()` function
   - Builds the LangGraph workflow that connects the supervisor with worker agents
   - Handles message flow between agents and the supervisor

2. **Handoff Mechanism** (`handoff.py`):
   - Implements the agent-to-agent communication protocol
   - Creates specialized tools for agent handoff and message forwarding
   - Manages message history throughout handoff operations

3. **Agent Name Handling** (`agent_name.py`):
   - Handles agent naming conventions and visibility
   - Provides utilities for displaying agent names in different formats

### Data Flow

1. User sends a request to the supervisor agent
2. Supervisor decides which specialized agent should handle the request
3. System hands off control to the selected agent using handoff tools
4. Agent processes the request and returns control to the supervisor
5. Supervisor either forwards the result to the user or delegates to another agent

### Message History Management

The library provides two modes for managing message history:
- `full_history`: Include the entire conversation history from each agent
- `last_message`: Include only the final message from each agent (default)

### Integration with LangGraph

LangGraph Supervisor is built on top of LangGraph and extends it with:
- Tool-based agent handoff mechanism
- Flexible message history management
- Support for multi-level hierarchical agent structures

## Project Structure

```
langgraph_supervisor/
├── __init__.py          # Package exports
├── agent_name.py        # Agent naming utilities
├── handoff.py           # Inter-agent communication tools
└── supervisor.py        # Main supervisor implementation

tests/
├── test_agent_name.py
├── test_supervisor.py
└── test_supervisor_functional_api.py
```