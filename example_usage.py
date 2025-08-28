
import os
from dotenv import load_dotenv
from langchain_litellm import ChatLiteLLM

# LEARNING: LangGraph Supervisor provides the hierarchical agent orchestration pattern
# TLDR: These imports let us create a team of AI helpers with a boss that coordinates them
from langgraph_supervisor import create_supervisor, LangfuseSupervisorTracer

# LEARNING: ReAct agent pattern: Reasoning + Acting cycle for deliberative agents
# TLDR: This helps create AI helpers that can think step-by-step and use tools
from langgraph.prebuilt import create_react_agent

# Load environment variables
load_dotenv()

# LEARNING: Tool definitions for the math agent
# TLDR: These are special abilities we're giving to our AI helpers
def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b

def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b

# LEARNING: Knowledge retrieval tool for the research agent
# TLDR: This tool lets our AI pretend to search the internet
def web_search(query: str) -> str:
    """Simulate searching the web."""
    return f"Search results for '{query}': Example data..."

# LEARNING: Model configuration with provider abstraction via LiteLLM
# TLDR: Choose which version of Claude we want to use
claude_model_id = os.getenv("CLAUDE_MODEL_ID", "us.anthropic.claude-sonnet-4-20250514-v1:0")
model_provider = os.getenv("MODEL_PROVIDER", "anthropic")

# LEARNING: LiteLLM configuration for model access
# TLDR: Set up a way to connect to any AI model service
lite_llm_base_url = os.getenv("LITE_LLM_BASE_URL")
lite_llm_api_key = os.getenv("LITE_LLM_SECRET_KEY")

if not lite_llm_base_url or not lite_llm_api_key:
    print("Warning: LITE_LLM_BASE_URL or LITE_LLM_SECRET_KEY not set in environment.")
    print("Please set these variables in your .env file.")
    exit(1)

# LEARNING: Initialize a single model instance to be shared across all agents
# TLDR: Create one AI brain that all our helpers will share
claude_model = ChatLiteLLM(
    model=claude_model_id, 
    model_provider=model_provider,
    api_base=lite_llm_base_url, 
    api_key=lite_llm_api_key,
    temperature=0.3  # Lower temperature for more deterministic agent behavior
)

# LEARNING: Create specialized agents using the ReAct pattern
# TLDR: Create AI helpers that are good at different things

# Math specialist agent with numerical operation tools
math_agent = create_react_agent(
    model=claude_model,
    tools=[add, multiply],  # Tool selection defines agent capabilities
    name="math_expert",  # Names are required for routing in multi-agent systems
    prompt="You are a math expert who can perform calculations."  # Role-specific instructions
)

# Research specialist agent with information retrieval tools
research_agent = create_react_agent(
    model=claude_model,
    tools=[web_search],
    name="researcher",
    prompt="You are a research expert who can find information online."
)

# LEARNING: Create supervisor workflow for hierarchical agent orchestration
# TLDR: Create a boss AI that will decide which helper to use for each question
workflow = create_supervisor(
    [math_agent, research_agent],  # Agent registry for delegation targets
    model=claude_model,  # Using same model for reasoning consistency across system
    prompt="""You are a supervisor agent that coordinates specialized expert agents.
    You have access to the following agents:
    - math_expert: For solving mathematical calculations
    - researcher: For finding information online
    
    When given a task, decide which agent is best suited to handle it, 
    then use the handoff tools to delegate the task to that agent.
    """,
    # Note: Handoff tools are automatically generated for each registered agent
)

# LEARNING: Compile the workflow into an executable graph
# TLDR: Turn our team design into a working program
app = workflow.compile(name="MixedTaskSolver")

# LEARNING: Configure Langfuse tracing for agent orchestration observability
# TLDR: Set up a special tool that watches how our AI team works together
tracer = LangfuseSupervisorTracer(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST", "http://localhost:3000")
)

# LEARNING: Apply tracing wrapper to the workflow using decorator pattern
# TLDR: Turn on the recording system for our AI team
traced_app = tracer.trace_workflow(app)

# LEARNING: Execute the workflow with a multi-task query
# TLDR: Ask our AI team a question that needs both math and research
result = traced_app.invoke({
    "messages": [{
        "role": "user",
        "content": "I need to calculate 123 × 456 and find information about the largest rainforest."
    }]
})
# LEARNING: Task decomposition and routing process:
# TLDR: Behind the scenes:
# 1. The boss will see there's a math problem and ask the math helper
# 2. The boss will see there's a research question and ask the research helper
# 3. The boss will combine their answers into one complete response

# LEARNING: Access workflow execution results from the returned state
# TLDR: Show the answer our AI team came up with
print("Workflow completed!")
print(f"Final message: {result['messages'][-1].content}")  # Final message in conversation history
print("Trace available at: http://localhost:3000")  # Langfuse trace visualization URL