import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langgraph_supervisor import create_supervisor, LangfuseSupervisorTracer
from langgraph.prebuilt import create_react_agent

# Load environment variables
load_dotenv()

# CEO Coffee Expert Tools
def get_ceo_coffee_order() -> str:
    """Get the CEO's preferred coffee order."""
    return "CEO's order: Double shot oat milk cortado, extra hot, no foam, with a dash of cinnamon. Served in the blue ceramic mug from Italy."

def prepare_ceo_coffee_schedule(time: str, meeting_type: str) -> str:
    """Prepare coffee timing based on CEO's schedule."""
    if "board" in meeting_type.lower():
        return f"For {meeting_type} at {time}: Prepare the CEO's cortado 5 minutes before meeting. Use the special boardroom blend beans."
    else:
        return f"For {meeting_type} at {time}: Standard cortado ready 2 minutes before meeting starts."

def check_ceo_coffee_preferences(occasion: str) -> str:
    """Check if CEO has special coffee preferences for different occasions."""
    occasions = {
        "stressful": "Switch to decaf cortado with extra cinnamon and a biscotti",
        "celebration": "Upgrade to single-origin Ethiopian beans with gold leaf garnish",
        "morning": "Regular cortado, but make it extra strong if it's before 7am"
    }
    return occasions.get(occasion.lower(), "Standard cortado preparation applies")

# Dog Coffee Expert Tools  
def get_dog_coffee_order() -> str:
    """Get the CEO's dog's coffee order (puppuccino)."""
    return "Barkley's order: Puppuccino (whipped cream in a small cup) with a dash of cinnamon, served at room temperature. No caffeine!"

def prepare_dog_coffee_treat(occasion: str) -> str:
    """Prepare special dog coffee treat for occasions."""
    return f"For {occasion}: Barkley gets his puppuccino with a dog biscuit on the side and extra whipped cream swirl."

def check_dog_coffee_schedule() -> str:
    """Check when the dog should get his coffee treat."""
    return "Barkley gets his puppuccino: Every Tuesday and Friday at 3pm, and whenever the CEO has his afternoon coffee."

# Model setup
claude_model_id = os.getenv("CLAUDE_MODEL_ID", "us.anthropic.claude-sonnet-4-20250514-v1:0")
model_provider = os.getenv("MODEL_PROVIDER", "openai")
lite_llm_base_url = os.getenv("LITE_LLM_BASE_URL")
lite_llm_api_key = os.getenv("LITE_LLM_SECRET_KEY")

model = init_chat_model(
    claude_model_id,
    model_provider=model_provider,
    base_url=lite_llm_base_url,
    api_key=lite_llm_api_key,
    temperature=0.3
)

# Create coffee specialists
ceo_coffee_expert = create_react_agent(
    model=model,
    tools=[get_ceo_coffee_order, prepare_ceo_coffee_schedule, check_ceo_coffee_preferences],
    name="ceo_coffee_expert",
    prompt="You are the CEO's personal coffee specialist. You know everything about the CEO's coffee preferences, timing, and special requirements for different occasions."
)

dog_coffee_expert = create_react_agent(
    model=model,
    tools=[get_dog_coffee_order, prepare_dog_coffee_treat, check_dog_coffee_schedule],
    name="dog_coffee_expert", 
    prompt="You are Barkley the dog's coffee specialist. You handle all puppuccino orders and special treats for the CEO's beloved Golden Retriever."
)

# Create executive assistant supervisor
executive_assistant = create_supervisor(
    [ceo_coffee_expert, dog_coffee_expert],
    model=model,
    prompt="""You are the CEO's executive assistant. You manage two coffee runners:
    
    - ceo_coffee_expert: Handles all of the CEO's coffee needs, preferences, and scheduling
    - dog_coffee_expert: Manages Barkley the dog's puppuccino orders and special treats
    
    When requests come in about coffee, determine who needs what and coordinate with the appropriate specialist.
    You're known for your attention to detail and never mixing up the CEO's cortado with Barkley's puppuccino!"""
)

# Compile and set up tracing
app = executive_assistant.compile(name="ExecutiveAssistant")

tracer = LangfuseSupervisorTracer(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST", "http://localhost:3000")
)

traced_app = tracer.trace_workflow(app)

# Demo: Executive assistant handling a complex coffee situation
print("\n=== ☕️ Welcome to E.A. Coffee Demo 👩‍💼 ===\n")
print("CEO's EA gets request: 'I have the board meeting at 10am this Friday, and it might be stressful. Can you prepare coffee for both me and Barkley?'\n")

result = traced_app.invoke({
    "messages": [{
        "role": "user", 
        "content": "We have the board meeting at 10am tomorrow, and it might be stressful. Can you prepare coffee for both me and Barkley?"
    }]
})

print("Executive Assistant Response:")
print(f"{result['messages'][-1].content}")
print(f"\n☕ View detailed delegation trace at: http://localhost:3000")

# This demonstrates:
# 1. Executive assistant receives complex request involving BOTH CEO and dog
# 2. Recognizes need to consult CEO coffee expert for stressful board meeting prep
# 3. Consults dog coffee expert for Barkley's special treat
# 4. Coordinates timing and preparation for both
# 5. Delivers comprehensive coffee plan