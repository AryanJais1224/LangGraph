import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

# Tool for multiplying two integers
@tool
def multiply(a: int, b: int) -> int:
    """Multiplies two integers."""
    return a * b

# Tool for adding two integers
@tool
def add(a: int, b: int) -> int:
    """Adds two integers."""
    return a + b

# Function that runs an example math conversation session
def run_math_session():

    # Initialize OpenAI chat model
    llm = ChatOpenAI(
        model="gpt-4o"
    )

    # Register available tools
    tools = [multiply, add]

    # Initialize memory checkpointing
    memory = MemorySaver()

    # Create ReAct agent with tools and memory
    agent = create_react_agent(
        llm,
        tools,
        checkpointer=memory
    )

    # Configure thread ID for memory persistence
    config = {"configurable": {"thread_id": "math_user_1"}}

    # First user query
    q1 = "What is 50 + 50?"
    print(f"User: {q1}")

    res1 = agent.invoke(
        {"messages": [("user", q1)]},
        config
    )

    print(f"Agent: {res1['messages'][-1].content}\n")

    # Second query that depends on previous answer
    q2 = "Multiply that result by 2"
    print(f"User: {q2}")

    res2 = agent.invoke(
        {"messages": [("user", q2)]},
        config
    )

    print(f"Agent: {res2['messages'][-1].content}")

# Script entry point
if __name__ == "__main__":
    run_math_session()
