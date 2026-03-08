import os

# Import message types
from langchain_core.messages import AIMessage, HumanMessage, AnyMessage

# Import Groq chat model
from langchain_groq import ChatGroq

# Import typing utilities
from typing_extensions import TypedDict
from typing import Annotated

# Import LangGraph utilities
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# Import tool node utilities
from langgraph.prebuilt import ToolNode, tools_condition

# Initialize LLM model
llm = ChatGroq(model="qwen-qwq-32b")


# Define addition tool
def add(a: int, b: int) -> int:
    return a + b


# Bind tool with LLM
llm_with_tools = llm.bind_tools([add])


# Define graph state
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


# Define LLM node
def llm_tool(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


# List of available tools
tools = [add]


# Initialize graph builder
builder = StateGraph(State)

# Add LLM node
builder.add_node("llm_tool", llm_tool)

# Add tool execution node
builder.add_node("tools", ToolNode(tools))

# Define start edge
builder.add_edge(START, "llm_tool")

# Define conditional routing for tool calls
builder.add_conditional_edges(
    "llm_tool",
    tools_condition
)

# Return to LLM after tool execution
builder.add_edge("tools", "llm_tool")

# Compile the graph
graph = builder.compile()


# Run chatbot loop
while True:

    # Take user input
    user_input = input("\nUser: ")

    # Exit condition
    if user_input.lower() == "exit":
        break

    # Invoke graph
    result = graph.invoke({"messages": [HumanMessage(content=user_input)]})

    # Print assistant response
    print("\nAssistant:", result["messages"][-1].content)
