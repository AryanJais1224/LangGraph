import os

# Import OpenAI chat model
from langchain_openai import ChatOpenAI

# Import tool decorator
from langchain_core.tools import tool

# Import LangGraph components
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage
from langgraph.graph.message import add_messages
from typing import TypedDict, Annotated

# Initialize chat model
llm = ChatOpenAI(model="gpt-4o-mini")


# Define graph state
class State(TypedDict):
    messages: Annotated[list, add_messages]


# Define addition tool
@tool
def add(a: int, b: int) -> int:
    return a + b


# Bind tools with the model
llm_with_tools = llm.bind_tools([add])

# Create tool execution node
tool_node = ToolNode([add])


# LLM router node
def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


# Router condition to decide tool usage
def route_tools(state: State):
    last_message = state["messages"][-1]
    return "tools" if last_message.tool_calls else END


# Initialize graph builder
builder = StateGraph(State)

# Add chatbot node
builder.add_node("chatbot", chatbot)

# Add tool node
builder.add_node("tools", tool_node)

# Define graph flow
builder.add_edge(START, "chatbot")
builder.add_conditional_edges("chatbot", route_tools)
builder.add_edge("tools", "chatbot")

# Compile the graph
graph = builder.compile()


# Run interactive chatbot
while True:

    # Take user input
    user_input = input("\nUser: ")

    # Exit condition
    if user_input.lower() == "exit":
        break

    # Invoke graph
    result = graph.invoke({"messages": [HumanMessage(content=user_input)]})

    # Print final response
    print("\nAssistant:", result["messages"][-1].content)
