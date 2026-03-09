import os
from typing import Annotated
from dotenv import load_dotenv
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from langchain_groq import ChatGroq

# Define graph state structure
class ChatState(TypedDict):
    messages: Annotated[list, add_messages]

# Node that calls the Groq LLM
def call_model(state: ChatState):

    # Initialize Groq chat model
    llm = ChatGroq(
        model="qwen-qwq-32b"
    )

    # Invoke model with conversation messages
    response = llm.invoke(state["messages"])

    return {"messages": [response]}

# Initialize LangGraph builder
builder = StateGraph(ChatState)

# Add chatbot node
builder.add_node("chatbot", call_model)

# Define execution flow
builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", END)

# Compile graph into executable application
graph = builder.compile()

# Run example query
if __name__ == "__main__":
    result = graph.invoke({
        "messages": [
            ("user", "What is Machine Learning?")
        ]
    })

    print(result["messages"][-1].content)
