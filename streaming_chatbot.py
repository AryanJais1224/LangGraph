import os
import asyncio
from typing import Annotated
from typing_extensions import TypedDict
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

from langchain_groq import ChatGroq

# Define graph state schema
class State(TypedDict):
    messages: Annotated[list, add_messages]

# Initialize Groq LLM model
llm_groq = ChatGroq(
    model="qwen-qwq-32b"
)

# Node that generates chatbot response
def chatbot_node(state: State):
    response = llm_groq.invoke(state["messages"])
    return {"messages": [response]}

# Create LangGraph workflow
workflow = StateGraph(State)

# Add chatbot node to graph
workflow.add_node("agent", chatbot_node)

# Define execution flow
workflow.add_edge(START, "agent")
workflow.add_edge("agent", END)

# Initialize memory checkpointing
memory = MemorySaver()

# Compile graph with memory support
graph = workflow.compile(checkpointer=memory)

# Async function to stream responses from the model
async def main():

    # Thread configuration for memory persistence
    config = {"configurable": {"thread_id": "1"}}

    # Example user query
    user_input = "Hi, I'm a developer. Can you explain streaming in LangGraph?"

    # Stream events from LangGraph execution
    async for event in graph.astream_events(
        {"messages": [user_input]},
        config,
        version="v2"
    ):
        if event["event"] == "on_chat_model_stream":
            content = event["data"]["chunk"].content
            if content:
                print(content, end="", flush=True)

# Run async event loop
if __name__ == "__main__":
    asyncio.run(main())
