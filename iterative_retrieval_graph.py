import os
from typing import List, Literal
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

# Define state used by the graph
class IterativeState(BaseModel):
    query: str
    context: List[str] = []
    iterations: int = 0
    is_sufficient: bool = False

# Routing function that decides whether to retrieve more or generate answer
def check_sufficiency(state: IterativeState) -> Literal["retrieve", "generate"]:
    if state.iterations > 2 or state.is_sufficient:
        return "generate"
    return "retrieve"

# Node that simulates retrieval from an external source
def retrieve_node(state: IterativeState):
    return {
        "context": state.context + ["New retrieved information"],
        "iterations": state.iterations + 1
    }

# Node that generates the final answer using an LLM
def generate_node(state: IterativeState):

    # Initialize chat model
    llm = ChatOpenAI(model="gpt-4o-mini")

    # Combine retrieved context
    context = "\n".join(state.context)

    # Create generation prompt
    prompt = f"Answer the query: {state.query}\nUsing context:\n{context}"

    # Generate response
    response = llm.invoke(prompt)

    return {"context": state.context, "is_sufficient": True}

# Initialize graph workflow
workflow = StateGraph(IterativeState)

# Add nodes to graph
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("generate", generate_node)

# Define graph entry point
workflow.add_edge(START, "retrieve")

# Add conditional routing logic
workflow.add_conditional_edges(
    "retrieve",
    check_sufficiency,
    {
        "retrieve": "retrieve",
        "generate": "generate"
    }
)

# End graph after generation
workflow.add_edge("generate", END)

# Compile workflow into executable graph
app = workflow.compile()
