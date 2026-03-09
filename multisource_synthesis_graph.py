import os
from typing import List
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

# Define state structure used by the graph
class SynthesisState(BaseModel):
    question: str
    retrieved_content: List[str] = []
    final_answer: str = ""

# Node that simulates retrieving content from multiple sources
def retrieve_from_multiple_sources(state: SynthesisState):

    # Simulated retrieval from different sources
    sources = [
        "Source A: Transformer architectures improve contextual reasoning.",
        "Source B: Agentic loops allow iterative reasoning and tool usage."
    ]

    return {"retrieved_content": sources}

# Node that synthesizes the final answer using an LLM
def synthesize_answer(state: SynthesisState):

    # Initialize chat model
    llm = ChatOpenAI(
        model="gpt-4o-mini"
    )

    # Combine retrieved context
    context = "\n".join(state.retrieved_content)

    # Create synthesis prompt
    prompt = (
        f"Synthesize a coherent answer for the question:\n{state.question}\n\n"
        f"Using the following context:\n{context}"
    )

    # Generate final answer
    response = llm.invoke(prompt)

    return {"final_answer": response.content}

# Initialize LangGraph workflow
workflow = StateGraph(SynthesisState)

# Add nodes to graph
workflow.add_node("retrieve", retrieve_from_multiple_sources)
workflow.add_node("synthesize", synthesize_answer)

# Define graph execution flow
workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "synthesize")
workflow.add_edge("synthesize", END)

# Compile the workflow into an executable graph
app = workflow.compile()
