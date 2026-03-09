import os
from typing import List
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

# Define graph state structure
class CoTRAGState(BaseModel):
    question: str
    reasoning_steps: List[str] = []
    final_answer: str = ""

# Node that plans reasoning steps for the question
def plan_steps(state: CoTRAGState):

    # Simulated Chain-of-Thought planning
    steps = [
        "Step 1: Define Transformers",
        "Step 2: Identify important variants",
        "Step 3: Summarize differences"
    ]

    return {"reasoning_steps": steps}

# Node that executes reasoning steps to produce the final answer
def execute_reasoning(state: CoTRAGState):

    # Initialize chat model
    llm = ChatOpenAI(model="gpt-4o-mini")

    # Combine reasoning steps into context
    steps_text = "\n".join(state.reasoning_steps)

    # Build prompt for final reasoning
    prompt = (
        f"Question: {state.question}\n\n"
        f"Follow these reasoning steps:\n{steps_text}\n\n"
        f"Provide the final answer."
    )

    # Generate final answer
    response = llm.invoke(prompt)

    return {"final_answer": response.content}

# Initialize LangGraph workflow
workflow = StateGraph(CoTRAGState)

# Add nodes to graph
workflow.add_node("plan", plan_steps)
workflow.add_node("execute", execute_reasoning)

# Define graph execution flow
workflow.add_edge(START, "plan")
workflow.add_edge("plan", "execute")
workflow.add_edge("execute", END)

# Compile graph into executable application
app = workflow.compile()
