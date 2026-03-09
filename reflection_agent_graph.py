import os
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

# Define graph state structure
class ReflectionState(BaseModel):
    question: str
    answer: str = ""
    reflection: str = ""
    quality_score: int = 0

# Node that generates an initial answer
def generate_answer(state: ReflectionState):

    # Initialize chat model
    llm = ChatOpenAI(model="gpt-4o")

    # Generate answer for the question
    response = llm.invoke(state.question)

    return {"answer": response.content}

# Node that critiques the generated answer
def reflect(state: ReflectionState):

    # Initialize chat model
    llm = ChatOpenAI(model="gpt-4o")

    # Ask model to critique the answer
    feedback = llm.invoke(f"Critique this answer:\n{state.answer}")

    # Return critique and mock quality score
    return {
        "reflection": feedback.content,
        "quality_score": 5
    }

# Initialize LangGraph workflow
workflow = StateGraph(ReflectionState)

# Add nodes to graph
workflow.add_node("generate", generate_answer)
workflow.add_node("reflect", reflect)

# Define execution flow
workflow.add_edge(START, "generate")
workflow.add_edge("generate", "reflect")
workflow.add_edge("reflect", END)

# Compile graph into executable application
app = workflow.compile()
