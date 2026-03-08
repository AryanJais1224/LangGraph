# Import TypedDict for state structure
from typing_extensions import TypedDict

# Define graph state
class State(TypedDict):
    graph_info: str


# Start node function
def start_play(state: State):
    print("Start_Play node has been called")
    return {"graph_info": state["graph_info"] + " I am planning to play"}


# Cricket node function
def cricket(state: State):
    print("Cricket node has been called")
    return {"graph_info": state["graph_info"] + " Cricket"}


# Badminton node function
def badminton(state: State):
    print("Badminton node has been called")
    return {"graph_info": state["graph_info"] + " Badminton"}


# Import random for routing
import random
from typing import Literal


# Router function to randomly choose next node
def random_play(state: State) -> Literal["cricket", "badminton"]:
    
    if random.random() > 0.5:
        return "cricket"
    else:
        return "badminton"


# Import LangGraph components
from langgraph.graph import StateGraph, START, END


# Initialize graph builder
graph = StateGraph(State)

# Add nodes to graph
graph.add_node("start_play", start_play)
graph.add_node("cricket", cricket)
graph.add_node("badminton", badminton)

# Define graph edges
graph.add_edge(START, "start_play")

# Add conditional routing
graph.add_conditional_edges("start_play", random_play)

# Define end edges
graph.add_edge("cricket", END)
graph.add_edge("badminton", END)

# Compile the graph
graph_builder = graph.compile()


# Run graph
result = graph_builder.invoke({"graph_info": "Hey!"})

# Print final state
print(result)
