import os
from langchain_community.tools.tavily_search import TavilySearchResults

# Import Wikipedia tool
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun

# Import Arxiv research paper tool
from langchain_community.tools.arxiv.tool import ArxivQueryRun

# Import OpenAI chat model
from langchain_openai import ChatOpenAI

# Import LangGraph utilities
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

# Initialize the language model
llm = ChatOpenAI(model="gpt-4o-mini")


# Add two numbers
def add(a: int, b: int) -> int:
    return a + b


# Multiply two numbers
def multiply(a: int, b: int) -> int:
    return a * b


# Initialize Wikipedia search
wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

# Initialize Arxiv paper search
arxiv = ArxivQueryRun()

# Initialize Tavily web search
tavily = TavilySearchResults(max_results=3)

# Combine all tools
tools = [add, multiply, wiki, arxiv, tavily]

# Initialize memory checkpoint
memory = MemorySaver()

# Create the LangGraph agent
agent = create_react_agent(llm, tools, checkpointer=memory)


# Run interactive chatbot
def run_chat():

    print("Chatbot ready. Type 'exit' to quit.")

    while True:

        # Take user input
        user_input = input("\nUser: ")

        # Exit condition
        if user_input.lower() == "exit":
            break

        # Invoke agent with user message
        response = agent.invoke({"messages": [("user", user_input)]})

        # Print final response
        print("\nAssistant:", response["messages"][-1].content)


# Run program
if __name__ == "__main__":
    run_chat()
