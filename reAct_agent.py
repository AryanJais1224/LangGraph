import os
from dotenv import load_dotenv

from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

# Function to run the ReAct agent
def run_agent():

    # Initialize Wikipedia API wrapper
    api_wrapper = WikipediaAPIWrapper(
        top_k_results=1,
        doc_content_chars_max=1000
    )

    # Create Wikipedia search tool
    wiki_tool = WikipediaQueryRun(api_wrapper=api_wrapper)

    # Register tools for the agent
    tools = [wiki_tool]

    # Initialize OpenAI chat model
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0
    )

    # Create ReAct agent with tools
    agent = create_react_agent(llm, tools)

    # Execute agent with user query
    response = agent.invoke({
        "messages": [
            ("user", "Who is Lilian Weng and what does she do?")
        ]
    })

    # Print final response from the agent
    print("\nFinal Answer:")
    print(response["messages"][-1].content)

# Entry point for script execution
if __name__ == "__main__":
    run_agent()
