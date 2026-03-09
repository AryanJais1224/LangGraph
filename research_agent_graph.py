import os
from typing import Literal
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain.chat_models import init_chat_model
from langchain.agents import Tool

from langgraph.prebuilt import create_react_agent
from langgraph.graph import MessagesState, END
from langgraph.types import Command

# Initialize chat model
llm = init_chat_model("openai:gpt-4o-mini")

# Utility function that builds a retriever tool from a local text file
def make_retriever_tool_from_text(file_path, name, desc):

    # Create dummy file if it doesn't exist
    if not os.path.exists(file_path):
        with open(file_path, "w") as f:
            f.write("Sample research data about Transformer variants.")

    # Load documents from file
    loader = TextLoader(file_path, encoding="utf-8")
    docs = loader.load()

    # Split documents into chunks
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    ).split_documents(docs)

    # Create FAISS vector store with embeddings
    vs = FAISS.from_documents(
        chunks,
        OpenAIEmbeddings()
    )

    # Create retriever interface
    retriever = vs.as_retriever()

    # Define tool function for retrieval
    def tool_func(query: str) -> str:
        results = retriever.invoke(query)
        return "\n\n".join(doc.page_content for doc in results)

    # Return LangChain tool
    return Tool(
        name=name,
        description=desc,
        func=tool_func
    )

# Initialize internal research retriever tool
internal_tool = make_retriever_tool_from_text(
    "internal_docs.txt",
    "InternalResearchNotes",
    "Search internal notes."
)

# Helper function to build system prompt
def make_system_prompt(suffix: str) -> str:
    return (
        "You are a helpful AI assistant. Use tools to progress. "
        "Prefix with 'FINAL ANSWER' when done.\n"
        f"{suffix}"
    )

# Create research agent focused on internal documents
research_agent = create_react_agent(
    llm,
    [internal_tool],
    state_modifier=make_system_prompt("You focus on internal documents.")
)

# Node that executes the researcher agent
def run_researcher(state: MessagesState) -> Command[Literal["manager", "__end__"]]:

    # Run agent with current conversation state
    result = research_agent.invoke(state)

    # Extract latest message
    last_msg = result["messages"][-1]

    # End graph if final answer is produced
    if "FINAL ANSWER" in last_msg.content:
        return Command(
            update={"messages": [last_msg]},
            goto=END
        )

    # Otherwise return control to manager node
    return Command(
        update={"messages": [last_msg]},
        goto="manager"
    )
