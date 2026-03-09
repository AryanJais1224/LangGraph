import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.tools.retriever import create_retriever_tool

from langgraph.prebuilt import create_react_agent

# Simulated internal research notes
raw_notes = [
    "Internal Note: We are testing LLaMA2 for reduced latency in our RAG pipeline.",
    "Internal Note: LoRA tuning achieved a 60% reduction in GPU memory usage.",
    "Internal Note: FAISS is preferred over Weaviate for our current scale."
]

# Convert text notes into LangChain documents
docs = [Document(page_content=text) for text in raw_notes]

# Create FAISS vector store using OpenAI embeddings
vectorstore = FAISS.from_documents(
    docs,
    OpenAIEmbeddings()
)

# Create retriever interface for similarity search
retriever = vectorstore.as_retriever()

# Convert retriever into a tool usable by the agent
retriever_tool = create_retriever_tool(
    retriever,
    "search_research_notes",
    "Searches internal research notes about LLaMA, LoRA, and FAISS."
)

# Initialize OpenAI chat model
llm = ChatOpenAI(
    model="gpt-4o"
)

# Create ReAct agent with retriever tool
agent = create_react_agent(
    llm,
    [retriever_tool]
)

# Execute agent with example query
if __name__ == "__main__":
    query = "What do our internal research notes say about GPU memory?"
    
    result = agent.invoke({
        "messages": [
            ("human", query)
        ]
    })

    print(result["messages"][-1].content)
