import os
from typing import List
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.schema import Document
from langgraph.graph import StateGraph, START, END

# Graph state schema for passing data between nodes
class GraphState(BaseModel):
    question: str
    context: List[Document] = Field(default_factory=list)
    answer: str = ""

# URLs used as knowledge sources
urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/"
]

# Load documents from web pages
docs = [WebBaseLoader(url).load() for url in urls]

# Flatten nested document lists
docs_list = [doc for sublist in docs for doc in sublist]

# Split documents into smaller chunks
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250,
    chunk_overlap=0
)

# Generate document chunks
doc_splits = text_splitter.split_documents(docs_list)

# Create vector store using OpenAI embeddings
vectorstore = FAISS.from_documents(
    documents=doc_splits,
    embedding=OpenAIEmbeddings()
)

# Create retriever interface
retriever = vectorstore.as_retriever()

# Initialize language model
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0
)

# Node responsible for retrieving relevant documents
def retrieve(state: GraphState):
    documents = retriever.get_relevant_documents(state.question)
    return {
        "context": documents,
        "question": state.question
    }

# Node responsible for generating final answer
def generate(state: GraphState):
    context_text = "\n".join([doc.page_content for doc in state.context])
    response = llm.invoke(
        f"Context:\n{context_text}\n\nQuestion:\n{state.question}"
    )
    return {
        "answer": response.content
    }

# Initialize LangGraph workflow
workflow = StateGraph(GraphState)

# Add nodes to graph
workflow.add_node("retrieve", retrieve)
workflow.add_node("generate", generate)

# Define graph execution flow
workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

# Compile the graph into executable app
app = workflow.compile()

# Run pipeline with a sample query
if __name__ == "__main__":
    inputs = {"question": "What are the components of an LLM agent?"}

    for output in app.stream(inputs):
        print(output)
