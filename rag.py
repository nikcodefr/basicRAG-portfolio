import bs4
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
from components.vector_store import vector_store
from components.chat_model import llm
import os
from dotenv import load_dotenv

load_dotenv()

WEB_SOURCE=os.getenv("WEB_SOURCE")

loader = WebBaseLoader(
    web_paths=(WEB_SOURCE,),
    bs_kwargs=dict(
        parse_only = bs4.SoupStrainer(
            class_ = ("source text")
        )
    ),
)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
all_splits = text_splitter.split_documents(docs)

_ = vector_store.add_documents(documents=all_splits)

prompt = hub.pull("rlm/rag-prompt")

class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}

def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}

graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

while True:
    q = input("Ask a question about Nikhil: (e to exit)\n")
    if q.lower() == "e":
        print("Catch you later. Have a great day!")
        break
    else:
        response = graph.invoke({"question": q})
        print(f"Response: {response["answer"]}\n")