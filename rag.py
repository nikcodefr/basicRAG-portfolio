import bs4
import os
from dotenv import load_dotenv
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict

from components.vector_store import vector_store
from components.chat_model import llm

load_dotenv()

class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

class RAGPipeline:
    def __init__(self):
        WEB_SOURCE = os.getenv("WEB_SOURCE")
        loader = WebBaseLoader(
            web_paths=(WEB_SOURCE,),
            bs_kwargs=dict(parse_only=bs4.SoupStrainer(class_="source text")),
        )
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = splitter.split_documents(docs)

        vector_store.add_documents(documents=splits)

        self.prompt = hub.pull("rlm/rag-prompt")
        self.graph = self._build_graph()

    def _build_graph(self):
        def retrieve(state: State):
            retrieved_docs = vector_store.similarity_search(state["question"])
            return {"context": retrieved_docs}

        def generate(state: State):
            docs_content = "\n\n".join(doc.page_content for doc in state["context"])
            messages = self.prompt.invoke({
                "question": state["question"],
                "context": docs_content,
            })
            response = llm.invoke(messages)
            return {"answer": response.content}

        graph_builder = StateGraph(State).add_sequence([retrieve, generate])
        graph_builder.add_edge(START, "retrieve")
        return graph_builder.compile()

    def answer(self, question: str) -> str:
        result = self.graph.invoke({"question": question})
        return result["answer"]

pipeline_instance = None

def get_pipeline():
    global pipeline_instance
    if pipeline_instance is None:
        pipeline_instance = RAGPipeline()
    return pipeline_instance

