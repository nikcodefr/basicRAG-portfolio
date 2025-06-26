from langchain_core.vectorstores import InMemoryVectorStore
from components.embeddings_model import embeddings

vector_store = InMemoryVectorStore(embeddings)