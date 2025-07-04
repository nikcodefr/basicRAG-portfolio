from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from rag import RAGPipeline

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag_pipeline = RAGPipeline()

class Query(BaseModel):
    query: str

@app.post("/chat/")
def chatbot(query: Query):
    response = rag_pipeline.answer(query.query)
    return {"response": response}
