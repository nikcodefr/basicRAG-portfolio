from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from rag import get_pipeline

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    query: str

@app.post("/chat/")
def chatbot(query: Query):
    try:
        rag = get_pipeline()
        response = rag.answer(query.query)
        return {"response": response}
    except Exception as e:
        return {"response": f"Something went wrong: {str(e)}"}

