import os
from dotenv import load_dotenv

load_dotenv()

LANGSMITH_TRACING=os.getenv("LANGSMITH_TRACING")
LANGSMITH_API_KEY=os.getenv("LANGSMITH_API_KEY")