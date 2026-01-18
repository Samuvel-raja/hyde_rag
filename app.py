from fastapi import FastAPI
from upload_api import router as upload_router
from rag_api import router as rag_router

app = FastAPI(title="Production HyDE RAG")

app.include_router(upload_router)
app.include_router(rag_router)
