from fastapi import APIRouter, UploadFile, File, Request, HTTPException
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tempfile
import os

from qdrant_vector_store import create_qdrant_vector_store
from hyderag import ProductionHyDE

router = APIRouter()

@router.post("/upload")
async def upload_file(
    request: Request,
    file: UploadFile = File(...)
):
    # Lazy initialization
    vectorstore = getattr(request.app.state, "vectorstore", None)

    if not vectorstore:
        vectorstore = await create_qdrant_vector_store()
        request.app.state.vectorstore = vectorstore
        request.app.state.hyde_rag = ProductionHyDE(
            vectorstore.as_retriever(search_kwargs={"k": 5})
        )

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    loader = PyPDFLoader(tmp_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(docs)

    vectorstore.add_documents(chunks)
    os.remove(tmp_path)

    return {
        "status": "uploaded",
        "filename": file.filename,
        "chunks": len(chunks)
    }
