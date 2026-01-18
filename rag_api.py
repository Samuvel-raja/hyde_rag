from fastapi import APIRouter, Request, HTTPException

router = APIRouter()

@router.post("/rag")
async def rag_query(request: Request, query: str):
    hyde_rag = getattr(request.app.state, "hyde_rag", None)

    if not hyde_rag:
        raise HTTPException(
            status_code=400,
            detail="RAG not initialized. Upload documents first."
        )

    result = await hyde_rag(query)

    return {
        "answer": result,
        "model": "gpt-4o-mini+HyDE"
    }
