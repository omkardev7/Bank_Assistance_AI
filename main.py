# main.py
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional
import uvicorn
import rag_engine
import config

logger = config.get_logger(__name__)

app = FastAPI(
    title="Bank of Maharashtra Loan Assistant",
    description="AI-powered assistant for loan queries",
    version="2.0"
)

# Request Models
class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500)
    session_id: Optional[str] = Field(default="default", max_length=100)

class ClearHistoryRequest(BaseModel):
    session_id: str = Field(..., max_length=100)

# Response Models
class QueryResponse(BaseModel):
    answer: str
    context_used: List[str]
    sources: List[str]
    session_id: str

class StatusResponse(BaseModel):
    status: str
    message: str

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": "An unexpected error occurred. Please try again."
        }
    )

@app.on_event("startup")
async def startup_event():
    """Initialize RAG system on startup"""
    try:
        logger.info("Starting up server...")
        rag_engine.initialize_system()
        logger.info("Server ready to handle requests")
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise

@app.get("/", response_model=StatusResponse)
async def root():
    """Health check endpoint"""
    return StatusResponse(
        status="ok",
        message="Bank of Maharashtra Loan Assistant API is running"
    )

@app.get("/health", response_model=StatusResponse)
async def health_check():
    """Detailed health check"""
    try:
        # Verify system is initialized
        if rag_engine._retriever is None:
            raise HTTPException(
                status_code=503,
                detail="System not initialized"
            )
        
        return StatusResponse(
            status="healthy",
            message="All systems operational"
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=503,
            detail="Service unavailable"
        )

@app.post("/query", response_model=QueryResponse)
async def query_loan_assistant(request: QueryRequest):
    """
    Process a loan-related query
    
    - **question**: Your question about loans (3-500 characters)
    - **session_id**: Optional session identifier for conversation history
    """
    try:
        logger.info(f"Received query from session: {request.session_id}")
        
        result = rag_engine.process_query(
            question=request.question,
            session_id=request.session_id
        )
        
        return QueryResponse(
            answer=result["answer"],
            context_used=result["context_used"],
            sources=result["sources"],
            session_id=request.session_id
        )
        
    except ValueError as e:
        logger.warning(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Query processing error: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to process query. Please try again."
        )

@app.post("/clear-history", response_model=StatusResponse)
async def clear_conversation_history(request: ClearHistoryRequest):
    """
    Clear conversation history for a session
    
    - **session_id**: Session identifier to clear
    """
    try:
        success = rag_engine.clear_conversation(request.session_id)
        
        if success:
            return StatusResponse(
                status="success",
                message=f"Conversation history cleared for session: {request.session_id}"
            )
        else:
            return StatusResponse(
                status="info",
                message="No conversation history found for this session"
            )
    except Exception as e:
        logger.error(f"Error clearing history: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to clear conversation history"
        )

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )