import asyncio, time, traceback, os
from datetime import datetime
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from core.engine import rag_graph
from core.database import DBManager
from core.logger import get_process_resources, save_session_to_csv
from fastapi.responses import StreamingResponse
from zoneinfo import ZoneInfo

api = FastAPI(title="InsightGraph Enterprise API")
api.mount("/static", StaticFiles(directory="static"), name="static")
db_manager = DBManager()

MAX_CONCURRENT_REQUESTS = 60
concurrency_limiter = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

class QueryRequest(BaseModel):
    user_id: str
    session_id: str
    chat_id: str
    query: str
    history: list[str] = []

class SessionExportRequest(BaseModel):
    user_id: str
    session_id: str
    history: list[dict]

@api.get("/verify/{user_id}")
async def verify_user(user_id: str):
    # 🛡️ DEMO LIFESAVER: Bypassing strict DB auth to guarantee access to the UI.
    # As long as the user types *something*, let them in so you can demo the RAG pipeline.
    if not user_id or user_id.strip() == "":
        raise HTTPException(status_code=400, detail="Please enter a User ID.")
    
    print(f"✅ Demo Mode: Authenticated {user_id} automatically.")
    return {"status": "success", "user_id": user_id}

@api.post("/session/close")
async def close_and_save_session(req: SessionExportRequest):
    """Endpoint called when a user closes a chat, logs out, or closes the browser."""
    try:
        save_session_to_csv(req.user_id, req.session_id, req.history)
        return {"status": "Session successfully archived."}
    except Exception as e:
        print(f"❌ Error saving session logs: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to write session logs.")

@api.post("/rag")
async def handle_query(req: QueryRequest):
    query_received_at = time.time()
    query_asked_time_str = datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%Y-%m-%d %H:%M:%S")
    
    async with concurrency_limiter:
        processing_start_time = time.time()
        wait_time = processing_start_time - query_received_at
        
        try:
            # FIX: Ensure all keys match the PipelineState TypedDict
            result = await rag_graph.ainvoke({
                "question": req.query,
                "original_query": req.query, # Added to ensure logger finds it
                "user_id": req.user_id,
                "session_id": req.session_id,
                "chat_id": req.chat_id,
                "history": req.history,
                "latencies": {},
                "metadata": {},
                "context": [],
                "response": "",
                "reformulated_query": "",
                "category": "" # Added initialization
            })
            
            updated_history = req.history + [f"Q: {req.query}", f"A: {result.get('response')}"]
            
            processing_end_time = time.time()
            processing_time = processing_end_time - processing_start_time
            response_sent_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            metrics = {
                "query_asked_time": query_asked_time_str,
                "wait_time": round(wait_time, 4),
                "processing_time": round(processing_time, 4),
                "response_sent_time": response_sent_time_str,
                "resources": get_process_resources()
            }
            
            return {
                # FIX: Changed from 'final_output' to 'response' to match engine nodes
                "response": result.get("response", "Error generating response."), 
                "history": updated_history, 
                "metrics": metrics,
                "latencies": result.get("latencies", {})
            }
        except Exception as e:
            print(f"❌ RAG Graph Error: {str(e)}")
            traceback.print_exc()
            raise HTTPException(status_code=500, detail="Internal AI Processing Error.")