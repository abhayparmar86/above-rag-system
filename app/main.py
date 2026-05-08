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

from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.panel import Panel
import logging
# Silence standard library logs that might print to stdout
logging.getLogger("uvicorn").setLevel(logging.ERROR)
logging.getLogger("uvicorn.access").setLevel(logging.ERROR)
logging.getLogger("fastapi").setLevel(logging.ERROR)

# console = Console()
console = Console(force_terminal=True, color_system="truecolor", width=120)

api = FastAPI(title="InsightGraph Enterprise API")
api.mount("/static", StaticFiles(directory="static"), name="static")
db_manager = DBManager()

@api.on_event("startup")
async def startup_event():
    console.print(Panel("[bold green]🚀 InsightGraph API Engine Online[/]\n"
                        "Port: 9001 | Status: Ready for Inference", 
                        expand=False, style="green"))

MAX_CONCURRENT_REQUESTS = 20
concurrency_limiter = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

class QueryRequest(BaseModel):
    user_id: str
    session_id: str
    chat_id: str
    query: str
    query_id: str
    history: list[str] = []

class SessionExportRequest(BaseModel):
    user_id: str
    session_id: str
    history: list[dict]

@api.get("/verify/{user_id}")
async def verify_user(user_id: str):
    if not user_id or user_id.strip() == "":
        raise HTTPException(status_code=400, detail="Please enter a User ID.")
    
    print(f"✅ Demo Mode: Authenticated {user_id} automatically.")
    return {"status": "success", "user_id": user_id}

@api.post("/session/close")
async def close_and_save_session(req: SessionExportRequest):
    try:
        save_session_to_csv(req.user_id, req.session_id, req.history)
        return {"status": "Session successfully archived."}
    except Exception as e:
        print(f"❌ Error saving session logs: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to write session logs.")

@api.post("/rag")
async def handle_query(req: QueryRequest):
    # Log incoming request with colors and style matching the client
    console.print(f"[dim][{datetime.now().strftime('%H:%M:%S.%f')[:-3]}][/] [cyan][bold]→ RECIEVED[/bold][/]  [magenta][{req.query_id}][/] [dim]({req.user_id})[/] [white]{req.query[:60]}{'...' if len(req.query)>60 else ''}[/]")
    
    query_received_at = time.time()
    query_asked_time_str = datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%Y-%m-%d %H:%M:%S")
    
    async with concurrency_limiter:
        processing_start_time = time.time()
        wait_time = processing_start_time - query_received_at
        
        try:
            result = await rag_graph.ainvoke({
                "question": req.query,
                "original_query": req.query,
                "user_id": req.user_id,
                "session_id": req.session_id,
                "chat_id": req.chat_id,
                "history": req.history,
                "latencies": {},
                "metadata": {},
                "context": [],
                "response": "",
                "reformulated_query": "",
                "category": ""
            })
            
            updated_history = req.history + [f"Q: {req.query}", f"A: {result.get('response')}"]
            
            processing_end_time = time.time()
            processing_time = processing_end_time - processing_start_time
            gpu_usage = get_process_resources()
            
            metrics = {
                "query_asked_time": query_asked_time_str,
                "wait_time": round(wait_time, 4),
                "processing_time": round(processing_time, 4),
                "response_sent_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "resources": gpu_usage
            }
            
            # Colorful Success Log: Includes Query snippet as requested
            console.print(
                f"[dim][{datetime.now().strftime('%H:%M:%S.%f')[:-3]}][/] [green][bold]✓ DONE[/bold][/]  [magenta][{req.query_id}][/] "
                f"Q: [white]{req.query[:40]}{'...' if len(req.query)>40 else ''}[/] | "
                f"wait=[yellow]{wait_time:.2f}s[/]  proc=[blue]{processing_time:.2f}s[/]  gpu=[yellow]{gpu_usage}[/]"
            )
            
            return {
                "response": result.get("response", "Error generating response."), 
                "history": updated_history, 
                "metrics": metrics,
                "latencies": result.get("latencies", {})
            }
        except Exception as e:
            console.print(f"[dim][{datetime.now().strftime('%H:%M:%S.%f')[:-3]}][/] [red][bold]✗ ERR[/bold][/]   [magenta][{req.query_id}][/] Q: [white]{req.query[:45]}[/] | [red]{str(e)[:80]}[/]")
            traceback.print_exc()
            raise HTTPException(status_code=500, detail="Internal AI Processing Error.")