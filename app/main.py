import asyncio, time, traceback, os
from urllib.request import Request, urlopen
from datetime import datetime
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from core.engine import rag_graph
from core.logger import get_process_resources, save_session_to_csv
from fastapi.responses import StreamingResponse
from zoneinfo import ZoneInfo
from rich.console import Console
import logging
import torch
from core.database import DBManager, embedder, util

api = FastAPI(title="Above RAG System API")
api.mount("/static", StaticFiles(directory="static"), name="static")
db_manager = DBManager()

MAX_CONCURRENT_REQUESTS = 60
concurrency_limiter = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)


semantic_cache = {}
CACHE_THRESHOLD = 0.95
MAX_CACHE_PER_USER = 50


# VLLM Buffer Control
vllm_ready_event = asyncio.Event()
startup_time = None

console = Console(force_terminal=True, color_system="truecolor", width=120)

logging.getLogger("uvicorn").setLevel(logging.ERROR)
logging.getLogger("uvicorn.access").setLevel(logging.ERROR)
logging.getLogger("fastapi").setLevel(logging.ERROR)

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
    chat_id: str
    history: list[dict]

@api.on_event("startup")
async def startup_event():
    global startup_time
    startup_time = time.time()
    ist_time = datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%Y-%m-%d %H:%M:%S")
    print(f"🚀 [{ist_time}] API Server started. Waiting for vLLM to boot...")
    asyncio.create_task(check_vllm_health())

async def check_vllm_health():
    vllm_url = os.environ.get("VLLM_URL", "http://vllm_retrieval:8005/v1") + "/models"
    while True:
        try:
            def fetch():
                req = Request(vllm_url, method="GET")
                with urlopen(req, timeout=2.0) as response:
                    return response.status
            status = await asyncio.to_thread(fetch)
            if status == 200:
                print(f"✅ [SYSTEM] vLLM is UP and Ready! Boot time: {time.time() - startup_time:.2f}s")
                vllm_ready_event.set()
                break
        except Exception:
            pass
        print("⏳ [SYSTEM] vLLM not ready yet. Requests will be buffered. Retrying in 5 seconds...")
        await asyncio.sleep(5)

@api.get("/verify/{user_id}")
async def verify_user(user_id: str):
    if not user_id or user_id.strip() == "":
        raise HTTPException(status_code=400, detail="Please enter a User ID.")
    print(f"✅ Demo Mode: Authenticated {user_id} automatically.")
    return {"status": "success", "user_id": user_id}

@api.post("/session/close")
async def close_and_save_session(req: SessionExportRequest):
    try:
        save_session_to_csv(req.user_id, req.session_id, req.chat_id, req.history)
        return {"status": "Session successfully archived."}
    except Exception as e:
        print(f"❌ Error saving session logs: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to write session logs.")


@api.post("/rag")
async def handle_query(req: QueryRequest):
    query_received_at = time.time()
    query_asked_time_str = datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%Y-%m-%d %H:%M:%S")

    # Server-side parallel logging: Entering the buffer
    console.print(
        f"\n[dim][{datetime.now().strftime('%H:%M:%S.%f')[:-3]}][/] "
        f"[cyan][bold]📥 BUFFER IN[/bold][/] "
        f"[magenta][{req.query_id}][/] "
        f"[white]{req.query}[/]"
    )

    # ---------------------------------------------------------
    # 1. SEMANTIC CACHE CHECK
    # ---------------------------------------------------------
    # Generate embedding for the incoming query without blocking the async loop
    query_vec = await asyncio.to_thread(embedder.encode, req.query)
    
    user_cache = semantic_cache.get(req.user_id, [])
    cache_hit_response = None

    if user_cache:
        # Extract all cached vectors for this user
        cached_vectors = [item['vector'] for item in user_cache]
        
        # Calculate cosine similarity against all cached vectors at once
        similarities = util.cos_sim(query_vec, cached_vectors)[0]
        
        # Find the most similar cached query
        max_sim_idx = torch.argmax(similarities).item()
        max_sim_value = similarities[max_sim_idx].item()
        
        if max_sim_value >= CACHE_THRESHOLD:
            cache_hit_response = user_cache[max_sim_idx]['response']
            console.print(f"[green][bold]⚡ CACHE HIT[/bold][/] [magenta][{req.query_id}][/] Similarity: {max_sim_value:.3f}")

    # If cache hit, return immediately!
    if cache_hit_response:
        processing_time = time.time() - query_received_at
        updated_history = req.history + [f"Q: {req.query}", f"A: {cache_hit_response}"]
        
        metrics = {
            "query_asked_time": query_asked_time_str,
            "wait_time": 0.0,
            "processing_time": round(processing_time, 4),
            "response_sent_time": datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%Y-%m-%d %H:%M:%S"),
            "resources": "In-Memory Cache (CPU)"
        }
        
        console.print(
            f"[green][bold]📤 CACHE RESPONSE SENT[/bold][/] "
            f"[magenta][{req.query_id}][/] "
            f"proc=[blue]{processing_time:.2f}s[/]"
        )
        
        return {
            "response": cache_hit_response,
            "history": updated_history,
            "metrics": metrics,
            "latencies": {"cache_retrieval": processing_time}
        }
    # ---------------------------------------------------------

    # Buffer requests if VLLM isn't up yet
    if not vllm_ready_event.is_set():
        console.print(
            f"[yellow][bold]⚠ BUFFER WAIT[/bold][/] "
            f"[magenta][vLLM is still booting. Query {req.query_id} is waiting in queue.][/] "
            f"[white]{req.query}[/]"
            )
        await vllm_ready_event.wait()
        console.print(
            f"[green][bold]🟢 BUFFER RELEASE[/bold][/] "
            f"[magenta][ vLLM is ready. Processing Query {req.query_id}][/] "
            f"[white]{req.query}[/]"
        )

    async with concurrency_limiter:
        processing_start_time = time.time()
        wait_time = processing_start_time - query_received_at
        console.print(
            f"[blue][bold]⚙ PROCESSING START[/bold][/] "
            f"[magenta][{req.query_id}][/] "
            f"[yellow] Buffer Wait Time: {wait_time:.2f}s[/] "
            f"Q:[white]{req.query[:50]}[/]"
        )
        
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
            
            # --- THIS IS THE LINE THAT WAS MISSING ---
            final_response = result.get("response", "Error generating response.")
            
            updated_history = req.history + [f"Q: {req.query}", f"A: {final_response}"]
            
            # ---------------------------------------------------------
            # 2. CACHE UPDATE (Missed cache, so save the new answer)
            # ---------------------------------------------------------
            if req.user_id not in semantic_cache:
                semantic_cache[req.user_id] = []
                
            semantic_cache[req.user_id].append({
                "vector": query_vec, 
                "question": req.query,
                "response": final_response
            })
            
            # Keep cache from growing infinitely
            if len(semantic_cache[req.user_id]) > MAX_CACHE_PER_USER:
                semantic_cache[req.user_id].pop(0) # Remove oldest query
            # ---------------------------------------------------------
            
            processing_end_time = time.time()
            processing_time = processing_end_time - processing_start_time
            response_sent_time_str = datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%Y-%m-%d %H:%M:%S")
            
            resources = get_process_resources()
            metrics = {
                "query_asked_time": query_asked_time_str,
                "wait_time": round(wait_time, 4),
                "processing_time": round(processing_time, 4),
                "response_sent_time": response_sent_time_str,
                "resources": resources
            }
            
            # Server-side parallel logging: Sending Response
            console.print(
                f"[green][bold]📤 RESPONSE SENT[/bold][/] "
                f"[magenta][{req.query_id}][/] "
                f"proc=[blue]{processing_time:.2f}s[/] "
                f"gpu=[yellow]{resources}[/]"
            )
            
            return {
                "response": final_response, 
                "history": updated_history, 
                "metrics": metrics,
                "latencies": result.get("latencies", {})
            }
        except Exception as e:
            console.print(
                f"[red][bold]❌ RAG ERROR[/bold][/] "
                f"[magenta][{req.query_id}][/] "
                f"[red]{str(e)}[/]"
            )
            traceback.print_exc()
            raise HTTPException(status_code=500, detail="Internal AI Processing Error.")