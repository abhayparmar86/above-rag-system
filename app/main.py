import asyncio, time, traceback, os
from urllib.request import Request, urlopen
from datetime import datetime
from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from core.logger import get_process_resources, save_session_to_csv, setup_logging, get_logger
from core.reminders import get_daily_reminders_and_mark 

# Logging must be set up before importing modules that log at import time
# (stt.py loads Whisper, translation.py loads NLLB — both print on import).
setup_logging()
logger = get_logger(__name__)

from core.engine import rag_graph
from fastapi.responses import StreamingResponse
from zoneinfo import ZoneInfo
from rich.console import Console
import logging
import torch
from core.database import DBManager, embedder, util
from core.translation import detect_language, translate_to_native, is_language_supported, get_bcp47, list_supported_languages
from core.stt import transcribe  # NEW: Whisper STT
import concurrent.futures
import httpx

try:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass 

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
    english_history: list[str] = []
    # "auto" = use whatever detect_language() finds. Any other value must be one of
    # the SUPPORTED_LANGUAGES names (see core/translation.py for the current list).
    # Per-chat, not global — the frontend stores these on the session object, not a page-wide default.
    input_language: str = "auto"
    output_language: str = "auto"


class SessionExportRequest(BaseModel):
    user_id: str
    session_id: str
    chat_id: str
    history: list[dict]

class SessionEventRequest(BaseModel):
    user_id: str
    session_id: str
    chat_id: str
    event_type: str  # "switch" | "create" | "delete"
    chat_name: str = ""

@api.on_event("startup")
async def startup_event():
    global startup_time
    startup_time = time.time()
    loop = asyncio.get_event_loop()
    loop.set_default_executor(
        concurrent.futures.ThreadPoolExecutor(max_workers=16)
    )
    ist_time = datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%Y-%m-%d %H:%M:%S")
    print(f"🚀 [{ist_time}] API Server started. Waiting for vLLM to boot...")
    logger.info("API server startup initiated | resources=%s", get_process_resources())
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
                boot_time = time.time() - startup_time
                print(f"✅ [SYSTEM] vLLM is UP and Ready! Boot time: {boot_time:.2f}s")
                logger.info("vLLM ready | boot_time_s=%.2f | resources=%s", boot_time, get_process_resources())
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
    logger.info("User verified | user_id=%s", user_id)
    
    # Clean workspace reminder check on login
    mcp_url = os.environ.get("CALENDAR_MCP_URL", "http://calendar_mcp:8090/sse")
    today_reminders = await get_daily_reminders_and_mark(user_id, db_manager, mcp_url)
    
    return {
        "status": "success", 
        "user_id": user_id,
        "reminders": today_reminders
    }

@api.get("/languages")
async def get_supported_languages():
    """
    Returns the full list of supported languages with their locale info, so the
    frontend's input/output dropdowns and TTS voice selection read from one
    source of truth instead of keeping a second hardcoded copy in JS.
    """
    return {"languages": list_supported_languages()}

@api.post("/session/close")
async def close_and_save_session(req: SessionExportRequest):
    try:
        save_session_to_csv(req.user_id, req.session_id, req.chat_id, req.history)
        logger.info(
            "Session archived | user_id=%s session_id=%s chat_id=%s turns=%d",
            req.user_id, req.session_id, req.chat_id, len(req.history)
        )
        return {"status": "Session successfully archived."}
    except Exception as e:
        print(f"❌ Error saving session logs: {str(e)}")
        logger.exception("Failed to archive session | user_id=%s chat_id=%s", req.user_id, req.chat_id)
        raise HTTPException(status_code=500, detail="Failed to write session logs.")


# =============================================================================
# NEW: /log/session-event — records chat/session switching from the UI.
# The frontend fires this on switch/create/delete so these events show up
# in the server-side log, not just the browser. Fire-and-forget from the UI
# side — failures here should never block the user's workflow.
# =============================================================================
@api.post("/log/session-event")
async def log_session_event(req: SessionEventRequest):
    logger.info(
        "Session event | type=%s user_id=%s session_id=%s chat_id=%s chat_name=%s",
        req.event_type, req.user_id, req.session_id, req.chat_id, req.chat_name
    )
    return {"status": "logged"}
# =============================================================================

# =============================================================================
# NEW: /stt endpoint — Speech to Text
# Accepts an audio blob from the browser (WebM from MediaRecorder),
# runs it through faster-whisper, returns the transcript as plain text.
# The frontend then populates the #query input and lets the user send normally.
# Zero impact on existing /rag flow.
# =============================================================================
@api.post("/stt")
async def speech_to_text(audio: UploadFile = File(...), language: str = Form("auto")):
    """
    Receives an audio file upload from the browser's MediaRecorder API.
    `language`: the chat's current input-language dropdown value ("auto" or a
    SUPPORTED_LANGUAGES name). Passed through as the Whisper language hint —
    a correct hint measurably reduces mis-transcription vs. auto-detect alone.
    Returns: { "transcript": "..." }
    """
    try:
        audio_bytes = await audio.read()
        if not audio_bytes:
            raise HTTPException(status_code=400, detail="Empty audio file received.")

        console.print(f"[cyan]🎙️  STT request received[/cyan] | size={len(audio_bytes)} bytes | type={audio.content_type} | lang_hint={language}")
        logger.debug("STT request received | size_bytes=%d content_type=%s lang_hint=%s", len(audio_bytes), audio.content_type, language)

        # Run Whisper in a thread so we don't block the async event loop
        # (same pattern as asyncio.to_thread used for DB calls throughout the codebase)
        transcript = await asyncio.to_thread(transcribe, audio_bytes, language)

        if not transcript:
            # Audio was silent or unintelligible — return empty so frontend can handle gracefully
            console.print("[yellow]⚠️  STT: Empty transcript (silent audio?)[/yellow]")
            logger.warning("STT returned empty transcript | size_bytes=%d", len(audio_bytes))
            return {"transcript": ""}

        console.print(f"[green]✅ STT transcript:[/green] [white]{transcript}[/white]")
        logger.info("STT transcript generated | length=%d", len(transcript))
        return {"transcript": transcript}

    except Exception as e:
        console.print(f"[red]❌ STT Error: {str(e)}[/red]")
        traceback.print_exc()
        logger.exception("STT processing failed")
        raise HTTPException(status_code=500, detail=f"STT processing failed: {str(e)}")
# =============================================================================


@api.post("/rag")
async def handle_query(req: QueryRequest):
    query_received_at = time.time()
    query_asked_time_str = datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%Y-%m-%d %H:%M:%S")
    
    detected_language, detection_supported = detect_language(req.query)

    # Resolve input language: explicit dropdown choice wins; "auto" falls back to detection.
    if req.input_language and req.input_language.lower() != "auto" and is_language_supported(req.input_language):
        input_language = req.input_language
        language_warning = None
    else:
        input_language = detected_language
        # Only warn when we actually detected something outside our supported set —
        # not when detection simply failed on short/ambiguous text.
        language_warning = (
            f"Detected language isn't fully supported yet — continuing in English. "
            f"Supported languages: {', '.join(l['name'] for l in list_supported_languages())}."
            if not detection_supported else None
        )

    # Resolve output language: explicit dropdown choice wins; "auto" mirrors the input language
    # (i.e. by default we reply in whatever language the question came in).
    if req.output_language and req.output_language.lower() != "auto" and is_language_supported(req.output_language):
        output_language = req.output_language
    else:
        output_language = input_language

    console.print(f"[cyan]🌐 Language:[/cyan] [bold white]in={input_language} out={output_language}[/]" + (f" [yellow](fallback: {detected_language} not supported)[/]" if language_warning else ""))
    logger.debug(
        "Query received | query_id=%s user_id=%s input_language=%s output_language=%s detection_supported=%s",
        req.query_id, req.user_id, input_language, output_language, detection_supported
    )
    
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
    
    # BYPASS CACHE FOR TOOL CALL QUERIES
    query_lower = req.query.lower()
    is_tool_query = any(kw in query_lower for kw in [
        "schedule", "calendar", "meeting", "event", "task", "todo", "to-do", 
        "doc", "document", "reminder", "remind", "diary"
    ])
    
    query_vec = None
    cache_hit_response = None
    
    if not is_tool_query:
        
        # Generate embedding for the incoming query without blocking the async loop
        query_vec = await asyncio.to_thread(embedder.encode, req.query)    
        user_cache = semantic_cache.get(req.user_id, [])

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
        # Translate the cache hit response to the resolved output language
        if output_language.lower() not in ['en', 'english']:
            cache_hit_response = await translate_to_native(cache_hit_response, output_language)
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
        logger.info(
            "Cache hit | query_id=%s user_id=%s processing_time_s=%.4f",
            req.query_id, req.user_id, processing_time
        )
        
        return {
            "response": cache_hit_response,
            "history": updated_history,
            "metrics": metrics,
            "latencies": {"cache_retrieval": processing_time},
            "language_warning": language_warning,
            "output_locale": get_bcp47(output_language)
        }
    # ---------------------------------------------------------

    # Buffer requests if VLLM isn't up yet
    if not vllm_ready_event.is_set():
        console.print(
            f"[yellow][bold]⚠ BUFFER WAIT[/bold][/] "
            f"[magenta][vLLM is still booting. Query {req.query_id} is waiting in queue.][/] "
            f"[white]{req.query}[/]"
            )
        logger.warning("Query buffered — vLLM not ready | query_id=%s", req.query_id)
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
                "english_history": req.english_history,
                "input_language": input_language,
                "output_language": output_language,
                "english_question": "",
                "english_response": "",
                "latencies": {},
                "metadata": {},
                "context": [],
                "response": "",
                "reformulated_query": "",
                "category": "" 
            })
            
            # --- THIS IS THE LINE THAT WAS MISSING ---
            # final_response = result.get("response", "Error generating response.")
            # updated_history = req.history + [f"Q: {req.query}", f"A: {final_response}"]
            
            final_response = result.get("response","Error generating response.")
            category = result.get("category", "")
            
            if "out_of_bounds" in category:
                updated_history = req.history
                console.print(f"[yellow][GUARD] Out of bounds query detected. Skipping history update to maintain context hygiene.[/]")
            else:
                updated_history = req.history + [f"Q: {req.query}", f"A: {final_response}"]    
            
            # ---------------------------------------------------------
            # 2. CACHE UPDATE (Missed cache, so save the new answer)
            # ---------------------------------------------------------
            if not is_tool_query:
            # Ensure the vector is generated if it was bypassed initially
                if query_vec is None:
                    query_vec = await asyncio.to_thread(embedder.encode, req.query)
                    
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
            logger.info(
                "Response sent | query_id=%s user_id=%s wait_time_s=%.4f processing_time_s=%.4f resources=%s",
                req.query_id, req.user_id, wait_time, processing_time, resources
            )
            
            return {
                "response": final_response,
                "english_question": result.get("english_question", req.query),
                "english_response": result.get("english_response", final_response), 
                "history": updated_history, 
                "metrics": metrics,
                "latencies": result.get("latencies", {}),
                "language_warning": language_warning,
                "output_locale": get_bcp47(output_language)
            }
        except Exception as e:
            console.print(
                f"[red][bold]❌ RAG ERROR[/bold][/] "
                f"[magenta][{req.query_id}][/] "
                f"[red]{str(e)}[/]"
            )
            traceback.print_exc()
            logger.exception("RAG pipeline failed | query_id=%s user_id=%s", req.query_id, req.user_id)
            raise HTTPException(status_code=500, detail="Internal AI Processing Error.")