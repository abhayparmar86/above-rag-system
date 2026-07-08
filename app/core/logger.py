import csv
import os
import logging
import sys
import psutil
import pynvml
from datetime import datetime
import json
from zoneinfo import ZoneInfo

# ---------------------------------------------------------------------------
# System logging — levels + persistent file, separate from the CSV metrics
# below. Console shows LOG_LEVEL and above; the file captures everything,
# so a quiet demo terminal doesn't mean a quiet audit trail.
# ---------------------------------------------------------------------------
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
SYSTEM_LOG_PATH = "/logs/system.log"

def setup_logging():
    os.makedirs(os.path.dirname(SYSTEM_LOG_PATH), exist_ok=True)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(LOG_LEVEL)
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(SYSTEM_LOG_PATH)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)

# ---------------------------------------------------------------------------
# Initialize GPU monitoring
# ---------------------------------------------------------------------------
try:
    pynvml.nvmlInit()
    GPU_AVAILABLE = True
except Exception:
    GPU_AVAILABLE = False

LOG_DIR = "/logs/sessions"

def get_process_resources():
    """Captures CPU, RAM, and specifically VLLM::EngineCore GPU VRAM usage."""
    try:
        process = psutil.Process(os.getpid())
        cpu = process.cpu_percent()
        ram = psutil.virtual_memory().percent
        gpu_mem = 0
        
        if GPU_AVAILABLE:
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                # Fetch ONLY Compute (C) processes, ignoring Graphics (G) like Xorg/gnome
                compute_processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                vllm_mem = 0
                
                for p in compute_processes:
                    try:
                        # Try to get the process name to match VLLM::EngineCore
                        proc_name = pynvml.nvmlSystemGetProcessName(p.pid)
                        if isinstance(proc_name, bytes):
                            proc_name = proc_name.decode('utf-8')
                            
                        if 'vllm' in proc_name.lower() or 'enginecore' in proc_name.lower() or 'python' in proc_name.lower():
                            vllm_mem += p.usedGpuMemory
                    except pynvml.NVMLError:
                        # Permission restrictions in Docker might block process name resolution
                        pass
                
                # Fallback: If name matching fails due to Docker PID isolation,
                # VLLM is guaranteed to be the largest compute process in our architecture.
                if vllm_mem == 0 and compute_processes:
                    vllm_mem = max([p.usedGpuMemory for p in compute_processes])
                    
                gpu_mem = round(vllm_mem / 1024**2, 2)
            except Exception:
                gpu_mem = 0
                
        return f"CPU:{cpu}%|RAM:{ram}%|GPU_VRAM:{gpu_mem}MB"
    except Exception:
        return "CPU:N/A|RAM:N/A|GPU_VRAM:N/A"

def get_session_path(user_id, session_id, chat_id):
    """Ensures session directory exists."""
    path = os.path.join(LOG_DIR, str(session_id), str(user_id), str(chat_id))
    os.makedirs(path, exist_ok=True)
    return path

def save_session_to_csv(user_id: str, session_id: str, chat_id: str, history_data: list):
    """Updates the specific session file, correctly mapped to the session hierarchy."""
    session_dir = get_session_path(user_id, session_id, chat_id)
    filepath = os.path.join(session_dir, "session_metrics.csv")
    
    ist_time = datetime.now(ZoneInfo("Asia/Kolkata"))
    
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            "Question_Asked", "Response_Generated", "Query_Asked_Time", 
            "Wait_Time_s", "Processing_Time_s", "Response_Sent_Time", 
            "Process_Resource_Usage"
        ])
        
        for entry in history_data:
            writer.writerow([
                entry.get("question", ""),
                entry.get("response", "").replace("\n", " "),  
                entry.get("query_asked_time", ""),
                f"{entry.get('wait_time', 0):.4f}",
                f"{entry.get('processing_time', 0):.4f}",
                entry.get("response_sent_time", ist_time.strftime("%Y-%m-%d %H:%M:%S")),
                entry.get("resources", "")
            ])
            
    return filepath

def log_step_to_csv(state, node_name, prompt, english_response, native_response):
    """
    Logs the full pipeline step (Query, Reformulation, Prompt, Response, etc) 
    to pipeline_logs.csv inside the user session folder.
    """
    session_dir = get_session_path(state.get('user_id'), state.get('session_id'), state.get('chat_id'))
    filepath = os.path.join(session_dir, "pipeline_logs.csv")
    
    ist_time = datetime.now(ZoneInfo("Asia/Kolkata"))
    simple_time = ist_time.strftime("%Y-%m-%d %H:%M:%S")
    
    latencies = json.dumps(state.get('latencies', {}))
    
    file_exists = os.path.isfile(filepath)
    with open(filepath, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "timestamp", "node", "original_query", "translated_q", "reformed_query", 
                "intent", "metadata", "prompt", "english_response", "native_response", "latencies", "resources"
            ])
        writer.writerow([
            simple_time,
            node_name,
            state.get('original_query', ''),
            state.get('english_question',''),
            state.get('reformulated_query', ''),
            state.get('category', ''),
            str(state.get('metadata', {})),
            prompt,
            english_response,
            native_response,
            latencies,
            get_process_resources()
        ])

def save_detailed_log(state, prompt, node_name, response=None):
    """Saves a human-readable trace for debugging."""
    session_dir = get_session_path(
        state.get('user_id'), 
        state.get('session_id'), 
        state.get('chat_id')
    )
    path = os.path.join(session_dir, "detailed_trace.txt")
    with open(path, "a", encoding='utf-8') as f:
        f.write(f"\n{'='*20} {datetime.now().isoformat()} | NODE: {node_name} {'='*20}\n")
        f.write(f"Original Query: {state.get('original_query', 'N/A')}\n")
        f.write(f"Reformed Query: {state.get('reformulated_query', 'N/A')}\n")
        f.write(f"Intent/Category: {state.get('category', 'N/A')}\n")
        f.write(f"Metadata: {str(state.get('metadata', {}))}\n")
        f.write(f"Prompt Sent:\n{prompt}\n")
        f.write(f"Response:\n{response if response else 'N/A'}\n")
        f.write("-" * 60 + "\n")