import csv
import os
import psutil
import pynvml
from datetime import datetime
import json
from zoneinfo import ZoneInfo

# Initialize GPU monitoring
try:
    pynvml.nvmlInit()
    GPU_AVAILABLE = True
except Exception:
    GPU_AVAILABLE = False

LOG_DIR = "/logs/sessions"

def get_process_resources():
    """Captures CPU, RAM, and GPU VRAM usage."""
    try:
        process = psutil.Process(os.getpid())
        cpu = process.cpu_percent()
        ram = psutil.virtual_memory().percent
        gpu_mem = 0
        
        if GPU_AVAILABLE:
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                gpu_mem = round(mem.used / 1024**2, 2)
            except:
                gpu_mem = 0
                
        return f"CPU:{cpu}%|RAM:{ram}%|GPU_VRAM:{gpu_mem}MB"
    except Exception:
        return "CPU:N/A|RAM:N/A|GPU_VRAM:N/A"

def get_session_path(user_id, session_id, chat_id):
    """Ensures session directory exists."""
    # path = os.path.join(LOG_DIR, str(user_id), str(session_id))
    # os.makedirs(path, exist_ok=True)
    path = os.path.join(LOG_DIR, str(session_id), str(user_id), str(chat_id))
    os.makedirs(path, exist_ok=True)
    return path

def save_session_to_csv(user_id: str, session_id: str, history_data: list):
    """Updates the specific session file, strictly overwriting it to prevent file spam."""
    user_dir = os.path.join(LOG_DIR, "users", user_id)
    os.makedirs(user_dir, exist_ok=True)
    
    # FIX: Removed the timestamp. Now, 1 session ID = 1 constantly updated CSV file.
    filename = f"session_{session_id}.csv"
    filepath = os.path.join(user_dir, filename)
    
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
                ist_time.strftime("%Y-%m-%d %H:%M:%S"),
                entry.get("resources", "")
            ])
            
    return filepath

from zoneinfo import ZoneInfo
def log_step_to_csv(state, node_name, prompt, response):
    """
    Logs the full pipeline step (Query, Reformulation, Prompt, Response, etc) 
    to pipeline_logs.csv inside the user session folder.
    """
    session_dir = get_session_path(state.get('user_id'), state.get('session_id'), state.get('chat_id'))
    filepath = os.path.join(session_dir, "pipeline_logs.csv")
    
    ist_time = datetime.now(ZoneInfo("Asia/Kolkata"))
    simple_time = ist_time.strftime("%Y-%m-%d %H:%M:%S")
    
    # Extract latencies and convert to JSON string
    latencies = json.dumps(state.get('latencies', {}))
    
    file_exists = os.path.isfile(filepath)
    with open(filepath, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # Headers: timestamp, node, original, reformed, intent, metadata, prompt, response, resources
        if not file_exists:
            writer.writerow([
                "timestamp", "node", "original_query", "reformed_query", 
                "intent", "metadata", "prompt", "final_response", "latencies", "resources"
            ])
        writer.writerow([
            simple_time,
            node_name,
            state.get('original_query', ''),
            state.get('reformulated_query', ''),
            state.get('category', ''),
            str(state.get('metadata', {})),
            prompt,
            response,
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