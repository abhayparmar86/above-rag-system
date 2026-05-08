import asyncio
import aiohttp
import time
import uuid
import pandas as pd
import signal
from datetime import datetime

# --- CONFIG ---
API_URL = "http://localhost:9001/rag"
CONCURRENT_REQUESTS = 50
TOTAL_REQUESTS_TO_SEND = 1000   # None = infinite, or e.g. 500
HEALTH_CHECK_URL = "http://localhost:9001/docs"
# --- END CONFIG ---

TEST_DATA = [
    {"user_id": "user_001", "query": "What did Rahul work on yesterday and what are his plans for today?"},
    {"user_id": "user_001", "query": "Which team member is responsible for reviewing the OAuth PR, and when will they do it?"},
    {"user_id": "user_001", "query": "What testing gaps exist in Rahul's authentication module implementation?"},
    {"user_id": "user_001", "query": "What specific performance problems is the notification service experiencing at scale?"},
    {"user_id": "user_001", "query": "What concerns does the team have about adopting Kafka, and what alternatives were proposed?"},
    {"user_id": "user_001", "query": "What are the five main services in production and what communication problems exist between them?"},
    {"user_id": "user_001", "query": "What are the specific migration criteria the team established for moving from RabbitMQ to Kafka?"},
    {"user_id": "user_001", "query": "What work of Rahul is Priya going to review and what will she work on further"},
    {"user_id": "user_002", "query": "What restaurant did the team plan to visit and what dish was specifically mentioned as being good?"},
    {"user_id": "user_002", "query": "Which team members were invited but couldn't attend, and why?"},
    {"user_id": "user_002", "query": "What user feedback is driving the need for improved content recommendations?"},
    {"user_id": "user_002", "query": "Describe the collaborative filtering approach proposed as the initial solution."},
    {"user_id": "user_002", "query": "How did the team structure their A/B testing strategy and what metrics would they track?"},
    {"user_id": "user_002", "query": "What time did the group plan to meet and why did they choose that specific time?"},
    {"user_id": "user_002", "query": "What privacy concerns were raised about tracking user behavior for personalization?"},
    {"user_id": "user_003", "query": "How long has the speaker been at their current company and why are they considering leaving?"},
    {"user_id": "user_003", "query": "What advice did Divya give about job searching and taking career risks?"},
    {"user_id": "user_003", "query": "What specific symptoms of stress and anxiety is the speaker experiencing?"},
    {"user_id": "user_003", "query": "What analogy does Nisha use to explain why therapy isn't a sign of weakness?"},
    {"user_id": "user_003", "query": "What concrete resources and next steps does Nisha provide for finding a therapist?"},
    {"user_id": "user_003", "query": "What specific memory from college triggered the speaker's reflection about being present?"},
    {"user_id": "user_003", "query": "What small, achievable commitment does the speaker make regarding meditation practice?"},
    {"user_id": "user_003", "query": "What Instagram quote triggered the philosophical discussion and what was its meaning?"},
    {"user_id": "user_003", "query": "What existential question does the speaker raise about daily routines and 'going through the motions'?"},
    {"user_id": "user_003", "query": "Explain the disagreement about the starfish story - why does one person find it annoying?"}
]

# ── ANSI colors ───────────────────────────────────────────────────────────────
RESET   = "\033[0m";  BOLD    = "\033[1m";  DIM     = "\033[2m"
CYAN    = "\033[96m"; GREEN   = "\033[92m"; YELLOW  = "\033[93m"
RED     = "\033[91m"; BLUE    = "\033[94m"; MAGENTA = "\033[95m"
WHITE   = "\033[97m"

def ts():
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]

def log_send(query_id, user_id, query):
    print(f"{DIM}[{ts()}]{RESET} {CYAN}{BOLD}→ SEND{RESET}  {MAGENTA}[{query_id}]{RESET} {DIM}({user_id}){RESET} {WHITE}{query[:60]}{'...' if len(query)>60 else ''}{RESET}")

def log_waiting(query_id, query):
    print(f"{DIM}[{ts()}]{RESET} {YELLOW}{BOLD}⏳ WAIT{RESET}  {MAGENTA}[{query_id}]{RESET} {DIM}Processing:{RESET} {WHITE}{query[:55]}{'...' if len(query)>55 else ''}{RESET}")

def log_success(query_id, query, response, wait_time, processing_time, total_latency):
    print(
        f"{DIM}[{ts()}]{RESET} {GREEN}{BOLD}✓ DONE{RESET}  {MAGENTA}[{query_id}]{RESET} "
        f"Q: {WHITE}{query[:40]}{'...' if len(query)>40 else ''}{RESET} | "
        f"R: {DIM}{response[:50]}{'...' if len(response)>50 else ''}{RESET}\n"
        f"{'':>28}wait={YELLOW}{wait_time:.2f}s{RESET}  proc={BLUE}{processing_time:.2f}s{RESET}  total={CYAN}{total_latency:.2f}s{RESET}"
    )

def log_error(query_id, query, error):
    print(f"{DIM}[{ts()}]{RESET} {RED}{BOLD}✗ ERR{RESET}   {MAGENTA}[{query_id}]{RESET} Q: {WHITE}{query[:45]}{'...' if len(query)>45 else ''}{RESET} | {RED}{error}{RESET}")

def log_batch(batch_num, dispatched, total_so_far, limit):
    limit_str = str(limit) if limit else "∞"
    print(
        f"\n{DIM}{'─'*70}{RESET}\n"
        f"{BOLD}{BLUE}[{ts()}] 🚀 BATCH #{batch_num} — "
        f"Queued {dispatched} requests  "
        f"[Total sent: {total_so_far}/{limit_str}]{RESET}\n"
        f"{DIM}{'─'*70}{RESET}\n"
    )

def log_system_waiting(attempt):
    print(f"{DIM}[{ts()}]{RESET} {YELLOW}⚠  System not ready — retrying... (attempt {attempt}){RESET}")

def log_system_ready():
    print(f"{DIM}[{ts()}]{RESET} {GREEN}{BOLD}✓ System is UP — starting stress test{RESET}\n")

# ── Globals ───────────────────────────────────────────────────────────────────
results         = []
request_count   = 0
batch_number    = 0
active_requests = 0

# ── Report ────────────────────────────────────────────────────────────────────
def save_report():
    if not results:
        print(f"\n{YELLOW}  No results collected — CSV not written.{RESET}")
        return

    successful = [r for r in results if r.get("status") == "Success"]
    failed     = [r for r in results if r.get("status") != "Success"]

    print(f"\n{BOLD}{CYAN}{'═'*70}{RESET}")
    print(f"{BOLD}  STRESS TEST SUMMARY{RESET}")
    print(f"{CYAN}{'═'*70}{RESET}")
    print(f"  Total Requests : {BOLD}{len(results)}{RESET}")
    print(f"  {GREEN}Successful{RESET}     : {BOLD}{GREEN}{len(successful)}{RESET}")
    print(f"  {RED}Failed/Errors{RESET}  : {BOLD}{RED}{len(failed)}{RESET}")

    if successful:
        df = pd.DataFrame(successful)
        print(f"\n  {BOLD}⏱  Latency (seconds):{RESET}")
        print(f"  {'Metric':<22} {'Min':>6}  {'Max':>6}  {'Avg':>6}")
        print(f"  {'─'*44}")
        for col, label in [
            ("total_api_latency", "API (total)"),
            ("processing_time",   "Processing"),
            ("wait_time",         "Queue wait"),
        ]:
            print(f"  {label:<22} {df[col].min():>6.2f} {df[col].max():>6.2f} {df[col].mean():>6.2f}")

    pd.DataFrame(results).to_csv("stress_test_report.csv", index=False)
    print(f"\n  {GREEN}✓ Report saved → stress_test_report.csv{RESET}")
    print(f"{CYAN}{'═'*70}{RESET}\n")

# ── Health check ──────────────────────────────────────────────────────────────
async def wait_for_system(session, stop_event):
    attempt = 0
    while not stop_event.is_set():
        attempt += 1
        try:
            async with session.get(HEALTH_CHECK_URL, timeout=aiohttp.ClientTimeout(total=5)) as r:
                if r.status < 500:
                    log_system_ready()
                    return True
        except Exception:
            pass
        log_system_waiting(attempt)
        await asyncio.sleep(3)
    return False

# ── Worker ────────────────────────────────────────────────────────────────────
async def worker(queue, session, stop_event, lock):
    global active_requests

    while True:
        # Check if we should stop only when the queue is empty
        try:
            test_case = await asyncio.wait_for(queue.get(), timeout=0.5)
        except asyncio.TimeoutError:
            if stop_event.is_set():
                break
            continue

        async with lock:
            active_requests += 1

        query    = test_case["query"]
        user_id  = test_case["user_id"]
        seq      = test_case["seq"]
        query_id = f"Q{seq:04d}-{uuid.uuid4().hex[:4].upper()}"

        payload = {
            "query_id":   query_id,
            "user_id":    user_id,
            "session_id": f"test_{uuid.uuid4().hex[:6]}",
            "chat_id":    f"chat_{uuid.uuid4().hex[:6]}",
            "query":      query,
            "history":    []
        }

        log_send(query_id, user_id, query)
        log_waiting(query_id, query)

        start_req = time.time()
        try:
            async with session.post(
                API_URL, json=payload, timeout=aiohttp.ClientTimeout(total=300)
            ) as response:
                res_data      = await response.json()
                total_latency = time.time() - start_req

                if response.status == 200:
                    metrics         = res_data.get("metrics", {})
                    wait_time       = metrics.get("wait_time", 0)
                    processing_time = metrics.get("processing_time", 0)
                    resp_text       = res_data.get("response", "")
                    log_success(query_id, query, resp_text, wait_time, processing_time, total_latency)
                    results.append({
                        "query_id":          query_id,
                        "user":              user_id,
                        "query":             query,
                        "status":            "Success",
                        "total_api_latency": round(total_latency, 4),
                        "processing_time":   round(processing_time, 4),
                        "wait_time":         round(wait_time, 4),
                        "response":          resp_text
                    })
                else:
                    log_error(query_id, query, f"HTTP {response.status}")
                    results.append({"query_id": query_id, "query": query, "status": f"Failed: {response.status}"})

        except Exception as e:
            log_error(query_id, query, str(e)[:80])
            results.append({"query_id": query_id, "query": query, "status": f"Error: {str(e)}"})
        finally:
            queue.task_done()
            async with lock:
                active_requests -= 1

# ── Refill ────────────────────────────────────────────────────────────────────
def refill_queue(queue, batch_num, total_queued, limit):
    data_len   = len(TEST_DATA)
    batch_size = CONCURRENT_REQUESTS
    remaining  = (limit - total_queued) if limit else batch_size
    chunk_size = min(batch_size, remaining)
    if chunk_size <= 0:
        return batch_num, total_queued, 0
    for i in range(chunk_size):
        item = dict(TEST_DATA[i % data_len])
        total_queued += 1
        item["seq"] = total_queued
        queue.put_nowait(item)
    batch_num += 1
    log_batch(batch_num, chunk_size, total_queued, limit)
    return batch_num, total_queued, chunk_size

# ── Async body ────────────────────────────────────────────────────────────────
async def run(stop_event):
    global batch_number, request_count
    lock = asyncio.Lock()
    connector = aiohttp.TCPConnector(limit=CONCURRENT_REQUESTS + 10)

    async with aiohttp.ClientSession(connector=connector) as session:
        ready = await wait_for_system(session, stop_event)
        if not ready: return

        queue = asyncio.Queue()
        batch_number, request_count, _ = refill_queue(queue, batch_number, request_count, TOTAL_REQUESTS_TO_SEND)

        workers = [asyncio.create_task(worker(queue, session, stop_event, lock)) for _ in range(CONCURRENT_REQUESTS)]

        while not stop_event.is_set():
            if TOTAL_REQUESTS_TO_SEND and request_count >= TOTAL_REQUESTS_TO_SEND:
                break
            if queue.qsize() < CONCURRENT_REQUESTS:
                batch_number, request_count, added = refill_queue(queue, batch_number, request_count, TOTAL_REQUESTS_TO_SEND)
            await asyncio.sleep(0.5)

        print(f"\n{YELLOW}  Draining in-flight requests — please wait...{RESET}")
        stop_event.set()
        await queue.join()
        
        for w in workers: w.cancel()
        await asyncio.gather(*workers, return_exceptions=True)

# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    print(f"\n{BOLD}{CYAN}{'═'*70}{RESET}")
    print(f"{BOLD}{CYAN}  InsightGraph RAG — Stress Test Client{RESET}")
    print(f"{BOLD}{CYAN}  Concurrency : {CONCURRENT_REQUESTS}  |  Target: {API_URL}{RESET}")
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    stop_event = asyncio.Event()

    def handle_sigint(*_):
        print(f"\n{RED}{BOLD}🛑 SIGINT received — finishing in-flight requests then saving report...{RESET}")
        stop_event.set()

    signal.signal(signal.SIGINT, handle_sigint)

    try:
        loop.run_until_complete(run(stop_event))
    finally:
        loop.close()
        save_report()

if __name__ == "__main__":
    main()