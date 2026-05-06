import asyncio
import aiohttp
import time
import uuid
import pandas as pd
import signal

# --- CONFIG ---
API_URL = "http://localhost:9001/rag"
CONCURRENT_REQUESTS = 50 
TOTAL_REQUESTS_TO_SEND = None # Set to None for infinite
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

results = []
stop_event = asyncio.Event()
request_count = 0

def handle_sigint(*args):
    print("\n🛑 Stopping test...")
    stop_event.set()

signal.signal(signal.SIGINT, handle_sigint)

async def worker(queue, session):
    global request_count
    while not stop_event.is_set():
        if TOTAL_REQUESTS_TO_SEND and request_count >= TOTAL_REQUESTS_TO_SEND:
            stop_event.set()
            break

        test_case = await queue.get()
        request_count += 1
        start_req = time.time()
        
        payload = {"user_id": test_case["user_id"], "session_id": f"test_{uuid.uuid4().hex[:6]}", "query": test_case["query"], "history": []}
        
        try:
            async with session.post(API_URL, json=payload, timeout=300) as response:
                res_data = await response.json()
                latency = time.time() - start_req
                
                if response.status == 200:
                    metrics = res_data.get("metrics", {})
                    results.append({
                        "user": test_case["user_id"],
                        "query": test_case["query"],
                        "status": "Success",
                        "total_api_latency": latency,
                        "processing_time": metrics.get("processing_time", 0),
                        "wait_time": metrics.get("wait_time", 0),
                        "response": res_data.get("response", "")
                    })
                    print(f"✅ [{request_count}] Latency: {latency:.2f}s | Success")
                else:
                    results.append({"status": f"Failed: {response.status}"})
                    print(f"❌ [{request_count}] Failed: {response.status}")
        except Exception as e:
            results.append({"status": f"Error: {str(e)}"})
            print(f"⚠️ [{request_count}] Error: {str(e)}")
        
        queue.task_done()

async def main():
    queue = asyncio.Queue()
    for case in TEST_DATA: queue.put_nowait(case)
    
    print(f"🚀 Starting Test (Concurrency: {CONCURRENT_REQUESTS})...")
    async with aiohttp.ClientSession() as session:
        workers = [asyncio.create_task(worker(queue, session)) for _ in range(CONCURRENT_REQUESTS)]
        while not stop_event.is_set():
            if queue.empty():
                for case in TEST_DATA: queue.put_nowait(case)
            await asyncio.sleep(1)
        await asyncio.gather(*workers)
    
    if results:
        df = pd.DataFrame([r for r in results if r.get("status") == "Success"])
        print("\n" + "="*40)
        print("⏱️ Performance Metrics (Seconds):")
        print(f"API Latency   -> Min: {df['total_api_latency'].min():.2f} | Max: {df['total_api_latency'].max():.2f} | Avg: {df['total_api_latency'].mean():.2f}")
        print(f"Processing    -> Min: {df['processing_time'].min():.2f} | Max: {df['processing_time'].max():.2f} | Avg: {df['processing_time'].mean():.2f}")
        pd.DataFrame(results).to_csv("stress_test_report.csv", index=False)
        print("\n✅ Report saved to 'stress_test_report.csv'")

if __name__ == "__main__":
    asyncio.run(main())