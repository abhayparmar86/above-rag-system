# RAG-Chat Application - ABOVE based

This project focuses on a high-performance, resource-optimized **RAG (Retrieval-Augmented Generation)** pipeline designed to process internal knowledge queries. It leverages an **8GB VRAM constraint** using quantized models and dynamic concurrency management.

## Key Characteristics
*   **High-Throughput Inference:** Powered by `vLLM` with AWQ quantization for optimized GPU utilization.
*   **Capacity-Safe Scheduling:** Semaphore-based admission control prevents GPU overload and ensures stability under high concurrency.
*   **Robust RAG Pipeline:** Multi-node `LangGraph` architecture featuring automated query reformulation, intent routing, and metadata extraction.
*   **Asynchronous Processing:** Non-blocking `FastAPI` worker handles concurrent requests with intelligent backpressure.
*   **Observability:** Structured CSV logging for every pipeline step, tracking latency, GPU/RAM/CPU usage, and processing metrics per node.

---

## Directory Structure
```text
Project/
├── docker/
│   ├── Dockerfile.retrieval  # API orchestration container
│   ├── Dockerfile.vllm       # Inference server definition
│   └── docker-compose.yml    # Orchestration & Resource limits
├── app/
│   ├── core/
│   │   ├── engine.py         # LangGraph workflow nodes & logic
│   │   ├── database.py       # SurrealDB & Vector retrieval
│   │   └── logger.py         # Structured logging & metrics
│   ├── main.py               # FastAPI entry point
│   └── static/index.html     # Frontend UI
├── test/
│   └── stress_test_loop.py   # Load testing script
├── requirements.txt
├── README.md
└── CHANGELOG.md
```

### Component Breakdown
*   **Core Engine (`app/core/`)**
    *   `engine.py`: The "brain." Contains the LangGraph definition, workflow nodes, and LLM integration.
    *   `database.py`: Handles SurrealDB connections, vector embeddings, and history/fact retrieval logic.
    *   `logger.py`: Manages performance monitoring and the hierarchical logging structure (`logs/sessions/{session_id}/{user_id}/{chat_id}/`).
*   **Infrastructure (`docker/`)**
    *   `Dockerfile.retrieval`: Configures the FastAPI environment for the orchestration layer.
    *   `Dockerfile.vllm`: Configures the optimized vLLM inference server.
    *   `docker-compose.yml`: Defines the network, resource reservations, and environmental variables.
*   **Testing & UI**
    *   `app/main.py`: The API entry point. Manages concurrent request limits via `asyncio.Semaphore`.
    *   `app/static/index.html`: The Frontend UI. Handles authentication and maintains persistent browser sessions.
    *   `test/stress_test_loop.py`: A continuous load-testing script to simulate high concurrency.

---

## Job Flow
1. **Frontend/UI:** User authenticates and initiates a query via `index.html`.
2. **FastAPI (main.py):** Receives the request; `asyncio.Semaphore` regulates concurrent access.
3. **LangGraph (engine.py):**
    *   **Reformulator:** Cleans query and resolves history.
    *   **Router:** Classifies intent (casual, factual, or historical).
    *   **Extractor:** Pulls metadata for database filtering.
4. **Backend (database.py):** Executes vector search or fact lookup in **SurrealDB**.
5. **Inference (vLLM):** Processes the prompt and generates a response using the context-injected template.
6. **Logging:** Results and performance metrics are written to `/logs/sessions/{session_id}/{user_id}/{chat_id}/`.

---

## Usage Guide

### Launching the System
```bash
# Build and Start
docker compose -f docker/docker-compose.yml up --build

# Full cleanup if code changes
docker builder prune -a -f
docker compose -f docker/docker-compose.yml build --no-cache api_retrieval
```

### UI Testing
Navigate to `http://localhost:9001/static/index.html`. Authenticate with any ID to start an investigation. Metrics are displayed per response, and chat logs are auto-saved to `/logs/sessions/`.

### Stress Testing
The `test/stress_test_loop.py` script tests the pipeline under heavy load.
*   **Configure:** Set `CONCURRENT_REQUESTS` and `TOTAL_REQUESTS_TO_SEND` (set to `None` for infinite loop) in the script.
*   **Run:** `python test/stress_test_loop.py`
*   **Output:** Generates `stress_test_report.csv` which benchmarks `total_api_latency` vs. `processing_time`.
*   **Exit:** Press `Ctrl+C` for a graceful shutdown and report generation.

---

## Tech Stack
*   **Inference Engine:** `vLLM` (Mistral-7B-Instruct-v0.2-AWQ, Quantized)
*   **Orchestration:** `LangGraph` & `LangChain`
*   **API Framework:** `FastAPI` (Asynchronous, Semaphore-throttled)
*   **Database:** `SurrealDB` (Graph-based retrieval)
*   **Infrastructure:** `Docker` & `Docker Compose`
*   **Monitoring:** `psutil` & `pynvml` (Real-time GPU/CPU/RAM tracking)

## Performance Notes
*   **VRAM:** Optimized for ~8GB VRAM using `AWQ` quantization and `--enforce-eager`.
*   **Isolation:** Every UI visit generates a unique `session_id`, ensuring logs are isolated in the `/logs/` directory per session.
*   **Throughput:** Managed by an API-side `Semaphore` and `vLLM` sequence queuing to balance latency and memory stability.