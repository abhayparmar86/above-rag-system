# Above RAG System

A high-performance, asynchronous **Retrieval-Augmented Generation (RAG)** system designed for production-grade internal knowledge retrieval under strict **8GB VRAM hardware constraints**.

The system combines:

- **FastAPI** for asynchronous API orchestration
- **LangGraph** for multi-node reasoning workflows
- **SurrealDB** for isolated retrieval pipelines
- **vLLM + AWQ Quantization** for efficient LLM inference
- **Semaphore-based admission control** for concurrency safety
- **Process-aware observability** for accurate GPU monitoring

---

# Key Characteristics

- **High-Throughput Inference:** Powered by `vLLM` running `Mistral-7B-Instruct-v0.2-AWQ` in eager mode with flash attention.
- **Instantaneous Cold-Boots:** Local model volume mapping bypasses HuggingFace downloads, reducing startup times from ~6.5 minutes to **< 2 seconds**.
- **Capacity-Safe Scheduling:** `asyncio.Semaphore` admission control prevents GPU KV-cache overload during high concurrency.
- **Intelligent Startup Buffering:** Incoming requests are safely queued while vLLM profiles its KV cache during startup.
- **Robust RAG Pipeline:** Multi-node `LangGraph` architecture featuring query reformulation, intent routing, and metadata extraction.
- **Asynchronous Processing:** Non-blocking FastAPI workers handle concurrent requests with intelligent backpressure.
- **Process-Aware Observability:** GPU metrics isolate only the `VLLM::EngineCore` process, eliminating desktop graphical overhead.
- **Hierarchical Logging:** Structured session logs capture end-to-end latency, GPU usage, wait times, and pipeline traces.

---

# System Architecture

```text
Frontend UI
     │
     ▼
FastAPI Worker (Semaphore Controlled)
     │
     ▼
LangGraph Engine
 ├── Reformulator
 ├── Router
 └── Extractor
     │
     ▼
SurrealDB Retrieval
     │
     ▼
vLLM Inference Server
     │
     ▼
Structured Logging & Metrics
```

---

# Directory Structure

```text
above-rag-system/
├── .github/
│   └── workflows/
│       └── main.yml          # CI/CD linting & Docker validation
├── docker/
│   ├── Dockerfile.retrieval  # API orchestration container
│   ├── Dockerfile.vllm       # vLLM inference container
│   └── docker-compose.yml    # Service orchestration & resource limits
├── app/
│   ├── core/
│   │   ├── engine.py         # LangGraph workflow nodes & orchestration
│   │   ├── database.py       # SurrealDB retrieval logic
│   │   └── logger.py         # Metrics & observability pipeline
│   ├── main.py               # FastAPI entry point
│   └── static/index.html     # Frontend UI
├── test/
│   └── loop_stress_test.py   # Async stress testing script
├── requirements.txt
├── README.md
└── CHANGELOG.md
```

---

# Component Breakdown

## docker/

Contains Dockerfiles and orchestration configuration.

### Responsibilities

- Builds FastAPI orchestration containers
- Builds isolated vLLM inference containers
- Mounts local AWQ models directly into containers
- Configures networking and GPU resource reservations

---

## app/main.py (FastAPI Worker)

Main API entry point.

### Responsibilities

- Manages global concurrency using `asyncio.Semaphore`
- Handles startup buffering using `asyncio.Event`
- Prevents dropped requests during vLLM initialization
- Routes requests into the LangGraph pipeline

---

## app/core/engine.py (LangGraph Engine)

The orchestration layer of the RAG system.

### Pipeline Nodes

- **Reformulator**
  - Rewrites queries using conversational history

- **Router**
  - Classifies intent:
    - Casual
    - Factual
    - Historical

- **Extractor**
  - Extracts metadata such as dates and entities for retrieval filtering

---

## app/core/database.py (Retrieval Layer)

Handles all SurrealDB communication and vector retrieval.

### Features

- Strict logical isolation boundaries

```sql
type::string(transcript.conversation.user) = $user_record
```

- Duplicate context filtering using cosine similarity checks

```python
util.cos_sim > 0.95
```

- Historical and factual retrieval pipelines

---

## app/core/logger.py (Observability Layer)

Tracks pipeline metrics and operational telemetry.

### Features

- Process-aware NVML GPU monitoring
- Tracks only `VLLM::EngineCore`
- Captures:
  - GPU utilization
  - RAM usage
  - CPU usage
  - Request latency
  - Node execution timings

### Outputs

- `pipeline_logs.csv`
- `session_metrics.csv`

### Log Structure

```text
logs/sessions/{session_id}/{user_id}/{chat_id}/
```

---

## test/loop_stress_test.py (Load Testing)

Asynchronous multi-worker client simulator used for stress testing.

### Capabilities

- Simulates concurrent users
- Generates sustained request pressure
- Measures:
  - API latency
  - Processing time
  - Queue delays
  - Throughput stability

### Output

```text
stress_test_report.csv
```

---

# Job Flow

1. **Frontend/UI**
   - User authenticates and submits a query via `index.html`

2. **Startup Buffer Check**
   - `main.py` verifies the `vllm_ready_event`
   - Requests wait safely while the model initializes

3. **Admission Control**
   - Request acquires a Semaphore slot

4. **LangGraph Execution**
   - Query reformulation
   - Intent routing
   - Metadata extraction

5. **Retrieval**
   - SurrealDB fetches isolated contextual vectors

6. **Inference**
   - Prompt and retrieved context are forwarded to vLLM

7. **Observability**
   - Metrics and timing data are captured

8. **Response Delivery**
   - Final response is returned to the UI

---

# Configuration & Environment Variables

Configured primarily inside:

```text
docker/docker-compose.yml
```

| Variable | Description | Default |
|---|---|---|
| `SURREAL_URL` | SurrealDB WebSocket endpoint | `ws://host.docker.internal:8000/rpc` |
| `VLLM_URL` | vLLM inference endpoint | `http://vllm_retrieval:8005/v1` |
| `VLLM_TARGET_GPU_MEMORY_GB` | Target VRAM constraint | `8` |

---

# GPU & Inference Configuration

Passed directly to the vLLM container.

| Argument | Purpose |
|---|---|
| `--max-model-len=8192` | Native context window |
| `--enforce-eager` | Disables CUDA graph capture for stable execution |
| `--served-model-name=mistral-local` | Matches LangChain endpoint configuration |

---

# Setup & Usage

## 1. Prerequisites

- Docker
- Docker Compose
- NVIDIA GPU
- NVIDIA Container Toolkit

---

## 2. Local Model Setup

Ensure the AWQ model exists locally:

```text
/home/android/Documents/abhay/models/Mistral-7B-Instruct-v0.2-AWQ
```

If your path differs, update the volume mapping in:

```text
docker/docker-compose.yml
```

---

## 3. Launching the System

### Build and Start

```bash
docker compose -f docker/docker-compose.yml up --build -d
```

### Monitor vLLM Boot Logs

```bash
docker logs -f rag_vllm_isolated
```

---

## 4. Full Cleanup (Optional)

Useful when Docker cache or containers become inconsistent.

```bash
docker builder prune -a -f

docker compose -f docker/docker-compose.yml build --no-cache api_retrieval
```

---

# UI Testing

Navigate to:

```text
http://localhost:9001/static/index.html
```

### Features

- Real-time response metrics
- Persistent browser sessions
- Auto-archived chat logs
- Request-lock protection during generation

---

# Stress Testing

Run from the project root:

```bash
python test/loop_stress_test.py
```

### Stress Test Configuration

Inside `test/loop_stress_test.py`:

- `CONCURRENT_REQUESTS`
- `TOTAL_REQUESTS_TO_SEND`

Set:

```python
TOTAL_REQUESTS_TO_SEND = None
```

for infinite load generation.

### Output

```text
stress_test_report.csv
```

### Graceful Exit

```bash
Ctrl + C
```

---

# Tech Stack

| Layer | Technology |
|---|---|
| Inference Engine | vLLM |
| Model | Mistral-7B-Instruct-v0.2-AWQ |
| Orchestration | LangGraph & LangChain |
| API Framework | FastAPI |
| Database | SurrealDB |
| Infrastructure | Docker & Docker Compose |
| Monitoring | psutil & pynvml |

---

# Performance Notes

- Optimized for ~8GB VRAM environments using AWQ quantization
- Uses `--enforce-eager` for predictable inference execution
- Semaphore-based admission control prevents GPU memory spikes
- vLLM continuous batching improves throughput under concurrency
- Every session generates isolated hierarchical logs
- Local model mounting reduces cold-boot times to under 2 seconds

---

# CHANGELOG

See:

```text
CHANGELOG.md
```

for release history and architectural evolution.