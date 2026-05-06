# Changelog

This document tracks the evolution of the InsightGraph Enterprise RAG System
from its initial prototype to the current production-ready version (v2.2).

# [2.2] - 2026-05-06

# Major Enhancements & Refactoring

  - VLLM Configuration:
      - Updated the VLLM model source from TheBloke/Mistral-7B-Instruct-v0.1-AWQ
        to a local path: /models/Mistral-7B-Instruct-v0.2-AWQ.
      - Increased max-model-len to 4096 and max-num-seqs to 12 for better
        throughput.
      - Enabled --enforce-eager mode and explicitly defined the served model
        name as mistral-local.
  - Retrieval Pipeline (RAG Engine):
      - Duplicate Filtering: Implemented a cosine similarity check (util.cos_sim
        > 0.95) in retrieve_historical to prevent redundant document chunks from
        being sent to the LLM.
      - Context Isolation: Enforced strict DB-level user isolation using
        type::string(transcript.conversation.user) = $user_record.
      - Graph Edge Mapping: Updated SurrealDB queries to use direct edge
        traversal (->has_fact->facts) to improve factual retrieval accuracy.
  - Logging System:
      - Directory Structure: Refactored log paths to a hierarchical format:
        /logs/sessions/{session_id}/{user_id}/{chat_id}/.
      - Timezone Localization: Integrated ZoneInfo("Asia/Kolkata") to ensure
        accurate local timestamping across logs, CSV metrics, and detailed
        traces.
      - Detailed Traces: Added logic to log entire pipeline steps, including
        node-specific prompts, responses, and resource metrics.
  - Frontend (UI/UX):
      - UI Modernization: Complete transition to Tailwind CSS with a dark-mode
        slate theme.
      - Chat Management: Introduced a dynamic chat_id system, allowing users to
        switch between multiple investigations in one session.
      - Session Persistence: Implemented robust session saving, manual export
        (CSV generation), and "keepalive" fetch requests to prevent data loss on
        tab closing.
      - Auth Flow: Added a persistent authentication modal with visual feedback
        (spinners) and error states.
  - Backend Stability:
      - Increased MAX_CONCURRENT_REQUESTS to 60 for better scalability.
      - Implemented asyncio.to_thread and ainvoke throughout the graph nodes to
        prevent blocking the Event Loop during intensive LLM tasks.

# [2.1] - 2026-05-06

# Backend Integration & Optimization

  - Graph Logic:
      - Formalized the PipelineState TypedDict to act as the single source of
        truth for variables across graph nodes.
      - Added dedicated reformulation_node with strict system instructions to
        force the LLM to output only the query text.
  - API Enhancements:
      - Added support for /session/close to decouple session management from
        message handling.
      - Updated main.py to support StreamingResponse preparation and better
        concurrency limiting via Semaphore.
  - Model Prompting:
      - Wrapped prompts in [INST] tags to comply with Mistral’s instruct
        template format, significantly improving instruction adherence.

# [2.0] - 2026-05-06

# Structural Overhaul

  - Directory Restructuring:
      - Moved from a monolithic script file to a professional app/core/ package
        structure (engine.py, database.py, logger.py).
  - State Management:
      - Replaced linear graph flow with LangGraph states, allowing for
        conditional routing based on query complexity (using an intent_node).
  - Database Management:
      - Encapsulated all SurrealDB logic inside a DBManager class, introducing
        standardized error handling and connection pooling/signin routines.
  - Containerization:
      - Added custom Dockerfiles for retrieval and VLLM servers, utilizing
        uvicorn and nvidia/cuda base images to optimize GPU compute
        capabilities.

# [1.1] - 2026-05-06

Initial Deployment & Dockerization

  - Environment Configuration:
      - Introduced docker-compose.yml to define the relationship between the
        Retrieval API and the VLLM model server.
      - Added extra_hosts and networks to resolve host.docker.internal
        connectivity issues between container layers.
  - Memory Management:
      - Added shell scripts in Docker commands to dynamically calculate
        gpu-memory-utilization based on available VRAM detected by nvidia-smi.

# [1.0] - 2026-05-06

# The Prototype

  - Core Functionality:
      - Initial integration of LangGraph, SurrealDB, and SentenceTransformers
        for a standard RAG pipeline.
      - Basic manual logging to a shared pipeline_metrics.csv.
      - In-memory chat history tracking using local file-based persistence
        (chat_history.txt).
      - Basic langgraph nodes for intent classification (casual/factual/historical), basic
        metadata extraction, llm_simple generation, llm_factual generation and llm_rag generation.
