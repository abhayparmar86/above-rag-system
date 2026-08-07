import os
import json
import time
import asyncio
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, field_validator
from mcp.client.sse import sse_client
from mcp import ClientSession
from langchain_openai import OpenAI

from core.logger import get_session_path, save_detailed_log, save_workspace_agent_log

# Configuration
CALENDAR_MCP_URL = os.environ.get("CALENDAR_MCP_URL", "http://calendar_mcp:8090/sse")
VLLM_URL = os.environ.get("VLLM_URL", "http://vllm_retrieval:8005/v1")

# Dedicated LLM instance for tool planning to avoid circular imports
llm = OpenAI(
    openai_api_base=VLLM_URL,
    openai_api_key="EMPTY",
    model_name="mistral-local",
    temperature=0.1,
    stop=["User:", "History:", "Query:", "query", "\n\nUser:", "AI:"]
)

# --- Pydantic Validation Schemas ---
class PlanStep(BaseModel):
    step: int
    tool: str
    args: dict
    depends_on: Optional[int] = None
    store_result_as: Optional[str] = None

class Plan(BaseModel):
    steps: list[PlanStep]

    @field_validator('steps')
    @classmethod
    def must_have_steps(cls, v):
        if not v:
            raise ValueError("Plan must have at least one step.")
        return v


# --- Helper Function with Placeholder Tracing ---
def resolve_placeholders_with_trace(args: dict, executor_state: dict) -> tuple[dict, list[str]]:
    """
    Replace $key or $key.field placeholders with values from state results.
    Automatically cleans and heals bracket indexing (like [0]) if appended by the LLM.
    """
    resolved = {}
    trace = []
    
    for k, v in args.items():
        if isinstance(v, str) and v.startswith("$"):
            placeholder = v[1:]  # strip $
            if "." in placeholder:
                result_key, field = placeholder.split(".", 1)
                
                # --- SELF-HEALING BRACKET CLEANER ---
                # If LLM appended bracket indexing to the field name, strip it on the fly
                # E.g. "task_id[0]" -> "task_id"
                if "[" in field:
                    clean_field = field.split("[")[0]
                    trace.append(f"Safety Gate Intercepted bracket index in field '{field}' -> Automatically cleaned to '{clean_field}'")
                    field = clean_field
                    
                stored = executor_state["results"].get(result_key, "")
                try:
                    stored_data = json.loads(stored) if isinstance(stored, str) else stored
                    if isinstance(stored_data, list):
                        stored_data = stored_data[0] if stored_data else {}
                    resolved[k] = stored_data.get(field, stored)
                    trace.append(f"Mapped key '{k}' from nested placeholder '${v}' -> resolved to: '{resolved[k]}'")
                except Exception as e:
                    resolved[k] = stored
                    trace.append(f"Failed to parse JSON for placeholder '${v}'. Defaulted to raw storage: '{stored}'. Error: {str(e)}")
            else:
                stored = executor_state["results"].get(placeholder, v)
                
                # --- AUTOMATIC JSON EXTRACTION SAFETY NET ---
                if k in ["event_id", "task_id"] and isinstance(stored, str) and (stored.startswith("[") or stored.startswith("{")):
                    try:
                        stored_data = json.loads(stored)
                        if isinstance(stored_data, list):
                            stored_data = stored_data[0] if stored_data else {}
                        
                        target_key = "event_id" if k == "event_id" else "task_id"
                        if isinstance(stored_data, dict) and target_key in stored_data:
                            stored = stored_data[target_key]
                            trace.append(f"Safety Gate Intercepted root placeholder '${v}' -> Automatically extracted raw ID: '{stored}'")
                    except Exception as e:
                        trace.append(f"Safety Gate failed to parse JSON for key '{k}'. Error: {str(e)}")
                
                resolved[k] = stored
                trace.append(f"Mapped key '{k}' from root placeholder '${v}' -> resolved to: '{stored}'")
        else:
            resolved[k] = v
            
    return resolved, trace

# --- LangGraph Execution Node ---
async def calendar_agent_node(state):
    total_start = time.time()
    question = state['question']
    
    session_dir = get_session_path(state['user_id'], state['session_id'], state['chat_id'])
    history_file = os.path.join(session_dir, "english_history.txt")
    
    # Telemetry collectors
    execution_trace = []
    latencies = {}
    
    try:
        async with sse_client(CALENDAR_MCP_URL) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                mcp_tools = await session.list_tools()
                tool_schemas = {t.name: t.input_schema for t in mcp_tools.tools}

                # Fetch current events/tasks context
                current_events = await session.call_tool("list_events", {"date_filter": ""})
                current_data = current_events.content[0].text if current_events.content else "No events."

                # PHASE 1 — PLANNER LLM CALL
                planner_start = time.time()
                planner_prompt = f"""[INST] You are a Google Workspace planning assistant.
                    Analyze the user request and produce an ordered execution plan.

                    Available tools and schemas:
                    {json.dumps(tool_schemas, indent=2)}

                    Current events context (use these real IDs for update/delete):
                    {current_data}

                    Today's date: {datetime.now().strftime('%Y-%m-%d')}
                    User request: {question}

                    STRICT RULES:
                    1. DIRECT CREATIONS (1 STEP): For adding/scheduling any NEW task or event, use exactly ONE step with 'add_task' or 'add_event' with the 'due_date'/'date' directly. Never search or list first.
                    2. MODIFICATIONS (MULTI-STEP): To update/modify/delete existing items, you must first search/list them ('list_events'/'list_tasks'/'search_events'/'search_tasks'), store the result, and pass its ID placeholder (e.g. '$today_tasks.task_id') to the action tool.
                    3. READ-ONLY (1 STEP): For simple listing or searching, use exactly ONE step ('list_events'/'list_tasks'/'search_events'). Never add speculative creation or update steps.
                    4. SYNONYMS: "meeting", "meetings", and "events" are identical. Map "list meetings" directly to 'list_events'.
                    5. STRICT FORMATTING: Use ONLY available tools. Never use fictional tools (like 'filter', 'map', 'select'). Never include comments, explanations, notes, or double slashes (//) in your JSON output.
                    6. PLACEHOLDERS: Stored results are flat JSON lists. Reference fields using only "$store_name.field" (e.g., "$my_search.event_id"). Never use '.items', 'items[0]', or '[0]'.
                    7. STRICT ID RULE: Parameters 'event_id' and 'task_id' must be raw alphanumeric IDs. Never pass a human-readable title (like 'Planning' or 'punch-out') as an ID.
                    8. NO OVER-PLANNING: For a simple, single-field update/delete on an already-uniquely-identified item, use only the minimum steps needed (find it, then act on it). Never add extra no-op steps that set a field to its own current value, and never create a new resource (e.g. via 'add_task'/'add_event') as a substitute for updating the existing one.
                    9. DISAMBIGUATION: When the user names a specific item by title/keyword (e.g., "ingestion-tool"), use 'search_tasks' or 'search_events' with that keyword to find the exact matching item. Only use 'list_tasks'/'list_events' when the user does NOT reference any specific title/keyword.

                    Respond ONLY with a JSON array of steps matching one of these structures:

                    Example 1 (Direct Creation): [{{ "step": 1, "tool": "add_task", "args": {{ "title": "Check email", "due_date": "2026-08-05", "due_time": "15:00" }}, "depends_on": null, "store_result_as": null }}]
                    Example 2 (Update Event): [{{ "step": 1, "tool": "list_events", "args": {{ "date_filter": "today" }}, "depends_on": null, "store_result_as": "today_meetings" }},{{ "step": 2, "tool": "update_event", "args": {{ "event_id": "$today_meetings.event_id", "field": "attendees", "value": "Priya" }}, "depends_on": 1, "store_result_as": null }}]
                    Example 3 (Subtask): [{{ "step": 1, "tool": "add_task", "args": {{ "title": "Planning" }}, "depends_on": null, "store_result_as": "parent_task" }},{{ "step": 2, "tool": "add_task", "args": {{ "title": "Design", "parent_id": "$parent_task.task_id" }}, "depends_on": 1, "store_result_as": null }}]
                    Example 4 (Update Named Task): [{{ "step": 1, "tool": "search_tasks", "args": {{ "keyword": "ingestion-tool" }}, "depends_on": null, "store_result_as": "target_task" }},{{ "step": 2, "tool": "update_task", "args": {{ "task_id": "$target_task.task_id", "field": "status", "value": "completed" }}, "depends_on": 1, "store_result_as": null }}]
                    JSON: [/INST]"""

                planner_schema = {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "step": {"type": "integer"},
                            "tool": {"type": "string", "enum": list(tool_schemas.keys())},
                            "args": {"type": "object"},
                            "depends_on": {"type": "integer"},
                            "store_result_as": {"type": "string"}
                        },
                        "required": ["step", "tool", "args"]
                    }
                }

                raw_plan = (await llm.ainvoke(
                    planner_prompt,
                    max_tokens=512,
                    extra_body={"guided_json": planner_schema}
                )).strip()

                # --- DUAL-LAYER JSON EXTRACTOR ---
                cleaned_plan = raw_plan
                start_idx = cleaned_plan.find("[")
                end_idx = cleaned_plan.rfind("]")
                if start_idx != -1 and end_idx != -1:
                    cleaned_plan = cleaned_plan[start_idx:end_idx + 1]

                try:
                    raw_steps = json.loads(cleaned_plan)
                except Exception:
                    last_complete = cleaned_plan.rfind("},")
                    if last_complete != -1:
                        salvaged = cleaned_plan[:last_complete + 1] + "]"
                        try:
                            raw_steps = json.loads(salvaged)
                        except Exception as e2:
                            raise RuntimeError(f"Failed to parse plan after truncation salvage: {e2}")
                    else:
                        raise RuntimeError("Failed to parse truncated plan json.")

                if isinstance(raw_steps, dict):
                    raw_steps = [raw_steps]
                plan_obj = Plan(steps=[PlanStep(**s) for s in raw_steps if "tool" in s and "args" in s])
                plan = [s.model_dump() for s in plan_obj.steps]
                
                latencies["planning"] = time.time() - planner_start

                # PHASE 2 — EXECUTOR LOOP WITH STATE
                executor_start = time.time()
                executor_state = {
                    "completed": [],
                    "failed": [],
                    "results": {}
                }

                all_tool_results = []
                sorted_plan = sorted(plan, key=lambda s: s.get("step", 0))
                i = 0

                while i < len(sorted_plan):
                    current_step = sorted_plan[i]
                    depends_on = current_step.get("depends_on")

                    if depends_on is not None and depends_on not in executor_state["completed"]:
                        executor_state["failed"].append(current_step["step"])
                        fail_msg = f"Aborted - Dependency step {depends_on} not completed."
                        
                        execution_trace.append({
                            "step": current_step["step"],
                            "tool": current_step["tool"],
                            "original_args": current_step.get("args", {}),
                            "placeholder_trace": ["Aborted due to dependency failure."],
                            "resolved_args": {},
                            "response": fail_msg,
                            "status": "ABORTED"
                        })
                        all_tool_results.append(f"{current_step['tool']} (step {current_step['step']}): {fail_msg}")
                        i += 1
                        continue

                    concurrent_batch = []
                    while i < len(sorted_plan):
                        step = sorted_plan[i]
                        dep = step.get("depends_on")
                        if dep is None or dep in executor_state["completed"]:
                            concurrent_batch.append(step)
                            i += 1
                        else:
                            break

                    if not concurrent_batch:
                        i += 1
                        continue

                    resolved_batch = []
                    for step in concurrent_batch:
                        resolved_args, p_trace = resolve_placeholders_with_trace(step.get("args", {}), executor_state)
                        resolved_batch.append((step, resolved_args, p_trace))

                    batch_results = await asyncio.gather(*[
                        session.call_tool(step.get("tool"), resolved_args)
                        for step, resolved_args, _ in resolved_batch
                    ], return_exceptions=True)

                    for ((step, resolved_args, p_trace), result) in zip(resolved_batch, batch_results):
                        step_num = step.get("step")
                        tool_name = step.get("tool")
                        store_as = step.get("store_result_as")
                        original_args = step.get("args", {})

                        if isinstance(result, Exception):
                            executor_state["failed"].append(step_num)
                            err_str = f"ERROR - {str(result)}"
                            
                            execution_trace.append({
                                "step": step_num,
                                "tool": tool_name,
                                "original_args": original_args,
                                "placeholder_trace": p_trace,
                                "resolved_args": resolved_args,
                                "response": err_str,
                                "status": "ERROR"
                            })
                            all_tool_results.append(f"{tool_name} (step {step_num}): {err_str}")
                        else:
                            result_text = result.content[0].text if result.content else "No result."
                            
                            lower_result = result_text.lower()
                            is_failure = (
                                result_text.startswith("Error") or 
                                "no events found" in lower_result or 
                                "no tasks found" in lower_result or 
                                "nothing found" in lower_result or
                                "no upcoming events" in lower_result or
                                "unknown tool" in lower_result
                            )
                            
                            if is_failure:
                                executor_state["failed"].append(step_num)
                                execution_trace.append({
                                    "step": step_num,
                                    "tool": tool_name,
                                    "original_args": original_args,
                                    "placeholder_trace": p_trace,
                                    "resolved_args": resolved_args,
                                    "response": result_text,
                                    "status": "FAILED"
                                })
                                all_tool_results.append(f"{tool_name} (step {step_num}): FAILED - {result_text}")
                            else:
                                executor_state["completed"].append(step_num)
                                execution_trace.append({
                                    "step": step_num,
                                    "tool": tool_name,
                                    "original_args": original_args,
                                    "placeholder_trace": p_trace,
                                    "resolved_args": resolved_args,
                                    "response": result_text,
                                    "status": "SUCCESS"
                                })
                                all_tool_results.append(f"{tool_name} (step {step_num}): {result_text}")
                                if store_as:
                                    executor_state["results"][store_as] = result_text

                latencies["execution"] = time.time() - executor_start

                # PHASE 3 — FINAL CONFIRMATION GENERATION
                response_start = time.time()
                tool_results_summary = "\n".join(all_tool_results)
                response_prompt = f"""[INST] You are a helpful personal assistant.
                The user asked: {question}
                Actions performed and results:
                {tool_results_summary}
                STRICT RULES:
                1. Only state facts explicitly present in the 'Actions performed and results' above. Never invent, assume, or restate the user's originally requested value as confirmed — always use the exact value returned by the tools.
                2. If a step is marked FAILED, ERROR, or ABORTED, clearly say so instead of implying success.
                3. Do not mention tool names, step numbers, or internal IDs.
                Write a short simple confirmation or summary grounded strictly in the above results. Be concise and to the point [/INST]"""

                response = (await llm.ainvoke(response_prompt)).strip()
                
                latencies["response"] = time.time() - response_start
                latencies["total"] = time.time() - total_start
                
                # Write English conversation history
                with open(history_file, "a") as f: 
                    f.write(f"Q: {question}\nA: {response}\n")

                # --- WRITE DETAILED TRACES & JSON TELEMETRY ---
                save_workspace_agent_log(
                    state=state,
                    planner_prompt=planner_prompt,
                    raw_plan=raw_plan,
                    plan=plan,
                    execution_trace=execution_trace,
                    response_prompt=response_prompt,
                    response=response,
                    latencies=latencies
                )

                new_latencies = {**state.get("latencies", {}), "workspace_execution": latencies["total"]}

                return {
                    "response": response,
                    "prompt": response_prompt,
                    "latencies": new_latencies
                }

    except Exception as e:
        
        import traceback
        traceback.print_exc()
        
        error_msg = f"I encountered an error connecting to or running the Workspace assistant. Detail: {str(e)}"
        with open(history_file, "a") as f: 
            f.write(f"Q: {question}\nA: {error_msg}\n")
        
        error_msg = f"I encountered an error connecting to or running the Workspace assistant. Detail: {str(e)}"
        with open(history_file, "a") as f: 
            f.write(f"Q: {question}\nA: {error_msg}\n")
            
        latencies["total"] = time.time() - total_start
        save_workspace_agent_log(
            state=state,
            planner_prompt=planner_prompt if 'planner_prompt' in locals() else "N/A (Crashed before prompt)",
            raw_plan=raw_plan if 'raw_plan' in locals() else "N/A (Crashed before planning completion)",
            plan=plan if 'plan' in locals() else [],
            execution_trace=execution_trace,
            response_prompt="N/A (Crashed)",
            response=error_msg,
            latencies=latencies
        )
        return {
            "response": error_msg,
            "prompt": "N/A (Execution Failure)",
            "latencies": {**state.get("latencies", {}), "workspace_execution": latencies["total"]}
        }