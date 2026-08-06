import os
import json
import asyncio
from datetime import datetime
from zoneinfo import ZoneInfo
from mcp.client.sse import sse_client
from mcp import ClientSession
from core.logger import get_logger

logger = get_logger(__name__)

async def get_daily_reminders_and_mark(user_id: str, db_manager, mcp_url: str) -> list[dict]:
    """
    Connects to the Workspace MCP server using the official client transport,
    queries today's tasks and calendar events, filters them against SurrealDB
    to isolate unnotified items, and marks them as notified.
    """
    try:
        today_str = datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%Y-%m-%d")
        
        # Connect using the native MCP SSE client transport
        async with sse_client(mcp_url) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                
                # Fetch events and tasks concurrently through the initialized session
                event_res = await session.call_tool("list_events", {"date_filter": today_str})
                task_res = await session.call_tool("list_tasks", {"date_filter": today_str})
                
                event_text = event_res.content[0].text if event_res.content else ""
                task_text = task_res.content[0].text if task_res.content else ""
                
                raw_events = []
                raw_tasks = []
                
                # Parse today's events if the response text contains valid data
                if "No upcoming events" not in event_text and event_text:
                    try:
                        raw_events = json.loads(event_text)
                    except Exception:
                        pass
                        
                # Parse today's tasks if the response text contains valid data
                if "No tasks" not in task_text and task_text:
                    try:
                        raw_tasks = json.loads(task_text)
                    except Exception:
                        pass
                
                # Add structural types for the frontend rendering engine
                for e in raw_events:
                    e["type"] = "event"
                for t in raw_tasks:
                    t["type"] = "task"
                    
                combined_items = raw_events + raw_tasks
                
                if not combined_items:
                    return []
                    
                # Filter and register the notified reminders inside SurrealDB
                notified_items = await asyncio.to_thread(
                    db_manager.filter_new_reminders_and_register, 
                    user_id, 
                    combined_items
                )
                return notified_items
                
    except Exception as e:
        logger.error("Proactive daily reminder aggregation failed | user_id=%s error=%s", user_id, str(e))
        return []