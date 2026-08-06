"""
Google Workspace MCP Server using FastMCP with SSE transport.
Supports Calendar, Tasks, and Docs integrations.
"""

from fastmcp import FastMCP
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from typing import Literal
from datetime import datetime, timedelta
import os, json

# Expanded scopes to support Calendar, Tasks, and Docs
SCOPES = [
    "https://www.googleapis.com/auth/calendar",
    "https://www.googleapis.com/auth/tasks",
    "https://www.googleapis.com/auth/documents"
]

BASE_DIR = os.path.dirname(__file__)
AUTH_DIR = "/app/auth" if os.path.exists("/app/auth") else BASE_DIR

CREDENTIALS_FILE = os.path.join(AUTH_DIR, "credentials.json")
TOKEN_FILE = os.path.join(AUTH_DIR, "token.json")
MAX_RESULTS = 10

MCP = FastMCP("google_workspace")

# --- Service Builders ---

def get_calendar_service():
    """Handles OAuth2 and returns a Google Calendar service instance."""
    creds = None
    if os.path.exists(TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
            creds = flow.run_local_server(port=0, open_browser=False)
        with open(TOKEN_FILE, "w") as f:
            f.write(creds.to_json())
    return build("calendar", "v3", credentials=creds)   

def get_tasks_service():
    """Handles OAuth2 and returns a Google Tasks service instance."""
    creds = None
    if os.path.exists(TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
            creds = flow.run_local_server(port=0, open_browser=False)
        with open(TOKEN_FILE, "w") as f:
            f.write(creds.to_json())
    return build("tasks", "v1", credentials=creds)

def get_docs_service():
    """Handles OAuth2 and returns a Google Docs service instance."""
    creds = None
    if os.path.exists(TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
            creds = flow.run_local_server(port=0, open_browser=False)
        with open(TOKEN_FILE, "w") as f:
            f.write(creds.to_json())
    return build("docs", "v1", credentials=creds)

# --- Helper Formatters ---

def _format_event(event: dict) -> str:
    """Formats a single calendar event as JSON string."""
    start = event.get("start", {}).get("dateTime", event.get("start", {}).get("date", "N/A"))
    end = event.get("end", {}).get("dateTime", event.get("end", {}).get("date", "N/A"))
    attendees = [a.get("email", "") for a in event.get("attendees", [])]
    return json.dumps({
        "event_id": event.get("id"),
        "title": event.get("summary", "No Title"),
        "start": start,
        "end": end,
        "attendees": attendees,
        "description": event.get("description", "")
    })

def _format_task(task: dict) -> dict:
    """Formats a Google Task object into a standardized dictionary."""
    return {
        "task_id": task.get("id"),
        "title": task.get("title", "No Title"),
        "due": task.get("due", "No Deadline"),
        "notes": task.get("notes", ""),
        "status": task.get("status", "needsAction")
    }

# --- DEFENSIVE AI PARAMETER SANITIZERS ---

def sanitize_date(date_str: str) -> str:
    """Resolves relative dates (today, tomorrow) and invalid text to strict YYYY-MM-DD."""
    date_str = date_str.strip().lower()
    today = datetime.now()
    
    if "today" in date_str:
        return today.strftime("%Y-%m-%d")
    if "tomorrow" in date_str:
        return (today + timedelta(days=1)).strftime("%Y-%m-%d")
        
    if "t" in date_str:
        date_str = date_str.split("t")[0]
        
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
        return date_str
    except ValueError:
        pass
        
    return today.strftime("%Y-%m-%d")

def sanitize_time(time_str: str) -> str:
    """Converts common time strings (3pm, 3 PM, 15, 10:00:00) to strict HH:MM."""
    time_str = time_str.strip().lower()
    if not time_str:
        return "12:00"
        
    if len(time_str) > 5 and ":" in time_str and "m" not in time_str:
        time_str = time_str[:5]
        
    try:
        datetime.strptime(time_str, "%H:%M")
        return time_str
    except ValueError:
        pass
        
    for fmt in ("%I%p", "%I %p", "%I:%M%p", "%I:%M %p"):
        try:
            dt = datetime.strptime(time_str, fmt)
            return dt.strftime("%H:%M")
        except ValueError:
            continue
            
    try:
        val = int(time_str.replace(":", ""))
        if val < 24:
            return f"{val:02d}:00"
    except ValueError:
        pass
        
    return "12:00"

def get_clean_timezone() -> str:
    """Returns a valid IANA timezone string acceptable by Google Calendar."""
    try:
        tz = str(datetime.now().astimezone().tzinfo)
        if tz in ["Coordinated Universal Time", "Local time zone", "None", ""] or " " in tz:
            return "UTC"
        return tz
    except Exception:
        return "UTC"

# =============================================================================
# CALENDAR TOOLS
# =============================================================================

@MCP.tool()
def list_events(date_filter: str = "") -> str:
    """List upcoming calendar events. Optionally filter by date (YYYY-MM-DD)."""
    service = get_calendar_service()
    now = datetime.utcnow().isoformat() + "Z"
    kwargs = {
        "calendarId": "primary",
        "maxResults": MAX_RESULTS,
        "singleEvents": True,
        "orderBy": "startTime",
        "timeMin": now
    }
    if date_filter:
        clean_date = sanitize_date(date_filter)
        kwargs["timeMin"] = f"{clean_date}T00:00:00Z"
        kwargs["timeMax"] = f"{clean_date}T23:59:59Z"
        
    events = service.events().list(**kwargs).execute().get("items", [])
    if not events:
        return "No upcoming events found."
    return json.dumps([json.loads(_format_event(e)) for e in events])   

@MCP.tool()
def add_event(title: str, date: str, start_time: str = "",
              end_time: str = "", attendees: str = "",
              description: str = "") -> str:
    """Add a new event to Google Calendar. date format: YYYY-MM-DD, time format: HH:MM."""
    clean_date = sanitize_date(date)
    clean_start = sanitize_time(start_time)
    clean_end = sanitize_time(end_time) if end_time else ""
    
    print(f"[DEBUG] Raw incoming arguments -> date: '{date}', start_time: '{start_time}', end_time: '{end_time}'")
    print(f"[DEBUG] Sanitized arguments -> date: '{clean_date}', start_time: '{clean_start}', end_time: '{clean_end}'")

    if not clean_end:
        start_dt = datetime.strptime(clean_start, "%H:%M")
        clean_end = (start_dt + timedelta(hours=1)).strftime("%H:%M")
        
    service = get_calendar_service()
    tz = get_clean_timezone()
    
    event = {
        "summary": title,
        "description": description,
        "start": {"dateTime": f"{clean_date}T{clean_start}:00", "timeZone": tz},
        "end":   {"dateTime": f"{clean_date}T{clean_end}:00",   "timeZone": tz},
    }
    if attendees:
        if isinstance(attendees, list):
            attendees = ", ".join(attendees)
        event["attendees"] = [{"email": e.strip()} for e in attendees.split(",") if e.strip()]
        
    created = service.events().insert(calendarId="primary", body=event).execute()
    return f"Event created: '{title}' on {clean_date} {clean_start}-{clean_end}. ID: {created.get('id')}"
                  
@MCP.tool()
def update_event(event_id: str,
                 field: Literal["title", "date", "start_time", "end_time", "description", "attendees"],
                 value: str) -> str:
    """Update a specific field of an existing event by its ID."""
    service = get_calendar_service()
    event = service.events().get(calendarId="primary", eventId=event_id).execute()
    tz = get_clean_timezone()

    if field == "title":
        event["summary"] = value
    elif field == "description":
        event["description"] = value
    elif field == "attendees":
        if isinstance(value, list):
            value = ", ".join(value)
        event["attendees"] = [{"email": e.strip()} for e in value.split(",") if e.strip()]
    elif field in ["date", "start_time", "end_time"]:
        if field == "date":
            value = sanitize_date(value)
        else:
            value = sanitize_time(value)
            
        start_dt = event["start"]["dateTime"]
        end_dt = event["end"]["dateTime"]
        start_date, start_time = start_dt[:10], start_dt[11:16]
        end_date, end_time = end_dt[:10], end_dt[11:16]
        
        if field == "date":
            start_date = end_date = value
        elif field == "start_time":
            orig_start = datetime.strptime(start_time, "%H:%M")
            orig_end = datetime.strptime(end_time, "%H:%M")
            duration = orig_end - orig_start
            new_start = datetime.strptime(value, "%H:%M")
            new_end = new_start + duration
            start_time = value
            end_time = new_end.strftime("%H:%M")
        elif field == "end_time":
            end_time = value
            
        event["start"] = {"dateTime": f"{start_date}T{start_time}:00", "timeZone": tz}
        event["end"]   = {"dateTime": f"{end_date}T{end_time}:00",     "timeZone": tz}

    updated = service.events().update(calendarId="primary", eventId=event_id, body=event).execute()
    return f"Event '{updated.get('summary')}' updated: {field} = {value}"

@MCP.tool()
def delete_event(event_id: str) -> str:
    """Delete a calendar event by its ID."""
    service = get_calendar_service()
    service.events().delete(calendarId="primary", eventId=event_id).execute()
    return f"Event {event_id} deleted successfully."

@MCP.tool()
def search_events(keyword: str) -> str:
    """Search calendar events by keyword in title or description."""
    service = get_calendar_service()
    events = service.events().list(
        calendarId="primary",
        q=keyword,
        maxResults=MAX_RESULTS,
        singleEvents=True,
        orderBy="startTime",
        timeMin=datetime.utcnow().isoformat() + "Z"
    ).execute().get("items", [])
    if not events:
        return f"No events found matching '{keyword}'."
    return json.dumps([json.loads(_format_event(e)) for e in events])

# =============================================================================
# GOOGLE TASKS TOOLS (TO-DOS)
# =============================================================================

@MCP.tool()
def list_tasks(date_filter: str = "", status_filter: Literal["all", "completed", "needsAction"] = "needsAction") -> str:
    """
    List outstanding tasks. Optionally filter by due date (YYYY-MM-DD).'status_filter' can be 'needsAction' (active/incomplete), 'completed', or 'all'. Defaults to 'needsAction'.
    """
    service = get_tasks_service()
    
    # Configure Google Tasks API query based on desired status
    kwargs = {"tasklist": "@default", "maxResults": 50}
    if status_filter in ["completed", "all"]:
        kwargs["showCompleted"] = True
        kwargs["showHidden"] = True  # Hidden tasks include completed ones
    else:
        kwargs["showCompleted"] = False
        kwargs["showHidden"] = False    
    
    tasks_data = service.tasks().list(tasklist="@default", maxResults=50).execute().get("items", [])
    
    formatted_tasks = [_format_task(t) for t in tasks_data]
    
    # Apply strict matching filters to the formatted results
    if status_filter == "completed":
        formatted_tasks = [t for t in formatted_tasks if t["status"] == "completed"]
    elif status_filter == "needsAction":
        formatted_tasks = [t for t in formatted_tasks if t["status"] == "needsAction"]
    
    if date_filter:
        clean_date = sanitize_date(date_filter)
        # Filter task to matching deadline date (matching YYYY-MM-DD in the RFC 3339 string)
        formatted_tasks = [t for t in formatted_tasks if clean_date in t["due"]]
        
    if not formatted_tasks:
        return "No tasks found."
        
    return json.dumps(formatted_tasks)

@MCP.tool()
def delete_task(task_id: str) -> str:
    """
    Delete a task from Google Tasks by its ID.
    """
    service = get_tasks_service()
    service.tasks().delete(tasklist="@default", task=task_id).execute()
    return f"Task {task_id} deleted successfully."

@MCP.tool()
def add_task(title: str, due_date: str = "", notes: str = "") -> str:
    """
    Add a task or to-do item with an optional deadline (YYYY-MM-DD).
    """
    service = get_tasks_service()
    task_body = {
        "title": title,
        "notes": notes
    }
    
    if due_date:
        clean_date = sanitize_date(due_date)
        # Google Tasks requires RFC 3339 formatting
        task_body["due"] = f"{clean_date}T00:00:00.000Z"
        
    created = service.tasks().insert(tasklist="@default", body=task_body).execute()
    return f"Task '{title}' created successfully. ID: {created.get('id')}"

@MCP.tool()
def update_task(task_id: str, field: Literal["title", "notes", "status", "due"], value: str) -> str:
    """
    Update a specific field of an existing task (e.g. status = 'completed' or 'needsAction').
    If updating 'due', value format must be YYYY-MM-DD.
    """
    service = get_tasks_service()
    task = service.tasks().get(tasklist="@default", task=task_id).execute()
    
    if field == "title":
        task["title"] = value
    elif field == "notes":
        task["notes"] = value
    elif field == "status":
        # Google Tasks statuses are either 'completed' or 'needsAction' (incomplete)
        task["status"] = "completed" if value.lower() in ["completed", "done", "complete"] else "needsAction"
    elif field == "due":
        clean_date = sanitize_date(value)
        task["due"] = f"{clean_date}T00:00:00.000Z"
        
    updated = service.tasks().update(tasklist="@default", task=task_id, body=task).execute()
    return f"Task '{updated.get('title')}' updated: {field} = {value}"

# =============================================================================
# GOOGLE DOCS TOOLS (NOTES / DIARY)
# =============================================================================

@MCP.tool()
def create_document(title: str) -> str:
    """
    Creates a new Google Document for notes or diary entries. Returns the document ID.
    """
    service = get_docs_service()
    # Google Docs requires utilizing the Drive API to create files, 
    # but we can initialize a blank file in Docs using document.create
    doc_body = {"title": title}
    doc = service.documents().create(body=doc_body).execute()
    return f"Document '{title}' created successfully. Document ID: {doc.get('documentId')}"

@MCP.tool()
def append_to_document(document_id: str, text: str) -> str:
    """
    Appends rich text onto the end of an existing Google Document.
    """
    service = get_docs_service()
    
    # Get current length of document to find the insertion end point
    doc = service.documents().get(documentId=document_id).execute()
    content = doc.get("body", {}).get("content", [])
    end_index = content[-1].get("endIndex", 1) - 1 if content else 1
    if end_index < 1:
        end_index = 1
        
    requests = [
        {
            "insertText": {
                "location": {"index": end_index},
                "text": f"\n{text}\n"
            }
        }
    ]
    
    service.documents().batchUpdate(documentId=document_id, body={"requests": requests}).execute()
    return f"Successfully appended notes to document ID: {document_id}"


if __name__ == "__main__":
    MCP.run(transport="sse", host="0.0.0.0", port=8090)