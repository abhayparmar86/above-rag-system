import os, traceback, json, threading
from surrealdb import Surreal
from sentence_transformers import SentenceTransformer, util
from core.logger import get_logger

logger = get_logger(__name__)

SURREAL_URL = os.environ.get("SURREAL_URL", "ws://host.docker.internal:19521/rpc")
NAMESPACE, DATABASE = 'insights_system', 'production'
embedder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device='cuda')

class DBManager:
    def __init__(self):
        # Initialize thread-local storage
        self._local = threading.local()
        # self.db = Surreal(SURREAL_URL)
        
    def _get_connection(self):
        """
        Retrieves or creates a persistent, authenticated Surreal connection 
        dedicated to the calling OS thread.
        """
        # If this specific thread doesn't have a connection yet, create one
        if not hasattr(self._local, "conn"):
            # print(f"[DB_DEBUG] Thread {threading.current_thread().name} is initializing a new DB connection.")
            conn = Surreal(SURREAL_URL)
            conn.__enter__()  # Establish persistent WebSocket
            conn.signin({"username": "root", "password": "root"})
            conn.use(NAMESPACE, DATABASE)
            self._local.conn = conn
            
        return self._local.conn    

    def check_user_exists(self, user_id: str) -> bool:
        try:
            db = Surreal(SURREAL_URL)
            with db as conn:
                conn.signin({"username": "root", "password": "root"})
                conn.use(NAMESPACE, DATABASE)
                res = conn.query("SELECT * FROM users WHERE id = $uid", {"uid": f"users:{user_id}"})
                return len(res[0].get('result', [])) > 0 if res else False
        except Exception:
            logger.warning("check_user_exists failed, demo bypass active | user_id=%s", user_id)
            return True # Demo bypass active

    def retrieve_historical(self, user_id, query, metadata):
        vec = embedder.encode(query).tolist()
        
        conn = self._get_connection()
        user_record = f"users:{user_id}"
        
        conn.signin({"username": "root", "password": "root"})
        conn.use(NAMESPACE, DATABASE)
        # print(f"[DEBUG] Using Namespace: {NAMESPACE}, Database: {DATABASE}")
        user_record = f"users:{user_id}"
        
        isolation = "type::string(transcript.conversation.user) = $user_record"
        
        entities = metadata.get("entities", [])
        timeframe = metadata.get("timeframe", [])
        # conditions, params = [], {"qvec": vec}
        # Combine with metadata filters
        # conditions = [isolation_clause]
        params = {"qvec": vec, "user_record": user_record}
        meta_filters = []
        
        if entities:
            res = conn.query("(SELECT VALUE in FROM mentions WHERE string::contains(string::lowercase(type::string(out)), $entity))", 
                                {"entity": entities[0].lower()})
            # print('-'*50)
            # print(res)
            if res:
                # params["allowed_ids"] = res
                # meta_filters.append("type::string(insight) IN $allowed_ids")
                if isinstance(res[0], dict) and 'result' in res[0]:
                    extracted_ids = res[0]['result']
                else:
                    extracted_ids = res
                
                # Convert RecordID objects to strings so they work with the IN clause
                params["allowed_ids"] = [str(record_id) for record_id in extracted_ids]
                meta_filters.append("type::string(insight) IN $allowed_ids")    
        
        if timeframe and len(timeframe) == 2:
            # conditions.append("created_at >= $start AND created_at <= $end")
            params.update({"start": timeframe[0], "end": timeframe[1]})
            # meta_conditions.append("created_at >= $start AND created_at <= $end")
            meta_filters.append("created_at >= $start AND created_at <= $end")

        
        if meta_filters:
            meta_str = "(" + " OR ".join(meta_filters) + ")"
            where_clause = f"WHERE ({isolation}) AND {meta_str}"
        else:
            # If no metadata filters, just filter by isolation
            where_clause = f"WHERE {isolation}"
        query_str = f"""
            SELECT *, source_text, id, vector::similarity::cosine(vector, $qvec) AS score 
            FROM embeddings 
            {where_clause}
            ORDER BY score DESC 
            LIMIT 10
        """
        # print('-'*50)
        # print(query_str)
        # print(params)
        print(f"[DEBUG] Querying for user: users:{user_id}")
        logger.debug("Querying historical embeddings | user_id=%s", user_id)
        # Check if the user exists in the DB right now
        exists = conn.query("SELECT * FROM users WHERE id = $id", {"id": f"users:{user_id}"})
        print(f"[DEBUG] User exists in DB: {bool(exists)}")
        logger.debug("User exists in DB | user_id=%s exists=%s", user_id, bool(exists))
        results = conn.query(query_str, params)
        data = results if results else []
        # print('#'*50)
        # print(results)
        
        
        # Logic for checking duplicate retrieved data 
        unique_chunks = []
        seen_embeddings = []
        
        for item in data:
            text = item.get('source_text','')
            chunk_vec = embedder.encode(text)
            
            is_duplicate = False
            for prev_vec in seen_embeddings:
                if util.cos_sim(chunk_vec, prev_vec) > 0.95:
                    is_duplicate = True
                    break
                
            if not is_duplicate:
                unique_chunks.append(f"[{len(unique_chunks)+1}] SOURCE_ID: {item.get('id')}\nSUMMARY: {text}")
                seen_embeddings.append(chunk_vec)
                
            if len(unique_chunks) >= 5:
                break        
        
        return unique_chunks

    def retrieve_factual(self, user_id, query):
        vec = embedder.encode(query).tolist()
        # Get the persistent connection dedicated to this thread
        conn = self._get_connection()
        query_str = """
        SELECT id, description, vector::similarity::cosine(vector, $qvec) AS score,
        (SELECT name, type, context FROM entities WHERE (name = $parent.subject OR name = $parent.object) AND owner = type::record('users', $user_id)) AS entity_details
        FROM facts WHERE id IN array::flatten((SELECT VALUE ->has_fact->facts FROM type::record('users', $user_id)))
        ORDER BY score DESC LIMIT 100;
        """
        results = conn.query(query_str, {"qvec": vec, "user_id": user_id})
        data = results if results else []
        
        # vector based deduplication logic
        unique_facts = []
        seen_embeddings = []
        
        for item in data:
            desc = item.get('description')
            if not desc:
                continue
        
            chunk_vec = embedder.encode(desc)
                
            is_duplicate = False
            for prev_vec in seen_embeddings:
                # Using the same 0.95 cosine similarity threshold
                if util.cos_sim(chunk_vec, prev_vec) > 0.95:
                    is_duplicate = True
                    break  
            
            if not is_duplicate:
                    unique_facts.append(f"[{len(unique_facts)+1}] Fact_ID: {item.get('id')}\nDescription: {desc}")
                    seen_embeddings.append(chunk_vec)
            
            if len(unique_facts) >= 10:
                break
            
        return "\n\n".join(unique_facts) if unique_facts else "No specific facts found."
    
       
    def filter_new_reminders_and_register(self, user_id: str, items: list[dict]) -> list[dict]:
        """
        Checks a list of calendar events or tasks against SurrealDB to filter out 
        any that have already been proactively notified to the user today. 
        Unnotified items are registered as 'notified' (along with their rich metadata)
        and returned to the frontend.
        """
        conn = self._get_connection()
        conn.signin({"username": "root", "password": "root"})
        conn.use(NAMESPACE, DATABASE)
        
        # Self-Healing: Explicitly define the table as schemaless to instantiate it
        try:
            conn.query("DEFINE TABLE notified_reminders SCHEMALESS")
        except Exception:
            pass
            
        unnotified_items = []
        
        for item in items:
            item_id = item.get("event_id") or item.get("task_id")
            if not item_id:
                continue
                
            record_id = f"notified_reminders:{user_id}_{item_id}"
            
            try:
                already_notified = False
                
                # 1. Check if record exists using a typed Record ID comparison
                try:
                    res = conn.query(
                        "SELECT * FROM notified_reminders WHERE id = type::record($rid)", 
                        {"rid": record_id}
                    )
                    already_notified = len(res[0].get('result', [])) > 0 if res else False
                except Exception as select_err:
                    if "does not exist" in str(select_err):
                        already_notified = False
                    else:
                        raise select_err
                
                # 2. If not already notified, write record with rich metadata to SurrealDB
                if not already_notified:
                    conn.query(
                        """
                        CREATE type::record($rid) SET 
                            user_id = $uid, 
                            item_id = $iid, 
                            item_type = $type,
                            title = $title,
                            notes = $notes,
                            scheduled_time = $time,
                            attendees = $attendees,
                            notified_at = time::now()
                        """,
                        {
                            "rid": record_id, 
                            "uid": f"users:{user_id}", 
                            "iid": item_id,
                            "type": item.get("type", "unknown"),
                            "title": item.get("title", "No Title"),
                            # Unifies Google Calendar "description" and Google Tasks "notes"
                            "notes": item.get("notes") or item.get("description") or "",
                            # Unifies Google Calendar "start" time and Google Tasks "due" date
                            "time": item.get("start") or item.get("due") or "",
                            "attendees": item.get("attendees") or []
                        }
                    )
                    unnotified_items.append(item)
            except Exception as e:
                # Fail-soft fallback
                logger.error("Failed to check/register notified reminder %s: %s", record_id, str(e))
                unnotified_items.append(item)
                
        return unnotified_items
    