import os, traceback, json
from surrealdb import Surreal
from sentence_transformers import SentenceTransformer, util

SURREAL_URL = os.environ.get("SURREAL_URL", "ws://host.docker.internal:8000/rpc")
NAMESPACE, DATABASE = 'insights_system', 'production'
embedder = SentenceTransformer('BAAI/bge-small-en-v1.5', device='cpu')

class DBManager:
    def __init__(self):
        self.db = Surreal(SURREAL_URL)

    def check_user_exists(self, user_id: str) -> bool:
        try:
            with self.db as conn:
                conn.signin({"username": "root", "password": "root"})
                conn.use(NAMESPACE, DATABASE)
                res = conn.query("SELECT * FROM users WHERE id = $uid", {"uid": f"users:{user_id}"})
                return len(res[0].get('result', [])) > 0 if res else False
        except Exception:
            return True # Demo bypass active

    def retrieve_historical(self, user_id, query, metadata):
        vec = embedder.encode(query).tolist()
        with self.db as conn:
            conn.signin({"username": "root", "password": "root"})
            conn.use(NAMESPACE, DATABASE)
            # print(f"[DEBUG] Using Namespace: {NAMESPACE}, Database: {DATABASE}")
            user_record = f"users:{user_id}"
            
            isolation = "type::string(transcript.conversation.user) = $user_record"
            
            entities = metadata.get("entities", [])
            timeframe = metadata.get("query_timeframe", [])
            # conditions, params = [], {"qvec": vec}
            # Combine with metadata filters
            # conditions = [isolation_clause]
            params = {"qvec": vec, "user_record": user_record}
            meta_filters = []
            
            if entities:
                res = conn.query("(SELECT VALUE type::string(in) FROM mentions WHERE string::contains(string::lowercase(type::string(out)), $entity))", 
                                 {"entity": entities[0].lower()})
                # print('-'*50)
                # print(res)
                if res:
                    params["allowed_ids"] = res
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
            # print(f"[DEBUG] Querying for user: users:{user_id}")
            # Check if the user exists in the DB right now
            exists = conn.query("SELECT * FROM users WHERE id = $id", {"id": f"users:{user_id}"})
            # print(f"[DEBUG] User exists in DB: {bool(exists)}")
            results = conn.query(query_str, params)
            data = results if results else []
            # print('#'*50)
            # print(results)
            
            unique_chunks = []
            seen_embeddings = []
            
            for item in data:
                
                if not isinstance(item, dict):
                    # print('#'*50)
                    # print(f"[DEBUG] Skipping non-dict item: {type(item)} - {item}")
                    continue
                
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
                    
                if len(unique_chunks) >= 3:
                    break        
            
            return unique_chunks

    def retrieve_factual(self, user_id, query):
        vec = embedder.encode(query).tolist()
        with self.db as conn:
            conn.signin({"username": "root", "password": "root"})
            conn.use(NAMESPACE, DATABASE)
            print(f"[DEBUG] Using Namespace: {NAMESPACE}, Database: {DATABASE}")
            query_str = """
            SELECT id, description, vector::similarity::cosine(vector, $qvec) AS score,
            (SELECT name, type, context FROM entities WHERE (name = $parent.subject OR name = $parent.object) AND owner = type::record('users', $user_id)) AS entity_details
            FROM facts WHERE id IN array::flatten((SELECT VALUE ->has_fact->facts FROM type::record('users', $user_id)))
            ORDER BY score DESC LIMIT 100;
            """
            results = conn.query(query_str, {"qvec": vec, "user_id": user_id})
            data = results if results else []
            
            ctx_list, seen = [], set()
            for i, item in enumerate(data, 1):
                desc = item.get('description')
                if desc and desc not in seen:
                    ctx_list.append(f"[{i}] Fact_ID: {item.get('id')}\nDescription: {desc}")
                    seen.add(desc)
            return "\n\n".join(ctx_list) if ctx_list else "No specific facts found."