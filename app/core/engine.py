import time, json, os, asyncio
from datetime import datetime, timezone
from langgraph.graph import StateGraph, END
from langchain_openai import OpenAI
from core.database import DBManager
from core.logger import log_step_to_csv, get_session_path, save_detailed_log
from typing import TypedDict, List, Dict
from core.translation import translate_to_english, translate_to_native

# Initialization
llm = OpenAI(openai_api_base="http://vllm_retrieval:8005/v1", openai_api_key="EMPTY", model_name="mistral-local", temperature=0.1, stop=["User:", "History:", "Query:", "query", "\n\nUser:", "AI:"])
db_manager = DBManager()

from rich.console import Console
console = Console(force_terminal=True, color_system="truecolor", width=120)

class PipelineState(TypedDict):
    question: str
    original_query: str
    category: str
    metadata: Dict
    context: List[str]
    history: List[str]
    english_history: List[str]
    response: str
    user_id: str
    session_id: str
    latencies: Dict
    reformulated_query: str
    chat_id: str
    language: str
    english_question: str
    english_response: str

# --- NODES ---


async def translate_in_node(state: PipelineState):
    start = time.time()
    lang = state.get("language","English").strip()
    text = state['question']
    
    if lang.lower() in ['en','english']:
        new_latencies = {**state.get("latencies",{}), "translate_in": time.time() - start}
        return {
            "english_question": text,
            "latencies": new_latencies
        }
              
    translated_q = await translate_to_english(llm, text, lang)
    
    new_latencies = {**state.get("latencies",{}), "translate_in": time.time() - start}
    return {
        "question": translated_q,
        "english_question": translated_q,
        "latencies": new_latencies
    }
    

async def translate_out_node(state: PipelineState):
    start = time.time()
    lang = state.get("language","English").strip()
    
    # Save Native History to txt
    session_dir = get_session_path(state['user_id'], state['session_id'], state['chat_id'])
    native_file = os.path.join(session_dir, "native_history.txt")
    
    if lang.lower() in ['en','english']:
        with open(native_file, "a", encoding="utf-8") as f:
            f.write(f"Q: {state['original_query']}\nA: {state['response']}\n")
            
        new_latencies = {**state.get('latencies',{}), "translate_out": time.time() - start}
                   
        return{
            "english_response": state['response'],
            "latencies": {**state.get('latencies',{}), "translate_out": time.time() - start}            
        }
            
    translated_res = await translate_to_native(llm, state['response'], lang)
    
    with open(native_file, "a", encoding="utf-8") as f:
        f.write(f"Q: {state['original_query']}\nA: {translated_res}\n")                              
    
    new_latencies = {**state.get("latencies",{}), "translate_out": time.time() - start}
        
    return {
        "english_response": state['response'],
        "response": translated_res,
        "latencies": new_latencies
    }

async def reformulation_node(state: PipelineState):
    start = time.time()
    messages = state.get("english_history", [])[-4:]
    
    if len(messages) == 0:
        history_str = "This is the first query of conversation. No history available."
    else:
        history_str = "\n".join(messages)
    
    prompt = f"""[INST] You are an expert Query Reformulator.
    Task: Rewrite the 'Latest Query' into a standalone, unambiguous question based on the 'History'.
    
    Rules:
    1. If the query is already standalone, or if it is a general/casual question (e.g., greetings, 'who are you'), output ONLY the query exactly as it is.
    2. If the query is a follow-up, resolve vague references (e.g., 'he', 'it', 'that') using the History.
    3. If the query is a general question (e.g., 'who are you', 'hello', 'what is AI') or changes the subject, IGNORE the history and output the exact query.
    3. STRICT RULE: Keep 'User', 'me', 'my', 'you', 'your' and specific names (John, Rahul) exactly as they appear in the query.
    4. STRICT RULE: Do NOT include citations, sources, or any conversation metadata.
    5. STRICT RULE: Return ONLY the reformulated query text. No conversational filler, no explanations, no headers.
    6. STRICT RULE: If the user changes the subject, DO NOT force a connection to the history.
    7. Do NOT write any explanation about your response, like 'No history provided', 'Exact as given in original query' etc. Just return the query without any additional explanation, notes or meta-talks or unuseful text.
    8. STRICT RULE: NEVER Change pronouns like 'you', 'your', 'yourself', 'me' or 'my' and nouns(names like 'Raj', 'Neha'). Keep them exactly as the user typed typed.

    History:
    {history_str}

    Latest Query: {state['question']}

    Just return the query. Do not return any other text apart from the query. 
    Do not give any explanations. You do not have to generate any response about the query, your job is to just generate and return the query.
    If there is no history provided, just return original query as it is without giving any explanation or generating any other text.
    Standalone Query: [/INST]"""
      
    reformed = (await llm.ainvoke(prompt)).strip()
    
    # Guardrail to prevent explanation text
    if "(" in reformed: 
        reformed = reformed.split("(")[0].strip()
    if "[" in reformed: 
        reformed = reformed.split("[")[0].strip()
    
    return {
        "question": reformed,
        "reformulated_query": reformed,
        "original_query": state.get("original_query", state['question']),
        "latencies": {**state.get("latencies", {}), "reformulation": time.time() - start}
    }

async def intent_node(state: PipelineState):
    start = time.time()
    prompt = f"""[INST] You are an expert AI Classifier for a RAG system. 
    Your goal is to categorize the user's query into exactly one of four categories: 'casual', 'factual', 'historical', or 'out_of_bounds' .

    DEFINITIONS:
    1. 'casual': Used for greetings, small talk, pleasantries, or general questions not related to the user, their specific data, or their work, or questions related to general knowledge and information, which people generally ask to web based search agent or chatbot.(Any general question not related to User personal information) (e.g., "Hello","How are you?", "Tell me a joke", "What is the weather in London?", "What is thermodynamics?").
    2. 'factual': Used for questions about the user's personal profile or stored facts like entities realted to user, projects/companies/clients user has worked for etc. NOTE: Facts here refer to User personal information and not general 'Facts' from world. This includes queries like "Who is my manager?" or "What is my company?" or "what is relation of Priya and John?" or "How is Kevin related to Jay?"
    3. 'historical': Used for queries that require searching through meeting transcripts, summaries, or past actions. This includes questions about specific past discussions, project progress updates, or details from a meeting (e.g., "What did Rahul say about the API module?", "What were the action items from yesterday's sync?").
    4. 'out_of_bounds': Used for ANY query asking to write code, solve math problems, discuss news, adopt a specific persona/role, or perform general world tasks unrelated to the user's personal data, meetings, and insights.
    
    INSTRUCTIONS:
    - Return ONLY the single word (casual, factual, or historical).
    - Do not output punctuation or extra text. Just answer in single word.
    - If in doubt between 'factual' and 'historical', default to 'historical'.
    - Do not generate any explanation. Just return the single word response, no other text or explanation.

    Query: {state['question']}
    Category: [/INST]"""    
    
    # GUARDAIL: Force vLLM to ONLY generate one of these three exact strings
    category = (await llm.ainvoke(
        prompt,
        extra_body={"guided_choice": ["casual", "factual", "historical", "out_of_bounds"]}
    )).strip().lower()
    
    # Guardrail for avoid explanation
    if "(" in category: 
        category = category.split("(")[0].strip()
    if "[" in category: 
        category = category.split("[")[0].strip()
    
    return {"category": category, "latencies": {**state.get("latencies", {}), "router": time.time() - start}}

async def extraction_node(state: PipelineState):
    current_date = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    start = time.time()
    prompt = f"""You are an expert Metadata Extraction agent.
    Your goal is to extract structured filters from the user query to aid in a database search.
    
    CURRENT UTC DATE: {current_date}
    
    INSTRUCTIONS:
    1. 'entities': Extract ONLY names of people(e.g., 'Raj'),place, company, or product.STRICT RULE: Do NOT put Project names here.
    2. 'topics': (OPTIONAL) Extract ONLY specific entity like project names(e.g., Project ABOVE),If mentioned any.It should be an entity,(not general words like 'project','task','topic').If no topic can be extracted, return []. STRICT RULE: If a project name is mentioned, it MUST go in 'topics', never in 'entities'.
    3. 'timeframe': You are experienced detective who can find out the user is asking question from which date. You are expert in reading between the lines, and find out timeframe, which includes results which the user is asking for and query refers to. REMEMBER: Based on User Query Intent you identify that which date/date range data could contain answer for the query, but you have to identify if the period/timeframe from query refers to query on which data was stored, or it is any timeframe/period that is just a planning or mentioned entity in the conversation(For example, 'What is John going to do next week' does not give us timeframe this data was stored on, it refers to timeframe in future, which is an entity, or just reference mentioned in data, which is not to be used for filtering our data based on timeframe.).  Calculate the date range relevant to the search. 
       - If the user asks about "yesterday", "last week", or "in Q3", calculate the specific ISO range [YYYY-MM-DD, YYYY-MM-DD]. Keep one date as buffer date in range for timeframe.
       - If the user asks about a "plan for Q4" or "future meeting", do NOT use Q4 as the search range. Instead, use the most recent 3-6 months as the search range because that's when the planning discussion likely happened.
       - If no time is implied, return []. 
       - Keep one date buffer before the start date of range as well as after end date of range of timeframe. for exampe if you get [2026-07-22, 2026-08-22], return the range [2026-07-21, 2026-08-23].
    4. QUALITY CONTROL: Id a field is not relevent to the query, return an empty list or null. Do not hallucinate topics.

    Return ONLY a JSON object with this structure:
    {{
        "entities": ["Name1", ...],
        "timeframe": ["YYYY-MM-DD", "YYYY-MM-DD"],
        "topics": [ topics extracted ]        
    }}      

    Query: {state['question']}
    JSON Output:""" 
    
    # GUARDRAIL: Define the exact JSON schema required
    extraction_schema = {
        "type": "object",
        "properties": {
            "entities": {"type": "array", "items": {"type": "string"}},
            "timeframe": {"type": "array", "items": {"type": "string"}},
            "topics": {"type": "array", "items": {"type": "string"}}
        },
        "required": ["entities", "timeframe", "topics"]
    }
       
    raw = (await llm.ainvoke(
        prompt,
        stop=["[/INST] ", "</s>"], 
        extra_body={"guided_json": extraction_schema}
    ))
    
    console.print(f"################ RAW Metadata: ################### \n {raw} ")
    
    try:
        metadata = json.loads(raw)
    except:
        metadata = {"entities": [], "topics": [], "timeframe": []}
    return {"metadata": metadata, "latencies": {**state.get("latencies", {}), "metadata": time.time() - start}}

async def retrieval_node(state: PipelineState):
    start = time.time()
    ctx = db_manager.retrieve_historical(state['user_id'], state['question'], state['metadata'])
    return {"context": ctx, "latencies": {**state.get("latencies", {}), "retrieval": time.time() - start}}

async def factual_retrieval_node(state: PipelineState):
    start = time.time()
    ctx = db_manager.retrieve_factual(state['user_id'], state['question'])
    return {"context": ctx, "latencies": {**state.get("latencies", {}), "retrieval": time.time() - start}}

async def llm_simple_node(state: PipelineState):
    start = time.time()
    messages = state.get("english_history", [])[-4:]
    if len(messages) == 0:
        history_str = "This is the first query of conversation. No history available."
    else:
        history_str = "\n".join(messages)
    prompt = f"<s>[INST] You are a helpful assistant.You have to answer general and casual queries of the user.Answer concisely.\nRespond to only casual queries.Do not perform complex tasks like code generation.If you are instructed to perform specific task like generate code, just respond: '''Sorry I am just for casual chat and not for specific complex tasks'''. \n\n History: {history_str}\n\nUser: {state['question']}\nAssistant: [/INST]"
    response = (await llm.ainvoke(prompt)).strip()
    
    new_latencies = {**state.get("latencies", {}), "llm-response": time.time() - start}
    log_state = {**state, "latencies": new_latencies}
    log_step_to_csv(log_state, "llm_simple", prompt, response)
    
    session_dir = get_session_path(state['user_id'], state['session_id'], state['chat_id'])
    history_file = os.path.join(session_dir, "english_history.txt")
    with open(history_file, "a") as f: f.write(f"Q: {state['question']}\nA: {response}\n")

    return {
        "response": response, 
        "latencies": new_latencies
    }
   
async def refusal_node(state: PipelineState):
    start = time.time()
    
    # Canned response - Zero LLM latency!
    response = "I am a personal assistant designed to answer questions about your daily conversations, meetings, and extracted insights. I cannot write code, solve math problems, role-play, or answer queries outside of this scope."
    
    new_latencies = {**state.get("latencies", {}), "refusal": time.time() - start}
    
    log_state = {**state, "latencies": new_latencies}
    log_step_to_csv(log_state, "refusal", "N/A (Input Guardrail Triggered)", response)
    
    session_dir = get_session_path(state['user_id'], state['session_id'], state['chat_id'])
    history_file = os.path.join(session_dir, "english_history.txt")
    with open(history_file, "a") as f: 
        f.write(f"Q: {state['question']}\nA: {response}\n")
    
    return {
        "response": response, 
        "latencies": new_latencies
    }

async def llm_factual_node(state: PipelineState):
    start = time.time()
    messages = state.get("english_history", [])[-4:]
    
    if len(messages) == 0:
        history_str = "This is the first query of conversation. No history available."
    else:
        history_str = "\n".join(messages)
    facts_ctx = state.get("context", "No specific facts found.")
    prompt = f"""You are a personal assistant with access to a knowledge graph of facts about the User.
     - If you use information, you MUST append the citation index (e.g., [1]) and the source id at the end of the sentence.
    - At the very end of your answer, list the "Sources" mapping the index to the Fact_ID (e.g., [1]: facts:fact_123).
    - Do not invent, hallucinate, or assume any facts.Do NOT use your own knowledge or general world knowledge to answer.
    - Ignore any previous interactions in 'History' when forming your answer; they are for context only, do not try to answer any query from it.
    
    History: {history_str}
    
    User Facts Context:
    {facts_ctx}
    
    Question: {state['question']}
    
    Answer based strictly on the User Facts Context provided:"""    
    response = (await llm.ainvoke(prompt)).strip()
    
    new_latencies = {**state.get("latencies", {}), "llm-response": time.time() - start}
    log_state = {**state, "latencies": new_latencies}
    log_step_to_csv(log_state, "llm_factual", prompt, response)
    session_dir = get_session_path(state['user_id'], state['session_id'], state['chat_id'])
    history_file = os.path.join(session_dir, "english_history.txt")
    with open(history_file, "a") as f: f.write(f"Q: {state['question']}\nA: {response}\n")

    return {
        "response": response, 
        "latencies": new_latencies
    }
    
async def llm_rag_node(state: PipelineState):
    start = time.time()
    messages = state.get("english_history", [])[-4:]

    if len(messages) == 0:
        history_str = "This is the first query of conversation. No history available."
    else:
        history_str = "\n".join(messages)
    context_str = "\n".join(state.get('context', []))
    prompt = f"""You are a helpful assistant. Answer the question using the provided context.\nINSTRUCTIONS:
    - Use the provided context to answer.
    - If the answer is not present in the context, you MUST explicitly state: "I'm sorry, but I couldn't find any information regarding this in the database.". Do NOT use your own knowledge or general world knowledge to answer.
    - Do not invent, hallucinate, or assume any facts.
    - Ignore any previous interactions in 'History' when forming your answer; they are for context only, do not try to answer any query from it.
    - After response generation, If you had used information from a specific document, append the corresponding citation ID (e.g., [1], [2]) and the source id at the end of the sentence.(Do NOT provide Source or Citation Text, ONLY give ID. Do not give more than 3 Source or Citation ID.)
    \n Chat History: {history_str}\n\nContext Provided:\n{context_str}\n\nQuestion: {state['question']}\nAnswer based on context:"""
    response = (await llm.ainvoke(prompt)).strip()
    new_latencies = {**state.get("latencies", {}), "llm-response": time.time() - start}

    log_state = {**state, "latencies": new_latencies}
    log_step_to_csv(log_state, "llm_rag", prompt, response)
    session_dir = get_session_path(state['user_id'], state['session_id'], state['chat_id'])
    history_file = os.path.join(session_dir, "english_history.txt")
    with open(history_file, "a") as f: f.write(f"Q: {state['question']}\nA: {response}\n")

    return {
        "response": response, 
        "latencies": new_latencies
    }
   
# --- GRAPH DEFINITION ---
workflow = StateGraph(PipelineState)
workflow.add_node("reformulator", reformulation_node)
workflow.add_node("router", intent_node)
workflow.add_node("extractor", extraction_node)
workflow.add_node("retriever", retrieval_node)
workflow.add_node("factual_retrieval", factual_retrieval_node)
workflow.add_node("llm_simple", llm_simple_node)
workflow.add_node("refusal", refusal_node)
workflow.add_node("llm_factual", llm_factual_node)
workflow.add_node("llm_rag", llm_rag_node)
workflow.add_node("translate_in", translate_in_node)
workflow.add_node("translate_out", translate_out_node)

def route_intent(state):
    cat = state["category"]
    if "out_of_bounds" in cat:
        return 'refusal'
    elif "casual" in cat:
        return 'llm_simple'
    elif "factual" in cat:
        return 'factual_retrieval'
    else:
        return 'extractor'     
    

# workflow.set_entry_point("reformulator")
workflow.set_entry_point("translate_in")
workflow.add_edge("translate_in","reformulator")
workflow.add_edge("reformulator", "router")

workflow.add_conditional_edges("router", route_intent)
workflow.add_edge("extractor", "retriever")
workflow.add_edge("retriever", "llm_rag")
workflow.add_edge("refusal", "translate_out")
workflow.add_edge("factual_retrieval", "llm_factual")
workflow.add_edge("llm_rag", "translate_out")
workflow.add_edge("llm_factual", "translate_out")
workflow.add_edge("llm_simple", "translate_out")
workflow.add_edge("translate_out", END)

rag_graph = workflow.compile()