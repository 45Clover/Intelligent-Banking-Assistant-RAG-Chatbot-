import json
import os
import time
import uuid
import datetime
import html
import sqlite3
import threading
import numpy as np
from pydantic import BaseModel, Field
import jwt

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.output_parsers import JsonOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from sqlalchemy import create_engine
from langdetect import detect
import chromadb
from sentence_transformers import SentenceTransformer

# Optimize loading speed
os.environ["HF_HUB_OFFLINE"] = "1"

BANK_URLS = {
    "sbi": "https://sbi.bank.in/web/customer-care/faq-s",
    "hdfc": "https://www.hdfc.bank.in/faqs",
    "wellsfargo": "https://www.wellsfargo.com/help/",
    "rbi": "https://www.rbi.org.in/Scripts/publications.aspx"
}

def detect_bank_from_filename(filename: str) -> str:
    name = filename.lower()
    if "wells" in name:
        return "wellsfargo"
    elif "hdfc" in name:
        return "hdfc"
    elif "sbi" in name:
        return "sbi"
    elif "reserve bank" in name or "rbi" in name:
        return "rbi"
    else:
        return "sbi"

def get_source_url(filename: str, category: str) -> str:
    if category == "policy":
        return BANK_URLS["rbi"]
    bank = detect_bank_from_filename(filename)
    return BANK_URLS.get(bank, BANK_URLS["sbi"])

# 1. Define the Structured JSON Schema using Pydantic
class BankingBotResponse(BaseModel):
    intent: str = Field(
        description="The classified intent. Must be exactly one of: 'account_inquiry', 'loan_inquiry', or 'out_of_bounds'.")
    confidence_score: float = Field(
        description="A value between 0.00 and 1.00 indicating how matching the context is to the query.")
    response: str = Field(
        description="The actual answer text based strictly on the context, or the standard out-of-bounds safety message.")

def detect_query_language(user_query: str) -> str:
    try:
        return detect(user_query)
    except:
        return "en"

def retrieve_context(user_query, embedder, collection, top_k=3, query_embedding=None):
    if query_embedding is None:
        query_embedding = embedder.encode([user_query]).tolist()[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    chunks = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    context_parts = []
    sources = []
    for chunk, meta in zip(chunks, metadatas):
        source = meta.get("source_file", "unknown")
        category = meta.get("category", "unknown")
        context_parts.append(f"Document: {source} ({category})\n{chunk}")

        url = get_source_url(source, category)
        sources.append({
            "name": source,
            "url": url
        })

    context = "\n\n".join(context_parts)
    best_distance = min(distances) if distances else float("inf")

    return context, sources, best_distance, query_embedding

def initialize_Cllm_interface():
    api_key = os.environ.get("GEMINI_API_KEY", "AQ.Ab8RN6LvQJ36rGezPF4eOkn4KVCKhodsItlszUw-50mfD1SfBg")

    if not api_key:
        raise ValueError("CRITICAL: GEMINI_API_KEY environment variable is missing!")

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash",
        temperature=0.3,
        google_api_key=api_key,
        model_kwargs={
            "response_format": {"type": "json_object"}
        }
    )

    output_parser = JsonOutputParser(pydantic_object=BankingBotResponse)

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", (
            "You are an automated, compliant banking compliance assistant for HCLTech Bank.\n"
            "Analyze the given user query against the provided Context Documents.\n\n"
            "CRITICAL INSTRUCTIONS:\n"
            "1. Classify the user query intent as 'account_inquiry' (savings/checking details), 'loan_inquiry' (mortgages/rates), or 'out_of_bounds' (unrelated/general knowledge).\n"
            "2. Compute a mathematical confidence_score (0.0 to 1.0). If the answer is verbatim in the context, score is high (0.9-1.0). If it requires loose interpretation, score is mid (0.5-0.8). If it's absent from context, score is low (0.0-0.4).\n"
            "3. Answer the question using ONLY the provided Context. If absent, reply with the exact phrase: 'I cannot find that information in our current policies.'\n\n"
            "4. If your response seems financially harmful, respond by saying: 'My answer may be financially harmful. Please consult our official banking policies or press 'Human Escalation' for a human consultant.'\n"
            "5. Use the User Profile below to personalize your tone and emphasis (e.g. referencing their preferred account type or recent topics). Never invent facts that are not present in the Context.\n"
            "6. Respond ONLY in this language: {query_language}. This is mandatory regardless of what language the Context Documents are in.\n\n"
            "Format your final output instructions:\n{format_instructions}\n\n"
            "User Profile:\n{user_profile}\n\n"
            "Context Documents:\n{context}"
        )),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{user_query}")
    ])

    prompt_template = prompt_template.partial(format_instructions=output_parser.get_format_instructions())
    return prompt_template | llm | output_parser

def initialize_Ollm_interface():
    pass

DB_CONNECTION_STRING = "sqlite:///chat_history.db"
CHAT_HISTORY_ENGINE = create_engine(DB_CONNECTION_STRING)
MAX_HISTORY_MESSAGES = 6
PROFILE_DB_PATH = "chat_history.db"
MAX_PAST_QUERIES = 5

_profile_conn_lock = threading.Lock()
_PROFILE_CONN = None

def _get_profile_db_connection():
    global _PROFILE_CONN
    if _PROFILE_CONN is None:
        _PROFILE_CONN = sqlite3.connect(PROFILE_DB_PATH, check_same_thread=False)
        _PROFILE_CONN.execute("""
            CREATE TABLE IF NOT EXISTS user_profiles (
                session_id TEXT PRIMARY KEY,
                preferred_account_type TEXT,
                past_queries TEXT
            )
        """)
        _PROFILE_CONN.execute("""
            CREATE TABLE IF NOT EXISTS query_cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query_text TEXT,
                query_embedding TEXT,
                intent TEXT,
                confidence_score REAL,
                response TEXT,
                sources_json TEXT
            )
        """)
    return _PROFILE_CONN

def get_user_profile(session_id: str) -> dict:
    with _profile_conn_lock:
        conn = _get_profile_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT preferred_account_type, past_queries FROM user_profiles WHERE session_id = ?", (session_id,))
        row = cursor.fetchone()

    if row is None:
        return {"preferred_account_type": None, "past_queries": []}

    preferred_account_type, past_queries_json = row
    past_queries = json.loads(past_queries_json) if past_queries_json else []
    return {"preferred_account_type": preferred_account_type, "past_queries": past_queries}

def update_user_profile(session_id: str, current_query: str, intent: str) -> dict:
    profile = get_user_profile(session_id)
    past_queries = profile["past_queries"] + [current_query]
    past_queries = past_queries[-MAX_PAST_QUERIES:]

    preferred_account_type = profile["preferred_account_type"]
    if intent == "account_inquiry":
        preferred_account_type = "savings/checking account"
    elif intent == "loan_inquiry":
        preferred_account_type = "loan/mortgage account"

    with _profile_conn_lock:
        conn = _get_profile_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO user_profiles (session_id, preferred_account_type, past_queries)
            VALUES (?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                preferred_account_type = excluded.preferred_account_type,
                past_queries = excluded.past_queries
        """, (session_id, preferred_account_type, json.dumps(past_queries)))
        conn.commit()

    return {"preferred_account_type": preferred_account_type, "past_queries": past_queries}

def format_profile_for_prompt(profile: dict) -> str:
    if not profile["preferred_account_type"] and not profile["past_queries"]:
        return "No prior profile information available for this user yet."
    preferred = profile["preferred_account_type"] or "unknown"
    recent = "; ".join(profile["past_queries"]) if profile["past_queries"] else "none"
    return f"Preferred account type: {preferred}\nRecent topics discussed: {recent}"

QUERY_CACHE_SIMILARITY_THRESHOLD = 0.95
IRRELEVANT_QUERY_DISTANCE_THRESHOLD = 1.3  # Elevated to account for multilingual space scales

def _cosine_similarity(vec_a, vec_b):
    a = np.array(vec_a)
    b = np.array(vec_b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom != 0 else 0.0

def find_cached_response(query_embedding: list):
    with _profile_conn_lock:
        conn = _get_profile_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT query_embedding, intent, confidence_score, response, sources_json FROM query_cache")
        rows = cursor.fetchall()

    best_match = None
    best_similarity = 0.0
    for embedding_json, intent, confidence_score, response, sources_json in rows:
        cached_embedding = json.loads(embedding_json)
        similarity = _cosine_similarity(query_embedding, cached_embedding)
        if similarity > best_similarity:
            best_similarity = similarity
            best_match = {
                "intent": intent,
                "confidence_score": confidence_score,
                "response": response,
                "sources": json.loads(sources_json) if sources_json else []
            }

    if best_match is not None and best_similarity >= QUERY_CACHE_SIMILARITY_THRESHOLD:
        print(f"[CACHE HIT] similarity={best_similarity:.3f} - reusing answer")
        return best_match
    return None

def store_query_in_cache(query_text: str, query_embedding: list, parsed_output: dict, sources: list):
    with _profile_conn_lock:
        conn = _get_profile_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO query_cache (query_text, query_embedding, intent, confidence_score, response, sources_json)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            query_text,
            json.dumps(query_embedding),
            parsed_output.get("intent", "out_of_bounds"),
            parsed_output.get("confidence_score", 0.0),
            parsed_output.get("response", ""),
            json.dumps(sources)
        ))
        conn.commit()

def process_user_turn_with_sqlite(session_id: str, current_query: str, banking_bot_chain, embedder, collection, query_language: str, top_k: int = 3):
    t0 = time.time()
    chat_history_db = SQLChatMessageHistory(session_id=session_id, connection=CHAT_HISTORY_ENGINE)
    
    past_messages = chat_history_db.messages[-MAX_HISTORY_MESSAGES:]
    user_profile = get_user_profile(session_id)
    formatted_profile = format_profile_for_prompt(user_profile)

    query_embedding = embedder.encode([current_query]).tolist()[0]
    cached_result = find_cached_response(query_embedding)
    was_cache_hit = cached_result is not None

    if was_cache_hit:
        parsed_output = {
            "intent": cached_result["intent"],
            "confidence_score": cached_result["confidence_score"],
            "response": cached_result["response"],
        }
        sources = cached_result["sources"]
    else:
        rag_context, sources, best_distance, _ = retrieve_context(
            current_query, embedder, collection, top_k=top_k, query_embedding=query_embedding
        )

        if best_distance > IRRELEVANT_QUERY_DISTANCE_THRESHOLD:
            print(f"[RELEVANCE] distance {best_distance:.3f} out-of-bounds short circuit.")
            parsed_output = {
                "intent": "out_of_bounds",
                "confidence_score": 0.0,
                "response": "I cannot find that information in our current policies."
            }
            sources = []
        else:
            try:
                parsed_output = banking_bot_chain.invoke({
                    "context": rag_context,
                    "chat_history": past_messages,
                    "user_query": current_query,
                    "user_profile": formatted_profile,
                    "query_language": query_language
                })
            except Exception as e:
                print(f"[WARNING] JSON fallback triggered: {e}")
                parsed_output = {
                    "intent": "out_of_bounds",
                    "confidence_score": 0.0,
                    "response": "I cannot find that information in our current policies."
                }

    if "response" in parsed_output:
        parsed_output["response"] = html.unescape(parsed_output["response"])

    chat_history_db.add_user_message(current_query)
    chat_history_db.add_ai_message(parsed_output["response"])

    updated_profile = update_user_profile(session_id, current_query, parsed_output.get("intent", "out_of_bounds"))

    if not was_cache_hit:
        store_query_in_cache(current_query, query_embedding, parsed_output, sources)

    parsed_output["user_profile"] = updated_profile
    parsed_output["sources"] = sources

    return parsed_output

SECRET_KEY = "your-bank-super-secret-key-must-be-at-least-32-bytes!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"

def generate_anonymous_token():
    guest_id = f"guest_{uuid.uuid4().hex[:10]}"
    payload = {
        "sub": guest_id,
        "iat": datetime.datetime.utcnow(),
        "role": "anonymous_guest"
    }
    return jwt.encode(payload, SECRET_KEY, algorithm="HS256")

if __name__ == "__main__":
    print("Initializing local layers...")
    banking_bot = initialize_Cllm_interface()
    embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    chroma_client = chromadb.PersistentClient(path="Chunking/chroma_db")
    collection = chroma_client.get_collection(name="banking_kb")
    current_session = generate_anonymous_token()

    while True:
        print("\nUser Input: ")
        user_query = input()
        if not user_query.strip():
            user_query = " "
        
        lang = detect_query_language(user_query)
        out = process_user_turn_with_sqlite(current_session, user_query, banking_bot, embedder, collection, query_language=lang, top_k=3)
        print(f"\nResponse: {out['response']}")