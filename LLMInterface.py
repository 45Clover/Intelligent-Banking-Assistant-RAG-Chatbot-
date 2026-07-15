import json
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import JsonOutputParser

from langchain_google_genai import ChatGoogleGenerativeAI

import chromadb

import os
os.environ["HF_HUB_OFFLINE"] = "1" #trying to make it faster
from sentence_transformers import SentenceTransformer

# import sqlite3

# conn = sqlite3.connect("chat_history.db")
# cursor = conn.cursor()

# # See what tables exist
# cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
# print("From Sqlite: ",cursor.fetchall())

import sqlite3  # used below for the user_profiles personalization table

import jwt
import datetime
import uuid
import time
import numpy as np  # CACHE FEATURE ADDITION: used for cosine similarity between query embeddings
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
        return "sbi"  # default fallback for generic/unlabeled FAQ files

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
        description="The actual answer text based strictly on the context")


# the retrieval function: takes user query, embeds it, gets back top k chunks
# CACHE/RELEVANCE ADDITION: accepts an optional pre-computed query_embedding so callers that
# already embedded the query (e.g. for a cache lookup) don't have to embed it a second time.
def retrieve_context(user_query, embedder, collection, top_k=3, query_embedding=None):
    if query_embedding is None:
        query_embedding = embedder.encode([user_query]).tolist()[0] #Converting Text to Numbers (Embedding)

    results = collection.query( #Searching the Vector Database (Find documents that are similar to the embedded user query)
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]  # RELEVANCE ADDITION: need distances to gauge how "banking-related" the query actually is
    )

    #Extracting Text and Metadata 
    chunks = results["documents"][0] #The actual raw text blocks extracted from your bank PDFs/FAQs.
    metadatas = results["metadatas"][0] #Accompanying key-value data about those blocks (like file name or category) that is saved during the ingestion phase.
    distances = results["distances"][0] #RELEVANCE ADDITION: lower distance = the retrieved chunk is a closer semantic match to the query

    #Iterating and Building the Context
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

    # RELEVANCE ADDITION: the closest (smallest) distance among the top_k hits. If even the
    # closest banking document is nowhere near the query, the query is probably off-topic.
    best_distance = min(distances) if distances else float("inf")

    return context, sources, best_distance, query_embedding

 
import re

def clean_json_markdown(ai_message) -> str:
    """Extracts raw JSON content from markdown code fences if present."""
    text = ai_message.content
    # Strip markdown backticks if Gemini wrapped it in ```json ... ```
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        return match.group(1)
    return text

def initialize_llm_interface(): #AQ.Ab8RN6LvQJ36rGezPF4eOkn4KVCKhodsItlszUw-50mfD1SfBg
    # Retrieve the API key from your environment or paste it here directly as a fallback string
    # api_key = os.environ.get("GEMINI_API_KEY", "AQ.Ab8RN6LvQJ36rGezPF4eOkn4KVCKhodsItlszUw-50mfD1SfBg")

    # if not api_key:
    #     raise ValueError(
    #         "CRITICAL: GEMINI_API_KEY environment variable is missing!\n"
    #         "Please set it in your terminal via:\n"
    #         "  Windows (CMD):  set GEMINI_API_KEY=AIzaSyYourKeyHere...\n"
    #         "  Windows (PS):   $env:GEMINI_API_KEY=\"AIzaSyYourKeyHere...\"\n"
    #         "  Linux/macOS:    export GEMINI_API_KEY=\"AIzaSyYourKeyHere...\""
    #     )

    # --- UPDATED: Swapped to ChatGoogleGenerativeAI with gemini-3.5-flash ---
    llm = ChatOllama(model = "llama3.2", temperature = 0.9)
    # llm = ChatGoogleGenerativeAI(
    #     model="gemini-3.5-flash",
    #     temperature=0.9,
    #     google_api_key=api_key,         # Explicitly passing credentials to fix the crash
    #     # Gemini handles JSON structure mapping natively via its response_format configuration
    #     model_kwargs={
    #     "response_format": {"type": "json_object"},
    #     # Force the model to bypass thinking steps for ultra-low latency
    #     "thinking_config": {"thinking_budget": 1} 
    # }
    # )

    # Initialize the output parser tied to our structure template
    output_parser = JsonOutputParser(pydantic_object=BankingBotResponse)

    # Build System Prompt Template injects JSON formatting layout guidelines dynamically
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", (
            "Analyze the given user query against the provided Context Documents.\n\n"
            "CRITICAL INSTRUCTIONS:\n"
            "1. Classify the user query intent as 'account_inquiry' (savings/checking details), 'loan_inquiry' (mortgages/rates), or 'out_of_bounds' (unrelated/general knowledge).\n"
            "2. Compute a mathematical confidence_score (0.0 to 1.0). If the answer is verbatim in the context, score is high (0.9-1.0). If it requires loose interpretation, score is mid (0.5-0.8). If it's absent from context, score is low (0.0-0.4).\n"
            # "3. Provide an answer to the questiob by summarizing the provided context"
            # "3. Just say the first 2 sentences you read in the document. DO NOT waste time reading the rest of the document"
            # "3. Answer the question using ONLY the provided Context.\n" #If absent, reply with the exact phrase: 'I cannot find that information in our current policies.'\n\n"
            # "3. If your response seems financially harmful, respond by saying: 'My answer may be financially harmful. Please consult our official banking policies or press 'Human Escalation' for a human consultant.'\n"
            # "4. Use the User Profile below to personalize your tone and emphasis (e.g. referencing their preferred account type or recent topics). Never invent facts that are not present in the Context.\n"
            "Format your final output instructions:\n{format_instructions}\n\n"
            # "User Profile:\n{user_profile}\n\n"
            "Context Documents:\n{context}"
        )),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{user_query}")
    ])

    # Add partial variable injecting format expectations into system prompt
    prompt_template = prompt_template.partial(format_instructions=output_parser.get_format_instructions())

    # Chain them together (Prompt -> LLM -> JSON Parser)
    llm_chain = prompt_template | llm | clean_json_markdown | output_parser
    return llm_chain


# Memory management for user sessions using SQLite
# Get the absolute path of the directory where LLMInterface.py lives
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# FORCE the database to exist in that exact directory
PROFILE_DB_PATH = os.path.join(BASE_DIR, "chat_history.db")
DB_CONNECTION_STRING = f"sqlite:///{PROFILE_DB_PATH}"
# DB_CONNECTION_STRING = "sqlite:///chat_history.db"

# Reuse one engine across requests instead of letting SQLChatMessageHistory open a fresh
# connection pool every single turn - opening/closing engines repeatedly was adding latency.
from sqlalchemy import create_engine
CHAT_HISTORY_ENGINE = create_engine(DB_CONNECTION_STRING)

# How many recent messages (user+ai combined) to send to the LLM as chat_history.
# Sending the entire unbounded history back on every turn slows down prompt processing
# as a conversation grows, so we cap it to a recent rolling window.
MAX_HISTORY_MESSAGES = 6

# Personalization: user profile storage (preferred account type + recent past queries)
# Reuses the same physical sqlite file as chat history, in its own table.
# PROFILE_DB_PATH = "chat_history.db"
MAX_PAST_QUERIES = 5  # how many recent queries we keep for personalization context


import threading
_profile_conn_lock = threading.Lock()
_PROFILE_CONN = None  # lazily created, reused across requests to avoid reconnect overhead


def _get_profile_db_connection():
    # Reuse one long-lived connection instead of opening/closing a new one every call -
    # repeated connect/close was adding avoidable latency to each turn.
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
        # CACHE FEATURE ADDITION: table backing the semantic query cache (see find_cached_response /
        # store_query_in_cache below). Stores every question we've already answered, along with the
        # embedding, so a near-duplicate question later can be answered instantly without RAG + the LLM.
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
    # Returns the stored personalization profile for a session, or sensible defaults if none exists yet.
    with _profile_conn_lock:
        # FIXED: Calling this helper first ensures the tables are created if they are missing!
        conn = _get_profile_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT preferred_account_type, past_queries FROM user_profiles WHERE session_id = ?",
            (session_id,)
        )
        row = cursor.fetchone()

    if row is None:
        return {"preferred_account_type": None, "past_queries": []}

    preferred_account_type, past_queries_json = row
    past_queries = json.loads(past_queries_json) if past_queries_json else []
    return {"preferred_account_type": preferred_account_type, "past_queries": past_queries}


def update_user_profile(session_id: str, current_query: str, intent: str) -> dict:
    # Updates preferred_account_type based on the classified intent of this turn,
    # and appends current_query to the rolling window of past_queries.
    profile = get_user_profile(session_id)

    past_queries = profile["past_queries"] + [current_query]
    past_queries = past_queries[-MAX_PAST_QUERIES:]  # keep only the most recent queries

    preferred_account_type = profile["preferred_account_type"]
    if intent == "account_inquiry":
        preferred_account_type = "savings/checking account"
    elif intent == "loan_inquiry":
        preferred_account_type = "loan/mortgage account"
    # if intent is "out_of_bounds", leave the existing preferred_account_type untouched

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
    # Turns the stored profile dict into a short block of text the system prompt can inject.
    if not profile["preferred_account_type"] and not profile["past_queries"]:
        return "No prior profile information available for this user yet."

    preferred = profile["preferred_account_type"] or "unknown"
    recent = "; ".join(profile["past_queries"]) if profile["past_queries"] else "none"
    return f"Preferred account type: {preferred}\nRecent topics discussed: {recent}"


# --- CACHE FEATURE ADDITIONS: SEMANTIC QUERY CACHE ---
# If a new question's embedding is nearly identical to one we've already answered, we skip
# RAG retrieval and the LLM call entirely and just reuse the prior answer/sources/confidence.

# Cosine similarity threshold for treating two queries as "the same question". 1.0 = identical
# vectors. Tune this up/down based on how strict you want the duplicate-detection to be.
QUERY_CACHE_SIMILARITY_THRESHOLD = 0.95

# --- RELEVANCE FEATURE ADDITION ---
# If even the closest banking document is farther than this from the query, we treat the
# query as off-topic/out-of-bounds and skip building context + calling the LLM entirely.
# NOTE: Chroma's default distance metric depends on how the collection was created (L2 vs
# cosine) - this value may need tuning against your own banking_kb collection.
IRRELEVANT_QUERY_DISTANCE_THRESHOLD = 1.3


def _cosine_similarity(vec_a, vec_b):
    # Small local helper so two embeddings can be compared without pulling in a heavier ML library.
    a = np.array(vec_a)
    b = np.array(vec_b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def find_cached_response(query_embedding: list):
    # Looks through every previously-answered question's embedding for the closest match.
    # Returns a cached parsed_output-style dict (intent/confidence_score/response/sources)
    # if a near-duplicate question is found above QUERY_CACHE_SIMILARITY_THRESHOLD, else None.
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
        print(f"[CACHE HIT] similarity={best_similarity:.3f} - reusing prior answer, skipping RAG + LLM")
        return best_match

    return None


def store_query_in_cache(query_text: str, query_embedding: list, parsed_output: dict, sources: list):
    # Saves this turn's question + answer so a future near-duplicate question can be
    # answered instantly from cache instead of re-running retrieval and the LLM.
    # query_text itself is stored only for debugging/inspection - matching is done purely on embeddings.
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


def process_user_turn_with_sqlite(session_id: str, current_query: str, banking_bot_chain, embedder, collection, top_k: int = 3):

    t0 = time.time()
    chat_history_db = SQLChatMessageHistory(
        session_id=session_id,
        connection=CHAT_HISTORY_ENGINE
    )
    t1 = time.time()
    print(f"[TIMING]   SQLChatMessageHistory init: {t1 - t0:.2f}s") #Test Run time

    past_messages = chat_history_db.messages[-MAX_HISTORY_MESSAGES:] #cap history window so prompt processing time doesn't grow with the conversation
    t2 = time.time()
    print(f"[TIMING]   fetch past_messages: {t2 - t1:.2f}s") #Test Run Time

    # Fetch this session's personalization profile (preferred account type + recent past queries)
    user_profile = get_user_profile(session_id)
    formatted_profile = format_profile_for_prompt(user_profile)

    # CACHE FEATURE ADDITION: embed the query once up front so we can check the semantic cache,
    # and (if we don't get a hit) reuse this same embedding for the relevance check below instead
    # of embedding the same text twice.
    query_embedding = embedder.encode([current_query]).tolist()[0]

    cached_result = find_cached_response(query_embedding) #This is where we get similarity
    was_cache_hit = cached_result is not None
    t2b = time.time()
    print(f"[TIMING]   cache lookup: {t2b - t2:.2f}s")

    if was_cache_hit:
        # --- CACHE HIT: we've already answered a near-identical question before, so reuse that
        # answer, confidence score, and source documents. RAG retrieval and the LLM are both skipped. ---
        parsed_output = {
            "intent": cached_result["intent"],
            "confidence_score": cached_result["confidence_score"],
            "response": cached_result["response"],
        }
        sources = cached_result["sources"]
    else:
        # RELEVANCE ADDITION: do the vector-store lookup once, and check how close the nearest
        # banking document actually is before deciding whether to build full context / call the LLM.

        t1rag = time.time()
        rag_context, sources, best_distance, _ = retrieve_context(
            current_query, embedder, collection, top_k=top_k, query_embedding=query_embedding
        )
        # rag_context = clean_text(rag_context)
        t2rag = time.time()
        print(f"[TIMING]   RAG: {t2rag - t1rag:.2f}s")

        # mock_rag_context = (
        # "Document: rbi_savings_policy.pdf (Page 3)\n"
        # "The minimum initial deposit required to open a Student Savings Account is $50. "
        # "Account holders under 18 receive an introductory rate of 3.0% APY."
        # )
        # print("HERE IS THE BANKING INFO AHHAHSAD SIAHDJ KASDJSADSA",rag_context)

        if best_distance > IRRELEVANT_QUERY_DISTANCE_THRESHOLD:
            # --- RELEVANCE SHORT-CIRCUIT: nothing in the knowledge base is even remotely close
            # to this query, so it's very likely off-topic/out-of-bounds. Skip calling the LLM
            # with the retrieved context and just answer directly with low confidence. ---
            print(f"[RELEVANCE] best_distance={best_distance:.3f} exceeds threshold {IRRELEVANT_QUERY_DISTANCE_THRESHOLD} - treating as out-of-bounds, skipping LLM call")
            parsed_output = {
                "intent": "out_of_bounds",
                "confidence_score": 0.0,
                "response": "IRRELEVANC" #I cannot find that information in our current policies."
            }
            sources = []  # no context was actually used to answer, so there's nothing to cite
        else:
            try:
                parsed_output = banking_bot_chain.invoke({ #gets the LLM response
                    "context": rag_context, #input context
                    "chat_history": past_messages, #input chat history
                    "user_query": current_query, #input user query
                    # "user_profile": formatted_profile #input personalization profile
                })
            except Exception as e:
                print(f"[WARNING] JSON parsing failed, using fallback response: {e}")
                parsed_output = {
                    "intent": "out_of_bounds",
                    "confidence_score": 0.0,
                    "response": "ERROR"#I cannot find that information in our current policies."
                }
                raise e
    t3 = time.time()
    print(f"[TIMING]   RAG + LLM chain (or cache/relevance shortcut): {t3 - t2b:.2f}s")
    print(f"[DEBUG] response length in chars: {len(str(parsed_output))}")
    chat_history_db.add_user_message(current_query) #add user query and LLM response to the SQLite database for future context [for the specific user session]
    chat_history_db.add_ai_message(parsed_output["response"])
    t4 = time.time()
    print(f"[TIMING]   write messages to DB: {t4 - t3:.2f}s")

    # Persist the updated personalization profile (preferred account type + rolling past queries) for next turn
    updated_profile = update_user_profile(session_id, current_query, parsed_output.get("intent", "out_of_bounds"))
    t5 = time.time()
    print(f"[TIMING]   update user profile: {t5 - t4:.2f}s")

    # CACHE FEATURE ADDITION: remember this question/answer so a future near-duplicate can be
    # served instantly from cache. Skip re-storing if this turn was itself already a cache hit,
    # so we don't keep piling up redundant near-identical rows.
    if not was_cache_hit:
        store_query_in_cache(current_query, query_embedding, parsed_output, sources)

    parsed_output["user_profile"] = updated_profile
    parsed_output["sources"] = sources

    return parsed_output

# import re

# def clean_text(text: str) -> str:
#     """
#     Strips useless noise, extra whitespaces, and repetitive symbols 
#     often left behind by PDF parsers.
#     """
#     # 1. Replace multiple consecutive newlines or spaces with a single space/newline
#     text = re.sub(r'\n+', '\n', text)
#     text = re.sub(r' +', ' ', text)
    
#     # 2. Remove typical PDF garbage (like raw page numbers, footers, or repeating dashes/underscores)
#     text = re.sub(r'[-_]{3,}', '', text)  # removes lines of dashes like '---'
#     text = re.sub(r'(Page\s+\d+\s+of\s+\d+|Confidential|HCLTech Bank)', '', text, flags=re.IGNORECASE)
    
#     return text.strip()

# Re-use your existing connection lock to keep threading safe
# _profile_conn_lock = threading.Lock()

# def delete_user_profile(session_id: str, db_path: str = "chat_history.db") -> dict:
#     """
#     Completely deletes all chat messages and profile details associated 
#     with a given session_id from the database.
    
#     Returns:
#         dict: A status report of the rows deleted.
#     """
#     status = {
#         "session_id": session_id,
#         "messages_deleted": 0,
#         "profiles_deleted": 0,
#         "success": False
#     }
    
#     # Acquire the lock to avoid thread collision / database locking errors
#     with _profile_conn_lock:
#         try:
#             # Set a 30-second timeout to handle pending concurrent write transactions safely
#             conn = sqlite3.connect(db_path, timeout=30)
            
#             with conn: # Context manager automatically handles COMMIT on success / ROLLBACK on error
#                 cursor = conn.cursor()
                
#                 # 1. Clear conversation history (LangChain's default SQL history table is named 'message_store')
#                 cursor.execute("DELETE FROM message_store WHERE session_id = ?", (session_id,))
#                 status["messages_deleted"] = cursor.rowcount
                
#                 # 2. Clear user personalization profile
#                 cursor.execute("DELETE FROM user_profiles WHERE session_id = ?", (session_id,))
#                 status["profiles_deleted"] = cursor.rowcount
                
#             status["success"] = True
#             print(f"[DATABASE] Successfully cleared session {session_id}. "
#                   f"Deleted {status['messages_deleted']} message(s) and {status['profiles_deleted']} profile record(s).")
                  
#         except sqlite3.OperationalError as e:
#             print(f"[DATABASE ERROR] Failed to wipe session {session_id} because database was locked: {e}")
#             status["error"] = str(e)
#         except Exception as e:
#             print(f"[DATABASE ERROR] An unexpected error occurred: {e}")
#             status["error"] = str(e)
            
#     return status
def delete_user_profile(session_id: str, db_path = "chat_history.db") -> None:
    if os.path.exists(db_path):
        os.remove(db_path)
        print("Entire database deleted. It will automatically rebuild empty tables on the next run.")
    # with _profile_conn_lock:
    #     conn = _get_profile_db_connection()
    #     cursor = conn.cursor()
    #     cursor.execute("DELETE FROM user_profiles WHERE session_id = ?", (session_id,))
    #     conn.commit()



##Generating secret tokens for anonymous guest users to access the banking bot without creating an account
# maybe a sign in option can be implemented later

SECRET_KEY = "your-bank-super-secret-key" #placeholder


def generate_anonymous_token():
    # Create a completely random user ID
    guest_id = f"guest_{uuid.uuid4().hex[:10]}"

    payload = {
        "sub": guest_id,
        "iat": datetime.datetime.utcnow(),
        # "exp": datetime.datetime.utcnow() + datetime.timedelta(minutes=45), adds expiration date
        "role": "anonymous_guest"
    }

    # 3. Sign it
    token = jwt.encode(payload, SECRET_KEY, algorithm="HS256")
    return token



# --- Test Execution ---
runTest = True #set to true if you want to test the LLM interface locally without running the FastAPI server.

if __name__ == "__main__" and runTest == True:
    print("Initializing structured local LLM interface layer...")
    banking_bot = initialize_llm_interface()
    memory_buffer = []

    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    chroma_client = chromadb.PersistentClient(path="Chunking/chroma_db")
    collection = chroma_client.get_collection(name="banking_kb")
    current_session = generate_anonymous_token()

    print(f"\nGenerated Anonymous Session Token: {current_session}\n")

    while True:
        print("User Input: ")
        user_query = input().lower()  # clean user input
        if user_query == "":
            user_query = " "  # to avoid errors in the LLM chain if user presses enter without typing anything

        # NOTE: retrieval now happens *inside* process_user_turn_with_sqlite (after the cache
        # check and before the relevance check), so we no longer call retrieve_context here directly.
        out = process_user_turn_with_sqlite(current_session, user_query, banking_bot, embedder, collection, top_k=3)
        print(f"\n[{current_session}] Response: \n{out['response']}")

        print("\n--- Structural JSON Output 1 ---")
        print(json.dumps(out, indent=2))

    # ==========================================
    # TURN 1: Initial Question
    # ==========================================
    # "How much money do I need to start a student account?"
    # ==========================================
    # TURN 2: Follow-up Question (Testing Memory)
    # ==========================================
    # "Great, and what is the interest rate for it?"
    # ==========================================
    # TURN 3: Guardrail Check (Out of Bounds)
    # ==========================================
    # "Can you tell me how to bake chocolate chip cookies?"