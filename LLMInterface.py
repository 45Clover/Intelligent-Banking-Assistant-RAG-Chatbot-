import json
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import JsonOutputParser


import chromadb
import html
import os

os.environ["HF_HUB_OFFLINE"] = "1"  # trying to make it faster
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
        description="The actual answer text based strictly on the context, or the standard out-of-bounds safety message.")


# the retrieval function: takes user query, embeds it, gets back top k chunks
def retrieve_context(user_query, embedder, collection, top_k=3):
    query_embedding = embedder.encode([user_query]).tolist()  # Converting Text to Numbers (Embedding)
    results = collection.query(
        # Searching the Vector Database (Find documents that are similar to the embedded user query)
        query_embeddings=query_embedding,
        n_results=top_k
    )

    # Extracting Text and Metadata
    chunks = results["documents"][0]  # The actual raw text blocks extracted from your bank PDFs/FAQs.
    metadatas = results["metadatas"][
        0]  # Accompanying key-value data about those blocks (like file name or category) that is saved during the ingestion phase.

    # Iterating and Building the Context
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
    return context, sources
# Manual language override — lets a user force the response language,
# bypassing detect_query_language entirely for that session until changed again.
SESSION_LANGUAGE_OVERRIDE = {}

LANGUAGE_COMMANDS = {
    "/en": "en",
    "/english": "en",
    "/fr": "fr",
    "/french": "fr",
}

def check_language_command(user_query: str):
    """Returns 'en'/'fr' if the query is a language-switch command, else None."""
    return LANGUAGE_COMMANDS.get(user_query.strip().lower())

def resolve_query_language(session_id: str, user_query: str) -> str:
    forced = check_language_command(user_query)
    if forced:
        SESSION_LANGUAGE_OVERRIDE[session_id] = forced
        return forced
    if session_id in SESSION_LANGUAGE_OVERRIDE:
        return SESSION_LANGUAGE_OVERRIDE[session_id]
    return detect_query_language(user_query)

from langdetect import detect_langs
def detect_query_language(user_query: str) -> str:
    try:
        result = detect_langs(user_query)[0]

        # Only trust detection if confidence is reasonable
        if result.prob > 0.90:
            if result.lang == "fr":
                return "fr"
            elif result.lang == "en":
                return "en"

        # fallback
        return "en"

    except:
        return "en"

def initialize_llm_interface():
    llm = ChatOllama(
        model="llama3.2",  # Connect to local Llama 3.2 model
        temperature=0.2,  # low temperature for deterministic output
        format="json",
        num_predict=300,  # caps generation length (caps how many tokens it can generate) — JSON answers don't need more
        num_ctx=2048,  # bounds context so prompt eval doesn't blow up/grow unbounded
        keep_alive="30m",  # keeps the model resident in Ollama so it isn't reloaded from disk between requests
    )

    # Initialize the output parser tied to our structure template
    output_parser = JsonOutputParser(pydantic_object=BankingBotResponse)

    # Build System Prompt Template injects JSON formatting layout guidelines dynamically
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", (
            "You are an automated, compliant banking compliance assistant for HCLTech Bank.\n"
            "Analyze the given user query against the provided Context Documents.\n\n"
            "CRITICAL INSTRUCTIONS:\n"
            "1. Classify the user query intent as 'account_inquiry' (savings/checking details), 'loan_inquiry' (mortgages/rates), or 'out_of_bounds' (unrelated/general knowledge).\n"
            "2. Compute a mathematical confidence_score (0.0 to 1.0). If the answer is verbatim in the context, score is high (0.9-1.0). If it requires loose interpretation, score is mid (0.5-0.8). If it's absent from context, score is low (0.0-0.4).\n"
            "3. Answer the question using ONLY the provided Context. If absent, reply with the exact phrase: 'I cannot find that information in our current policies.'\n\n"
            "4. If your reponse seems financially harmful, respond by saying: 'My answer may be financially harmful. Please consult our official banking policies or press 'Human Escalation' for a human consultant.'\n"
            "5. Use the User Profile below to personalize your tone and emphasis (e.g. referencing their preferred account type or recent topics). Never invent facts that are not present in the Context.\n"
            "6. LANGUAGE RULE:\n"
            "Current required response language: {query_language}\n"
            "STRICT RULE: If language is 'en', output ONLY English.\n"
            "STRICT RULE: If language is 'fr', output ONLY French.\n"
            "Never copy the language from Context Documents.\n"
            "Never copy the language from Chat History.\n"
            "Only follow the Current required response language.\n"
            "Ignore the language of previous assistant messages.\n"
            "Format your final output instructions:\n{format_instructions}\n\n"
            "User Profile:\n{user_profile}\n\n"
            "Context Documents:\n{context}"
        )),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{user_query}")
    ])

    # Add partial variable injecting format expectations into system prompt
    prompt_template = prompt_template.partial(format_instructions=output_parser.get_format_instructions())

    # Chain them together (Prompt -> LLM -> JSON Parser)
    llm_chain = prompt_template | llm | output_parser
    return llm_chain


# Memory management for user sessions using SQLite
DB_CONNECTION_STRING = "sqlite:///chat_history.db"

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
PROFILE_DB_PATH = "chat_history.db"
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
    return _PROFILE_CONN


def get_user_profile(session_id: str) -> dict:
    # Returns the stored personalization profile for a session, or sensible defaults if none exists yet.
    with _profile_conn_lock:
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


def process_user_turn_with_sqlite(session_id: str, current_query: str, banking_bot_chain, rag_context: str,
                                  query_language: str):
    t0 = time.time()
    chat_history_db = SQLChatMessageHistory(
        session_id=session_id,
        connection=CHAT_HISTORY_ENGINE
    )
    t1 = time.time()
    print(f"[TIMING]   SQLChatMessageHistory init: {t1 - t0:.2f}s")  # Test Run time

    past_messages = chat_history_db.messages[
        -MAX_HISTORY_MESSAGES:]  # cap history window so prompt processing time doesn't grow with the conversation
    t2 = time.time()
    print(f"[TIMING]   fetch past_messages: {t2 - t1:.2f}s")  # Test Run Time

    # Fetch this session's personalization profile (preferred account type + recent past queries)
    user_profile = get_user_profile(session_id)
    formatted_profile = format_profile_for_prompt(user_profile)

    try:
        parsed_output = banking_bot_chain.invoke({  # gets the LLM response
            "context": rag_context,  # input context
            "chat_history": past_messages,  # input chat history
            "user_query": current_query,  # input user query
            "user_profile": formatted_profile,  # input personalization profile
            "query_language": query_language  # Add this line
        })
    except Exception as e:
        print(f"[WARNING] JSON parsing failed, using fallback response: {e}")
        parsed_output = {
            "intent": "out_of_bounds",
            "confidence_score": 0.0,
            "response": "I cannot find that information in our current policies."
        }

    t3 = time.time()
    print(f"[TIMING]   LLM chain.invoke: {t3 - t2:.2f}s")
    print(f"[DEBUG] response length in chars: {len(str(parsed_output))}")

    if "response" in parsed_output:
        parsed_output["response"] = html.unescape(parsed_output["response"])
    chat_history_db.add_user_message(
        current_query)  # add user query and LLM response to the SQLite database for future context [for the specific user session]
    chat_history_db.add_ai_message(parsed_output["response"])
    t4 = time.time()
    print(f"[TIMING]   write messages to DB: {t4 - t3:.2f}s")

    # Persist the updated personalization profile (preferred account type + rolling past queries) for next turn
    updated_profile = update_user_profile(session_id, current_query, parsed_output.get("intent", "out_of_bounds"))
    t5 = time.time()
    print(f"[TIMING]   update user profile: {t5 - t4:.2f}s")

    parsed_output["user_profile"] = updated_profile

    return parsed_output


##Generating secret tokens for anonymous guest users to access the banking bot without creating an account
# maybe a sign in option can be implemented later

SECRET_KEY = "your-bank-super-secret-key"  # placeholder


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
runTest = True  # set to true if you want to test the LLM interface locally without running the FastAPI server.

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
        user_query = input()  # clean user input
        if user_query == "":
            user_query = " "  # to avoid errors in the LLM chain if user presses enter without typing anything
        query_lang = resolve_query_language(current_session, user_query)

        if check_language_command(user_query):
            print(f"[Language switched to: {query_lang}]")
            continue
        rag_context, _ = retrieve_context(user_query, embedder, collection, top_k=3)


        print("USER:", user_query)
        print("DETECTED LANGUAGE:", query_lang)
        out = process_user_turn_with_sqlite(current_session, user_query, banking_bot, rag_context, query_lang)

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