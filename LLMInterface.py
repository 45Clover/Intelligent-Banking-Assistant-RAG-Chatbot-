import json
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import JsonOutputParser

import chromadb

import os
os.environ["HF_HUB_OFFLINE"] = "1" #trying to make it faster
from sentence_transformers import SentenceTransformer


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
    query_embedding = embedder.encode([user_query]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k
    )

    chunks = results["documents"][0]
    metadatas = results["metadatas"][0]


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


def initialize_llm_interface():
    # Connect to local Llama 3.2 model
    llm = ChatOllama(model="llama3.2", temperature=0.2)  # 0 temperature for deterministic output

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
            "Format your final output instructions:\n{format_instructions}\n\n"
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


def process_user_turn_with_sqlite(session_id: str, current_query: str, banking_bot_chain, rag_context: str):
    t0 = time.time()
    chat_history_db = SQLChatMessageHistory(
        session_id=session_id,
        connection=DB_CONNECTION_STRING
    )
    t1 = time.time()
    print(f"[TIMING]   SQLChatMessageHistory init: {t1 - t0:.2f}s")

    past_messages = chat_history_db.messages
    t2 = time.time()
    print(f"[TIMING]   fetch past_messages: {t2 - t1:.2f}s")

    try:
        parsed_output = banking_bot_chain.invoke({
            "context": rag_context,
            "chat_history": past_messages,
            "user_query": current_query
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
    chat_history_db.add_user_message(current_query)
    chat_history_db.add_ai_message(parsed_output["response"])
    t4 = time.time()
    print(f"[TIMING]   write messages to DB: {t4 - t3:.2f}s")

    return parsed_output


##Generating secret tokens for anonymous guest users to access the banking bot without creating an account
# maybe a sign in option can be implemented later

SECRET_KEY = "your-bank-super-secret-key" #placeholder


def generate_anonymous_token():
    # Create a completely random user ID
    guest_id = f"guest_{uuid.uuid4().hex[:10]}"

    # Define payload with an expiration (45 minutes)
    payload = {
        "sub": guest_id,
        "iat": datetime.datetime.utcnow(),
        "exp": datetime.datetime.utcnow() + datetime.timedelta(minutes=45),
        "role": "anonymous_guest"
    }

    # 3. Sign it
    token = jwt.encode(payload, SECRET_KEY, algorithm="HS256")
    return token

runTest = False #set to true if you want to test the LLM interface locally without running the FastAPI server.
# --- Test Execution ---
if __name__ == "__main__" and runTest == True:
    print("Initializing structured local LLM interface layer...")
    banking_bot = initialize_llm_interface()
    memory_buffer = []

    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    chroma_client = chromadb.PersistentClient(path="Chunking/chroma_db")
    collection = chroma_client.get_collection(name="banking_kb")
    current_session = generate_anonymous_token()

    print(f"Generated Anonymous Session Token: {current_session}\n")

    while True:
        print("User Input: ")
        user_query = input().lower()  # clean user input
        if user_query == "":
            user_query = " "  # to avoid errors in the LLM chain if user presses enter without typing anything

        rag_context, _ = retrieve_context(user_query, embedder, collection, top_k=3)

        out = process_user_turn_with_sqlite(current_session, user_query, banking_bot, rag_context)
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