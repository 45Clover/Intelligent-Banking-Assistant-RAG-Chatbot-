import uuid
import datetime
import time
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import jwt

import chromadb
from sentence_transformers import SentenceTransformer

from LLMInterface import initialize_Ollm_interface, initialize_Cllm_interface, process_user_turn_with_sqlite, retrieve_context, generate_anonymous_token

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#testing run time of the LLM interface
t0 = time.time()
# response_bot = initialize_Ollm_interface()
banking_bot = initialize_Cllm_interface() #get the LLM chain initialized and ready to process user queries
print(f"[STARTUP] initialize_llm_interface: {time.time() - t0:.2f}s", flush=True )

t1 = time.time()
embedder = SentenceTransformer("all-MiniLM-L6-v2")
print(f"[STARTUP] SentenceTransformer load: {time.time() - t1:.2f}s", flush=True)

t2 = time.time()
chroma_client = chromadb.PersistentClient(path="Chunking/chroma_db")
collection = chroma_client.get_collection(name="banking_kb")
print(f"[STARTUP] Chroma client + collection: {time.time() - t2:.2f}s", flush=True)

t3 = time.time()
current_session = generate_anonymous_token()
print(f"[STARTUP] generate_anonymous_token: {time.time() - t3:.2f}s", flush=True)


class ChatPayload(BaseModel):
    user_query: str

SECRET_KEY = "your-bank-super-secret-key"


@app.get("/api/init-chat")
async def init_chat():
    guest_id = f"guest_{uuid.uuid4().hex[:10]}"
    payload = {
        "sub": guest_id,
        "iat": datetime.datetime.utcnow(),
        #"exp": datetime.datetime.utcnow() + datetime.timedelta(minutes=45) only add for expiration
    }
    token = jwt.encode(payload, SECRET_KEY, algorithm="HS256")
    return {"token": token}


def triggerQuadrails(parsedOutput):
    response = parsedOutput.get("response", "No response generated.")
    confidenceScore = parsedOutput.get("confidence_score", 0.0)

    if confidenceScore < 0.2 and response != "My answer may be financially harmful. Please press 'Source Documents' to refer to official banking policies or press 'Human Escalation' for a human consultant.":
        #if the user has a low confidence score and the response is not already a warning, replace it with a warning message
        pass # response = "I am uncertain about my answer. Please press 'Source Documents' to refer to official banking policies or press 'Human Escalation' for a human consultant."
    
    return response, confidenceScore


@app.post("/api/chat")
async def chat_endpoint(payload: ChatPayload, authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid token")

    try:
        token = authorization.split(" ")[1]
        decoded_payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        existing_guest_id = decoded_payload["sub"]

        t0 = time.time()
        
        # --- UPDATED TO MATCH LLMInterface.py GOALS ---
        # The new interface performs cache lookups, relevance threshold checks, and context 
        # retrieval internally. We now simply invoke process_user_turn_with_sqlite, passing 
        # the embedder, collection, and top_k directly.
        out = process_user_turn_with_sqlite(
            session_id=existing_guest_id,      # we want the output that is tied to a specific user
            current_query=payload.user_query,   # input user query
            banking_bot_chain=banking_bot,      # input the initialized LLM chain
            embedder=embedder,                  # passed through to handle internal embedding/caching
            collection=collection,              # passed through to handle internal vector lookups
            top_k=3
        )
        
        t1 = time.time()
        print(f"[TIMING] process_user_turn_with_sqlite (Cache + RAG + LLM): {t1 - t0:.2f}s")
        print(f"[TIMING] TOTAL: {t1 - t0:.2f}s")

        quadrailedResponse, quadrailedConfidenceScore = triggerQuadrails(out)

        return {
            "response": quadrailedResponse,
            "confidence_score": quadrailedConfidenceScore,
            "sources": out.get("sources", []), # Extracted directly from the unified process output
            "user_profile": out.get("user_profile", {"preferred_account_type": None, "past_queries": []})
        }

    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

#python -m uvicorn server:app --reload --port 8000