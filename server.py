import uuid
import datetime
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import jwt

import chromadb
from sentence_transformers import SentenceTransformer

# Import the logic you already created in your prior scripts
from LLMInterface import initialize_llm_interface, process_user_turn_with_sqlite, retrieve_context, generate_anonymous_token

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"], # Or ["*"] to allow everything for local dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

banking_bot = initialize_llm_interface()

embedder = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.PersistentClient(path="Chunking/chroma_db")
collection = chroma_client.get_collection(name="banking_kb")
current_session = generate_anonymous_token()



class ChatPayload(BaseModel):
    user_query: str

SECRET_KEY = "your-bank-super-secret-key" #place holder

# === FIX 1: ADD THIS MISSING ENDPOINT ===
@app.get("/api/init-chat")
async def init_chat():
    guest_id = f"guest_{uuid.uuid4().hex[:10]}"
    payload = {
        "sub": guest_id,
        "iat": datetime.timezone.utc() #token is permanent
        # "exp": datetime.datetime.utcnow() + datetime.timedelta(minutes=45)
    }
    token = jwt.encode(payload, SECRET_KEY, algorithm="HS256")
    return {"token": token}

@app.post("/api/chat")
async def chat_endpoint(payload: ChatPayload, authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        if authorization != None: print(authorization)
        else: print(authorization)
        raise HTTPException(status_code=401, detail="Missing or invalid token")
        
    try:
        token = authorization.split(" ")[1]
        decoded_payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        existing_guest_id = decoded_payload["sub"]
        
        rag_context = retrieve_context(payload.user_query, embedder, collection, top_k=3)

        out = process_user_turn_with_sqlite(
            session_id=existing_guest_id,
            current_query=payload.user_query,
            banking_bot_chain=banking_bot,
            rag_context=rag_context
        )
        
        # === FIX 2: ENSURE FRONTEND EXPECTED KEYS ARE EXPLICITLY RETURNED ===
        return {
            "response": out.get("response", "No response generated."),
            "confidence_score": out.get("confidence_score", 0.0),
            "sources": out.get("sources", ["Mock Document Context Layer"])
        }
        
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")
    
#first cd to the repo (for Chris) and then run what is below
#python -m uvicorn server:app --reload --port 8000
#ctrl + c to stop server.py