import json
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import JsonOutputParser

import nltk
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer
from nltk.tokenize import word_tokenize

import jwt
import datetime
import uuid

# 1. Define the Structured JSON Schema using Pydantic
class BankingBotResponse(BaseModel):
    intent: str = Field(description="The classified intent. Must be exactly one of: 'account_inquiry', 'loan_inquiry', or 'out_of_bounds'.")
    confidence_score: float = Field(description="A value between 0.00 and 1.00 indicating how matching the context is to the query.")
    response: str = Field(description="The actual answer text based strictly on the context, or the standard out-of-bounds safety message.")

def initialize_llm_interface():
    # Connect to local Llama 3.2 model
    llm = ChatOllama(model="llama3.2", temperature=0.2) #0 temperature for deterministic output
    
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

#Memory management for user sessions using SQLite
DB_CONNECTION_STRING = "sqlite:///chat_history.db"

def process_user_turn_with_sqlite(session_id: str, current_query: str, banking_bot_chain, rag_context: str):
    # 1. Automatically connect or initialize the SQLite table mapping for this user ID
    chat_history_db = SQLChatMessageHistory(
        session_id=session_id,
        connection=DB_CONNECTION_STRING
    )
    
    # 2. Extract their existing historical messages into a standard list format for your LLM chain
    past_messages = chat_history_db.messages
    
    # 3. Fire the query through your structured banking bot chain, passing the retrieved historical messages
    parsed_output = banking_bot_chain.invoke({
        "context": rag_context,
        "chat_history": past_messages,
        "user_query": current_query
    })
    
    # 4. Commit the new turn data directly into the database so it records permanently
    chat_history_db.add_user_message(current_query)
    chat_history_db.add_ai_message(parsed_output["response"])
    
    return parsed_output


##Generating secret tokens for anonymous guest users to access the banking bot without creating an account
#maybe a sign in option can be implemented later

SECRET_KEY = "your-bank-super-secret-key"

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

# --- Execution ---
if __name__ == "__main__":
    print("Initializing structured local LLM interface layer...")
    banking_bot = initialize_llm_interface()
    memory_buffer = []
    
    mock_rag_context = (
        "Document: rbi_savings_policy.pdf (Page 3)\n"
        "The minimum initial deposit required to open a Student Savings Account is $50. "
        "Account holders receive an introductory rate of 3.0% APY."
        "Investor accounts can be opened with a minimum of $1000 and have an interest rate of 2.5% APY.\n\n"
    )

    current_session = generate_anonymous_token()
    print(f"Generated Anonymous Session Token: {current_session}\n")

    while True:
        print("User Input: ")
        user_query = input().lower() #clean user input
        if user_query == "":
            user_query = " " #to avoid errors in the LLM chain if user presses enter without typing anythig
        
        out = process_user_turn_with_sqlite(current_session, user_query, banking_bot, mock_rag_context)
        print(f"\n[{current_session}] Response: \n{out['response']}")
        
        print("\n--- Structural JSON Output 1 ---")
        print(json.dumps(out, indent=2))
    
    # ==========================================
    # TURN 1: Initial Question
    # ==========================================
    #"How much money do I need to start a student account?"
    # ==========================================
    # TURN 2: Follow-up Question (Testing Memory)
    # ==========================================
    #"Great, and what is the interest rate for it?"
    # ==========================================
    # TURN 3: Guardrail Check (Out of Bounds)
    # ==========================================
    #"Can you tell me how to bake chocolate chip cookies?"