from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

def initialize_llm_interface():
    # Connect to locally running Llama 3.2 engine
    # Setting temperature=0.0 stops the model from being creative, making it reliable for banking data (eliminate hallucinations)
    llm = OllamaLLM(model="llama3.2", temperature=0.0)
    
    # Build the System Prompt Template
    # instructs the model on its identity, guardrails, and how to treat your context documents.
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", (
            "You are a compliant, secure banking assistant for HCLTech Bank.\n"
            "Use ONLY the provided Context Documents to answer the user's question.\n"
            "If the answer cannot be found in the context, strictly reply: 'I cannot find that information in our current policies.'\n"
            "Do not make up facts or use external knowledge under any circumstance.\n\n"
            "Context Documents:\n{context}"
        )),
        ("human", "{user_query}")
    ])
    
    # Chain them together using LangChain expression language
    llm_chain = prompt_template | llm
    return llm_chain

# --- Execution Simulation ---
if __name__ == "__main__":
    print("Initializing local LLM interface layer...")
    banking_bot = initialize_llm_interface()
    
    # PLACEHOLDER - Simulatating data retrieved from the ChromaDB RAG database step
    mock_rag_context = (
        "Document: rbi_savings_policy.pdf (Page 3)\n"
        "The minimum initial deposit required to open a Student Savings Account is $50. "
        "Account holders under 18 receive an introductory rate of 3.0% APY."
    )
    
    # Take user input
    user_question = "How much money do I need to start a student account?"
    print(f"\nUser Query: '{user_question}'")
    print("Processing (running entirely on your machine)...")
    
    # Run the query through the interface chain
    response = banking_bot.invoke({
        "context": mock_rag_context,
        "user_query": user_question
    })
    
    print("\n--- LLM Response ---")
    print(response)
    
    # Test guardrails with a question NOT in the context
    print("\n--- Testing Out-of-Bounds Guardrail ---")
    bad_question = "What is the capital of France?"
    print(f"User Query: '{bad_question}'")
    
    guardrail_response = banking_bot.invoke({
        "context": mock_rag_context,
        "user_query": bad_question
    })
    print(guardrail_response)