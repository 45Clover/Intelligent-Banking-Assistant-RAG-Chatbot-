from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

def initialize_llm_interface():
    # Connect to locally running Llama 3.2 engine
    llm = OllamaLLM(model="llama3.2", temperature=0.0)
    
    # Build the System Prompt Template including a placeholder for Chat History
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", (
            "You are a compliant, secure banking assistant for HCLTech Bank.\n"
            "Use ONLY the provided Context Documents to answer the user's question.\n"
            "If the answer cannot be found in the context, strictly reply: 'I cannot find that information in our current policies.'\n"
            "Do not make up facts or use external knowledge under any circumstance.\n\n"
            "Context Documents:\n{context}"
        )),
        # This dynamically injects the back-and-forth chat history array into the prompt
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{user_query}")
    ])
    
    # Chain them together using LangChain expression language
    llm_chain = prompt_template | llm
    return llm_chain

# --- Execution Simulation ---
if __name__ == "__main__":
    print("Initializing local LLM interface layer with Memory...")
    banking_bot = initialize_llm_interface()
    
    # Initialize an empty list to keep track of the chat history
    memory_buffer = []
    
    # Simulating data retrieved from the ChromaDB RAG database step
    mock_rag_context = (
        "Document: rbi_savings_policy.pdf (Page 3)\n"
        "The minimum initial deposit required to open a Student Savings Account is $50. "
        "Account holders receive an introductory rate of 3.0% APY."
    )
    
    #initial question
    query_1 = "How much money do I need to start a student account?"
    print(f"\nUser Query 1: '{query_1}'")
    
    response_1 = banking_bot.invoke({
        "context": mock_rag_context,
        "chat_history": memory_buffer, # Passing empty list initially
        "user_query": query_1
    })
    
    print("\n--- LLM Response 1 ---")
    print(response_1)
    
    # Update memory buffer manually with Turn 1 data
    memory_buffer.append(HumanMessage(content=query_1))
    memory_buffer.append(AIMessage(content=response_1))
    
    # (Testing Memory)
    # question relies entirely on the context of Turn 1 ("what is the rate for IT?")
    query_2 = "Great, and what is the interest rate for it?"
    print(f"\nUser Query 2: '{query_2}'")
    
    response_2 = banking_bot.invoke({
        "context": mock_rag_context,
        "chat_history": memory_buffer, # Passing the updated history list
        "user_query": query_2
    })
    
    print("\n--- LLM Response 2 ---")
    print(response_2)
    
    # Update memory buffer with Turn 2 data
    memory_buffer.append(HumanMessage(content=query_2))
    memory_buffer.append(AIMessage(content=response_2))