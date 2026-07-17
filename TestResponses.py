import json
import os
import time

# --- IMPORT YOUR ACTUAL SYSTEM ---
from LLMInterface import (
    initialize_llm_interface, 
    process_user_turn_with_sqlite,
    retrieve_context,
    detect_query_language
)
from sentence_transformers import SentenceTransformer
import chromadb

# --- INITIALIZATION ---
print("[INIT] Loading banking bot and database...")
banking_bot = initialize_llm_interface()
embedder = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.PersistentClient(path="Chunking/chroma_db")
collection = chroma_client.get_or_create_collection(name="banking_kb")

def load_test_cases(filename="test_rag.json"):
    # if not os.path.exists(filename):
    #     raise FileNotFoundError(f"Missing {filename}! Make sure it is in this folder.")
    # with open(filename, "r", encoding="utf-8") as f:
    #     return json.load(f)
    return [
  {
    "query": "Can a student open a savings account?",
    "answer": "Yes, students can open savings accounts. Many banks offer specialized savings accounts tailored specifically for students to help them start saving early with low minimum deposit requirements and reduced fees."
  },
  {
    "query": "Can I open an account online?",
    "answer": "Yes, you can apply for and open an account online. Most modern banks allow you to submit your application and retrieve status updates directly through their website or mobile application."
  },
  {
    "query": "What types of accounts do you offer?",
    "answer": "Banks offer a diverse range of accounts to suit different financial needs, including checking accounts for daily transactions, savings accounts and Certificates of Deposit (CDs) for earning interest, and specialized student or retirement accounts."
  },
  {
    "query": "What is the difference between a savings account and a checking account?",
    "answer": "A checking account is designed for everyday spending, allowing you to easily deposit money, write checks, and make debit card purchases. A savings account is meant for storing money over time to earn interest, often coming with limitations on frequent transactions."
  },
  {
    "query": "What is the interest rate on a savings account?",
    "answer": "Interest rates fluctuate and depend heavily on the specific type of savings account you choose, as well as prevailing market conditions. You can check the bank's website under their rates section for the most current savings and CD interest rates."
  },
  {
    "query": "Are there withdrawal limits on savings accounts?",
    "answer": "Yes. Savings accounts typically have restrictions on the frequency of withdrawals. For instance, withdrawing funds from certain interest-bearing time accounts like Certificates of Deposit (CDs) before their maturity term can result in an early withdrawal or Regulation D penalty."
  },
  {
    "query": "How can I check my account balance?",
    "answer": "You can check your available balance by signing on to your online banking portal or mobile app. Your available balance is the most current record of funds ready for use, accounting for recent postings, holds, and pending transactions."
  },
  {
    "query": "Can I access my account through mobile banking?",
    "answer": "Yes, you can securely access your account through mobile banking apps. This allows you to check balances, turn your debit card on or off, request replacement cards, and make mobile deposits on eligible devices."
  },
  {
    "query": "What should I do if I forget my banking PIN?",
    "answer": "If you forget your banking PIN, you should log in to your secure online banking portal or mobile app to request a PIN change or reset. Alternatively, you can visit a local physical branch with proper identification, or contact customer service securely. Never share your PIN with anyone."
  },
  {
    "query": "Can you tell me about student accounts?",
    "answer": "Student accounts are specifically designed bank accounts for young adults in school. They usually offer waived monthly service fees, low minimum opening balances, and educational tools to help students manage their money responsibly."
  },
  {
    "query": "Why does the bank require identity verification?",
    "answer": "Banks require identity verification to comply with strict regulatory laws, prevent identity theft, stop financial fraud, and ensure that they are opening accounts for the true owners of the provided identities."
  },
  {
    "query": "What is KYC?",
    "answer": "KYC stands for 'Know Your Customer'. It is a mandatory regulatory process where banks verify the identity, address, and legal status of their clients before or during the time they start doing business to prevent money laundering and fraud."
  },
  {
    "query": "Can I transfer money to Canada?",
    "answer": "Yes, you can transfer money to Canada using international wire transfers, global money transfer services, or specialized international remittance programs offered by your bank."
  },
  {
    "query": "Are there fees for international transfers?",
    "answer": "Yes, international transfers usually incur processing fees and currency exchange conversion costs. These charges vary depending on the destination country, the transfer method, and the specific banking fees associated with your account."
  },
  {
    "query": "Comment puis-je contacter le service clientèle ?",
    "answer": "Vous pouvez contacter notre service clientèle par téléphone, vous connecter à notre site web ou à notre application mobile pour obtenir de l'aide, ou encore vous rendre directement dans l'une de nos succursales."
  },
  {
    "query": "Quelles sont les politiques de confidentialité de la banque ?",
    "answer": "Les politiques de confidentialité de la banque décrivent comment nous recueillons, utilisons, partageons et protégeons vos informations personnelles et financières conformément aux réglementations de protection des données en vigueur."
  },
  {
    "query": "Comment puis-je verrouiller temporairement ma carte de débit ?",
    "answer": "Si vous avez égaré votre carte, vous pouvez la désactiver temporairement pour empêcher toute transaction non autorisée en vous connectant à votre espace en ligne ou à l'application mobile et en accédant à l'option d'activation/désactivation de la carte."
  },
  {
    "query": "Comment puis-je changer mon code PIN ?",
    "answer": "Vous pouvez facilement changer votre code PIN en ligne via notre site web, en utilisant notre application mobile de manière sécurisée, ou en vous rendant à un guichet automatique de la banque."
  },
  {
    "query": "Comment activer ma nouvelle carte de débit ?",
    "answer": "Pour activer votre nouvelle carte de débit, vous pouvez généralement effectuer un premier achat ou un retrait en saisissant votre code PIN dans un guichet automatique, l'activer en ligne via votre espace client, ou appeler le numéro d'activation sécurisé fourni avec la carte."
  },
  {
    "query": "Puis-je obtenir une carte de débit ?",
    "answer": "Oui, vous pouvez obtenir une carte de débit. Elle est généralement délivrée automatiquement lors de l'ouverture d'un compte courant (checking) ou sur demande pour vous permettre d'effectuer des achats et de retirer des espèces."
  }
]

# --- DETERMINISTIC EVALUATION METRICS ---
def calculate_word_overlap(text1: str, text2: str) -> float:
    """Calculates simple Jaccard similarity (word overlap ratio) between two texts."""
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    if not words1 or not words2:
        return 0.0
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    return len(intersection) / len(union)

def evaluate_guardrail(query: str, response: str, expected: str) -> bool:
    """
    Checks if an out-of-bounds query correctly triggered the safety fallback 
    instead of leaking information.
    """
    safety_triggers = ["cannot find", "not in our current policies", "je ne trouve pas"]
    is_out_of_bounds = any(trigger in expected.lower() for trigger in safety_triggers)
    
    if is_out_of_bounds:
        # Pass if the bot successfully refused to answer
        return any(trigger in response.lower() for trigger in safety_triggers)
    return True # Skip safety verification for normal banking queries

# --- RUN THE EVALUATION ENGINE ---
def run_evaluation():
    test_cases = load_test_cases()
    results = []
    
    passed_count = 0
    total_tests = len(test_cases)
    
    print(f"\n[START] Starting evaluation on {total_tests} test cases...")
    print("-" * 60)

    for idx, case in enumerate(test_cases, 1):
        query = case["query"]
        expected = case["answer"]#["expected_output"]
        
        # 1. Run through your actual RAG system
        start_time = time.time()
        
        rag_context, sources = retrieve_context(query, embedder, collection,
                                                top_k=3)  # identify the user to get their specific chat history

        query_lang = detect_query_language(query)

        bot_output = process_user_turn_with_sqlite(  # invoke the LLM chain to get the response
            session_id=f"eval_session_{idx}",  # we want the output that is tied to a specific user
            current_query=query,  # input user query
            banking_bot_chain=banking_bot,  # input the initialized LLM chain
            rag_context=rag_context,  # input the retrieved context
            query_language=query_lang  # input the detected language of the query
        )
        # bot_output = process_user_turn_with_sqlite(
        #     session_id=f"eval_session_{idx}",
        #     current_query=query,
        #     banking_bot_chain=banking_bot,
        #     embedder=embedder,
        #     collection=collection,
        #     query_language="en" if "Comment" not in query else "fr"
        # )
        
        actual_response = bot_output.get("response", "")
        latency = time.time() - start_time
        
        # 2. Score the Output
        overlap_score = calculate_word_overlap(actual_response, expected)
        guardrail_passed = evaluate_guardrail(query, actual_response, expected)
        
        # Define pass/fail criteria:
        # A test passes if the safety guardrails worked AND we got a reasonable overlap score (e.g., > 15% keyword match)
        is_pass = guardrail_passed and (overlap_score > 0.15 or "cannot find" in expected)
        
        if is_pass:
            passed_count += 1
            status = "✅ PASS"
        else:
            status = "❌ FAIL"
            
        print(f"[{idx}/{total_tests}] {status} | Latency: {latency:.2f}s | Query: {query[:40]}...")
        if not is_pass:
            print(f"   ↳ Expected: {expected}")
            print(f"   ↳ Got:      {actual_response}")
            
        results.append({
            "query": query,
            "expected": expected,
            "actual": actual_response,
            "overlap_score": overlap_score,
            "pass": is_pass,
            "latency": latency
        })

    # --- RENDER RESULTS SUMMARY ---
    print("=" * 60)
    print("                    EVALUATION REPORT")
    print("=" * 60)
    print(f"Total Tests Run: {total_tests}")
    print(f"Passed:          {passed_count}")
    print(f"Failed:          {total_tests - passed_count}")
    print(f"Success Rate:    {(passed_count / total_tests) * 100:.1f}%")
    print(f"Average Latency: {sum(r['latency'] for r in results)/total_tests:.2f}s")
    print("=" * 60)

if __name__ == "__main__":
    run_evaluation()\
    

#1 pass
#2 fail
#3 pass
#4 pass
#5 fail
#6 fail
#7 Pass
#8 pass
#9 fail
#10 fail
#11 fail
#12 pass
#13 fail
#14 faii
#15 pass
#16 pass
#17 fail
#18 fail
#19 fail
#20 fail