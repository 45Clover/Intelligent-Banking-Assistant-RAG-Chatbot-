import re
from abc import ABC, abstractmethod
from typing import List

class BaseChunker(ABC):
    @abstractmethod
    def split_text(self, text: str) -> List[str]: #function should return a list of strings, each string being a chunk of the original text
        raise NotImplementedError("Subclasses must implement split_text")

class SmallerContextChunker(BaseChunker): #subclass of BaseChunker that implements the split_text method to create smaller chunks with overlap
    def __init__(self, max_chunk_size: int = 150, overlap_size: int = 30):
        """
        :param max_chunk_size: Max length of a chunk (approx characters or words)
        :param overlap_size: How much context to carry over to the next chunk
        """
        self.max_chunk_size = max_chunk_size
        self.overlap_size = overlap_size

    def split_text(self, text: str) -> List[str]:
        # Clean up excessive newlines but keep rough boundaries
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Split text into rough "words" to simulate token/word-based small limits
        words = text.split(' ')
        chunks = []
        
        start_idx = 0
        while start_idx < len(words):
            # Define the end point for this smaller chunk
            end_idx = min(start_idx + self.max_chunk_size, len(words))
            
            # Extract and join the slice
            chunk_words = words[start_idx:end_idx]
            chunk_text = " ".join(chunk_words)
            chunks.append(chunk_text)
            
            # If we reached the end of the document, break
            if end_idx == len(words):
                break
                
            # Slide the window forward, but step back by the overlap amount
            # to preserve semantic context for the LLM
            start_idx = end_idx - self.overlap_size
            
            # Safety check to avoid infinite loops if parameters are bad
            if start_idx >= end_idx:
                start_idx = end_idx
                
        return chunks


# --- Example Usage ---
runExample = False #set to true if you want to test the chunker locally without running the FastAPI server.

if __name__ == "__main__" and runExample == True:
    sample_document = (
        "Retrieval-Augmented Generation (RAG) is a technique that enhances LLM accuracy "
        "by pulling relevant data from an external knowledge base. When chunks are too large, "
        "the LLM gets overwhelmed with noise. By breaking down the data into highly specific, "
        "smaller pieces, the model can synthesize information faster and with fewer hallucinations. "
        "Adding an overlap ensures no vital semantic details are dropped across boundaries."
    )
    
    # Configure it for tiny chunks (e.g., max 15 words, overlapping by 4 words)
    micro_chunker = SmallerContextChunker(max_chunk_size=15, overlap_size=4)
    result_chunks = micro_chunker.split_text(sample_document)
    
    print(f"Total chunks created: {len(result_chunks)}\n")
    for i, chunk in enumerate(result_chunks):
        print(f"Chunk {i+1}: \"{chunk}\"")