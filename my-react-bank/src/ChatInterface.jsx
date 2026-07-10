import React, { useState, useRef, useEffect } from 'react'; //#http://localhost:5174 // npm run dev

function ChatInterface() {
  const [userInput, setUserInput] = useState('');
  const [chatHistory, setChatHistory] = useState([]);
  const [openMenuIndex, setOpenMenuIndex] = useState(null);
  const [visibleSourcesIndex, setVisibleSourcesIndex] = useState(null);
  const [isLoading, setIsLoading] = useState(false); // Track server loading state
  const scrollRef = useRef(null);

  // --- STEP 3 MODIFICATION: INITIALIZE ANONYMOUS GUEST SESSION ON LAUNCH ---
  useEffect(() => {
    const initializeSession = async () => {
      let savedToken = localStorage.getItem("chat_token");
      
      // If no token exists in the browser storage, request a new one from the server
      if (!savedToken) {
        try {
          console.log("No guest session token found. Fetching a new anonymous token...");
          const response = await fetch("http://localhost:8000/api/init-chat"); // Make sure this endpoint exists on your FastAPI server
          if (response.ok) {
            const data = await response.json();
            localStorage.setItem("chat_token", data.token);
            console.log("Successfully initialized guest token.");
          }
        } catch (error) {
          console.error("Failed to initialize anonymous guest token:", error);
        }
      } else {
        console.log("Welcome back! Reusing valid token from local storage.");
      }
    };

    initializeSession();
  }, []);

  const handleInputChange = (event) => {
    setUserInput(event.target.value);
  };

  // --- STEP 3 MODIFICATION: CONVERT MOCK SUBMIT TO REAL ASYNC BACKEND CALL ---
 const handleSubmit = async (event) => {
    event.preventDefault();
    if (!userInput.trim() || isLoading) return;

    const currentQuery = userInput;
    setUserInput('');
    setIsLoading(true);

    setChatHistory((prev) => [
      ...prev,
      { role: 'user', text: currentQuery },
    ]);

    const callChatApi = async (token) => {
      return fetch("http://localhost:8000/api/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Authorization": `Bearer ${token || ""}`
        },
        body: JSON.stringify({ user_query: currentQuery }),
      });
    };

    const fetchNewToken = async () => {
      const initResponse = await fetch("http://localhost:8000/api/init-chat");
      if (!initResponse.ok) {
        throw new Error("Failed to fetch a new session token");
      }
      const initData = await initResponse.json();
      localStorage.setItem("chat_token", initData.token);
      return initData.token;
    };

    try {
      let savedToken = localStorage.getItem("chat_token") || "";

      // If we have no token at all, get one before even trying
      if (!savedToken) {
        savedToken = await fetchNewToken();
      }

      let response = await callChatApi(savedToken);

      // If the token was expired/invalid, refresh it once and retry
      if (response.status === 401) {
        console.log("Token expired or invalid — fetching a new one and retrying...");
        savedToken = await fetchNewToken();
        response = await callChatApi(savedToken);
      }

      if (!response.ok) {
        throw new Error(`Server responded with HTTP error status: ${response.status}`);
      }

      const data = await response.json();

      setChatHistory((prev) => [
        ...prev,
        {
          role: 'ai',
          text: data.response,
          confidence: data.confidence_score || 0.00,
          sources: data.sources || [],
        },
      ]);
    } catch (error) {
      console.error("Failed to fetch real LLM response:", error);

      setChatHistory((prev) => [
        ...prev,
        {
          role: 'ai',
          text: "System Error: Unable to establish connection with the secure banking backend pipeline.",
          confidence: 0,
          sources: ["Network Firewall / Offline Connection Error"],
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleEscalate = (index) => {
    // In a real application, you'd trigger your human hand-off logic here
    console.log('Escalating to human hand-off...', index);
    alert('Escalating to human hand-off... (Simulation)');
    setOpenMenuIndex(null);
  };

  const toggleSources = (index) => {
    setVisibleSourcesIndex((prev) => (prev === index ? null : index));
    setOpenMenuIndex(null);
  };

  const toggleMenu = (index) => {
    setOpenMenuIndex((prev) => (prev === index ? null : index));
  };

  // Auto-scroll to the latest message whenever the conversation updates
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [chatHistory, isLoading]);

  return (
    <div className="container mt-5" style={{ maxWidth: '800px' }}>
      <h1>Intelligent Banking Assistant UI</h1>

      <div
        ref={scrollRef}
        className="border rounded p-3 mt-4"
        style={{ height: '480px', overflowY: 'auto', backgroundColor: '#e0effb' }}
      >
        {chatHistory.length === 0 && (
          <p className="text-muted text-center mt-5">
            Your secure conversation will appear here. Ask a query regarding bank policies or internet setups.
          </p>
        )}

        {chatHistory.map((message, index) => {
          const isUser = message.role === 'user';
          return (
            <div
              key={index}
              className={`d-flex mb-3 ${isUser ? 'justify-content-end' : 'justify-content-start'}`}
            >
              <div style={{ maxWidth: '75%' }}>
                <div
                  className={`p-2 rounded ${
                    isUser ? 'bg-primary text-white' : 'bg-white border'
                  }`}
                >
                  {message.text}
                </div>

                {!isUser && (
                  <div className="mt-1 position-relative">
                    <div className="d-flex align-items-center justify-content-between">
                      <small className="text-muted">
                        Confidence: {message.confidence ? message.confidence.toFixed(2) : "0.00"}
                      </small>
                      <button
                        className="btn btn-sm btn-outline-secondary"
                        onClick={() => toggleMenu(index)}
                        aria-haspopup="true"
                        aria-expanded={openMenuIndex === index}
                      >
                        Options &#9662;
                      </button>
                    </div>

                    {openMenuIndex === index && (
                      <div
                        className="border rounded bg-white shadow-sm mt-1 position-absolute"
                        style={{ right: 0, zIndex: 10, minWidth: '200px' }}
                      >
                        <button
                          className="btn btn-sm btn-warning w-100 text-start rounded-0"
                          onClick={() => handleEscalate(index)}
                        >
                          Escalate to Human
                        </button>
                        <button
                          className="btn btn-sm btn-light w-100 text-start rounded-0"
                          onClick={() => toggleSources(index)}
                        >
                          {visibleSourcesIndex === index
                            ? 'Hide Source Documents'
                            : 'Show Source Documents'}
                        </button>
                      </div>
                    )}

                    {visibleSourcesIndex === index && (
                      <div className="mt-2 p-2 border rounded bg-light">
                        <h6 className="mb-1">Source Documents</h6>
                        <ul className="mb-0 ps-3">
                          {message.sources.map((src, linkIndex) => (
                            <li key={linkIndex}>
                              {src.url ? (
                                <a href={src.url} target="_blank" rel="noopener noreferrer">
                                  {src.name}
                                </a>
                              ) : (
                                <span className="text-secondary">{src.name}</span>
                              )}
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>
          );
        })}
        {/* Visual Typing indicator block */}
        {isLoading && (
          <p className="text-muted text-start ps-2 small italic animate-pulse">
            Assistant is consulting database...
          </p>
        )}
      </div>

      <form onSubmit={handleSubmit} className="mt-3">
        <div
          className="d-flex align-items-stretch"
          style={{ width: "100%"}}
        >
          <textarea
            className="form-control flex-grow-1"
            id="humanInput"
            rows="1"
            value={userInput}
            onChange={handleInputChange}
            placeholder={isLoading ? "Waiting for response..." : "Type your message..."}
            disabled={isLoading} // Lock entry area during network call
            style={{
              width: "700px",
              resize: "none",
              borderTopRightRadius: 0,
              borderBottomRightRadius: 0,
            }}
          />

          <button
            type="submit"
            className="btn btn-primary"
            disabled={isLoading}
            style={{
              borderTopLeftRadius: 0,
              borderBottomLeftRadius: 0,
              width: "90px",
              height: "100%",
              transform: "translateY(-6px)"
            }}
          >
            Send
          </button>
        </div>
      </form>

    </div>
  );
}

export default ChatInterface;
