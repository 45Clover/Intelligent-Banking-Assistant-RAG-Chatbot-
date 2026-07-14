import React, { useState, useRef, useEffect } from 'react'; //#http://localhost:5174 // npm run dev
import './ChatInterface.css';
function ChatInterface() {
  const [userInput, setUserInput] = useState('');
  const [chatHistory, setChatHistory] = useState([]);
  const [selectedIndex, setSelectedIndex] = useState(null); // which AI message's details are shown in the sidebar
  const [isLoading, setIsLoading] = useState(false); // Track server loading state
  const scrollRef = useRef(null);

  // --- VOICE FEATURE ADDITIONS: STATE ---
  const [isListening, setIsListening] = useState(false); // true while the mic is actively capturing speech
  const [voiceOutputEnabled, setVoiceOutputEnabled] = useState(false); // toggles auto-reading AI responses aloud
  const [speechSupported, setSpeechSupported] = useState(true); // whether this browser supports SpeechRecognition
  const recognitionRef = useRef(null); // holds the single SpeechRecognition instance across renders

  // --- STEP 3 MODIFICATION: INITIALIZE ANONYMOUS GUEST SESSION ON LAUNCH ---
  useEffect(() => {
    const initializeSession = async () => {
      let savedToken = localStorage.getItem("chat_token");

      if (!savedToken) {
        try {
          console.log("No guest session token found. Fetching a new anonymous token...");
          const response = await fetch("http://localhost:8000/api/init-chat");
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

  // --- VOICE FEATURE ADDITIONS: SET UP SPEECH RECOGNITION (VOICE -> TEXT) ONCE ON MOUNT ---
  useEffect(() => {
    const SpeechRecognitionAPI = window.SpeechRecognition || window.webkitSpeechRecognition;

    if (!SpeechRecognitionAPI) {
      console.warn("SpeechRecognition API not supported in this browser. Voice input disabled.");
      setSpeechSupported(false);
      return;
    }

    const recognition = new SpeechRecognitionAPI();
    recognition.continuous = false;
    recognition.interimResults = true;
    recognition.lang = "en-US";

    recognition.onstart = () => {
      setIsListening(true);
    };

    recognition.onresult = (event) => {
      let transcript = "";
      for (let i = 0; i < event.results.length; i++) {
        transcript += event.results[i][0].transcript;
      }
      setUserInput(transcript);
    };

    recognition.onerror = (event) => {
      console.error("SpeechRecognition error:", event.error);
      setIsListening(false);
    };

    recognition.onend = () => {
      setIsListening(false);
    };

    recognitionRef.current = recognition;

    return () => {
      recognition.abort();
    };
  }, []);

  const handleMicToggle = () => {
    const recognition = recognitionRef.current;
    if (!recognition || isLoading) return;

    if (isListening) {
      recognition.stop();
      setIsListening(false);
    } else {
      setUserInput('');
      try {
        recognition.start();
        setIsListening(true);
      } catch (err) {
        console.error("Could not start SpeechRecognition:", err);
      }
    }
  };

  const speakText = (text) => {
    if (!window.speechSynthesis || !text) return;

    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = "en-US";
    window.speechSynthesis.speak(utterance);
  };

  const handleInputChange = (event) => {
    setUserInput(event.target.value);
  };

  const handleKeyDown = (event) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      handleSubmit(event);
    }
  };

  // NEW FEATURE: True Backend Memory Erasure
  // --- BACKEND MEMORY ERASURE HANDLER (TOTAL STATE RESET) ---
  const handleEraseMemory = async () => {
    const savedToken = localStorage.getItem("chat_token");
    if (!savedToken) return;

    try {
      // setIsLoading(true);

      // 1. Force stop the microphone if it's currently listening
      if (recognitionRef.current && isListening) {
        try {
          recognitionRef.current.abort();
        } catch (e) {
          console.error("Failed to abort speech recognition:", e);
        }
      }
      setIsListening(false); // Explicitly clear microphone state flag

      const response = await fetch("http://localhost:8000/api/clear-history", {
        method: "POST",
        headers: { 
          "Authorization": `Bearer ${savedToken}` 
        }
      });

      if (!response.ok) {
        console.error("Failed to clear backend database memory.");
      }
    } catch (error) {
      console.error("Network error clearing history:", error);
    } finally {
      // 2. Clear ALL locking mechanisms immediately before the alert fires
      setIsLoading(false);
      setIsListening(false);
      setUserInput(''); // Reset text box value to empty string

      // 3. Clear local state arrays
      setChatHistory([]);
      setSelectedIndex(null);

      // 4. Trigger the alert inside a safe timeout window
      setTimeout(() => {
        alert("Success: The AI's conversational memory database has been wiped clean!");
      }, 50);
    }
  };

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

      if (!savedToken) {
        savedToken = await fetchNewToken();
      }

      let response = await callChatApi(savedToken);

      if (response.status === 401) {
        console.log("Token expired or invalid — fetching a new one and retrying...");
        savedToken = await fetchNewToken();
        response = await callChatApi(savedToken);
      }

      if (!response.ok) {
        throw new Error(`Server responded with HTTP error status: ${response.status}`);
      }

      const data = await response.json();

      setChatHistory((prev) => {
        const updated = [
          ...prev,
          {
            role: 'ai',
            text: data.response,
            confidence: data.confidence_score || 0.00,
            sources: data.sources || [],
          },
        ];
        setSelectedIndex(updated.length - 1);
        return updated;
      });

      if (voiceOutputEnabled) {
        speakText(data.response);
      }
    } catch (error) {
      console.error("Failed to fetch real LLM response:", error);

      setChatHistory((prev) => {
        const updated = [
          ...prev,
          {
            role: 'ai',
            text: "System Error: Unable to establish connection with the secure banking backend pipeline.",
            confidence: 0,
            sources: ["Network Firewall / Offline Connection Error"],
          },
        ];
        setSelectedIndex(updated.length - 1);
        return updated;
      });
    } finally {
      setIsLoading(false);
    }
  };

  const handleEscalate = (index) => {
    console.log('Escalating to human hand-off...', index);
    alert('Escalating to human hand-off... (Simulation)');
  };

  const handleClearHistory = () => {
    setChatHistory([]);
    setSelectedIndex(null);
  };

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [chatHistory, isLoading]);

  const selectedMessage = selectedIndex !== null ? chatHistory[selectedIndex] : null;

  return (
    <div className="app-shell">
      {/* top accent bar */}
      <div className="top-accent-bar" />

      <div className="app-body">
        {/* --- LEFT SIDEBAR --- */}
        <aside className="sidebar">
          <div className="sidebar-section">
            <div className="sidebar-eyebrow">Banking Assistant</div>
            <div className="sidebar-title">Chat and Information Console</div>
          </div>

          <div className="sidebar-divider" />

          <div className="sidebar-section d-flex align-items-center justify-content-between">
            <label className="sidebar-label mb-0" htmlFor="voiceOutputToggle">
              🔊 Voice replies
            </label>
            <div className="form-check form-switch mb-0">
              <input
                className="form-check-input"
                type="checkbox"
                role="switch"
                id="voiceOutputToggle"
                checked={voiceOutputEnabled}
                onChange={() => {
                  if (voiceOutputEnabled) {
                    window.speechSynthesis && window.speechSynthesis.cancel();
                  }
                  setVoiceOutputEnabled((prev) => !prev);
                }}
              />
            </div>
          </div>

          <div className="sidebar-divider" />

          <div className="sidebar-section flex-grow-1" style={{ overflowY: 'auto' }}>
            <div className="sidebar-label mb-2">Response Details</div>
            {selectedMessage ? (
              <>
                <p className="small mb-3" style={{ color: '#ccc' }}>
                  Confidence: {selectedMessage.confidence ? selectedMessage.confidence.toFixed(2) : '0.00'}
                </p>

                <div className="mb-2 fw-semibold small" style={{ color: '#999' }}>Source Documents</div>
                <div className="d-flex flex-column gap-2 mb-3">
                  {selectedMessage.sources && selectedMessage.sources.length > 0 ? (
                    selectedMessage.sources.map((sourceItem, linkIndex) => {
                      if (typeof sourceItem === 'string') {
                        return (
                          <span key={linkIndex} className="badge bg-secondary text-wrap p-2 text-start">
                            {sourceItem}
                          </span>
                        );
                      }
                      return (
                        <a
                          key={linkIndex}
                          href={sourceItem.url}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="btn btn-sm btn-outline-warning text-decoration-none text-start"
                        >
                          📄 {sourceItem.name || "View Document Reference"}
                        </a>
                      );
                    })
                  ) : (
                    <span className="small" style={{ color: '#888' }}>No source documents cited.</span>
                  )}
                </div>

                <button
                  className="btn btn-sm btn-warning w-100 mb-2"
                  onClick={() => handleEscalate(selectedIndex)}
                >
                  Escalate to Human
                </button>
                <button
                  className="btn btn-sm btn-outline-light w-100"
                  onClick={() => speakText(selectedMessage.text)}
                >
                  🔊 Read Aloud
                </button>
              </>
            ) : (
              <p className="small" style={{ color: '#888' }}>
                Click an assistant response to see its confidence score and sources here.
              </p>
            )}
          </div>

          <div className="sidebar-divider" />

          <button className="btn-clear-history" onClick={handleClearHistory}>
            Clear chat history
          </button>

          {/* NEW BUTTON PLACED RIGHT UNDERNEATH */}
          <button 
            className="btn-clear-history" 
            onClick={handleEraseMemory}
            disabled={isLoading}
            title="Erase AI conversational memory completely"
          >
            🧠 Erase AI Memory
          </button>
          
        </aside>

        {/* --- MAIN PANEL --- */}
        <main className="main-panel">
          <div ref={scrollRef} className="main-scroll">
            {chatHistory.length === 0 ? (
              <div className="empty-state">
                <div className="empty-state-icon"></div>
                <h1 className="empty-state-title">Intelligent Banking Assistant</h1>
                <p className="empty-state-sub">
                  Ask any relevant questions and I'll pull from the knowledge base.
                </p>
              </div>
            ) : (
              chatHistory.map((message, index) => {
                const isUser = message.role === 'user';
                const isSelected = !isUser && selectedIndex === index;
                return (
                  <div
                      key={index}
                      style={{
                        display: 'flex',
                        width: '100%',
                        marginBottom: '1.5rem',
                        justifyContent: isUser ? 'flex-end' : 'flex-start',
                      }}
                    >
                      <div
                        style={{
                          display: 'flex',
                          flexDirection: 'column',
                          maxWidth: '60%',
                          alignItems: isUser ? 'flex-end' : 'flex-start',
                        }}
                      >
                        <small className="mb-1 fw-semibold" style={{ color: '#999' }}>
                          {isUser ? 'You' : 'Assistant'}
                        </small>
                        <div
                          className={`bubble-${isUser ? 'user' : 'ai'}`}
                          onClick={() => !isUser && setSelectedIndex(index)}
                          style={{
                            padding: '10px 16px',
                            cursor: !isUser ? 'pointer' : 'default',
                            outline: isSelected ? '2px solid var(--brand-yellow)' : 'none',
                          }}
                        >
                          {message.text}
                        </div>
                      </div>
                    </div>
                );
              })
            )}

            {isLoading && (
              <p className="text-start ps-2 small fst-italic animate-pulse" style={{ color: '#888' }}>
                Assistant is consulting database...
              </p>
            )}
            {isListening && (
              <p className="text-start ps-2 small fst-italic animate-pulse" style={{ color: '#e05252' }}>
                🎤 Listening... speak now
              </p>
            )}
          </div>

          {/* --- PINNED INPUT BAR --- */}
          <form onSubmit={handleSubmit} className="input-bar-wrapper">
            <div className="input-bar">
              <textarea
                className="input-bar-textarea"
                id="humanInput"
                rows="1"
                value={userInput}
                onChange={handleInputChange}
                onKeyDown={handleKeyDown}
                placeholder={isLoading ? "Waiting for response..." : "Type your question here..."}
                disabled={isLoading}
              />
              

              {speechSupported && (
                <button
                  type="button"
                  className="input-bar-icon-btn"
                  onClick={handleMicToggle}
                  disabled={isLoading}
                  title={isListening ? "Stop listening" : "Speak your query"}
                  style={{ color: isListening ? 'var(--brand-yellow)' : '#999' }}
                >
                  {isListening ? '🎧' : '🎙️'}
                </button>
              )}

              <button
                type="submit"
                className="input-bar-send-btn"
                disabled={isLoading}
                title="Send"
              >
                ➤
              </button>
            </div>
          </form>
        </main>
      </div>
    </div>
  );
}

export default ChatInterface;