import React, { useState, useRef, useEffect } from 'react'; //#http://localhost:5174 // npm run dev

function ChatInterface() {
  const [userInput, setUserInput] = useState('');
  const [chatHistory, setChatHistory] = useState([]);
  const [openMenuIndex, setOpenMenuIndex] = useState(null);
  const [visibleSourcesIndex, setVisibleSourcesIndex] = useState(null);
  const [isLoading, setIsLoading] = useState(false); // Track server loading state
  const scrollRef = useRef(null);
  const TypicalBorderRadius = "12px"; // standard border radius for chat bubbles

  // --- VOICE FEATURE ADDITIONS: STATE ---
  const [isListening, setIsListening] = useState(false); // true while the mic is actively capturing speech
  const [voiceOutputEnabled, setVoiceOutputEnabled] = useState(false); // toggles auto-reading AI responses aloud
  const [speechSupported, setSpeechSupported] = useState(true); // whether this browser supports SpeechRecognition
  const recognitionRef = useRef(null); // holds the single SpeechRecognition instance across renders

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

  // --- VOICE FEATURE ADDITIONS: SET UP SPEECH RECOGNITION (VOICE -> TEXT) ONCE ON MOUNT ---
  useEffect(() => {
    // Chrome/Edge expose this under the "webkit" prefix, some browsers don't support it at all
    const SpeechRecognitionAPI = window.SpeechRecognition || window.webkitSpeechRecognition;

    if (!SpeechRecognitionAPI) {
      // No voice input support in this browser (e.g. Firefox) - hide mic UI gracefully
      console.warn("SpeechRecognition API not supported in this browser. Voice input disabled.");
      setSpeechSupported(false);
      return;
    }

    const recognition = new SpeechRecognitionAPI();
    recognition.continuous = false; // stop automatically after the user pauses speaking
    recognition.interimResults = true; // stream partial results so the textbox updates live as the user talks
    recognition.lang = "en-US";

    recognition.onstart = () => {
      setIsListening(true); //Only set listening to true when browser successfully activates the mic
    };

    // Fires repeatedly as speech is recognized (both interim and final chunks)
    recognition.onresult = (event) => {
      let transcript = "";
      for (let i = 0; i < event.results.length; i++) {
        transcript += event.results[i][0].transcript;
      }
      setUserInput(transcript); // mirror the spoken words into the existing text box, reusing the normal submit flow
    };

    // Fires on any recognition error (e.g. no mic permission, no speech detected)
    recognition.onerror = (event) => {
      console.error("SpeechRecognition error:", event.error);
      setIsListening(false);
    };

    // Fires when the recognition session ends (either naturally or via .stop())
    recognition.onend = () => {
      setIsListening(false);
    };

    recognitionRef.current = recognition;

    // Cleanup: abort any in-progress recognition if the component unmounts
    return () => {
      recognition.abort();
    };
  }, []);

  // --- VOICE FEATURE ADDITIONS: MIC BUTTON HANDLER (STARTS/STOPS LISTENING) ---
  const handleMicToggle = () => {
    const recognition = recognitionRef.current;
    if (!recognition || isLoading) return;

    if (isListening) {
      recognition.stop(); // user clicked the mic again to manually stop
      setIsListening(false);
    } else {
      setUserInput(''); // clear old text so the fresh transcript doesn't append onto stale input
      try {
        recognition.start();
        setIsListening(true);
      } catch (err) {
        // start() throws if recognition is already running - safe to ignore
        console.error("Could not start SpeechRecognition:", err);
      }
    }
  };

  // --- VOICE FEATURE ADDITIONS: TEXT -> SPEECH PLAYBACK FOR AI RESPONSES ---
  const speakText = (text) => {
    if (!window.speechSynthesis || !text) return;

    window.speechSynthesis.cancel(); // stop any currently playing speech before starting new speech
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = "en-US";
    window.speechSynthesis.speak(utterance);
  };

  const handleInputChange = (event) => {
    setUserInput(event.target.value);
  };

  // --- NEW KEYBIND ENTER HANDLER ---
  const handleKeyDown = (event) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault(); // Stop a new blank line from breaking layout inside the textarea
      handleSubmit(event);    // Force form submission
    }
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
      const initResponse = await fetch("http://localhost:8000/api/init-chat");  // localhost:8000 is the default address where the Python FastAPI (from server.py) server runs
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

      // --- VOICE FEATURE ADDITION: SPEAK THE RESPONSE OUT LOUD IF ENABLED ---
      if (voiceOutputEnabled) {
        speakText(data.response);
      }
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
      <div className="d-flex align-items-center justify-content-between">
        <h1>Intelligent Banking Assistant UI</h1>
        {/* VOICE FEATURE ADDITION: master toggle for auto-reading AI responses aloud */}
        <div className="form-check form-switch">
          <input
            className="form-check-input"
            type="checkbox"
            role="switch"
            id="voiceOutputToggle"
            checked={voiceOutputEnabled}
            onChange={() => {
              // Turning it off mid-speech should also stop whatever is currently playing
              if (voiceOutputEnabled) {
                window.speechSynthesis && window.speechSynthesis.cancel();
              }
              setVoiceOutputEnabled((prev) => !prev);
            }}
          />
          <label className="form-check-label small text-muted" htmlFor="voiceOutputToggle">
            🔊 Voice replies
          </label>
        </div>
      </div>

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
                        className="btn btn-sm btn-outline-secondary" //For the options button on the AI response
                        onClick={() => toggleMenu(index)}
                        style = {{borderRadius: TypicalBorderRadius}}
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
                        {/* VOICE FEATURE ADDITION: replay this specific response with text-to-speech */}
                        <button
                          className="btn btn-sm btn-light w-100 text-start rounded-0"
                          onClick={() => {
                            speakText(message.text);
                            setOpenMenuIndex(null);
                          }}
                        >
                          🔊 Read Aloud
                        </button>
                      </div>
                    )}

                    {visibleSourcesIndex === index && (
                      <div className="mt-2 p-2 border rounded bg-light">
                        <h6 className="mb-2 text-dark font-weight-bold">Source Documents</h6>
                        <div className="d-flex flex-wrap gap-2">
                          {message.sources && message.sources.length > 0 ? (
                            message.sources.map((sourceItem, linkIndex) => {
                              // If it's an error fallback string rather than an object
                              if (typeof sourceItem === 'string') {
                                return (
                                  <span key={linkIndex} className="badge bg-secondary text-wrap p-2 text-start d-block w-100">
                                    {sourceItem}
                                  </span>
                                );
                              }
                              // Normal parsed object case
                              return (
                                <a
                                  key={linkIndex}
                                  href={sourceItem.url}
                                  target="_blank"
                                  rel="noopener noreferrer"
                                  className="btn btn-xs btn-outline-primary text-decoration-none text-start p-1 px-2 m-1 bg-white bubble-source"
                                  style={{ borderRadius: "6px", fontSize: "0.75rem", display: "inline-block" }}
                                >
                                  📄 {sourceItem.name || "View Document Reference"}
                                </a>
                              );
                            })
                          ) : (
                            <span className="text-muted small">No source documents cited.</span>
                          )}
                        </div>
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
        {/* VOICE FEATURE ADDITION: Visual listening indicator block, mirrors the typing indicator above */}
        {isListening && (
          <p className="text-danger text-start ps-2 small italic animate-pulse">
            🎤 Listening... speak now
          </p>
        )}
      </div>

      <form onSubmit={handleSubmit} className="mt-3">
        <div
          className="d-flex align-items-stretch"
          style={{ width: "100%"}}
        >
          <textarea //for the text box where you enter user query
            className="form-control flex-grow-1"
            id="humanInput"
            rows="1"
            value={userInput}
            onChange={handleInputChange}
            onKeyDown={handleKeyDown} // <-- KEYBIND LINKED HERE
            placeholder={isLoading ? "Waiting for response..." : "Type your message..."}
            disabled={isLoading} // Lock entry area during network call
            style={{
              width: "700px",
              resize: "none",
              borderRadius: TypicalBorderRadius
            }}
          />

          {/* VOICE FEATURE ADDITION: mic button - click to start/stop dictating the query by voice */}
          {speechSupported && (
            <button
              type="button" // not a submit button - this only toggles listening, it doesn't send the message
              className={`btn ${isListening ? 'btn-danger' : 'btn-outline-secondary'} mx-1`}
              onClick={handleMicToggle}
              disabled={isLoading}
              title={isListening ? "Stop listening" : "Speak your query"}
              style={{
                borderRadius: TypicalBorderRadius,
                width: "50px",
                height: "100%",
                transform: "translateY(-6px)"
              }}
            >
              {isListening ? '⏹️' : '🎤'}
            </button>
          )}

          <button
            type="submit" //for the send button
            className="btn btn-primary"
            disabled={isLoading}
            style={{
              borderRadius: TypicalBorderRadius,
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