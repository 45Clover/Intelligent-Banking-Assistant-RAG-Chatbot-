import React, { useState, useRef, useEffect } from 'react'; //#http://localhost:5174 // npm run dev

function ChatInterface() {
  const [userInput, setUserInput] = useState('');
  const [chatHistory, setChatHistory] = useState([]);
  const [openMenuIndex, setOpenMenuIndex] = useState(null);
  const [visibleSourcesIndex, setVisibleSourcesIndex] = useState(null);
  const scrollRef = useRef(null);

  const handleInputChange = (event) => {
    setUserInput(event.target.value);
  };

  const handleSubmit = (event) => {
    event.preventDefault();
    if (!userInput.trim()) return;

    // Simulate AI response for demonstration
    // In a real application, you'd call your backend here
    const simulatedResponse = {
      text: `You asked: ${userInput}. Here's a simulated AI response.`,
      confidence: Math.random(), // Simulate a random confidence score
      sources: ['https://example.com/source1', 'https://example.com/source2'], // Simulated source links
    };

    setChatHistory((prev) => [
      ...prev,
      { role: 'user', text: userInput },
      {
        role: 'ai',
        text: simulatedResponse.text,
        confidence: simulatedResponse.confidence,
        sources: simulatedResponse.sources,
      },
    ]);
    setUserInput('');
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
  }, [chatHistory]);

  return (
    <div className="container mt-5" style={{ maxWidth: '800px' }}>
      <h1>Chat UI</h1>

      <div
        ref={scrollRef}
        className="border rounded p-3 mt-4"
        style={{ height: '480px', overflowY: 'auto', backgroundColor: '#e0effb' }}
      >
        {chatHistory.length === 0 && (
          <p className="text-muted text-center mt-5">
            Your conversation will appear here.
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
                        Confidence: {message.confidence.toFixed(2)}
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
                          {message.sources.map((link, linkIndex) => (
                            <li key={linkIndex}>
                              <a href={link} target="_blank" rel="noopener noreferrer">
                                {link}
                              </a>
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
      </div>

      <form onSubmit={handleSubmit} className="mt-3">
        <div
          className="d-flex align-items-stretch"
          style={{ width: "100%"}}
        >
          <textarea
            className="formControl flex-grow-1"
            id="humanInput"
            rows="1"
            value={userInput}
            onChange={handleInputChange}
            placeholder="Type your message..."
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
