import React, { useState } from 'react'; //#http://localhost:5174/

function ChatInterface() {
  const [userInput, setUserInput] = useState('');
  const [aiOutput, setAiOutput] = useState('');
  const [chatHistory, setChatHistory] = useState([]);
  const [confidenceScore, setConfidenceScore] = useState(0);
  const [sourceLinks, setSourceLinks] = useState([]);

  const handleInputChange = (event) => {
    setUserInput(event.target.value);
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    // Simulate AI response for demonstration
    // In a real application, you'd call your backend here
    const simulatedResponse = {
      text: `You asked: ${userInput}. Here's a simulated AI response.`,
      confidence: Math.random(), // Simulate a random confidence score
      sources: ['https://example.com/source1', 'https://example.com/source2'], // Simulated source links
    };

    setAiOutput(simulatedResponse.text);
    setConfidenceScore(simulatedResponse.confidence);
    setSourceLinks(simulatedResponse.sources);
    setChatHistory([...chatHistory, { user: userInput, ai: simulatedResponse.text }]);
    setUserInput('');
  };

  const handleEscalate = () => {
    // In a real application, you'd trigger your human hand-off logic here
    console.log('Escalating to human hand-off...');
    alert('Escalating to human hand-off... (Simulation)');
  };

  return (
    <div className="container mt-5">
      <h1>Chat UI</h1>

      <div className="row mt-4">
        <div className="col-md-6">
          <h3>Human Input</h3>
          <form onSubmit={handleSubmit}>
            <div className="mb-3">
              <textarea
                className="form-control"
                id="humanInput"
                rows="5"
                value={userInput}
                onChange={handleInputChange}
                placeholder="Enter your query here..."
              ></textarea>
            </div>
            <button type="submit" className="btn btn-primary">
              Send
            </button>
          </form>
        </div>

        <div className="col-md-6">
          <h3>AI Output</h3>
          <div className="mb-3">
            <textarea
              className="form-control"
              id="aiOutput"
              rows="5"
              value={aiOutput}
              readOnly
              placeholder="AI response will appear here..."
            ></textarea>
          </div>
          <div className="row">
            <div className="col-md-6">
              <p>Confidence Score: {confidenceScore.toFixed(2)}</p>
            </div>
            <div className="col-md-6 text-end">
              <button className="btn btn-warning" onClick={handleEscalate}>
                Escalate to Human
              </button>
            </div>
          </div>
          <div className="mt-3">
            <h5>Source Documents</h5>
            <ul>
              {sourceLinks.map((link, index) => (
                <li key={index}>
                  <a href={link} target="_blank" rel="noopener noreferrer">
                    {link}
                  </a>
                </li>
              ))}
            </ul>
          </div>
        </div>
      </div>

      <div className="mt-5">
        <h3>Chat History</h3>
        <div className="border p-3 rounded h-400 overflow-auto">
          {chatHistory.map((message, index) => (
            <div key={index} className={`mb-3 ${message.user ? 'text-end' : ''}`}>
              <div
                className={`d-inline-block p-2 rounded ${
                  message.user ? 'bg-primary text-white' : 'bg-light'
                }`}
              >
                {message.user ? `You: ${message.user}` : `AI: ${message.ai}`}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

export default ChatInterface;