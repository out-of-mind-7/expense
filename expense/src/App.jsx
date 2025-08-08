import React, { useState, useEffect, useRef } from "react";

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [socket, setSocket] = useState(null);
  const chatRef = useRef(null);

  useEffect(() => {
    const ws = new WebSocket("ws://localhost:8000/ws");

    ws.onopen = () => {
      setMessages((prev) => [...prev, { from: "system", text: "✅ Connected to server." }]);
    };

    ws.onmessage = (event) => {
      setMessages((prev) => [...prev, { from: "bot", text: event.data }]);
    };

    // ws.onerror = () => {
    //   setMessages((prev) => [...prev, { from: "system", text: "❌ Connection error." }]);
    // };

    // ws.onclose = () => {
    //   setMessages((prev) => [...prev, { from: "system", text: "🛑 Disconnected from server." }]);
    // };

    setSocket(ws);
    return () => ws.close();
  }, []);

  useEffect(() => {
    if (chatRef.current) {
      chatRef.current.scrollTop = chatRef.current.scrollHeight;
    }
  }, [messages]);

  const sendMessage = () => {
    const trimmed = input.trim();
    if (socket && socket.readyState === WebSocket.OPEN && trimmed) {
      socket.send(trimmed);
      setMessages((prev) => [...prev, { from: "user", text: trimmed }]);
      setInput("");
    }
  };

  return (
    <div style={{ padding: "20px", maxWidth: "600px", margin: "auto", fontFamily: "Arial" }}>
      <h2>💬 Expense Tracker Chatbot</h2>
      <div ref={chatRef} style={{ border: "1px solid #ccc", padding: "10px", height: "300px", overflowY: "auto", backgroundColor: "#f9f9f9" }}>
        {messages.map((msg, index) => (
          <div key={index} style={{ marginBottom: "8px", color: msg.from === "system" ? "#888" : "#000" }}>
            <strong>{msg.from}:</strong> {msg.text}
          </div>
        ))}
      </div>
      <div style={{ marginTop: "10px", display: "flex", gap: "6px" }}>
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && sendMessage()}
          placeholder="Type a message..."
          style={{ flex: 1, padding: "8px" }}
        />
        <button
          onClick={sendMessage}
          style={{ padding: "8px 12px", backgroundColor: "#007bff", color: "#fff", border: "none", cursor: "pointer" }}
        >
          Send
        </button>
      </div>
    </div>
  );
}

export default App;