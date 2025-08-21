import React, { useState, useEffect, useRef, useCallback } from "react";

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [socket, setSocket] = useState(null);
  const [expenses, setExpenses] = useState([]);
  const [activeTab, setActiveTab] = useState("chat");
  const [loading, setLoading] = useState(false);
  const [connectionStatus, setConnectionStatus] = useState("connecting");
  const [isReconnecting, setIsReconnecting] = useState(false);
  const [newExpense, setNewExpense] = useState({
    date: new Date().toISOString().split('T')[0],
    amount: "",
    category: "",
    note: ""
  });

  const chatRef = useRef(null);
  const userId = useRef(`user_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`);
  const reconnectTimeout = useRef(null);
  const pingInterval = useRef(null);
  const isComponentMounted = useRef(true);
  const connectionAttempts = useRef(0);
  const maxReconnectAttempts = 5;
  const isConnecting = useRef(false);
  const currentSocket = useRef(null);
  
  const API_BASE = "http://localhost:8000";

  // Enhanced cleanup function
  const cleanupConnection = useCallback(() => {
    console.log("🧹 Cleaning up connection...");
    
    if (reconnectTimeout.current) {
      clearTimeout(reconnectTimeout.current);
      reconnectTimeout.current = null;
      console.log("🧹 Cleared reconnect timeout");
    }
    
    if (pingInterval.current) {
      clearInterval(pingInterval.current);
      pingInterval.current = null;
      console.log("🧹 Cleared ping interval");
    }
    
    isConnecting.current = false;
  }, []);

  // Force close any existing WebSocket connections
  const forceCloseSocket = useCallback((reason = "Force close") => {
    if (currentSocket.current) {
      console.log(`🔌 Force closing existing socket: ${reason}`);
      try {
        currentSocket.current.close(1000, reason);
      } catch (error) {
        console.warn("Warning closing socket:", error);
      }
      currentSocket.current = null;
    }
    
    if (socket) {
      console.log(`🔌 Force closing state socket: ${reason}`);
      try {
        socket.close(1000, reason);
      } catch (error) {
        console.warn("Warning closing state socket:", error);
      }
      setSocket(null);
    }
  }, [socket]);

  const connectWebSocket = useCallback(() => {
    // Prevent multiple simultaneous connections
    if (isConnecting.current || !isComponentMounted.current) {
      console.log("🚫 Already connecting or component unmounted");
      return;
    }

    // Force close any existing connections first
    forceCloseSocket("New connection starting");
    
    // Clear any existing intervals/timeouts
    cleanupConnection();

    isConnecting.current = true;
    console.log("🔌 Creating new WebSocket connection...");
    setConnectionStatus("connecting");
    
    const ws = new WebSocket("ws://localhost:8000/ws");
    currentSocket.current = ws;

    ws.onopen = () => {
      if (!isComponentMounted.current || currentSocket.current !== ws) {
        console.log("🚫 Aborting connection - component unmounted or socket replaced");
        ws.close(1000, "Component unmounted or socket replaced");
        isConnecting.current = false;
        return;
      }

      console.log("✅ WebSocket connected successfully");
      setConnectionStatus("connected");
      setIsReconnecting(false);
      connectionAttempts.current = 0;
      isConnecting.current = false;
      
      setSocket(ws);

      const userInfoMessage = {
        type: "user_info",
        data: {
          user_id: userId.current,
          timestamp: new Date().toISOString()
        }
      };
      
      try {
        ws.send(JSON.stringify(userInfoMessage));
        console.log("📤 Sent user info:", userInfoMessage);
      } catch (error) {
        console.error("❌ Failed to send user info:", error);
      }

      if (!pingInterval.current) {
        console.log("🏓 Starting new ping interval");
        pingInterval.current = setInterval(() => {
          if (!isComponentMounted.current) {
            console.log("🧹 Component unmounted, stopping ping");
            if (pingInterval.current) {
              clearInterval(pingInterval.current);
              pingInterval.current = null;
            }
            return;
          }
          
          if (currentSocket.current !== ws || ws.readyState !== WebSocket.OPEN) {
            console.log("🔌 Socket mismatch or not open, stopping ping");
            if (pingInterval.current) {
              clearInterval(pingInterval.current);
              pingInterval.current = null;
            }
            return;
          }

          const pingMessage = {
            type: "ping",
            data: {
              user_id: userId.current,
              timestamp: new Date().toISOString()
            }
          };
          
          try {
            ws.send(JSON.stringify(pingMessage));
            console.log("🏓 Ping sent at", new Date().toISOString());
          } catch (error) {
            console.error("❌ Failed to send ping:", error);
            if (pingInterval.current) {
              clearInterval(pingInterval.current);
              pingInterval.current = null;
            }
          }
        }, 30000);
      }
    };

    ws.onmessage = (event) => {
      if (!isComponentMounted.current || currentSocket.current !== ws) {
        console.log("🚫 Ignoring message - component unmounted or socket replaced");
        return;
      }

      console.log("📥 Raw message received:", event.data);
      
      try {
        const data = JSON.parse(event.data);
        console.log("📥 Parsed JSON message:", data);
        
        if (data.type === "system_message") {
          setMessages((prev) => [...prev, { 
            from: "system", 
            text: data.data.message || data.data,
            timestamp: new Date(data.data.timestamp || new Date()),
            messageType: "system"
          }]);
        } else if (data.type === "ai_response") {
          setMessages((prev) => [...prev, { 
            from: "bot", 
            text: data.data.message || data.data.response || data.data,
            timestamp: new Date(data.data.timestamp || new Date()),
            messageType: "ai"
          }]);
        } else if (data.type === "expense_added") {
          setMessages((prev) => [...prev, { 
            from: "system", 
            text: `✅ Expense added: ₹${(data.data.amount / 100).toFixed(2)} for ${data.data.category}`,
            timestamp: new Date(data.data.timestamp || new Date()),
            messageType: "expense"
          }]);
        } else if (data.type === "error") {
          setMessages((prev) => [...prev, { 
            from: "system", 
            text: `❌ Error: ${data.data.message || data.message || "Unknown error"}`,
            timestamp: new Date(),
            messageType: "error"
          }]);
        } else if (data.type === "pong") {
          console.log("🏓 Pong received - connection alive");
        } else if (data.type === "user_info_received") {
          console.log("✅ User info acknowledged by server");
        } else {
          const messageText = data.data?.message || data.message || JSON.stringify(data);
          setMessages((prev) => [...prev, { 
            from: "bot", 
            text: messageText,
            timestamp: new Date(),
            messageType: "basic"
          }]);
        }
      } catch (parseError) {
        console.log("📥 Plain text message:", event.data);
        if (isComponentMounted.current && currentSocket.current === ws) {
          setMessages((prev) => [...prev, { 
            from: "bot", 
            text: event.data,
            timestamp: new Date(),
            messageType: "basic"
          }]);
        }
      }
    };

    ws.onerror = (error) => {
      console.error("❌ WebSocket error:", error);
      isConnecting.current = false;
      
      if (!isComponentMounted.current || currentSocket.current !== ws) return;
      
      setConnectionStatus("error");
      setMessages((prev) => [...prev, { 
        from: "system", 
        text: "❌ Connection error occurred", 
        timestamp: new Date(),
        messageType: "error"
      }]);
    };

    ws.onclose = (event) => {
      console.log("🔌 WebSocket closed:", event.code, event.reason);
      isConnecting.current = false;
      
      if (currentSocket.current === ws) {
        currentSocket.current = null;
        setSocket(null);
      }
      
      cleanupConnection();
      
      if (!isComponentMounted.current) {
        console.log("🔌 Component unmounted, skipping reconnect");
        return;
      }
      
      setConnectionStatus("disconnected");
      
      if (event.code !== 1000 && connectionAttempts.current < maxReconnectAttempts && !isReconnecting) {
        connectionAttempts.current++;
        
        setMessages((prev) => [...prev, { 
          from: "system", 
          text: `🔌 Connection lost. Reconnecting in 3 seconds... (Attempt ${connectionAttempts.current}/${maxReconnectAttempts})`, 
          timestamp: new Date(),
          messageType: "system"
        }]);
        
        setIsReconnecting(true);
        
        const delay = Math.min(3000 * Math.pow(2, connectionAttempts.current - 1), 30000);
        
        reconnectTimeout.current = setTimeout(() => {
          if (isComponentMounted.current && connectionStatus !== "connected") {
            setIsReconnecting(false);
            connectWebSocket();
          }
        }, delay);
      } else if (connectionAttempts.current >= maxReconnectAttempts) {
        setMessages((prev) => [...prev, { 
          from: "system", 
          text: "❌ Maximum reconnection attempts reached. Please refresh the page.", 
          timestamp: new Date(),
          messageType: "error"
        }]);
        setIsReconnecting(false);
      }
    };
  }, [forceCloseSocket, cleanupConnection, isReconnecting, connectionStatus]);

  // Initialize WebSocket with proper cleanup
  useEffect(() => {
    console.log("🚀 Component mounting, initializing WebSocket...");
    isComponentMounted.current = true;
    
    const initTimeout = setTimeout(() => {
      if (isComponentMounted.current) {
        connectWebSocket();
      }
    }, 100);

    return () => {
      console.log("🧹 Component unmounting, cleaning up...");
      isComponentMounted.current = false;
      
      clearTimeout(initTimeout);
      cleanupConnection();
      forceCloseSocket("Component unmounting");
    };
  }, []);

  // Auto-scroll chat
  useEffect(() => {
    if (chatRef.current) {
      chatRef.current.scrollTop = chatRef.current.scrollHeight;
    }
  }, [messages]);

  // Load expenses when expenses tab is opened
  useEffect(() => {
    if (activeTab === "expenses") {
      loadExpenses();
    }
  }, [activeTab]);

  // Manual reconnect with better duplicate prevention
  const manualReconnect = useCallback(() => {
    if (!isComponentMounted.current || isConnecting.current) {
      console.log("🚫 Cannot reconnect - already connecting or unmounted");
      return;
    }
    
    console.log("🔄 Manual reconnect requested");
    connectionAttempts.current = 0;
    setIsReconnecting(false);
    
    forceCloseSocket("Manual reconnect");
    cleanupConnection();
    
    setConnectionStatus("connecting");
    
    setTimeout(() => {
      if (isComponentMounted.current && !isConnecting.current) {
        connectWebSocket();
      }
    }, 500);
  }, [forceCloseSocket, cleanupConnection, connectWebSocket]);

  // Enhanced message sending
  const sendMessage = useCallback(() => {
    const trimmed = input.trim();
    if (!trimmed) return;

    if (!isComponentMounted.current || !socket || socket.readyState !== WebSocket.OPEN) {
      setMessages((prev) => [...prev, { 
        from: "system", 
        text: "❌ Not connected to server. Please wait for reconnection.", 
        timestamp: new Date(),
        messageType: "error"
      }]);
      return;
    }

    const message = {
      type: "chat_message",
      data: {
        message: trimmed,
        user_id: userId.current,
        timestamp: new Date().toISOString()
      }
    };

    try {
      socket.send(JSON.stringify(message));
      console.log("📤 Sent message:", message);
      
      setMessages((prev) => [...prev, { 
        from: "user", 
        text: trimmed, 
        timestamp: new Date(),
        messageType: "user"
      }]);
      
      setInput("");
    } catch (error) {
      console.error("❌ Failed to send message:", error);
      setMessages((prev) => [...prev, { 
        from: "system", 
        text: "❌ Failed to send message. Please try again.", 
        timestamp: new Date(),
        messageType: "error"
      }]);
    }
  }, [input, socket]);

  const loadExpenses = async () => {
    setLoading(true);
    try {
      const response = await fetch(`${API_BASE}/api/expenses?limit=20`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
        mode: 'cors',
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      
      if (data.success) {
        setExpenses(data.expenses || []);
        console.log(`✅ Loaded ${data.expenses?.length || 0} expenses`);
      } else {
        console.error("Failed to load expenses:", data);
        setMessages((prev) => [...prev, { 
          from: "system", 
          text: `❌ Failed to load expenses: ${data.message || "Unknown error"}`, 
          timestamp: new Date(),
          messageType: "error"
        }]);
      }
    } catch (error) {
      console.error("Failed to load expenses:", error);
      const errorMessage = error.message.includes('CORS') 
        ? "❌ CORS error: Please check if the backend server is running and configured properly."
        : `❌ Failed to load expenses: ${error.message || "Network error"}`;
      
      setMessages((prev) => [...prev, { 
        from: "system", 
        text: errorMessage, 
        timestamp: new Date(),
        messageType: "error"
      }]);
    } finally {
      setLoading(false);
    }
  };

  // Helper function to format date for API
  const formatDateForAPI = (dateInput) => {
    if (!dateInput) return new Date().toISOString().split('T')[0];
    
    if (typeof dateInput === 'string') {
      if (/^\d{4}-\d{2}-\d{2}$/.test(dateInput)) {
        return dateInput;
      }
      return new Date(dateInput).toISOString().split('T')[0];
    }
    
    if (dateInput instanceof Date) {
      return dateInput.toISOString().split('T')[0];
    }
    
    return new Date().toISOString().split('T')[0];
  };

  // Format amount to ensure proper decimal precision
  const formatAmountForAPI = (amount) => {
    if (!amount || amount === '') return "0.00";
    
    // Handle string input carefully
    const cleaned = amount.toString().replace(/[^0-9.]/g, '');
    
    // Parse as number and validate
    const numAmount = parseFloat(cleaned);
    if (isNaN(numAmount) || numAmount <= 0) return "0.00";
    
    // Use toFixed(2) to ensure exactly 2 decimal places
    return numAmount.toFixed(2);
  };

  const addExpense = async () => {
    try {
      console.log('📝 Raw form data:', newExpense);
      
      // Properly format amount to prevent precision issues
      const formattedAmount = formatAmountForAPI(newExpense.amount);
      
      const expenseData = {
        date: formatDateForAPI(newExpense.date),
        amount: formattedAmount, // Send as properly formatted string
        category: newExpense.category?.toLowerCase() || 'other',
        note: newExpense.note || ''
      };
      
      console.log('📤 Formatted expense data:', expenseData);
      
      // Validate before sending
      if (!expenseData.date) {
        throw new Error('Date is required');
      }
      
      // Validate the formatted amount
      const amountNum = Number(formattedAmount);
      if (isNaN(amountNum) || amountNum <= 0) {
        throw new Error('Amount must be a positive number');
      }
      
      if (!expenseData.category) {
        throw new Error('Category is required');
      }

      setLoading(true);

      console.log('📤 Sending payload:', JSON.stringify(expenseData, null, 2));
      console.log('📤 Amount being sent:', formattedAmount, typeof formattedAmount);

      const response = await fetch('http://localhost:8000/api/expenses', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json'
        },
        body: JSON.stringify(expenseData)
      });

      console.log('📥 Response status:', response.status);

      if (!response.ok) {
        const errorText = await response.text();
        console.error('❌ Error response:', errorText);
        
        try {
          const errorJson = JSON.parse(errorText);
          console.error('❌ Parsed error:', errorJson);
          throw new Error(errorJson.detail || errorJson.message || `HTTP ${response.status}`);
        } catch (e) {
          console.error('❌ Raw error text:', errorText);
          throw new Error(`HTTP ${response.status}: ${errorText}`);
        }
      }

      const result = await response.json();
      console.log('✅ Success:', result);
      
      // Success feedback - display the exact amount that was saved
      setMessages((prev) => [...prev, { 
        from: "system", 
        text: `✅ Expense added: ₹${formattedAmount} for ${expenseData.category}`, 
        timestamp: new Date(),
        messageType: "expense"
      }]);
      
      // Reset form
      setNewExpense({
        date: new Date().toISOString().split('T')[0],
        amount: "",
        category: "",
        note: ""
      });

      // Refresh expenses if on expenses tab
      if (activeTab === "expenses") {
        loadExpenses();
      }
      
      console.log('✅ Expense added successfully with amount:', formattedAmount);
      
    } catch (error) {
      console.error('❌ Form submission error:', error);
      
      setMessages((prev) => [...prev, { 
        from: "system", 
        text: `❌ Failed to add expense: ${error.message}`, 
        timestamp: new Date(),
        messageType: "error"
      }]);
      
      alert(`Error adding expense: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  const testAPIConnection = async () => {
    setMessages((prev) => [...prev, { 
      from: "system", 
      text: "🔍 Testing API connection...", 
      timestamp: new Date(),
      messageType: "system"
    }]);

    try {
      const response = await fetch(`${API_BASE}/api/expenses?limit=1`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
        mode: 'cors',
      });
      
      if (response.ok) {
        const data = await response.json();
        setMessages((prev) => [...prev, { 
          from: "system", 
          text: `✅ API connection successful! Server is responding.`, 
          timestamp: new Date(),
          messageType: "system"
        }]);
      } else {
        setMessages((prev) => [...prev, { 
          from: "system", 
          text: `❌ API connection failed with status: ${response.status}`, 
          timestamp: new Date(),
          messageType: "error"
        }]);
      }
    } catch (error) {
      setMessages((prev) => [...prev, { 
        from: "system", 
        text: `❌ API connection error: ${error.message}`, 
        timestamp: new Date(),
        messageType: "error"
      }]);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const handleAmountChange = (e) => {
    const value = e.target.value;
    
    // Allow empty string
    if (value === '') {
      setNewExpense({...newExpense, amount: ''});
      return;
    }
    
    // More robust regex that handles various input scenarios
    // Allows: 123, 123., 123.4, 123.45, .50, 0.50, etc.
    if (/^\d*\.?\d{0,2}$/.test(value)) {
      setNewExpense({...newExpense, amount: value});
    }
    // If user tries to enter more than 2 decimal places, truncate
    else if (/^\d*\.\d{3,}$/.test(value)) {
      const truncated = value.substring(0, value.indexOf('.') + 3);
      setNewExpense({...newExpense, amount: truncated});
    }
  };

  const quickQuestions = [
    "How much did I spend this month?",
    "Show me my expense summary", 
    "What's my biggest expense?",
    "Compare this month to last month",
    "Show food expenses",
    "What's my spending pattern?"
  ];

  const categories = [
    "Food", "Transport", "Groceries", "Entertainment", 
    "Shopping", "Bills", "Medical", "Education", "Other"
  ];

  const formatCurrency = (amount) => {
    return new Intl.NumberFormat('en-IN', {
      style: 'currency',
      currency: 'INR',
      minimumFractionDigits: 2
    }).format(amount);
  };

  const formatDate = (dateString) => {
    return new Date(dateString).toLocaleDateString('en-IN');
  };

  const getConnectionStatusColor = () => {
    switch(connectionStatus) {
      case "connected": return "#10b981";
      case "connecting": return "#f59e0b";
      case "disconnected": return "#ef4444";
      case "error": return "#ef4444";
      default: return "#6b7280";
    }
  };

  const getConnectionStatusText = () => {
    switch(connectionStatus) {
      case "connected": return "🟢 Connected";
      case "connecting": return "🟡 Connecting...";
      case "disconnected": return "🔴 Disconnected";
      case "error": return "🔴 Error";
      default: return "🟡 Unknown";
    }
  };

  const getMessageTypeIcon = (messageType) => {
    switch(messageType) {
      case "ai": return "🤖";
      case "system": return "ℹ️";
      case "expense": return "💰";
      case "error": return "❌";
      case "user": return "👤";
      default: return "💬";
    }
  };

  // Validate amount properly without floating point conversion during typing
  const isAmountValid = () => {
    if (!newExpense.amount || newExpense.amount === '') return false;
    
    const amount = parseFloat(newExpense.amount);
    return !isNaN(amount) && amount > 0 && /^\d*\.?\d{0,2}$/.test(newExpense.amount);
  };

  return (
    <div style={{
      fontFamily: 'system-ui, -apple-system, sans-serif',
      minHeight: '100vh',
      background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
      margin: 0,
      padding: 0
    }}>
      <header style={{
        background: 'rgba(255, 255, 255, 0.1)',
        backdropFilter: 'blur(20px)',
        borderBottom: '1px solid rgba(255, 255, 255, 0.2)',
        padding: '1.5rem 2rem',
        textAlign: 'center'
      }}>
        <h1 style={{
          color: 'white',
          margin: 0,
          fontSize: '2.5rem',
          fontWeight: '700',
          textShadow: '0 2px 4px rgba(0,0,0,0.3)'
        }}>🤖 Expense AI Assistant</h1>
        
        <nav style={{
          marginTop: '1.5rem',
          display: 'flex',
          justifyContent: 'center',
          gap: '1rem',
          flexWrap: 'wrap'
        }}>
          {[
            { key: "chat", label: "💬 AI Chat", desc: "Talk to your AI assistant" },
            { key: "add", label: "➕ Add Expense", desc: "Record a new expense" },
            { key: "expenses", label: "📊 View Expenses", desc: "See your recent expenses" }
          ].map(tab => (
            <button 
              key={tab.key}
              style={{
                padding: '1rem 1.5rem',
                border: 'none',
                borderRadius: '15px',
                background: activeTab === tab.key 
                  ? 'rgba(255, 255, 255, 0.25)' 
                  : 'rgba(255, 255, 255, 0.1)',
                color: 'white',
                fontWeight: '600',
                cursor: 'pointer',
                transition: 'all 0.3s ease',
                backdropFilter: 'blur(10px)',
                transform: activeTab === tab.key ? 'translateY(-3px)' : 'none',
                boxShadow: activeTab === tab.key ? '0 8px 25px rgba(0,0,0,0.2)' : 'none',
                textAlign: 'center',
                minWidth: '160px'
              }}
              onClick={() => setActiveTab(tab.key)}
              title={tab.desc}
            >
              <div>{tab.label}</div>
              <div style={{ fontSize: '0.75rem', opacity: 0.8, marginTop: '0.25rem' }}>
                {tab.desc}
              </div>
            </button>
          ))}
        </nav>
      </header>

      <main style={{
        padding: '2rem',
        maxWidth: '1200px',
        margin: '0 auto'
      }}>
        
        {/* AI Chat Tab */}
        {activeTab === "chat" && (
          <div style={{
            background: 'rgba(255, 255, 255, 0.95)',
            borderRadius: '20px',
            padding: '2rem',
            boxShadow: '0 20px 40px rgba(0, 0, 0, 0.1)',
            minHeight: '600px',
            display: 'flex',
            flexDirection: 'column'
          }}>
            <div style={{ display: 'flex', gap: '2rem', flex: 1 }}>
              
              {/* Chat Area */}
              <div style={{ flex: 2, display: 'flex', flexDirection: 'column' }}>
                <div style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  marginBottom: '1.5rem',
                  flexWrap: 'wrap',
                  gap: '1rem'
                }}>
                  <h2 style={{ 
                    margin: 0, 
                    color: '#374151', 
                    display: 'flex', 
                    alignItems: 'center', 
                    gap: '0.5rem',
                    fontSize: '1.5rem'
                  }}>
                    🤖 AI Assistant
                    <span style={{ fontSize: '0.9rem', fontWeight: 'normal', opacity: 0.7 }}>
                      - Ask about your expenses!
                    </span>
                  </h2>
                  
                  {/* Connection Status & Controls */}
                  <div style={{ display: 'flex', gap: '1rem', alignItems: 'center' }}>
                    <div style={{
                      fontSize: '0.9rem',
                      color: getConnectionStatusColor(),
                      fontWeight: '600',
                      padding: '0.5rem 1rem',
                      borderRadius: '20px',
                      background: `${getConnectionStatusColor()}20`,
                      border: `1px solid ${getConnectionStatusColor()}40`
                    }}>
                      {getConnectionStatusText()}
                    </div>
                    
                    {connectionStatus !== "connected" && (
                      <button
                        onClick={manualReconnect}
                        disabled={isConnecting.current}
                        style={{
                          padding: '0.5rem 1rem',
                          background: isConnecting.current ? '#9ca3af' : '#667eea',
                          color: 'white',
                          border: 'none',
                          borderRadius: '12px',
                          cursor: isConnecting.current ? 'not-allowed' : 'pointer',
                          fontSize: '0.9rem',
                          fontWeight: '600'
                        }}
                      >
                        🔄 Reconnect
                      </button>
                    )}
                    
                    <button
                      onClick={testAPIConnection}
                      style={{
                        padding: '0.5rem 1rem',
                        background: '#059669',
                        color: 'white',
                        border: 'none',
                        borderRadius: '12px',
                        cursor: 'pointer',
                        fontSize: '0.9rem',
                        fontWeight: '600'
                      }}
                    >
                      🔍 Test API
                    </button>
                  </div>
                </div>
                
                <div 
                  ref={chatRef}
                  style={{
                    flex: 1,
                    border: '2px solid #e5e7eb',
                    borderRadius: '15px',
                    padding: '1.5rem',
                    overflowY: 'auto',
                    marginBottom: '1.5rem',
                    background: '#f9fafb',
                    minHeight: '400px'
                  }}
                >
                  {messages.length === 0 && (
                    <div style={{
                      textAlign: 'center', 
                      color: '#6b7280', 
                      padding: '2rem',
                      fontSize: '1.1rem'
                    }}>
                      👋 Start a conversation with your AI assistant!<br/>
                      <span style={{ fontSize: '0.9rem' }}>
                        Ask about your expenses, get summaries, or chat about anything.
                      </span>
                    </div>
                  )}
                  
                  {messages.map((msg, i) => (
                    <div 
                      key={i} 
                      style={{
                        marginBottom: '1.5rem',
                        display: 'flex',
                        flexDirection: msg.from === 'user' ? 'row-reverse' : 'row'
                      }}
                    >
                      <div style={{
                        maxWidth: '80%',
                        padding: '1rem 1.25rem',
                        borderRadius: '18px',
                        background: msg.from === 'user' 
                          ? 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)'
                          : msg.messageType === 'system' || msg.messageType === 'error'
                          ? msg.messageType === 'error' ? '#ef4444' : '#10b981'
                          : msg.messageType === 'ai' 
                          ? '#e0e7ff'
                          : '#e5e7eb',
                        color: msg.from === 'user' || msg.messageType === 'system' || msg.messageType === 'error' ? 'white' : '#374151',
                        fontSize: '0.95rem',
                        lineHeight: '1.5',
                        whiteSpace: 'pre-wrap'
                      }}>
                        <div style={{ display: 'flex', alignItems: 'flex-start', gap: '0.5rem' }}>
                          <span style={{ fontSize: '1.1rem' }}>{getMessageTypeIcon(msg.messageType)}</span>
                          <div style={{ flex: 1 }}>
                            {msg.text}
                            <div style={{
                              fontSize: '0.75rem',
                              opacity: 0.7,
                              marginTop: '0.5rem'
                            }}>
                              {msg.timestamp.toLocaleTimeString()}
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>

                {/* Input Area */}
                <div style={{ display: 'flex', gap: '0.75rem', alignItems: 'flex-end' }}>
                  <textarea
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyDown={handleKeyPress}
                    placeholder="Ask about your expenses or anything else..."
                    rows={2}
                    style={{
                      flex: 1,
                      padding: '1rem 1.25rem',
                      borderRadius: '15px',
                      border: '2px solid #e5e7eb',
                      fontSize: '1rem',
                      outline: 'none',
                      resize: 'vertical',
                      fontFamily: 'inherit',
                      transition: 'border-color 0.2s'
                    }}
                    onFocus={(e) => e.target.style.borderColor = '#667eea'}
                    onBlur={(e) => e.target.style.borderColor = '#e5e7eb'}
                  />
                  <button
                    onClick={sendMessage}
                    disabled={!input.trim() || connectionStatus !== "connected"}
                    style={{
                      padding: '1rem 2rem',
                      background: (input.trim() && connectionStatus === "connected")
                        ? 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)' 
                        : '#9ca3af',
                      color: 'white',
                      border: 'none',
                      borderRadius: '15px',
                      cursor: (input.trim() && connectionStatus === "connected") ? 'pointer' : 'not-allowed',
                      fontWeight: '600',
                      fontSize: '1rem',
                      transition: 'all 0.2s',
                      height: 'fit-content'
                    }}
                  >
                    Send 🚀
                  </button>
                </div>
              </div>

              {/* Quick Questions Sidebar */}
              <div style={{ flex: 1 }}>
                <h3 style={{ marginBottom: '1rem', color: '#374151' }}>💡 Try asking:</h3>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
                  {quickQuestions.map((q, i) => (
                    <button
                      key={i}
                      onClick={() => setInput(q)}
                      disabled={connectionStatus !== "connected"}
                      style={{
                        padding: '1rem',
                        background: connectionStatus === "connected" ? 'rgba(102, 126, 234, 0.1)' : 'rgba(156, 163, 175, 0.1)',
                        border: `1px solid ${connectionStatus === "connected" ? 'rgba(102, 126, 234, 0.3)' : 'rgba(156, 163, 175, 0.3)'}`,
                        borderRadius: '12px',
                        cursor: connectionStatus === "connected" ? 'pointer' : 'not-allowed',
                        textAlign: 'left',
                        fontSize: '0.9rem',
                        color: connectionStatus === "connected" ? '#4f46e5' : '#6b7280',
                        transition: 'all 0.2s ease',
                        lineHeight: '1.4',
                        opacity: connectionStatus === "connected" ? 1 : 0.6
                      }}
                      onMouseEnter={(e) => {
                        if (connectionStatus === "connected") {
                          e.target.style.background = 'rgba(102, 126, 234, 0.15)';
                          e.target.style.transform = 'translateY(-2px)';
                        }
                      }}
                      onMouseLeave={(e) => {
                        if (connectionStatus === "connected") {
                          e.target.style.background = 'rgba(102, 126, 234, 0.1)';
                          e.target.style.transform = 'translateY(0)';
                        }
                      }}
                    >
                      {q}
                    </button>
                  ))}
                </div>
                
                {/* Debug Info */}
                <div style={{ marginTop: '2rem', padding: '1rem', background: '#f3f4f6', borderRadius: '12px', fontSize: '0.8rem' }}>
                  <h4 style={{ margin: '0 0 0.5rem 0', color: '#374151' }}>🔧 Debug Info</h4>
                  <div><strong>User ID:</strong> {userId.current}</div>
                  <div><strong>Status:</strong> {connectionStatus}</div>
                  <div><strong>Messages:</strong> {messages.length}</div>
                  <div><strong>WebSocket:</strong> {socket ? 'Created' : 'None'}</div>
                  <div><strong>Reconnecting:</strong> {isReconnecting ? 'Yes' : 'No'}</div>
                  <div><strong>Attempts:</strong> {connectionAttempts.current}/{maxReconnectAttempts}</div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Add Expense Tab */}
        {activeTab === "add" && (
          <div style={{
            background: 'rgba(255, 255, 255, 0.95)',
            borderRadius: '20px',
            padding: '2rem',
            boxShadow: '0 20px 40px rgba(0, 0, 0, 0.1)',
            maxWidth: '600px',
            margin: '0 auto'
          }}>
            <h2 style={{ 
              color: '#374151', 
              marginBottom: '2rem', 
              textAlign: 'center',
              fontSize: '2rem'
            }}>➕ Add New Expense</h2>
            
            <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
              <div>
                <label style={{ 
                  display: 'block', 
                  marginBottom: '0.75rem', 
                  fontWeight: '600', 
                  color: '#374151',
                  fontSize: '1.1rem'
                }}>
                  📅 Date
                </label>
                <input
                  type="date"
                  value={newExpense.date}
                  onChange={(e) => setNewExpense({...newExpense, date: e.target.value})}
                  style={{
                    width: '100%',
                    padding: '1rem 1.25rem',
                    borderRadius: '12px',
                    border: '2px solid #e5e7eb',
                    fontSize: '1rem',
                    outline: 'none',
                    transition: 'border-color 0.2s'
                  }}
                  onFocus={(e) => e.target.style.borderColor = '#667eea'}
                  onBlur={(e) => e.target.style.borderColor = '#e5e7eb'}
                />
              </div>

              <div>
                <label style={{ 
                  display: 'block', 
                  marginBottom: '0.75rem', 
                  fontWeight: '600', 
                  color: '#374151',
                  fontSize: '1.1rem'
                }}>
                  💰 Amount (₹)
                </label>
                <input
                  type="text"
                  value={newExpense.amount}
                  onChange={handleAmountChange}
                  placeholder="0.00"
                  pattern="[0-9]+(\.[0-9]{1,2})?"
                  inputMode="decimal"
                  style={{
                    width: '100%',
                    padding: '1rem 1.25rem',
                    borderRadius: '12px',
                    border: '2px solid #e5e7eb',
                    fontSize: '1rem',
                    outline: 'none',
                    transition: 'border-color 0.2s'
                  }}
                  onFocus={(e) => e.target.style.borderColor = '#667eea'}
                  onBlur={(e) => e.target.style.borderColor = '#e5e7eb'}
                />
                {/* Add validation feedback */}
                {newExpense.amount && !isAmountValid() && (
                  <div style={{
                    marginTop: '0.5rem',
                    fontSize: '0.85rem',
                    color: '#ef4444',
                    fontWeight: '500'
                  }}>
                    Please enter a valid amount (e.g., 10.50)
                  </div>
                )}
              </div>

              <div>
                <label style={{ 
                  display: 'block', 
                  marginBottom: '0.75rem', 
                  fontWeight: '600', 
                  color: '#374151',
                  fontSize: '1.1rem'
                }}>
                  🏷️ Category
                </label>
                <select
                  value={newExpense.category}
                  onChange={(e) => setNewExpense({...newExpense, category: e.target.value})}
                  style={{
                    width: '100%',
                    padding: '1rem 1.25rem',
                    borderRadius: '12px',
                    border: '2px solid #e5e7eb',
                    fontSize: '1rem',
                    outline: 'none',
                    background: 'white',
                    transition: 'border-color 0.2s'
                  }}
                  onFocus={(e) => e.target.style.borderColor = '#667eea'}
                  onBlur={(e) => e.target.style.borderColor = '#e5e7eb'}
                >
                  <option value="">Select category</option>
                  {categories.map((cat, i) => (
                    <option key={i} value={cat}>{cat}</option>
                  ))}
                </select>
              </div>

              <div>
                <label style={{ 
                  display: 'block', 
                  marginBottom: '0.75rem', 
                  fontWeight: '600', 
                  color: '#374151',
                  fontSize: '1.1rem'
                }}>
                  📝 Note (optional)
                </label>
                <textarea
                  value={newExpense.note}
                  onChange={(e) => setNewExpense({...newExpense, note: e.target.value})}
                  placeholder="Add a note about this expense..."
                  rows={4}
                  style={{
                    width: '100%',
                    padding: '1rem 1.25rem',
                    borderRadius: '12px',
                    border: '2px solid #e5e7eb',
                    fontSize: '1rem',
                    outline: 'none',
                    resize: 'vertical',
                    fontFamily: 'inherit',
                    transition: 'border-color 0.2s'
                  }}
                  onFocus={(e) => e.target.style.borderColor = '#667eea'}
                  onBlur={(e) => e.target.style.borderColor = '#e5e7eb'}
                />
              </div>

              <button
                onClick={addExpense}
                disabled={loading || !newExpense.amount || !newExpense.category || parseFloat(newExpense.amount) <= 0}
                style={{
                  padding: '1.25rem 2rem',
                  background: (!loading && newExpense.amount && newExpense.category && parseFloat(newExpense.amount || 0) > 0)
                    ? 'linear-gradient(135deg, #10b981 0%, #059669 100%)'
                    : '#9ca3af',
                  color: 'white',
                  border: 'none',
                  borderRadius: '15px',
                  fontSize: '1.2rem',
                  fontWeight: '600',
                  cursor: (!loading && newExpense.amount && newExpense.category && parseFloat(newExpense.amount || 0) > 0) ? 'pointer' : 'not-allowed',
                  transition: 'all 0.3s ease',
                  marginTop: '1rem',
                  transform: (!loading && newExpense.amount && newExpense.category && parseFloat(newExpense.amount || 0) > 0) ? 'none' : 'scale(0.98)'
                }}
              >
                {loading ? '⏳ Saving...' : '💾 Save Expense'}
              </button>
            </div>
          </div>
        )}

        {/* View Expenses Tab */}
        {activeTab === "expenses" && (
          <div style={{
            background: 'rgba(255, 255, 255, 0.95)',
            borderRadius: '20px',
            padding: '2rem',
            boxShadow: '0 20px 40px rgba(0, 0, 0, 0.1)'
          }}>
            <div style={{ 
              display: 'flex', 
              justifyContent: 'space-between', 
              alignItems: 'center', 
              marginBottom: '2rem' 
            }}>
              <h2 style={{ color: '#374151', margin: 0, fontSize: '2rem' }}>
                📊 Recent Expenses
              </h2>
              <button
                onClick={loadExpenses}
                disabled={loading}
                style={{
                  padding: '0.75rem 1.5rem',
                  background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                  color: 'white',
                  border: 'none',
                  borderRadius: '12px',
                  cursor: loading ? 'not-allowed' : 'pointer',
                  fontWeight: '600',
                  opacity: loading ? 0.7 : 1
                }}
              >
                {loading ? '⏳ Loading...' : '🔄 Refresh'}
              </button>
            </div>

            {loading ? (
              <div style={{ 
                textAlign: 'center', 
                padding: '3rem', 
                color: '#6b7280',
                fontSize: '1.1rem'
              }}>
                <div style={{ fontSize: '3rem', marginBottom: '1rem' }}>⏳</div>
                Loading your expenses...
              </div>
            ) : expenses.length === 0 ? (
              <div style={{ 
                textAlign: 'center', 
                padding: '3rem', 
                color: '#6b7280' 
              }}>
                <div style={{ fontSize: '4rem', marginBottom: '1rem' }}>📝</div>
                <h3 style={{ marginBottom: '0.5rem' }}>No expenses found</h3>
                <p>Start by adding your first expense!</p>
                <button
                  onClick={() => setActiveTab("add")}
                  style={{
                    marginTop: '1rem',
                    padding: '0.75rem 1.5rem',
                    background: 'linear-gradient(135deg, #10b981 0%, #059669 100%)',
                    color: 'white',
                    border: 'none',
                    borderRadius: '12px',
                    cursor: 'pointer',
                    fontWeight: '600'
                  }}
                >
                  ➕ Add First Expense
                </button>
              </div>
            ) : (
              <div style={{ overflowX: 'auto' }}>
                <table style={{ 
                  width: '100%', 
                  borderCollapse: 'collapse',
                  background: 'white',
                  borderRadius: '12px',
                  overflow: 'hidden',
                  boxShadow: '0 4px 6px rgba(0, 0, 0, 0.05)'
                }}>
                  <thead>
                    <tr style={{ background: '#f8fafc' }}>
                      <th style={{ 
                        padding: '1.25rem', 
                        textAlign: 'left', 
                        fontWeight: '700',
                        color: '#374151',
                        fontSize: '1rem'
                      }}>📅 Date</th>
                      <th style={{ 
                        padding: '1.25rem', 
                        textAlign: 'left', 
                        fontWeight: '700',
                        color: '#374151',
                        fontSize: '1rem'
                      }}>🏷️ Category</th>
                      <th style={{ 
                        padding: '1.25rem', 
                        textAlign: 'right', 
                        fontWeight: '700',
                        color: '#374151',
                        fontSize: '1rem'
                      }}>💰 Amount</th>
                      <th style={{ 
                        padding: '1.25rem', 
                        textAlign: 'left', 
                        fontWeight: '700',
                        color: '#374151',
                        fontSize: '1rem'
                      }}>📝 Note</th>
                    </tr>
                  </thead>
                  <tbody>
                    {expenses.map((expense, i) => (
                      <tr 
                        key={expense.id || i} 
                        style={{ 
                          borderBottom: '1px solid #e5e7eb',
                          transition: 'background-color 0.2s'
                        }}
                        onMouseEnter={(e) => e.currentTarget.style.backgroundColor = '#f9fafb'}
                        onMouseLeave={(e) => e.currentTarget.style.backgroundColor = 'transparent'}
                      >
                        <td style={{ 
                          padding: '1.25rem',
                          fontWeight: '500'
                        }}>
                          {formatDate(expense.date)}
                        </td>
                        <td style={{ padding: '1.25rem' }}>
                          <span style={{
                            padding: '0.4rem 0.875rem',
                            background: 'rgba(102, 126, 234, 0.1)',
                            borderRadius: '20px',
                            fontSize: '0.875rem',
                            color: '#4f46e5',
                            fontWeight: '500'
                          }}>
                            {expense.category}
                          </span>
                        </td>
                        <td style={{ 
                          padding: '1.25rem', 
                          textAlign: 'right', 
                          fontWeight: '700',
                          color: '#059669',
                          fontSize: '1.1rem'
                        }}>
                          {formatCurrency(expense.amount)}
                        </td>
                        <td style={{ 
                          padding: '1.25rem', 
                          fontSize: '0.9rem', 
                          color: '#6b7280',
                          maxWidth: '200px',
                          overflow: 'hidden',
                          textOverflow: 'ellipsis',
                          whiteSpace: 'nowrap'
                        }}>
                          {expense.note || '-'}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                
                <div style={{ 
                  textAlign: 'center', 
                  marginTop: '2rem',
                  color: '#6b7280',
                  fontSize: '0.9rem'
                }}>
                  Showing {expenses.length} recent expenses
                </div>
              </div>
            )}
          </div>
        )}
      </main>
    </div>
  );
}

export default App;