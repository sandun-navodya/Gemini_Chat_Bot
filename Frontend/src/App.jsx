import React, { useState, useEffect, useRef } from 'react';
import './App.css';
import axios from 'axios';

// --- CONFIGURATION ---
const API_BASE_URL = 'http://127.0.0.1:5000';

function App() {
  // Theme state: initialized based on system preference or defaults to false (light)
  const [isDarkMode, setIsDarkMode] = useState(
    window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches
  );
  const [messages, setMessages] = useState([]);
  const [userInput, setUserInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [uploadMessage, setUploadMessage] = useState('No document loaded.');
  const [pdfFile, setPdfFile] = useState(null);
  const [chatHistory, setChatHistory] = useState([]);
  const [currentChatId, setCurrentChatId] = useState(null);
  const [showHistory, setShowHistory] = useState(false);
  const messagesEndRef = useRef(null);

  // --- UI UTILITIES ---
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  const handleThemeToggle = () => {
    setIsDarkMode(prevMode => !prevMode);
  };

  // --- EFFECT HOOKS ---
  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Initialize chat on first load
  useEffect(() => {
    startNewChat();
    loadChatHistory();
  }, []);

  // Load chat history from localStorage
  const loadChatHistory = () => {
    const savedHistory = localStorage.getItem('chatHistory');
    if (savedHistory) {
      setChatHistory(JSON.parse(savedHistory));
    }
  };

  // Save chat to history
  const saveChatToHistory = (chatId, messages, title) => {
    const newChat = {
      id: chatId,
      title: title || getSmartTitle(messages),
      messages: messages,
      timestamp: new Date().toISOString(),
      documentName: uploadMessage !== 'No document loaded.' ? uploadMessage : null
    };

    const updatedHistory = [newChat, ...chatHistory.filter(chat => chat.id !== chatId)];
    setChatHistory(updatedHistory);
    localStorage.setItem('chatHistory', JSON.stringify(updatedHistory));
  };

  // Generate smart title from first user message
  const getSmartTitle = (msgs) => {
    const firstUserMsg = msgs.find(m => m.role === 'user');
    if (firstUserMsg) {
      const content = firstUserMsg.content;
      return content.length > 40 ? content.substring(0, 40) + '...' : content;
    }
    return `Chat ${new Date().toLocaleString()}`;
  };

  // Load a chat from history
  const loadChat = (chatId) => {
    const chat = chatHistory.find(c => c.id === chatId);
    if (chat) {
      setMessages(chat.messages);
      setCurrentChatId(chatId);
      if (chat.documentName) {
        setUploadMessage(chat.documentName);
      }
      setShowHistory(false);
    }
  };

  // Delete a chat from history
  const deleteChat = (chatId, e) => {
    e.stopPropagation();
    const updatedHistory = chatHistory.filter(chat => chat.id !== chatId);
    setChatHistory(updatedHistory);
    localStorage.setItem('chatHistory', JSON.stringify(updatedHistory));
    
    if (currentChatId === chatId) {
      startNewChat();
    }
  };

  // Clear all history
  const clearAllHistory = () => {
    if (window.confirm('Are you sure you want to clear all chat history?')) {
      setChatHistory([]);
      localStorage.removeItem('chatHistory');
    }
  };

  // --- API CALLS ---

  const startNewChat = async () => {
    setIsLoading(true);
    try {
      // Clear the document from backend when starting new chat
      await axios.post(`${API_BASE_URL}/clear_document`);
      await axios.post(`${API_BASE_URL}/new_chat`);
      
      const newChatId = Date.now().toString();
      setCurrentChatId(newChatId);
      setMessages([{ role: 'bot', content: 'Hello! I am ready to start a new conversation. Please upload a PDF or ask me a general question.' }]);
      setUploadMessage('No document loaded.');
      setPdfFile(null); // Clear the file input
    } catch (error) {
      setMessages([{ role: 'bot', content: 'Error starting new chat. Check backend connectivity.' }]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleUpload = async () => {
    if (!pdfFile) return;

    setIsLoading(true);
    setUploadMessage('Uploading and processing...');
    
    const formData = new FormData();
    formData.append('pdf_file', pdfFile);

    try {
      const response = await axios.post(`${API_BASE_URL}/upload`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      setUploadMessage(response.data.message);
      
      // Don't call startNewChat here - just refresh the chat session
      // to work with the new document context
      await axios.post(`${API_BASE_URL}/new_chat`);
      
      // Add a system message indicating document is loaded
      setMessages([
        ...messages, 
        { role: 'bot', content: `Document "${pdfFile.name}" has been loaded successfully! You can now ask questions about it.` }
      ]);
      
    } catch (error) {
      console.error("Upload error:", error);
      setUploadMessage('Upload failed. Check console or API key.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleSendMessage = async (e) => {
    e.preventDefault();
    if (!userInput.trim() || isLoading) return;

    const messageToSend = userInput.trim();
    const newMessages = [...messages, { role: 'user', content: messageToSend }];
    setMessages(newMessages);
    setUserInput('');
    setIsLoading(true);

    try {
      const response = await axios.post(`${API_BASE_URL}/chat`, { message: messageToSend });
      const updatedMessages = [...newMessages, { role: 'bot', content: response.data.response }];
      setMessages(updatedMessages);
      
      // Save to history after getting response
      saveChatToHistory(currentChatId, updatedMessages);
    } catch (error) {
      const errorMessages = [...newMessages, { role: 'bot', content: 'Sorry, I ran into an error getting the response. Check API logs.' }];
      setMessages(errorMessages);
      saveChatToHistory(currentChatId, errorMessages);
    } finally {
      setIsLoading(false);
    }
  };
  
  // --- RENDERING ---

  return (
    // Apply the theme class to the main container
    <div className={`chatbot-app ${isDarkMode ? 'dark-mode' : 'light-mode'}`}>
      {/* History Sidebar */}
      <div className={`history-sidebar ${showHistory ? 'open' : ''}`}>
        <div className="history-header">
          <h2>Chat History</h2>
          <button 
            className="close-history-btn" 
            onClick={() => setShowHistory(false)}
            aria-label="Close history"
          >
            ✕
          </button>
        </div>
        
        <div className="history-actions">
          <button 
            className="clear-history-btn"
            onClick={clearAllHistory}
            disabled={chatHistory.length === 0}
          >
            🗑️ Clear All
          </button>
        </div>

        <div className="history-list">
          {chatHistory.length === 0 ? (
            <p className="no-history">No chat history yet</p>
          ) : (
            chatHistory.map((chat) => (
              <div 
                key={chat.id} 
                className={`history-item ${currentChatId === chat.id ? 'active' : ''}`}
                onClick={() => loadChat(chat.id)}
              >
                <div className="history-item-content">
                  <h3>{chat.title}</h3>
                  <p className="history-timestamp">
                    {new Date(chat.timestamp).toLocaleString()}
                  </p>
                  {chat.documentName && (
                    <p className="history-document">📄 {chat.documentName}</p>
                  )}
                </div>
                <button 
                  className="delete-chat-btn"
                  onClick={(e) => deleteChat(chat.id, e)}
                  aria-label="Delete chat"
                >
                  🗑️
                </button>
              </div>
            ))
          )}
        </div>
      </div>

      {/* Overlay when sidebar is open */}
      {showHistory && (
        <div 
          className="history-overlay" 
          onClick={() => setShowHistory(false)}
        />
      )}

      <div className="chatbot-container">
        <header className="chat-header">
          <div className="header-left">
            <button 
              className="history-toggle-btn"
              onClick={() => setShowHistory(!showHistory)}
              aria-label="Toggle history"
            >
              📚
            </button>
            <h1>Gemini Chatbot</h1>
          </div>
          
          <div className="header-controls">
            {/* Theme Toggle Switch */}
            <label className="theme-switch">
              <input type="checkbox" checked={isDarkMode} onChange={handleThemeToggle} disabled={isLoading} />
              <span className="slider round"></span>
            </label>

            <button onClick={startNewChat} disabled={isLoading} className="new-chat-btn">
              {isLoading ? 'Loading...' : '🚀 New Chat'}
            </button>
          </div>
        </header>

        <div className="rag-status">
          <p className="status-text">Document Status: <strong>{uploadMessage}</strong></p>
          <div className="upload-group">
            <input 
              type="file" 
              accept="application/pdf" 
              onChange={(e) => setPdfFile(e.target.files[0])} 
              disabled={isLoading}
              className="file-input"
              id="pdf-upload"
            />
             <label htmlFor="pdf-upload" className="file-label">
                {pdfFile ? pdfFile.name : 'Choose PDF'}
            </label>
            <button onClick={handleUpload} disabled={isLoading || !pdfFile} className="process-btn">
              {isLoading ? 'Processing...' : 'Process Document'}
            </button>
          </div>
        </div>

        <div className="chat-history">
          {messages.map((msg, index) => (
            <div key={index} className={`message ${msg.role}`}>
              <strong>{msg.role === 'user' ? 'You:' : '🤖'}</strong>
              <p>{msg.content}</p>
            </div>
          ))}
          {isLoading && <div className="message bot loading"><strong>🤖</strong> Typing...</div>}
          <div ref={messagesEndRef} />
        </div>

        <form onSubmit={handleSendMessage} className="chat-input-form">
          <input
            type="text"
            value={userInput}
            onChange={(e) => setUserInput(e.target.value)}
            placeholder="Ask a question..."
            disabled={isLoading}
          />
          <button type="submit" disabled={isLoading}>Send</button>
        </form>
      </div>
    </div>
  );
}

export default App;