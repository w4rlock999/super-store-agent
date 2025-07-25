"use client";

import React, { useState } from "react";
import styles from "./page.module.css";
import ReactMarkdown from "react-markdown";

interface Message {
  text: string;
  isUser: boolean;
  timestamp: Date;
}

export default function ChatPage() {
  const [inputText, setInputText] = useState<string>("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setInputText(e.target.value);
  };

  const handleKeyPress = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      submitMessage();
    }
  };

  const submitMessage = async () => {
    if (!inputText.trim()) return;

    // Add user message
    const userMessage: Message = {
      text: inputText,
      isUser: true,
      timestamp: new Date()
    };
    setMessages(prev => [...prev, userMessage]);
    setInputText("");
    setIsLoading(true);

    try {
      const response = await fetch("/api/agent-invoke", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ text: inputText }),
      });

      if (response.ok) {
        const result = await response.json();
        // Add bot response
        const botMessage: Message = {
          text: result.data.response || JSON.stringify(result),
          isUser: false,
          timestamp: new Date()
        };
        setMessages(prev => [...prev, botMessage]);
      } else {
        const error = await response.json();
        // Add error message
        const errorMessage: Message = {
          text: `Error: ${error.error || "Something went wrong."}`,
          isUser: false,
          timestamp: new Date()
        };
        setMessages(prev => [...prev, errorMessage]);
      }
    } catch (err) {
      // Add error message
      const errorMessage: Message = {
        text: `Error: ${err}`,
        isUser: false,
        timestamp: new Date()
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className={styles.chatContainer}>
      {/* <div className={styles.chatHeader}>
        <h1>Finance Agent!</h1>
      </div> */}
      
      <div className={styles.messagesContainer}>
        <div className={styles.messagesInner}>
        {messages.map((message, index) => (
          message.isUser ? (
            <div
              key={index}
              className={`${styles.messageBubble} ${styles.userMessage}`}
            >
              <div className={styles.userMessageContent}>{message.text}</div>
            </div>
          ) : (
            <div
              key={index}
              className={`${styles.messageBubble} ${styles.botMessage}`}
              style={{ display: 'flex', alignItems: 'flex-start', gap: '2.1rem', background: 'none', boxShadow: 'none', padding: 0, width: '90%', maxWidth: '100%' }}
            >
              <div style={{display: 'flex', flexDirection: 'column', alignItems: 'flex-start'}}>
                <div className={styles.botIcon} style={{marginBottom: '4px'}}></div>
              </div>
              <div className={styles.botMessageContent}>
                <ReactMarkdown>
                  {message.text}
                </ReactMarkdown>
              </div>
            </div>
          )
        ))}
        {isLoading && (
          <div className={`${styles.messageBubble} ${styles.botMessage}`} style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', background: 'none', boxShadow: 'none', padding: 0 }}>
            <div className={styles.botIcon}></div>
            <div className={styles.botMessageContent}>
              <div className={styles.typingIndicator}>
                <span></span>
                <span></span>
                <span></span>
              </div>
            </div>
          </div>
        )}
        </div>
      </div>
      <div className={styles.bottomGradient}></div>

      <div className={styles.inputContainer}>
          <input
            className={styles.chatInput}
            placeholder="Type your message..."
            value={inputText}
            onChange={handleInputChange}
            onKeyPress={handleKeyPress}
            autoComplete="just-something-to-make-it-work"
            name="actually-chat-input-for-finance-agent"
          />
        <button 
          className={styles.sendButton}
          onClick={submitMessage}
          disabled={!inputText.trim() || isLoading}
        >
          Send
        </button>
      </div>
    </div>
  );
}
