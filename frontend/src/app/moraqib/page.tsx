'use client';

import { useState, useRef, useEffect } from 'react';
import AILoadingAnimation from '@/components/AILoadingAnimation';

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  metadata?: {
    reports_count?: number;
    reports_used?: string[];
  };
}

export default function MoraqibPage() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      role: 'assistant',
      content: "Welcome to Moraqib! I'm your AI assistant for the Mirqab detection system. Ask me anything about detection reports in our database.",
      timestamp: new Date()
    }
  ]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isHydrated, setIsHydrated] = useState(false);
  const [shouldAutoScroll, setShouldAutoScroll] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Only set hydrated flag after component mounts on client
  useEffect(() => {
    setIsHydrated(true);
  }, []);

  // Auto-scroll to bottom only when explicitly requested (not on initial load)
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    if (shouldAutoScroll) {
      scrollToBottom();
      setShouldAutoScroll(false);
    }
  }, [messages, shouldAutoScroll]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!input.trim() || isLoading) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: input,
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);
    setShouldAutoScroll(true); // Enable auto-scroll for user messages

    try {
      const formData = new FormData();
      formData.append('query', input);

      const response = await fetch('http://localhost:8000/api/moraqib_query', {
        method: 'POST',
        body: formData
      });

      const data = await response.json();

      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: data.answer || "I'm sorry, I couldn't process your question.",
        timestamp: new Date(),
        metadata: {
          reports_count: data.reports_count,
          reports_used: data.reports_used
        }
      };

      setMessages(prev => [...prev, assistantMessage]);
      setShouldAutoScroll(true); // Enable auto-scroll for assistant response
    } catch (error) {
      console.error('Moraqib query error:', error);
      
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: "❌ I'm sorry, I encountered a connection error. Please make sure the backend server is running.",
        timestamp: new Date()
      };

      setMessages(prev => [...prev, errorMessage]);
      setShouldAutoScroll(true); // Enable auto-scroll for error message
    } finally {
      setIsLoading(false);
    }
  };

  // Example questions - Updated to match RAG capabilities
  const exampleQuestions = [
    "Give me a summary of the last report",
    "What's the average number of soldiers in the last 10 reports?",
    "What are all the devices used in the database?",
    "How many detections were made yesterday?",
    "Show me the last 5 reports",
    "What environments were detected in the last week?"
  ];

  const askExample = (question: string) => {
    setInput(question);
  };

  return (
    <>
      {/* AI Loading Animation */}
      <AILoadingAnimation />
      
      <main className="min-h-screen bg-gradient-to-br from-gray-900 via-blue-900 to-gray-900">
        {/* Header */}
        <div className="bg-gray-800/50 backdrop-blur-md border-b border-gray-700">
          <div className="max-w-5xl mx-auto px-4 py-6">
            <div className="flex items-center gap-3">
              <div className="w-12 h-12 bg-gradient-to-br from-blue-500 to-purple-600 rounded-lg flex items-center justify-center">
                <svg className="w-7 h-7 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                </svg>
              </div>
              <div>
                <h1 className="text-3xl font-bold text-white">Moraqib Assistant</h1>
                <p className="text-gray-400">AI-powered detection report analysis</p>
              </div>
            </div>
          </div>
        </div>

      {/* Chat Container */}
      <div className="max-w-5xl mx-auto px-4 py-8">
        <div className="bg-gray-800/30 backdrop-blur-md rounded-xl border border-gray-700 overflow-hidden shadow-2xl">
          {/* Messages Area */}
          <div className="h-[600px] overflow-y-auto p-6 space-y-4">
            {messages.map((message) => (
              <div
                key={message.id}
                className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
              >
                <div
                  className={`max-w-[80%] rounded-2xl px-6 py-4 ${
                    message.role === 'user'
                      ? 'bg-gradient-to-r from-blue-600 to-blue-500 text-white'
                      : 'bg-gray-700/50 text-gray-100'
                  }`}
                >
                  {/* Message Content */}
                  <p className="text-sm md:text-base whitespace-pre-wrap leading-relaxed">
                    {message.content}
                  </p>

                  {/* Metadata */}
                  {message.metadata && message.metadata.reports_count !== undefined && (
                    <div className="mt-3 pt-3 border-t border-gray-600/50">
                      <p className="text-xs text-gray-400">
                        📊 Analyzed {message.metadata.reports_count} report{message.metadata.reports_count !== 1 ? 's' : ''}
                      </p>
                      {message.metadata.reports_used && message.metadata.reports_used.length > 0 && (
                        <p className="text-xs text-gray-400 mt-1">
                          📋 Used: {message.metadata.reports_used.slice(0, 3).join(', ')}
                          {message.metadata.reports_used.length > 3 && ` +${message.metadata.reports_used.length - 3} more`}
                        </p>
                      )}
                    </div>
                  )}

                  {/* Timestamp - Only render on client after hydration */}
                  {isHydrated && (
                    <p className="text-xs mt-2 opacity-60">
                      {message.timestamp.toLocaleTimeString()}
                    </p>
                  )}
                </div>
              </div>
            ))}

            {/* Loading Indicator */}
            {isLoading && (
              <div className="flex justify-start">
                <div className="bg-gray-700/50 rounded-2xl px-6 py-4">
                  <div className="flex items-center gap-2">
                    <div className="flex gap-1">
                      <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></div>
                      <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></div>
                      <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></div>
                    </div>
                    <span className="text-sm text-gray-400 ml-2">Moraqib is thinking...</span>
                  </div>
                </div>
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>

          {/* Example Questions */}
          {messages.length === 1 && (
            <div className="px-6 py-4 bg-gray-800/50 border-t border-gray-700">
              <p className="text-sm text-gray-400 mb-3">💡 Try asking:</p>
              <div className="flex flex-wrap gap-2">
                {exampleQuestions.map((question, idx) => (
                  <button
                    key={idx}
                    onClick={() => askExample(question)}
                    className="text-xs px-3 py-2 bg-gray-700/50 hover:bg-gray-600/50 text-gray-300 rounded-lg transition-colors"
                  >
                    {question}
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Input Area */}
          <form onSubmit={handleSubmit} className="p-6 bg-gray-800/50 border-t border-gray-700">
            <div className="flex gap-3">
              <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Ask about detection reports..."
                className="flex-1 px-6 py-4 bg-gray-700/50 text-white rounded-xl border border-gray-600 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent placeholder-gray-400"
                disabled={isLoading}
              />
              <button
                type="submit"
                disabled={!input.trim() || isLoading}
                className="px-8 py-4 bg-gradient-to-r from-blue-600 to-blue-500 text-white rounded-xl font-semibold hover:from-blue-700 hover:to-blue-600 disabled:opacity-50 disabled:cursor-not-allowed transition-all shadow-lg hover:shadow-xl"
              >
                {isLoading ? '...' : 'Send'}
              </button>
            </div>
          </form>
        </div>

        {/* Info Box */}
        <div className="mt-6 p-4 bg-blue-900/20 border border-blue-700/30 rounded-xl">
          <div className="flex items-start gap-3">
            <svg className="w-6 h-6 text-blue-400 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <div className="text-sm text-gray-300">
              <p className="font-semibold text-white mb-2">About Moraqib RAG System:</p>
              <p>
                Moraqib uses advanced <strong>Retrieval-Augmented Generation (RAG)</strong> to intelligently query your detection reports database and provide accurate, context-aware answers.
              </p>
            </div>
          </div>
        </div>

        {/* Capability Cards */}
        <div className="mt-6 grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="p-4 bg-gradient-to-br from-purple-900/30 to-blue-900/30 border border-purple-700/30 rounded-xl">
            <div className="flex items-center gap-2 mb-2">
              <h3 className="font-semibold text-white">Statistics</h3>
            </div>
            <p className="text-xs text-gray-400">
              Get averages, totals, and counts from your detection data
            </p>
          </div>
          
          <div className="p-4 bg-gradient-to-br from-blue-900/30 to-cyan-900/30 border border-blue-700/30 rounded-xl">
            <div className="flex items-center gap-2 mb-2">
              <h3 className="font-semibold text-white">Smart Search</h3>
            </div>
            <p className="text-xs text-gray-400">
              Find reports by time, device, environment, or any criteria
            </p>
          </div>
          
          <div className="p-4 bg-gradient-to-br from-cyan-900/30 to-teal-900/30 border border-cyan-700/30 rounded-xl">
            <div className="flex items-center gap-2 mb-2">
              <h3 className="font-semibold text-white">Summaries</h3>
            </div>
            <p className="text-xs text-gray-400">
              Get natural language summaries of detection reports
            </p>
          </div>
        </div>
      </div>
      </main>
    </>
  );
}
