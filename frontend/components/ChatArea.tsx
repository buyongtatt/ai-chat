'use client';
import { useState, useRef, useCallback, useEffect } from 'react';
import { v4 as uuid } from 'uuid';
import { Message, AttachedFile } from '@/types';
import { streamAnswer, cancelSession, fetchSources } from '@/lib/api';
import MessageBubble from '@/components/MessageBubble';
import InputBar from '@/components/InputBar';
import TopBar from '@/components/TopBar';
import styles from './ChatArea.module.css';
import { BookOpen } from 'lucide-react';

interface Props {
  theme: 'dark' | 'light';
  onToggleTheme: () => void;
  onToggleSidebar: () => void;
  apiOnline: boolean;
}

export default function ChatArea({ theme, onToggleTheme, onToggleSidebar, apiOnline }: Props) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSend = useCallback(async (question: string, file: File | null) => {
    if (!question.trim() || isStreaming) return;

    const sessionId = uuid();
    setActiveSessionId(sessionId);

    // Build user message
    let attachedFile: AttachedFile | undefined;
    if (file) {
      attachedFile = {
        name: file.name,
        type: file.type.startsWith('image/') ? 'image' : 'document',
        previewUrl: file.type.startsWith('image/') ? URL.createObjectURL(file) : undefined,
        size: file.size,
      };
    }

    const userMsg: Message = {
      id: uuid(),
      role: 'user',
      content: question,
      timestamp: new Date(),
      attachedFile,
    };

    const aiMsg: Message = {
      id: uuid(),
      role: 'assistant',
      content: '',
      timestamp: new Date(),
      isStreaming: true,
    };

    setMessages(prev => [...prev, userMsg, aiMsg]);
    setIsStreaming(true);

    // Fetch sources in parallel (for image chips)
    fetchSources(question).then(data => {
      setMessages(prev =>
        prev.map(m =>
          m.id === aiMsg.id
            ? { ...m, sources: data.sources, sourceImages: data.images }
            : m
        )
      );
    });

    // Stream
    const abortController = new AbortController();
    abortRef.current = abortController;
    let accum = '';

    try {
      for await (const token of streamAnswer(question, sessionId, file, abortController.signal)) {
        accum += token;
        const captured = accum;
        setMessages(prev =>
          prev.map(m => (m.id === aiMsg.id ? { ...m, content: captured } : m))
        );
      }
    } catch (err: unknown) {
      if (err instanceof Error && err.name === 'AbortError') {
        // cancelled
      } else {
        accum += '\n\n⚠️ Connection error. Is the API running?';
        setMessages(prev =>
          prev.map(m => (m.id === aiMsg.id ? { ...m, content: accum } : m))
        );
      }
    } finally {
      setMessages(prev =>
        prev.map(m => (m.id === aiMsg.id ? { ...m, isStreaming: false } : m))
      );
      setIsStreaming(false);
      setActiveSessionId(null);
      abortRef.current = null;
    }
  }, [isStreaming]);

  const handleCancel = useCallback(async () => {
    if (!activeSessionId) return;
    // Signal cancellation both ways
    abortRef.current?.abort();
    await cancelSession(activeSessionId);
    setMessages(prev =>
      prev.map(m => (m.isStreaming ? { ...m, isStreaming: false, cancelled: true } : m))
    );
    setIsStreaming(false);
    setActiveSessionId(null);
  }, [activeSessionId]);

  const handleClear = useCallback(() => {
    if (isStreaming) handleCancel();
    setMessages([]);
  }, [isStreaming, handleCancel]);

  return (
    <div className={styles.chatArea}>
      <TopBar
        theme={theme}
        onToggleTheme={onToggleTheme}
        onToggleSidebar={onToggleSidebar}
        onClear={handleClear}
        hasMessages={messages.length > 0}
        apiOnline={apiOnline}
      />

      <div className={styles.messages}>
        {messages.length === 0 ? (
          <EmptyState />
        ) : (
          messages.map(msg => (
            <MessageBubble key={msg.id} message={msg} />
          ))
        )}
        <div ref={bottomRef} />
      </div>

      <InputBar
        onSend={handleSend}
        onCancel={handleCancel}
        isStreaming={isStreaming}
        disabled={!apiOnline}
      />
    </div>
  );
}

function EmptyState() {
  return (
    <div className={styles.emptyState}>
      <div className={styles.emptyIcon}>
        <BookOpen size={36} />
      </div>
      <h2>DocMind</h2>
      <p>Ask anything about your document library.</p>
      <p>Powered by qwen3-vl running locally.</p>
      <div className={styles.suggestions}>
        {[
          'Summarize the main findings',
          'What images are in the documents?',
          'List all key dates mentioned',
        ].map(s => (
          <span key={s} className={styles.suggestionChip}>{s}</span>
        ))}
      </div>
    </div>
  );
}
