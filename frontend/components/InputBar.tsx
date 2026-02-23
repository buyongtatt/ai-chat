'use client';
import { useState, useRef, useCallback, KeyboardEvent } from 'react';
import { Paperclip, Send, StopCircle, X, Image as ImgIcon, FileText } from 'lucide-react';
import styles from './InputBar.module.css';
import clsx from 'clsx';

interface Props {
  onSend: (question: string, file: File | null) => void;
  onCancel: () => void;
  isStreaming: boolean;
  disabled: boolean;
}

const MAX_FILE_SIZE = 10 * 1024 * 1024; // 10MB
const ALLOWED_TYPES = [
  'image/png', 'image/jpeg', 'image/webp', 'image/gif',
  'text/plain', 'text/markdown', 'application/pdf',
];

export default function InputBar({ onSend, onCancel, isStreaming, disabled }: Props) {
  const [input, setInput] = useState('');
  const [file, setFile] = useState<File | null>(null);
  const [filePreview, setFilePreview] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileSelect = useCallback((selected: File | null) => {
    if (!selected) return;
    if (!ALLOWED_TYPES.includes(selected.type)) {
      alert(`Unsupported file type. Allowed: images, .txt, .md, .pdf`);
      return;
    }
    if (selected.size > MAX_FILE_SIZE) {
      alert('File too large. Max 10MB.');
      return;
    }
    setFile(selected);
    if (selected.type.startsWith('image/')) {
      setFilePreview(URL.createObjectURL(selected));
    } else {
      setFilePreview(null);
    }
  }, []);

  const clearFile = useCallback(() => {
    if (filePreview) URL.revokeObjectURL(filePreview);
    setFile(null);
    setFilePreview(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
  }, [filePreview]);

  const handleSend = useCallback(() => {
    const q = input.trim();
    if (!q || isStreaming || disabled) return;
    onSend(q, file);
    setInput('');
    clearFile();
    textareaRef.current?.focus();
  }, [input, isStreaming, disabled, file, onSend, clearFile]);

  const handleKeyDown = useCallback((e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  }, [handleSend]);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    const dropped = e.dataTransfer.files[0];
    if (dropped) handleFileSelect(dropped);
  }, [handleFileSelect]);

  // Auto-resize textarea
  const handleInput = useCallback((e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInput(e.target.value);
    const ta = e.target;
    ta.style.height = 'auto';
    ta.style.height = Math.min(ta.scrollHeight, 160) + 'px';
  }, []);

  const canSend = input.trim().length > 0 && !disabled;

  return (
    <div className={styles.zone}>
      {/* File attachment preview */}
      {file && (
        <div className={styles.attachBar}>
          <div className={styles.attachedFile}>
            {filePreview ? (
              <img src={filePreview} alt="preview" className={styles.attachedThumb} />
            ) : (
              <div className={styles.attachedIcon}>
                <FileText size={16} />
              </div>
            )}
            <div className={styles.attachedInfo}>
              <span className={styles.attachedName}>{file.name}</span>
              <span className={styles.attachedSize}>{(file.size / 1024).toFixed(1)} KB</span>
            </div>
            <button className={styles.removeFile} onClick={clearFile}>
              <X size={13} />
            </button>
          </div>
        </div>
      )}

      {/* Cancel banner */}
      {isStreaming && (
        <div className={styles.cancelBanner}>
          <button className={styles.cancelBtn} onClick={onCancel}>
            <StopCircle size={15} />
            Stop Generating
          </button>
        </div>
      )}

      {/* Input row */}
      <div
        className={clsx(styles.inputRow, dragOver && styles.dragOver)}
        onDragOver={e => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={handleDrop}
      >
        <button
          className={styles.attachBtn}
          onClick={() => fileInputRef.current?.click()}
          disabled={isStreaming || disabled}
          title="Attach file or image"
        >
          <Paperclip size={17} />
        </button>

        <input
          ref={fileInputRef}
          type="file"
          accept={ALLOWED_TYPES.join(',')}
          className={styles.hiddenInput}
          onChange={e => handleFileSelect(e.target.files?.[0] ?? null)}
        />

        <textarea
          ref={textareaRef}
          className={styles.textarea}
          placeholder={disabled ? 'API offline — start backend first' : 'Message AI Assistant... (Shift+Enter for newline)'}
          value={input}
          onChange={handleInput}
          onKeyDown={handleKeyDown}
          disabled={isStreaming || disabled}
          rows={1}
        />

        <button
          className={clsx(styles.sendBtn, canSend && styles.sendActive)}
          onClick={handleSend}
          disabled={!canSend}
          title="Send (Enter)"
        >
          <Send size={16} />
        </button>
      </div>

      <p className={styles.disclaimer}>
        AI can make mistakes. Verify important information with source documents.
      </p>
    </div>
  );
}
