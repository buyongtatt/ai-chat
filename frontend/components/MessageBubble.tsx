'use client';
import { Message } from '@/types';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneDark } from 'react-syntax-highlighter/dist/cjs/styles/prism';
import { oneLight } from 'react-syntax-highlighter/dist/cjs/styles/prism';
import { useState, useCallback } from 'react';
import { Copy, Check, FileText, Image as ImageIcon } from 'lucide-react';
import styles from './MessageBubble.module.css';
import clsx from 'clsx';

interface Props { message: Message; }

export default function MessageBubble({ message }: Props) {
  const isUser = message.role === 'user';
  const [copied, setCopied] = useState(false);
  const isDark = typeof document !== 'undefined'
    ? document.documentElement.getAttribute('data-theme') === 'dark'
    : true;

  const handleCopy = useCallback(async () => {
    await navigator.clipboard.writeText(message.content);
    setCopied(true);
    setTimeout(() => setCopied(false), 1800);
  }, [message.content]);

  return (
    <div className={clsx(styles.wrapper, isUser && styles.userWrapper, 'animate-fadeSlideUp')}>
      {/* Avatar */}
      <div className={clsx(styles.avatar, isUser ? styles.userAvatar : styles.aiAvatar)}>
        {isUser ? 'U' : 'AI'}
      </div>

      <div className={clsx(styles.bubbleCol, isUser && styles.userBubbleCol)}>
        {/* Attached file preview (user side) */}
        {isUser && message.attachedFile && (
          <div className={styles.attachPreview}>
            {message.attachedFile.type === 'image' && message.attachedFile.previewUrl ? (
              <img
                src={message.attachedFile.previewUrl}
                alt={message.attachedFile.name}
                className={styles.attachThumb}
              />
            ) : (
              <FileText size={16} className={styles.attachIcon} />
            )}
            <div className={styles.attachInfo}>
              <span className={styles.attachName}>{message.attachedFile.name}</span>
              <span className={styles.attachSize}>
                {(message.attachedFile.size / 1024).toFixed(1)} KB
              </span>
            </div>
          </div>
        )}

        {/* Main bubble */}
        <div className={clsx(styles.bubble, isUser ? styles.userBubble : styles.aiBubble)}>
          {message.content ? (
            isUser ? (
              <p className={styles.userText}>{message.content}</p>
            ) : (
              <div className="prose">
                <ReactMarkdown
                  remarkPlugins={[remarkGfm]}
                  components={{
                    code({ node, className, children, ...props }) {
                      const match = /language-(\w+)/.exec(className || '');
                      const isBlock = !!match;
                      if (isBlock) {
                        return (
                          <div className={styles.codeBlock}>
                            <div className={styles.codeHeader}>
                              <span className={styles.codeLang}>{match![1]}</span>
                              <CopyCode code={String(children)} />
                            </div>
                            <SyntaxHighlighter
                              style={isDark ? oneDark : oneLight}
                              language={match![1]}
                              PreTag="div"
                              customStyle={{ margin: 0, background: 'transparent', padding: '12px 16px' }}
                            >
                              {String(children).replace(/\n$/, '')}
                            </SyntaxHighlighter>
                          </div>
                        );
                      }
                      return <code className={className} {...props}>{children}</code>;
                    },
                  }}
                >
                  {message.content}
                </ReactMarkdown>
                {message.isStreaming && <span className={styles.cursor} />}
              </div>
            )
          ) : (
            message.isStreaming && <ThinkingDots />
          )}

          {message.cancelled && (
            <span className={styles.cancelledNote}>— cancelled</span>
          )}
        </div>

        {/* Source images from knowledge base */}
        {!isUser && message.sourceImages && message.sourceImages.length > 0 && !message.isStreaming && (
          <div className={styles.sourceImages}>
            <div className={styles.sourceImagesTitle}>
              <ImageIcon size={12} /> Referenced document images
            </div>
            <div className={styles.imageGrid}>
              {message.sourceImages.map((img, i) => (
                <div key={i} className={styles.imageCard}>
                  <img
                    src={img.url}
                    alt={`${img.doc_name} p.${img.page + 1}`}
                    className={styles.docImage}
                    loading="lazy"
                  />
                  <div className={styles.imageCaption}>
                    <span className={styles.imageName}>{img.doc_name}</span>
                    <span className={styles.imagePage}>p.{img.page + 1}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Source chips */}
        {!isUser && message.sources && message.sources.length > 0 && !message.isStreaming && (
          <div className={styles.sources}>
            {message.sources.map((src, i) => (
              <span key={i} className={styles.sourceChip}>
                {src.source_type === 'image' ? <ImageIcon size={10} /> : <FileText size={10} />}
                {src.doc_name}
              </span>
            ))}
          </div>
        )}

        {/* Timestamp + copy */}
        {!isUser && message.content && !message.isStreaming && (
          <div className={styles.actions}>
            <span className={styles.time}>
              {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
            </span>
            <button className={styles.copyBtn} onClick={handleCopy}>
              {copied ? <Check size={13} /> : <Copy size={13} />}
              {copied ? 'Copied' : 'Copy'}
            </button>
          </div>
        )}
        {isUser && (
          <div className={clsx(styles.actions, styles.userActions)}>
            <span className={styles.time}>
              {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
            </span>
          </div>
        )}
      </div>
    </div>
  );
}

function ThinkingDots() {
  return (
    <div className={styles.thinking}>
      <span className={styles.dot1} />
      <span className={styles.dot2} />
      <span className={styles.dot3} />
      <span className={styles.thinkingText}>Analyzing...</span>
    </div>
  );
}

function CopyCode({ code }: { code: string }) {
  const [copied, setCopied] = useState(false);
  return (
    <button
      className={styles.codeCopyBtn}
      onClick={async () => {
        await navigator.clipboard.writeText(code);
        setCopied(true);
        setTimeout(() => setCopied(false), 1500);
      }}
    >
      {copied ? <Check size={12} /> : <Copy size={12} />}
      {copied ? 'Copied' : 'Copy'}
    </button>
  );
}
