'use client';
import { Document, Stats } from '@/types';
import styles from './Sidebar.module.css';
import { FileText, Image, BookOpen, Database, ChevronLeft } from 'lucide-react';
import clsx from 'clsx';

interface Props {
  documents: Document[];
  stats: Stats | null;
  apiOnline: boolean;
  open: boolean;
  onToggle: () => void;
}

const FILE_ICONS: Record<string, React.ReactNode> = {
  pdf:  <span className={styles.badge} style={{background:'#ef4444'}}>PDF</span>,
  docx: <span className={styles.badge} style={{background:'#3b82f6'}}>DOC</span>,
  txt:  <span className={styles.badge} style={{background:'#10b981'}}>TXT</span>,
  md:   <span className={styles.badge} style={{background:'#8b5cf6'}}>MD</span>,
  png:  <span className={styles.badge} style={{background:'#f59e0b'}}>IMG</span>,
  jpg:  <span className={styles.badge} style={{background:'#f59e0b'}}>IMG</span>,
};

export default function Sidebar({ documents, stats, apiOnline, open, onToggle }: Props) {
  return (
    <aside className={clsx(styles.sidebar, !open && styles.collapsed)}>
      <div className={styles.header}>
        <div className={styles.logo}>
          <div className={styles.logoIcon}>
            <BookOpen size={16} />
          </div>
          {open && <span className={styles.logoText}>DocMind</span>}
        </div>
        <button className={styles.collapseBtn} onClick={onToggle} title="Toggle sidebar">
          <ChevronLeft size={16} className={clsx(styles.chevron, !open && styles.rotated)} />
        </button>
      </div>

      {open && (
        <>
          {/* Status */}
          <div className={styles.statusRow}>
            <span className={clsx(styles.dot, apiOnline && styles.online)} />
            <span className={styles.statusText}>
              {apiOnline ? 'llama3.2-vision • online' : 'API offline'}
            </span>
          </div>

          {/* Stats */}
          {stats && (
            <div className={styles.statsGrid}>
              <div className={styles.statCard}>
                <Database size={13} />
                <div>
                  <div className={styles.statNum}>{stats.total_documents}</div>
                  <div className={styles.statLabel}>Documents</div>
                </div>
              </div>
              <div className={styles.statCard}>
                <BookOpen size={13} />
                <div>
                  <div className={styles.statNum}>{stats.total_chunks}</div>
                  <div className={styles.statLabel}>Chunks</div>
                </div>
              </div>
            </div>
          )}

          {/* Document list */}
          <div className={styles.sectionTitle}>Knowledge Base</div>
          <div className={styles.docList}>
            {documents.length === 0 ? (
              <div className={styles.empty}>
                <FileText size={28} />
                <p>No documents indexed</p>
                <span>Run the indexer to get started</span>
              </div>
            ) : (
              documents.map(doc => (
                <div key={doc.id} className={styles.docItem}>
                  <div className={styles.docIcon}>
                    {FILE_ICONS[doc.type] ?? <FileText size={14} />}
                  </div>
                  <div className={styles.docInfo}>
                    <div className={styles.docName} title={doc.name}>{doc.name}</div>
                    <div className={styles.docMeta}>
                      <span>{doc.chunks} chunks</span>
                      {doc.image_chunks > 0 && (
                        <span className={styles.imgBadge}>
                          <Image size={10} /> {doc.image_chunks}
                        </span>
                      )}
                    </div>
                  </div>
                </div>
              ))
            )}
          </div>
        </>
      )}
    </aside>
  );
}
