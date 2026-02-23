'use client';
import { useState, useEffect, useCallback } from 'react';
import { fetchStats, fetchDocuments } from '@/lib/api';
import { Document, Stats } from '@/types';
import Sidebar from '@/components/Sidebar';
import ChatArea from '@/components/ChatArea';
import styles from './page.module.css';

export default function Home() {
  const [theme, setTheme] = useState<'dark' | 'light'>('dark');
  const [documents, setDocuments] = useState<Document[]>([]);
  const [stats, setStats] = useState<Stats | null>(null);
  const [apiOnline, setApiOnline] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(true);

  useEffect(() => {
    const saved = localStorage.getItem('docmind-theme') as 'dark' | 'light' | null;
    if (saved) setTheme(saved);
  }, []);

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('docmind-theme', theme);
  }, [theme]);

  const loadData = useCallback(async () => {
    try {
      const [statsData, docsData] = await Promise.all([fetchStats(), fetchDocuments()]);
      setStats(statsData);
      setDocuments(docsData.documents || []);
      setApiOnline(true);
    } catch {
      setApiOnline(false);
    }
  }, []);

  useEffect(() => {
    loadData();
    const interval = setInterval(loadData, 30000);
    return () => clearInterval(interval);
  }, [loadData]);

  const toggleTheme = () => setTheme(t => (t === 'dark' ? 'light' : 'dark'));

  return (
    <div className={styles.layout}>
      <Sidebar
        documents={documents}
        stats={stats}
        apiOnline={apiOnline}
        open={sidebarOpen}
        onToggle={() => setSidebarOpen(o => !o)}
      />
      <ChatArea
        theme={theme}
        onToggleTheme={toggleTheme}
        onToggleSidebar={() => setSidebarOpen(o => !o)}
        apiOnline={apiOnline}
      />
    </div>
  );
}
