const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api/v1';

export async function fetchStats() {
  const res = await fetch(`${API_BASE}/stats`);
  if (!res.ok) throw new Error('API unavailable');
  return res.json();
}

export async function fetchDocuments() {
  const res = await fetch(`${API_BASE}/documents`);
  if (!res.ok) throw new Error('Failed to fetch documents');
  return res.json();
}

export async function fetchSources(query: string) {
  const res = await fetch(`${API_BASE}/sources?q=${encodeURIComponent(query)}`);
  if (!res.ok) return { sources: [], images: [] };
  return res.json();
}

export async function cancelSession(sessionId: string) {
  try {
    await fetch(`${API_BASE}/cancel/${sessionId}`, { method: 'POST' });
  } catch (e) {
    console.error('Cancel failed:', e);
  }
}

export async function* streamAnswer(
  question: string,
  sessionId: string,
  file?: File | null,
  signal?: AbortSignal
): AsyncGenerator<string> {
  const form = new FormData();
  form.append('question', question);
  form.append('session_id', sessionId);
  if (file) form.append('file', file);

  const res = await fetch(`${API_BASE}/ask`, {
    method: 'POST',
    body: form,
    signal,
  });

  if (!res.ok) {
    throw new Error(`API error: ${res.status}`);
  }

  const reader = res.body?.getReader();
  if (!reader) throw new Error('No response body');

  const decoder = new TextDecoder();
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    yield decoder.decode(value, { stream: true });
  }
}
