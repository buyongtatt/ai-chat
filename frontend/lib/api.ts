const API_BASE =
  process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api/v1";

const METADATA_PREFIX = "@@METADATA@@";
const METADATA_SUFFIX = "@@END@@";

export interface StreamMetadata {
  sources: { doc_id: string; doc_name: string }[];
  cited_images: {
    image_label: string;
    url: string;
    doc_name: string;
    page: number;
    label: string;
  }[];
  total_images_sent: number;
}

export async function fetchStats() {
  const res = await fetch(`${API_BASE}/stats`);
  if (!res.ok) throw new Error("API unavailable");
  return res.json();
}

export async function fetchDocuments() {
  const res = await fetch(`${API_BASE}/documents`);
  if (!res.ok) throw new Error("Failed to fetch documents");
  return res.json();
}

export async function fetchSources(query: string) {
  const res = await fetch(`${API_BASE}/sources?q=${encodeURIComponent(query)}`);
  if (!res.ok) return { sources: [], images: [] };
  return res.json();
}

export async function cancelSession(sessionId: string) {
  try {
    await fetch(`${API_BASE}/cancel/${sessionId}`, { method: "POST" });
  } catch (e) {
    console.error("Cancel failed:", e);
  }
}

/**
 * Stream answer tokens. The last chunk from the server contains a metadata
 * footer: @@METADATA@@{...json...}@@END@@
 * This generator strips it out and instead resolves the onMetadata callback
 * so the answer text is always clean.
 */
export async function* streamAnswer(
  question: string,
  sessionId: string,
  file?: File | null,
  signal?: AbortSignal,
  onMetadata?: (meta: StreamMetadata) => void,
): AsyncGenerator<string> {
  const form = new FormData();
  form.append("question", question);
  form.append("session_id", sessionId);
  if (file) form.append("file", file);

  const res = await fetch(`${API_BASE}/ask`, {
    method: "POST",
    body: form,
    signal,
  });

  if (!res.ok) throw new Error(`API error: ${res.status}`);

  const reader = res.body?.getReader();
  if (!reader) throw new Error("No response body");

  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });

    // Check if the metadata footer has fully arrived
    const metaStart = buffer.indexOf(METADATA_PREFIX);
    if (metaStart !== -1) {
      const metaEnd = buffer.indexOf(METADATA_SUFFIX, metaStart);
      if (metaEnd !== -1) {
        // Yield clean text before the footer (strip leading newline)
        const textBefore = buffer.slice(0, metaStart).replace(/\n$/, "");
        if (textBefore) yield textBefore;

        // Parse and fire metadata callback
        try {
          const jsonStr = buffer.slice(
            metaStart + METADATA_PREFIX.length,
            metaEnd,
          );
          const meta: StreamMetadata = JSON.parse(jsonStr);
          onMetadata?.(meta);
        } catch (e) {
          console.error("Failed to parse stream metadata:", e);
        }

        buffer = buffer.slice(metaEnd + METADATA_SUFFIX.length);
        continue;
      }
      // Footer started but not complete yet — hold buffer and wait for more chunks
      const textBefore = buffer.slice(0, metaStart);
      if (textBefore) {
        yield textBefore;
        buffer = buffer.slice(metaStart);
      }
      continue;
    }

    // No metadata marker — yield everything except the last 20 chars
    // (guard against METADATA_PREFIX arriving split across chunks)
    if (buffer.length > 20) {
      yield buffer.slice(0, -20);
      buffer = buffer.slice(-20);
    }
  }

  // Flush remaining buffer (shouldn't contain metadata at this point)
  if (buffer && !buffer.includes(METADATA_PREFIX)) {
    yield buffer;
  }
}
