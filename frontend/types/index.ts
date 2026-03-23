export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: Date;
  attachedFile?: AttachedFile;
  sourceImages?: SourceImage[];
  sources?: Source[];
  isStreaming?: boolean;
  cancelled?: boolean;
}

export interface AttachedFile {
  name: string;
  type: "image" | "document";
  previewUrl?: string; // object URL for images
  size: number;
}

export interface Source {
  doc_id: string;
  doc_name: string;
  source_type: string;
}

export interface SourceImage {
  url: string;
  doc_name: string;
  page: number;
  summary: string;
  label?: string; // e.g. "Image 2" — matches [Image N] in model answer
}

export interface Document {
  id: string;
  name: string;
  type: string;
  chunks: number;
  image_chunks: number;
}

export interface Stats {
  total_documents: number;
  total_chunks: number;
  model: string;
}
