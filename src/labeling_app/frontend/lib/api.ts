const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export type SampleListItem = {
  id: string;
  order_index: number;
  source_dataset: string;
  speaker_id: string | null;
  duration_sec: number | null;
  status: "pending" | "done";
};

export type SampleDetail = SampleListItem & {
  audio_url: string;
  asr_google: string | null;
  asr_internal: string | null;
  asr_phowhisper: string | null;
  asr_rover: string | null;
  final_asr_text: string | null;
  updated_at: string | null;
};

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${API_URL}${path}`, {
    ...options,
    headers: { "Content-Type": "application/json", ...(options?.headers || {}) },
  });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`${res.status} ${res.statusText}: ${body}`);
  }
  return res.json() as Promise<T>;
}

export function listSamples(): Promise<SampleListItem[]> {
  return request<SampleListItem[]>("/api/samples");
}

export function getSample(id: string): Promise<SampleDetail> {
  return request<SampleDetail>(`/api/samples/${encodeURIComponent(id)}`);
}

export function submitSample(id: string, finalAsrText: string): Promise<SampleDetail> {
  return request<SampleDetail>(`/api/samples/${encodeURIComponent(id)}/submit`, {
    method: "POST",
    body: JSON.stringify({ final_asr_text: finalAsrText }),
  });
}

export function resetSample(id: string): Promise<SampleDetail> {
  return request<SampleDetail>(`/api/samples/${encodeURIComponent(id)}/reset`, {
    method: "POST",
  });
}

export function audioSrc(audioUrl: string): string {
  return `${API_URL}${audioUrl}`;
}

export function exportCsvUrl(): string {
  return `${API_URL}/api/export/asr-check.csv`;
}
