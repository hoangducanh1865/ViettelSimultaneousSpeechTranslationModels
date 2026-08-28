"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import {
  SampleDetail,
  SampleListItem,
  audioSrc,
  exportCsvUrl,
  getSample,
  listSamples,
  resetSample,
  submitSample,
} from "../lib/api";

export default function Page() {
  const [items, setItems] = useState<SampleListItem[]>([]);
  const [currentId, setCurrentId] = useState<string | null>(null);
  const [detail, setDetail] = useState<SampleDetail | null>(null);
  const [draft, setDraft] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const refreshList = useCallback(async () => {
    const list = await listSamples();
    setItems(list);
    return list;
  }, []);

  useEffect(() => {
    refreshList().then((list) => {
      if (list.length > 0) setCurrentId(list[0].id);
    });
  }, [refreshList]);

  useEffect(() => {
    if (!currentId) return;
    setLoading(true);
    setError(null);
    getSample(currentId)
      .then((d) => {
        setDetail(d);
        setDraft(d.final_asr_text ?? d.asr_rover ?? d.asr_internal ?? d.asr_google ?? d.asr_phowhisper ?? "");
      })
      .catch((e) => setError(String(e)))
      .finally(() => setLoading(false));
  }, [currentId]);

  const currentIndex = useMemo(
    () => items.findIndex((it) => it.id === currentId),
    [items, currentId]
  );
  const doneCount = useMemo(() => items.filter((it) => it.status === "done").length, [items]);

  const goTo = (index: number) => {
    if (index < 0 || index >= items.length) return;
    setCurrentId(items[index].id);
  };

  const handleSubmit = async () => {
    if (!currentId || !draft.trim()) return;
    setLoading(true);
    try {
      const updated = await submitSample(currentId, draft.trim());
      setDetail(updated);
      const list = await refreshList();
      const idx = list.findIndex((it) => it.id === currentId);
      if (idx >= 0 && idx + 1 < list.length) {
        setCurrentId(list[idx + 1].id);
      }
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  };

  const handleReset = async () => {
    if (!currentId) return;
    setLoading(true);
    try {
      const updated = await resetSample(currentId);
      setDetail(updated);
      setDraft(updated.asr_rover ?? updated.asr_internal ?? updated.asr_google ?? updated.asr_phowhisper ?? "");
      await refreshList();
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  };

  const candidates: { label: string; text: string | null }[] = detail
    ? [
        { label: "ROVER (đã hợp nhất 3 nguồn)", text: detail.asr_rover },
        { label: "ASR nội bộ (Viettel)", text: detail.asr_internal },
        { label: "Google Cloud STT", text: detail.asr_google },
        { label: "PhoWhisper", text: detail.asr_phowhisper },
      ]
    : [];

  return (
    <div className="layout">
      <aside className="sidebar">
        <h2>
          Sample ({doneCount}/{items.length} done)
        </h2>
        <a className="export-link" href={exportCsvUrl()} target="_blank" rel="noreferrer">
          Tải CSV kết quả ↓
        </a>
        <div style={{ marginTop: 16 }}>
          {items.map((it) => (
            <div
              key={it.id}
              className={`sidebar-item ${it.id === currentId ? "active" : ""}`}
              onClick={() => setCurrentId(it.id)}
              title={it.id}
            >
              <span>
                {it.source_dataset} · {it.speaker_id ?? "-"}
              </span>
              <span className={`status-dot ${it.status}`} />
            </div>
          ))}
        </div>
      </aside>

      <main className="main">
        <div className="top-bar">
          <div className="nav-buttons">
            <button disabled={currentIndex <= 0} onClick={() => goTo(currentIndex - 1)}>
              ← Back
            </button>
            <button
              disabled={currentIndex < 0 || currentIndex >= items.length - 1}
              onClick={() => goTo(currentIndex + 1)}
            >
              Next →
            </button>
          </div>
          <div className="progress">
            Sample {currentIndex + 1} / {items.length}
          </div>
        </div>

        {error && <div className="card" style={{ color: "#d03b3b" }}>{error}</div>}

        {detail && (
          <>
            <div className="card">
              <h3>Audio</h3>
              <audio key={detail.id} controls src={audioSrc(detail.audio_url)} />
              <div style={{ marginTop: 8, fontSize: 13, color: "#6b6a63" }}>
                {detail.id} · {detail.duration_sec?.toFixed(2)}s
                <span className={`status-badge ${detail.status}`}>
                  {detail.status === "done" ? "Đã submit" : "Chưa submit"}
                </span>
              </div>
            </div>

            <div className="card">
              <h3>3 bản ASR ứng viên (bấm để copy vào ô sửa bên dưới)</h3>
              {candidates.map((c) => (
                <div
                  key={c.label}
                  className="asr-candidate"
                  onClick={() => c.text && setDraft(c.text)}
                >
                  <div className="label">{c.label}</div>
                  {c.text ? c.text : <span className="empty">(không có / lỗi)</span>}
                </div>
              ))}
            </div>

            <div className="card">
              <h3>Bản ASR final (sửa tay tại đây)</h3>
              <textarea value={draft} onChange={(e) => setDraft(e.target.value)} />
              <div className="action-buttons">
                <button onClick={handleReset} className="danger" disabled={loading}>
                  Làm lại
                </button>
                <button onClick={handleSubmit} className="primary" disabled={loading || !draft.trim()}>
                  Submit
                </button>
              </div>
            </div>
          </>
        )}
      </main>
    </div>
  );
}
