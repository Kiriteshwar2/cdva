// TrainPanel.jsx — Start training and watch epoch/loss live
import { useState, useEffect, useRef } from "react";
import axios from "axios";
import LossChart from "./LossChart";

const API = "http://localhost:8000";

export default function TrainPanel({ onLog }) {
  const [epochs, setEpochs]     = useState(300);
  const [status, setStatus]     = useState(null);
  const [training, setTraining] = useState(false);
  const [error, setError]       = useState("");
  const pollRef = useRef(null);

  const startPoll = () => {
    pollRef.current = setInterval(async () => {
      try {
        const res = await axios.get(`${API}/train-status`);
        setStatus(res.data);
        if (res.data.done) {
          setTraining(false);
          clearInterval(pollRef.current);
          onLog(res.data.error
            ? `Training error: ${res.data.error}`
            : `Training done! Final loss: ${res.data.loss}`);
        }
      } catch {
        setError("Unable to read training status.");
      }
    }, 1000);
  };

  const handleTrain = async () => {
    setError(""); setTraining(true);
    try {
      await axios.post(`${API}/train`, { epochs: Number(epochs) });
      onLog(`Training started for ${epochs} epochs…`);
      startPoll();
    } catch (e) {
      const msg = e.response?.data?.detail || e.message;
      setError(msg); setTraining(false); onLog(`Error: ${msg}`);
    }
  };

  useEffect(() => () => clearInterval(pollRef.current), []);

  const pct = status
    ? Math.round((status.epoch / Math.max(status.total_epochs - 1, 1)) * 100)
    : 0;

  return (
    <div className="card" style={{ display: "flex", flexDirection: "column", gap: 0 }}>
      {/* Header */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 }}>
        <div className="card-title" style={{ marginBottom: 0 }}>
          <span className="card-title-icon">📈</span>
          Training Progress
        </div>
        {status && (
          <span style={{ fontSize: "0.75rem", fontWeight: 600, color: "var(--accent)" }}>
            Epoch {status.epoch} / {status.total_epochs}
          </span>
        )}
      </div>

      {/* Controls row */}
      <div style={{ display: "flex", gap: 10, alignItems: "flex-end", marginBottom: 16 }}>
        <div style={{ flex: 1 }}>
          <label className="label">Epochs</label>
          <input
            type="number" min={10} max={2000} value={epochs}
            onChange={(e) => setEpochs(e.target.value)}
            className="input-field"
          />
        </div>
        <button className="btn-primary" onClick={handleTrain} disabled={training}
          style={{ whiteSpace: "nowrap", height: 38 }}>
          {training
            ? <><span className="animate-spin" style={{ display: "inline-block" }}>↻</span> Training…</>
            : "▶ Train Model"}
        </button>
      </div>

      {error && <p style={{ color: "var(--red)", fontSize: "0.78rem", marginBottom: 8 }}>{error}</p>}

      {/* Progress */}
      {status && (
        <>
          <div className="progress-track" style={{ marginBottom: 6 }}>
            <div className="progress-fill" style={{ width: `${pct}%` }} />
          </div>

          {/* Loss metrics */}
          <div style={{
            display: "grid", gridTemplateColumns: "repeat(3,1fr)",
            gap: 10, margin: "12px 0",
          }}>
            {[
              ["Total Loss",  status.loss],
              ["Recon Loss",  status.recon_loss],
              ["KL Loss",     status.kl_loss],
            ].map(([label, val]) => (
              <div key={label} style={{
                background: "var(--content-bg)", borderRadius: 8, padding: "10px 12px",
                border: "1px solid var(--border)", textAlign: "center",
              }}>
                <div style={{ fontSize: "0.68rem", color: "var(--text-muted)", marginBottom: 4 }}>{label}</div>
                <div style={{ fontSize: "1rem", fontWeight: 700, color: "var(--accent)", fontVariantNumeric: "tabular-nums" }}>
                  {typeof val === "number" ? val.toFixed(4) : "—"}
                </div>
              </div>
            ))}
          </div>

          {/* Loss chart */}
          {status.history?.length > 1 && <LossChart data={status.history} />}

          {status.done && !training && (
            <p style={{ color: "var(--green)", fontWeight: 600, fontSize: "0.82rem", marginTop: 8 }}>
              ✓ Training completed successfully!
            </p>
          )}
        </>
      )}

      {/* Empty state chart placeholder */}
      {!status && (
        <div style={{
          height: 140, background: "var(--content-bg)", borderRadius: 8,
          border: "1px dashed var(--border)",
          display: "flex", alignItems: "center", justifyContent: "center",
          color: "var(--text-light)", fontSize: "0.78rem",
        }}>
          Training chart will appear here
        </div>
      )}
    </div>
  );
}
