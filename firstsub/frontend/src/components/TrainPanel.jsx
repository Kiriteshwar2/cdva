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
      } catch (_) {}
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
    <div className="card" style={{ display:"flex", flexDirection:"column", gap:"1rem" }}>
      <h2 style={{ color:"var(--accent)", fontWeight:700, fontSize:"1.05rem" }}>🧠 Train Model</h2>

      <div style={{ display:"flex", alignItems:"flex-end", gap:"0.75rem", flexWrap:"wrap" }}>
        <div>
          <label className="label">Epochs</label>
          <input type="number" min={10} max={2000} value={epochs}
            onChange={(e) => setEpochs(e.target.value)}
            className="input-dark" style={{ width:90 }} />
        </div>
        <button className="btn-primary" onClick={handleTrain} disabled={training}>
          {training ? "Training…" : "Train Model"}
        </button>
      </div>

      {error && <p style={{ color:"#f87171", fontSize:"0.82rem" }}>{error}</p>}

      {status && (
        <div style={{ display:"flex", flexDirection:"column", gap:"0.5rem" }}>
          {/* Progress bar */}
          <div style={{ display:"flex", justifyContent:"space-between", fontSize:"0.75rem", color:"#9ca3af" }}>
            <span>Epoch {status.epoch} / {status.total_epochs}</span>
            <span>{pct}%</span>
          </div>
          <div style={{ width:"100%", height:8, borderRadius:4, backgroundColor:"var(--border)" }}>
            <div style={{
              width:`${pct}%`, height:"100%", borderRadius:4,
              backgroundColor:"var(--accent)", transition:"width 0.3s ease",
            }} />
          </div>

          {/* Loss metrics */}
          <div style={{ display:"grid", gridTemplateColumns:"repeat(3,1fr)", gap:"0.5rem" }}>
            {[["Total Loss", status.loss], ["Recon Loss", status.recon_loss], ["KL Loss", status.kl_loss]]
              .map(([label, val]) => (
              <div className="metric-card" key={label}>
                <div className="metric-label">{label}</div>
                <div className="metric-value">
                  {typeof val === "number" ? val.toFixed(4) : "—"}
                </div>
              </div>
            ))}
          </div>

          {/* Loss chart */}
          {status.history?.length > 1 && <LossChart data={status.history} />}

          {status.done && !training && (
            <p style={{ color:"#6ee7b7", fontWeight:600, fontSize:"0.85rem" }}>
              Training complete — model saved!
            </p>
          )}
        </div>
      )}
    </div>
  );
}
