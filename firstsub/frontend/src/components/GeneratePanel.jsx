// GeneratePanel.jsx — Antigravity slider + generate crystal button
import { useState } from "react";
import axios from "axios";

const API = "http://localhost:8000";

export default function GeneratePanel({ onGenerate, onLog }) {
  const [score, setScore]     = useState(0);
  const [element, setElement] = useState("C");
  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState("");

  const handleGenerate = async () => {
    setLoading(true); setError("");
    try {
      const res = await axios.post(`${API}/generate`, {
        antigravity_score: score, element,
      });
      onGenerate(res.data);
      onLog(`Generated crystal | score=${score} | atoms=${res.data.params.num_atoms}`);
    } catch (e) {
      const msg = e.response?.data?.detail || e.message;
      setError(msg); onLog(`Generate error: ${msg}`);
    } finally { setLoading(false); }
  };

  return (
    <div className="card" style={{ display: "flex", flexDirection: "column", gap: 0 }}>
      <div className="card-title">
        <span className="card-title-icon">⚙</span>
        Generation Controls
      </div>

      {/* Antigravity Score Slider */}
      <div className="slider-row">
        <div className="slider-header">
          <span className="slider-label">Antigravity Score</span>
          <span className="slider-value">{score.toFixed(2)}</span>
        </div>
        <input
          type="range" min={-2} max={2} step={0.05}
          value={score}
          onChange={(e) => setScore(parseFloat(e.target.value))}
        />
        <div className="slider-bounds"><span>-2 Stable</span><span>0</span><span>+2 Exotic</span></div>
      </div>

      {/* Num Atoms (derived from score for display — purely UI) */}
      <div className="slider-row">
        <div className="slider-header">
          <span className="slider-label">Lattice Bias (Å)</span>
          <span className="slider-value">{(5 + (score + 2) * 2.5).toFixed(1)}</span>
        </div>
        <input
          type="range" min={5} max={20} step={0.5}
          value={5 + (score + 2) * 2.5}
          readOnly
          style={{ pointerEvents: "none" }}
        />
        <div className="slider-bounds"><span>5.0</span><span>20.0</span></div>
      </div>

      {/* Temperature slider (visual only) */}
      <div className="slider-row">
        <div className="slider-header">
          <span className="slider-label">Temperature (Sampling)</span>
          <span className="slider-value">{(1 + score * 0.25).toFixed(1)}</span>
        </div>
        <input
          type="range" min={0.1} max={2} step={0.1}
          value={1 + score * 0.25}
          readOnly
          style={{ pointerEvents: "none" }}
        />
        <div className="slider-bounds"><span>0.1</span><span>2.0</span></div>
      </div>

      {/* Element picker */}
      <div style={{ marginBottom: 20 }}>
        <label className="label">Element</label>
        <select
          value={element}
          onChange={(e) => setElement(e.target.value)}
          className="input-field"
          style={{ width: "100%" }}
        >
          {["C","Si","Ge","Fe","Cu","Al","Ti","N","O"].map(el => (
            <option key={el} value={el}>{el}</option>
          ))}
        </select>
      </div>

      {/* Generate button */}
      <button className="btn-primary" onClick={handleGenerate} disabled={loading}
        style={{ width: "100%", justifyContent: "center", padding: "12px" }}>
        {loading
          ? <><span className="animate-spin" style={{ display:"inline-block" }}>↻</span> Generating…</>
          : "⊕ Generate Crystal"}
      </button>

      {error && (
        <p style={{ color: "var(--red)", fontSize: "0.78rem", marginTop: 8 }}>{error}</p>
      )}
    </div>
  );
}
