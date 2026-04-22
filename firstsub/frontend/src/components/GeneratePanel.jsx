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

  const accent = score < -0.5 ? "#60a5fa" : score > 0.5 ? "#8b5cf6" : "#14b8a6";

  return (
    <div className="card" style={{ display:"flex", flexDirection:"column", gap:"1rem" }}>
      <h2 style={{ color:"var(--accent)", fontWeight:700, fontSize:"1.05rem" }}>⚡ Generate Crystal</h2>

      {/* Slider */}
      <div>
        <label className="label">
          Antigravity Score &nbsp;
          <span style={{ color:"#fff", fontWeight:700, fontFamily:"monospace" }}>
            {score.toFixed(2)}
          </span>
        </label>
        <input type="range" min={-2} max={2} step={0.05}
          value={score} onChange={(e) => setScore(parseFloat(e.target.value))}
          style={{ width:"100%", accentColor: accent }} />
        <div style={{ display:"flex", justifyContent:"space-between",
                      fontSize:"0.65rem", color:"#6b7280", marginTop:2 }}>
          <span>-2 Dense/Stable</span><span>0 Neutral</span><span>+2 Exotic</span>
        </div>
        <div style={{
          marginTop:"0.5rem", padding:"0.5rem 0.75rem", borderRadius:8,
          background:`${accent}22`, border:`1px solid ${accent}55`,
          fontSize:"0.82rem", color:"#e5e7eb",
        }}>
          {score < -0.5 ? "Dense, stable-like structures"
           : score > 0.5 ? "Exotic, high-energy structures"
           : "Balanced latent space sampling"}
        </div>
      </div>

      {/* Element picker */}
      <div>
        <label className="label">Element</label>
        <select value={element} onChange={(e) => setElement(e.target.value)}
          className="input-dark">
          {["C","Si","Ge","Fe","Cu","Al","Ti","N","O"].map(el =>
            <option key={el} value={el}>{el}</option>)}
        </select>
      </div>

      <button className="btn-primary" onClick={handleGenerate} disabled={loading}>
        {loading ? "Generating…" : "Generate Material"}
      </button>

      {error && <p style={{ color:"#f87171", fontSize:"0.82rem" }}>{error}</p>}
    </div>
  );
}
