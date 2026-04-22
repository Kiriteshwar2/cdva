// MultiGenPanel.jsx — Generate 5 diverse crystal samples at once
import { useState } from "react";
import axios from "axios";

const API = "http://localhost:8000";

const COLS = {
  num_atoms:"Atoms", volume:"Vol(Å³)", a:"a(Å)", b:"b(Å)",
  c:"c(Å)", alpha:"α(°)", beta:"β(°)", gamma:"γ(°)"
};

export default function MultiGenPanel({ antigravityScore, onLog }) {
  const [samples, setSamples] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState("");

  const handleMultiGen = async () => {
    setLoading(true); setError("");
    try {
      const res = await axios.get(`${API}/generate-multiple`, {
        params: { n:5, antigravity_score: antigravityScore },
      });
      setSamples(res.data.samples);
      onLog(`Generated 5 samples at score=${antigravityScore}`);
    } catch (e) {
      const msg = e.response?.data?.detail || e.message;
      setError(msg); onLog(`Multi-gen error: ${msg}`);
    } finally { setLoading(false); }
  };

  return (
    <div className="card" style={{ display:"flex", flexDirection:"column", gap:"1rem" }}>
      <div style={{ display:"flex", justifyContent:"space-between", alignItems:"center", flexWrap:"wrap", gap:4 }}>
        <h2 style={{ color:"var(--accent)", fontWeight:700, fontSize:"1.05rem" }}>⚗️ Generate 5 Samples</h2>
        <span style={{ fontSize:"0.74rem", color:"#9ca3af" }}>
          score <span style={{ color:"var(--purple)", fontWeight:700 }}>{antigravityScore}</span>
        </span>
      </div>

      <button className="btn-secondary" onClick={handleMultiGen} disabled={loading}>
        {loading ? "Generating…" : "Generate 5 Diverse Samples"}
      </button>

      {error && <p style={{ color:"#f87171", fontSize:"0.82rem" }}>{error}</p>}

      {samples.length > 0 && (
        <div style={{ overflowX:"auto" }}>
          <table className="dark-table">
            <thead>
              <tr>
                <th>#</th>
                {Object.keys(COLS).map(k => <th key={k}>{COLS[k]}</th>)}
              </tr>
            </thead>
            <tbody>
              {samples.map((s) => (
                <tr key={s.index}>
                  <td>{s.index + 1}</td>
                  {Object.keys(COLS).map(k => (
                    <td key={k} className="right">
                      {typeof s[k] === "number" ? s[k].toFixed(3) : s[k]}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
