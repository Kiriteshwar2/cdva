// MultiGenPanel.jsx — Generate 5 diverse crystal samples at once
import { useState } from "react";
import axios from "axios";

const API = "http://localhost:8000";

const COLS = {
  num_atoms: "Atoms", volume: "Vol (Å³)", a: "a (Å)", b: "b (Å)",
  c: "c (Å)", alpha: "α (°)", beta: "β (°)", gamma: "γ (°)"
};

export default function MultiGenPanel({ antigravityScore, onLog }) {
  const [samples, setSamples] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState("");

  const handleMultiGen = async () => {
    setLoading(true); setError("");
    try {
      const res = await axios.get(`${API}/generate-multiple`, {
        params: { n: 5, antigravity_score: antigravityScore },
      });
      setSamples(res.data.samples);
      onLog(`Generated 5 samples at score=${antigravityScore}`);
    } catch (e) {
      const msg = e.response?.data?.detail || e.message;
      setError(msg); onLog(`Multi-gen error: ${msg}`);
    } finally { setLoading(false); }
  };

  return (
    <div className="card" style={{ display: "flex", flexDirection: "column", gap: 0 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 }}>
        <div className="card-title" style={{ marginBottom: 0 }}>
          <span className="card-title-icon">⚗</span>
          Batch Generate (5 Samples)
        </div>
        <span style={{
          fontSize: "0.72rem", color: "var(--accent)", fontWeight: 600,
          background: "rgba(91,94,244,0.08)", padding: "2px 8px", borderRadius: 5,
        }}>
          score {antigravityScore}
        </span>
      </div>

      <button
        className="btn-primary"
        onClick={handleMultiGen}
        disabled={loading}
        style={{ width: "100%", justifyContent: "center", marginBottom: 14 }}
      >
        {loading
          ? <><span className="animate-spin" style={{ display: "inline-block" }}>↻</span> Generating…</>
          : "⚗ Generate 5 Diverse Samples"}
      </button>

      {error && <p style={{ color: "var(--red)", fontSize: "0.78rem", marginBottom: 8 }}>{error}</p>}

      {samples.length > 0 && (
        <div style={{ overflowX: "auto" }}>
          <table className="data-table">
            <thead>
              <tr>
                <th>#</th>
                {Object.keys(COLS).map(k => <th key={k}>{COLS[k]}</th>)}
              </tr>
            </thead>
            <tbody>
              {samples.map((s) => (
                <tr key={s.index}>
                  <td style={{ fontWeight: 600, color: "var(--accent)" }}>{s.index + 1}</td>
                  {Object.keys(COLS).map(k => (
                    <td key={k} className="num">
                      {typeof s[k] === "number" ? s[k].toFixed(2) : s[k]}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {samples.length === 0 && !loading && (
        <div style={{
          textAlign: "center", padding: "24px",
          color: "var(--text-light)", fontSize: "0.78rem",
        }}>
          Click generate to produce 5 diverse crystal samples
        </div>
      )}
    </div>
  );
}
