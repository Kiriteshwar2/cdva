// ValidationPanel.jsx — Validate a generated CIF structure
import { useState } from "react";
import axios from "axios";

const API = "http://localhost:8000";

export default function ValidationPanel({ generatedData, onLog }) {
  const [result, setResult]   = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState("");

  const handleValidate = async () => {
    if (!generatedData?.cif_base64) { setError("Generate a crystal first!"); return; }
    setLoading(true); setError("");
    try {
      const cif_text = atob(generatedData.cif_base64);
      const res = await axios.post(`${API}/validate`, { cif_text });
      setResult(res.data);
      onLog(`Validated → ${res.data.is_valid ? "Valid" : "Invalid"} | dist=${res.data.min_distance}Å`);
    } catch (e) {
      const msg = e.response?.data?.detail || e.message;
      setError(msg); onLog(`Validation error: ${msg}`);
    } finally { setLoading(false); }
  };

  return (
    <div className="card" style={{ display:"flex", flexDirection:"column", gap:"1rem" }}>
      <h2 style={{ color:"var(--accent)", fontWeight:700, fontSize:"1.05rem" }}>🔍 Validate Structure</h2>

      <button className="btn-outline" onClick={handleValidate}
        disabled={loading || !generatedData}>
        {loading ? "Validating…" : "Validate Structure"}
      </button>

      {!generatedData && <p style={{ color:"#6b7280", fontSize:"0.82rem" }}>Generate a crystal first.</p>}
      {error && <p style={{ color:"#f87171", fontSize:"0.82rem" }}>{error}</p>}

      {result && (
        <div style={{ display:"flex", flexDirection:"column", gap:"0.75rem" }}>
          <div style={{ textAlign:"center" }}>
            {result.is_valid
              ? <span className="badge-valid">Valid Structure</span>
              : <span className="badge-invalid">Invalid Structure</span>}
          </div>
          <div style={{ display:"grid", gridTemplateColumns:"repeat(4,1fr)", gap:"0.5rem" }}>
            {[
              ["Min Distance", `${result.min_distance} Å`, result.min_distance > 0.5 ? "#6ee7b7" : "#f87171"],
              ["Density",      `${result.density} g/cm³`,  "var(--accent)"],
              ["Volume",       `${result.volume} Å³`,       "var(--accent)"],
              ["Atoms",         result.num_atoms,            "var(--accent)"],
            ].map(([label, val, color]) => (
              <div className="metric-card" key={label}>
                <div className="metric-label">{label}</div>
                <div className="metric-value" style={{ color }}>{val}</div>
              </div>
            ))}
          </div>
          <p style={{ textAlign:"center", fontSize:"0.82rem", color:"#d1d5db" }}>{result.message}</p>
        </div>
      )}
    </div>
  );
}
