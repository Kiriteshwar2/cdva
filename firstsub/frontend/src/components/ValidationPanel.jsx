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
    <div className="card" style={{ display: "flex", flexDirection: "column", gap: 0 }}>
      <div className="card-title">
        <span className="card-title-icon">🔍</span>
        Validate Structure
      </div>

      <button
        className="btn-outline"
        onClick={handleValidate}
        disabled={loading || !generatedData}
        style={{ width: "100%", justifyContent: "center", marginBottom: 12 }}
      >
        {loading ? "Validating…" : "Validate Structure"}
      </button>

      {!generatedData && (
        <p style={{ color: "var(--text-light)", fontSize: "0.78rem" }}>
          Generate a crystal first.
        </p>
      )}
      {error && <p style={{ color: "var(--red)", fontSize: "0.78rem" }}>{error}</p>}

      {result && (
        <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
          {/* Status */}
          <div style={{ textAlign: "center" }}>
            {result.is_valid
              ? <span className="badge-success">✓ Valid Structure</span>
              : <span className="badge-error">✗ Invalid Structure</span>}
          </div>

          {/* Metrics */}
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
            {[
              ["Min Distance", `${result.min_distance} Å`, result.min_distance > 0.5 ? "var(--green)" : "var(--red)"],
              ["Density",      `${result.density} g/cm³`,  "var(--accent)"],
              ["Volume",       `${result.volume} Å³`,       "var(--accent)"],
              ["Atoms",         result.num_atoms,            "var(--accent)"],
            ].map(([label, val, color]) => (
              <div key={label} style={{
                background: "var(--content-bg)",
                border: "1px solid var(--border)",
                borderRadius: 8, padding: "10px 12px", textAlign: "center",
              }}>
                <div style={{ fontSize: "0.68rem", color: "var(--text-muted)", marginBottom: 4 }}>{label}</div>
                <div style={{ fontSize: "0.9rem", fontWeight: 700, color }}>{val}</div>
              </div>
            ))}
          </div>

          <p style={{ textAlign: "center", fontSize: "0.78rem", color: "var(--text-muted)" }}>
            {result.message}
          </p>
        </div>
      )}
    </div>
  );
}
