// UploadPanel.jsx — Upload a CSV dataset and preview the first 5 rows
import { useState } from "react";
import axios from "axios";

const API = "http://localhost:8000";

export default function UploadPanel({ onLog }) {
  const [file, setFile]       = useState(null);
  const [preview, setPreview] = useState(null);
  const [rows, setRows]       = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState("");

  const handleUpload = async () => {
    if (!file) return;
    setLoading(true); setError("");
    const form = new FormData();
    form.append("file", file);
    try {
      const res = await axios.post(`${API}/upload`, form);
      setPreview(res.data.preview);
      setRows(res.data.rows);
      onLog(`Uploaded "${file.name}" — ${res.data.rows} rows`);
    } catch (e) {
      const msg = e.response?.data?.detail || e.message;
      setError(msg); onLog(`Upload error: ${msg}`);
    } finally { setLoading(false); }
  };

  return (
    <div className="card" style={{ display:"flex", flexDirection:"column", gap:"0.75rem" }}>
      <h2 style={{ color:"var(--accent)", fontWeight:700, fontSize:"1.05rem" }}>📂 Upload Dataset</h2>

      <div style={{ display:"flex", gap:"0.5rem", alignItems:"center", flexWrap:"wrap" }}>
        <label className="btn-outline" style={{ cursor:"pointer", fontSize:"0.85rem" }}>
          Choose CSV
          <input type="file" accept=".csv" style={{ display:"none" }}
            onChange={(e) => setFile(e.target.files[0])} />
        </label>
        {file && <span style={{ color:"#d1d5db", fontSize:"0.82rem" }}>{file.name}</span>}
        <button className="btn-primary" style={{ fontSize:"0.85rem" }}
          onClick={handleUpload} disabled={!file || loading}>
          {loading ? "Uploading…" : "Upload"}
        </button>
      </div>

      {error && <p style={{ color:"#f87171", fontSize:"0.82rem" }}>{error}</p>}

      {preview && (
        <div style={{ overflowX:"auto" }}>
          <p style={{ color:"#9ca3af", fontSize:"0.7rem", marginBottom:"0.25rem" }}>
            {rows} rows · showing first 5:
          </p>
          <table className="dark-table">
            <thead>
              <tr>{Object.keys(preview[0]).map(k =>
                <th key={k}>{k}</th>)}</tr>
            </thead>
            <tbody>
              {preview.map((row, i) => (
                <tr key={i}>{Object.values(row).map((v, j) =>
                  <td key={j}>{String(v).slice(0,60)}</td>)}</tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
