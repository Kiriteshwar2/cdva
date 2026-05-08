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
    <div className="card" style={{ display: "flex", flexDirection: "column", gap: 0 }}>
      <div className="card-title">
        <span className="card-title-icon">▣</span>
        Upload Dataset
      </div>

      {/* Drop zone */}
      <label className="upload-zone" style={{ cursor: "pointer" }}>
        <div className="upload-zone-icon">📂</div>
        <div className="upload-zone-text">
          {file
            ? <><strong style={{ color: "var(--accent)" }}>{file.name}</strong><br /><span>Click to change file</span></>
            : <><strong style={{ color: "var(--accent)" }}>Click to upload</strong> or drag & drop<br />CSV files only</>
          }
        </div>
        <input
          type="file" accept=".csv"
          style={{ display: "none" }}
          onChange={(e) => setFile(e.target.files[0])}
        />
      </label>

      <button
        className="btn-primary"
        onClick={handleUpload}
        disabled={!file || loading}
        style={{ width: "100%", justifyContent: "center", marginTop: 12 }}
      >
        {loading ? "Uploading…" : "Upload Dataset"}
      </button>

      {error && <p style={{ color: "var(--red)", fontSize: "0.78rem", marginTop: 8 }}>{error}</p>}

      {preview && (
        <div style={{ overflowX: "auto", marginTop: 12 }}>
          <p style={{ fontSize: "0.72rem", color: "var(--text-muted)", marginBottom: 6 }}>
            {rows} rows · showing first 5:
          </p>
          <table className="data-table">
            <thead>
              <tr>{Object.keys(preview[0]).map(k => <th key={k}>{k}</th>)}</tr>
            </thead>
            <tbody>
              {preview.map((row, i) => (
                <tr key={i}>
                  {Object.values(row).map((v, j) => (
                    <td key={j}>{String(v).slice(0, 60)}</td>
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
